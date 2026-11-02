from __future__ import annotations

import atexit
import threading
import time


class GpuError(RuntimeError):
    pass


def _text(value) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


class _PowerIntegrator(threading.Thread):

    def __init__(self, read_watts, hz: float = 20.0):
        super().__init__(daemon=True)
        self._read, self._period = read_watts, 1.0 / hz
        self._lock = threading.Lock()
        self._joules, self._t, self._w = 0.0, time.perf_counter(), read_watts()
        self._stop = threading.Event()
        self.start()

    def run(self) -> None:
        while not self._stop.wait(self._period):
            self._advance()

    def _advance(self) -> None:
        w, t = self._read(), time.perf_counter()
        with self._lock:
            self._joules += 0.5 * (w + self._w) * (t - self._t)
            self._t, self._w = t, w

    def energy_j(self) -> float:
        self._advance()
        with self._lock:
            return self._joules

    def stop(self) -> None:
        self._stop.set()


class NvmlGpu:
    def __init__(
        self,
        index: int = 0,
        knob: str = "auto",
        lock_mode: str = "range",
        min_clock_mhz: int = 210,
        nvml=None,
    ):
        if nvml is None:
            try:
                import pynvml as nvml
            except ImportError as err:
                raise GpuError("The NVML bindings are missing: pip install nvidia-ml-py") from err
        self.nv = nvml
        try:
            nvml.nvmlInit()
            self.handle = nvml.nvmlDeviceGetHandleByIndex(index)
        except nvml.NVMLError as err:
            raise GpuError(f"Cannot reach an NVIDIA GPU through NVML: {err}") from err
        self.name = _text(nvml.nvmlDeviceGetName(self.handle))
        self.driver = _text(nvml.nvmlSystemGetDriverVersion())
        self.lock_mode = lock_mode
        self.min_clock_mhz = min_clock_mhz
        self._capped = False
        self._power_limit_mw = self._try(nvml.nvmlDeviceGetPowerManagementLimit)
        self.has_energy_counter = self._try(nvml.nvmlDeviceGetTotalEnergyConsumption) is not None
        self._integrator = None
        if not self.has_energy_counter:
            try:
                self._integrator = _PowerIntegrator(self.power_w)
            except nvml.NVMLError as err:
                raise GpuError(f"{self.name} reports neither energy nor power through NVML: {err}") from err
        self.knob_errors: dict[str, str] = {}
        self.knob = self._choose_knob(knob)
        atexit.register(self.release)

    def _try(self, fn, *args):
        try:
            return fn(self.handle, *args)
        except self.nv.NVMLError:
            return None

    def energy_j(self) -> float:
        if self._integrator is not None:
            return self._integrator.energy_j()
        return self.nv.nvmlDeviceGetTotalEnergyConsumption(self.handle) / 1000.0

    def power_w(self) -> float:
        return self.nv.nvmlDeviceGetPowerUsage(self.handle) / 1000.0

    def clock_mhz(self) -> float | None:
        return self._try(self.nv.nvmlDeviceGetClockInfo, self.nv.NVML_CLOCK_GRAPHICS)

    def max_clock_mhz(self) -> float | None:
        return self._try(self.nv.nvmlDeviceGetMaxClockInfo, self.nv.NVML_CLOCK_GRAPHICS)

    def telemetry(self) -> dict:
        util = self._try(self.nv.nvmlDeviceGetUtilizationRates)
        return {
            "clock_mhz": self.clock_mhz(),
            "temp_c": self._try(self.nv.nvmlDeviceGetTemperature, self.nv.NVML_TEMPERATURE_GPU),
            "util_pct": getattr(util, "gpu", None),
        }

    def power_limits_w(self) -> dict:
        nv = self.nv
        bounds = self._try(nv.nvmlDeviceGetPowerManagementLimitConstraints)
        default = self._try(nv.nvmlDeviceGetPowerManagementDefaultLimit)
        return {
            "current": None if self._power_limit_mw is None else self._power_limit_mw / 1000.0,
            "default": None if default is None else default / 1000.0,
            "min": None if bounds is None else bounds[0] / 1000.0,
            "max": None if bounds is None else bounds[1] / 1000.0,
        }

    def energy_counter_period_s(self, seconds: float = 1.5) -> float | None:
        if not self.has_energy_counter:
            return None
        changes, last = [], self.energy_j()
        end = time.perf_counter() + seconds
        while time.perf_counter() < end:
            value = self.energy_j()
            if value != last:
                changes.append(time.perf_counter())
                last = value
            time.sleep(0.002)
        gaps = sorted(b - a for a, b in zip(changes, changes[1:]))
        return gaps[len(gaps) // 2] if gaps else None

    def _choose_knob(self, wanted: str) -> str:
        nv = self.nv
        usable = []
        for knob, test in (("clock", self._test_clock), ("power", self._test_power)):
            try:
                test()
                usable.append(knob)
            except nv.NVMLError_NoPermission:
                self.knob_errors[knob] = "no permission (start the terminal as administrator / use sudo)"
            except nv.NVMLError as err:
                self.knob_errors[knob] = str(err)
        self.usable_knobs = usable
        if wanted == "auto":
            return usable[0] if usable else "none"
        return wanted if wanted in usable else "none"

    def _test_clock(self) -> None:
        nv, top = self.nv, self.max_clock_mhz()
        candidates = ([top, round(0.8 * top)] if top else []) + [1500, 1200, 1000]
        error = None
        for mhz in candidates:
            try:
                self._lock_clock(mhz)
                nv.nvmlDeviceResetGpuLockedClocks(self.handle)
                return
            except (nv.NVMLError_NoPermission, nv.NVMLError_NotSupported):
                raise
            except nv.NVMLError as err:
                error = err
        raise error

    def _test_power(self) -> None:
        if self._power_limit_mw is None:
            raise self.nv.NVMLError(self.nv.NVML_ERROR_NOT_SUPPORTED)
        self.nv.nvmlDeviceSetPowerManagementLimit(self.handle, self._power_limit_mw)

    def _lock_clock(self, mhz: float) -> None:
        nv, mhz = self.nv, int(round(mhz))
        lock = nv.nvmlDeviceSetGpuLockedClocks
        if self.lock_mode == "range":  # may run slower than the cap when idle, never faster
            try:
                lock(self.handle, min(self.min_clock_mhz, mhz), mhz)
                return
            except nv.NVMLError_NoPermission:
                raise
            except nv.NVMLError as refused:
                try:
                    lock(self.handle, mhz, mhz)
                except nv.NVMLError:
                    raise refused from None
                self.lock_mode = "exact"  # this driver only accepts min == max
                return
        lock(self.handle, mhz, mhz)

    def require_knob(self) -> None:
        if self.knob == "none":
            reasons = "; ".join(f"{k}: {v}" for k, v in self.knob_errors.items()) or "nothing usable"
            raise GpuError(
                f"Cannot cap {self.name} ({reasons}). On Windows open PowerShell with "
                "'Run as administrator'; on Linux run the command with sudo."
            )

    def set_cap(self, cap: float | None) -> None:
        if cap is None:
            self.release()
            return
        self.require_knob()
        if self.knob == "clock":
            self._lock_clock(cap)
        else:
            self.nv.nvmlDeviceSetPowerManagementLimit(self.handle, int(round(cap * 1000)))
        self._capped = True

    def release(self) -> None:
        if not self._capped:
            return
        try:
            if self.knob == "clock":
                self.nv.nvmlDeviceResetGpuLockedClocks(self.handle)
            elif self._power_limit_mw is not None:
                self.nv.nvmlDeviceSetPowerManagementLimit(self.handle, self._power_limit_mw)
            self._capped = False
        except self.nv.NVMLError:
            pass

    def close(self) -> None:
        self.release()
        if self._integrator is not None:
            self._integrator.stop()

    def describe(self) -> dict:
        return {
            "name": self.name,
            "driver": self.driver,
            "energy_counter": self.has_energy_counter,
            "power_w": (self._try(self.nv.nvmlDeviceGetPowerUsage) or 0) / 1000.0,
            "clock_mhz": self.clock_mhz(),
            "max_clock_mhz": self.max_clock_mhz(),
            "power_limits_w": self.power_limits_w(),
            "usable_knobs": self.usable_knobs,
            "knob": self.knob,
            "knob_errors": self.knob_errors,
            **self.telemetry(),
        }
