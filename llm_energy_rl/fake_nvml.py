from __future__ import annotations

import time
from types import SimpleNamespace


class NVMLError(Exception):
    def __init__(self, code=0):
        super().__init__(f"fake NVML error {code}")
        self.value = code


class NVMLError_NoPermission(NVMLError):
    pass


class NVMLError_NotSupported(NVMLError):
    pass


class FakeNvml:
    NVMLError = NVMLError
    NVMLError_NoPermission = NVMLError_NoPermission
    NVMLError_NotSupported = NVMLError_NotSupported
    NVML_ERROR_NOT_SUPPORTED = 3
    NVML_CLOCK_GRAPHICS = 0
    NVML_TEMPERATURE_GPU = 0

    def __init__(
        self,
        permission: bool = True,
        clock_lock: bool = True,
        energy_counter: bool = True,
        range_lock: bool = True,
        range_runs_low: bool = False,
        max_clock: int = 1900,
        power_limit_w: float = 320.0,
    ):
        self.permission, self.clock_lock = permission, clock_lock
        self.energy_counter, self.range_lock = energy_counter, range_lock
        self.range_runs_low = range_runs_low 
        self.max_clock, self.default_limit = max_clock, power_limit_w
        self.locked: tuple[int, int] | None = None
        self.limit_w = power_limit_w
        self.calls: list[tuple] = []
        self._joules, self._t = 0.0, time.perf_counter()

    def _watts(self) -> float:
        clock = self.locked[1] if self.locked else self.max_clock
        return min(self.limit_w, 30.0 + 250.0 * (0.15 + 0.85 * (clock / self.max_clock) ** 2.3))

    def _advance(self) -> None:
        now = time.perf_counter()
        self._joules += self._watts() * (now - self._t)
        self._t = now

    def _write(self) -> None:
        if not self.permission:
            raise NVMLError_NoPermission(4)
        self._advance()

    def nvmlInit(self):
        pass

    def nvmlShutdown(self):
        pass

    def nvmlDeviceGetHandleByIndex(self, index):
        return index

    def nvmlDeviceGetName(self, handle):
        return "Fake GeForce RTX 3080"

    def nvmlSystemGetDriverVersion(self):
        return "0.0-fake"

    def nvmlDeviceGetTotalEnergyConsumption(self, handle):
        if not self.energy_counter:
            raise NVMLError_NotSupported(3)
        self._advance()
        return int(self._joules * 1000)

    def nvmlDeviceGetPowerUsage(self, handle):
        return int(self._watts() * 1000)

    def nvmlDeviceGetClockInfo(self, handle, kind):
        if not self.locked:
            return self.max_clock
        return self.locked[0] if self.range_runs_low else self.locked[1]

    def nvmlDeviceGetMaxClockInfo(self, handle, kind):
        return self.max_clock

    def nvmlDeviceGetTemperature(self, handle, sensor):
        return 55

    def nvmlDeviceGetUtilizationRates(self, handle):
        return SimpleNamespace(gpu=50, memory=20)

    def nvmlDeviceGetPowerManagementLimit(self, handle):
        return int(self.limit_w * 1000)

    def nvmlDeviceGetPowerManagementDefaultLimit(self, handle):
        return int(self.default_limit * 1000)

    def nvmlDeviceGetPowerManagementLimitConstraints(self, handle):
        return (100_000, int(self.default_limit * 1000 * 1.05))

    def nvmlDeviceSetGpuLockedClocks(self, handle, lo, hi):
        self._write()
        if not self.clock_lock:
            raise NVMLError_NotSupported(3)
        if lo != hi and not self.range_lock:
            raise NVMLError(2)
        self.locked = (int(lo), int(hi))
        self.calls.append(("lock", int(lo), int(hi)))

    def nvmlDeviceResetGpuLockedClocks(self, handle):
        self._write()
        self.locked = None
        self.calls.append(("unlock",))

    def nvmlDeviceSetPowerManagementLimit(self, handle, milliwatts):
        self._write()
        self.limit_w = milliwatts / 1000.0
        self.calls.append(("limit", self.limit_w))
