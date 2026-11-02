import time

import pytest

from llm_energy_rl.fake_nvml import FakeNvml
from llm_energy_rl.gpu import GpuError, NvmlGpu


def test_clock_knob_is_preferred_and_left_unlocked_after_the_self_test():
    nv = FakeNvml()
    gpu = NvmlGpu(nvml=nv)
    assert gpu.knob == "clock" and gpu.usable_knobs == ["clock", "power"]
    assert nv.locked is None 


def test_cap_locks_a_range_and_release_unlocks_once():
    nv = FakeNvml()
    gpu = NvmlGpu(nvml=nv)
    nv.calls.clear()
    gpu.set_cap(1200)
    assert nv.locked == (210, 1200)
    gpu.set_cap(None)
    gpu.release()
    gpu.release()
    assert nv.locked is None
    assert nv.calls == [("lock", 210, 1200), ("unlock",)]


def test_falls_back_to_an_exact_lock_when_ranges_are_refused():
    nv = FakeNvml(range_lock=False)
    gpu = NvmlGpu(nvml=nv)
    gpu.set_cap(1300)
    assert nv.locked == (1300, 1300) and gpu.lock_mode == "exact"


def test_power_knob_is_used_when_clock_locking_is_unsupported():
    nv = FakeNvml(clock_lock=False, power_limit_w=300.0)
    gpu = NvmlGpu(nvml=nv)
    assert gpu.knob == "power" and "clock" in gpu.knob_errors
    gpu.set_cap(180)
    assert nv.limit_w == 180.0
    gpu.release()
    assert nv.limit_w == 300.0 


def test_missing_permission_is_reported_clearly():
    gpu = NvmlGpu(nvml=FakeNvml(permission=False))
    assert gpu.knob == "none"
    with pytest.raises(GpuError, match="administrator"):
        gpu.set_cap(1200)
    assert gpu.energy_j() >= 0.0 


def test_energy_counter_grows_with_time_and_shrinks_with_the_cap():
    nv = FakeNvml()
    gpu = NvmlGpu(nvml=nv)

    def joules(seconds):
        start = gpu.energy_j()
        time.sleep(seconds)
        return gpu.energy_j() - start

    free = joules(0.2)
    gpu.set_cap(950)
    capped = joules(0.2)
    gpu.release()
    assert free == pytest.approx(280.0 * 0.2, rel=0.2)
    assert capped < 0.6 * free


def test_energy_is_integrated_from_power_without_a_counter():
    gpu = NvmlGpu(nvml=FakeNvml(energy_counter=False))
    assert not gpu.has_energy_counter
    start = gpu.energy_j()
    time.sleep(0.3)
    assert gpu.energy_j() - start == pytest.approx(280.0 * 0.3, rel=0.2)
    gpu.close()


def test_describe_lists_what_probe_prints():
    d = NvmlGpu(nvml=FakeNvml()).describe()
    assert d["name"].startswith("Fake") and d["knob"] == "clock"
    assert d["power_limits_w"]["min"] == 100.0 and d["max_clock_mhz"] == 1900


@pytest.mark.parametrize("fake, code, text", [
    (dict(), 0, "Ready"),
    (dict(permission=False), 2, "administrator"),
    (dict(clock_lock=False), 0, "cap the GPU by power"),
])
def test_probe_command_explains_what_it_found(monkeypatch, capsys, fake, code, text):
    import sys

    from llm_energy_rl.__main__ import main

    monkeypatch.setitem(sys.modules, "pynvml", FakeNvml(**fake))
    assert main(["probe"]) == code
    assert text in capsys.readouterr().out
