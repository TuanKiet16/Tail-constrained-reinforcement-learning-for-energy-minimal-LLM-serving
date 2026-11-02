import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from llm_energy_rl.backends import RealBackend  # noqa: E402
from llm_energy_rl.config import Config  # noqa: E402
from llm_energy_rl.engine import HFEngine  # noqa: E402
from llm_energy_rl.env import LLMServingEnv  # noqa: E402
from llm_energy_rl.experiment import run_episode  # noqa: E402
from llm_energy_rl.fake_nvml import FakeNvml  # noqa: E402
from llm_energy_rl.gpu import NvmlGpu  # noqa: E402
from llm_energy_rl.agents import FixedPolicy  # noqa: E402
from llm_energy_rl.measure import measure_profile  # noqa: E402
from llm_energy_rl.workload import Request  # noqa: E402


@pytest.fixture(scope="module")
def engine():
    return HFEngine("tiny-random", device="cpu", record=True)


def reference(model, prompt, n):
    ids, out = prompt[None, :].clone(), []
    with torch.inference_mode():
        for _ in range(n):
            token = model(input_ids=ids).logits[0, -1].argmax()
            out.append(int(token))
            ids = torch.cat([ids, token.view(1, 1)], 1)
    return out


def test_joining_and_leaving_a_batch_does_not_change_any_token(engine):
    reqs = [Request(0, 0, 7, 12), Request(1, 0, 19, 5), Request(2, 0, 3, 20),
            Request(3, 0, 30, 9), Request(4, 0, 11, 14), Request(5, 0, 2, 6)]
    engine.reset()
    engine.outputs.clear()
    active = []

    def retire():
        rows = [i for i, r in enumerate(active) if r.n_out >= r.out_len]
        if rows:
            engine.remove(rows)
            for i in sorted(rows, reverse=True):
                active.pop(i)

    def admit(group):
        engine.prefill(group)
        for r in group:
            r.n_out = 1
        active.extend(group)
        retire()

    def step(k):
        for _ in range(k):  
            if not active:
                break
            done, _ = engine.decode(1, float("inf"))
            for r in active:
                r.n_out += done
            retire()

    admit(reqs[0:2]); step(3); admit(reqs[2:3]); step(3); admit(reqs[3:5]); step(4); admit(reqs[5:6])
    while active:
        k = min(r.out_len - r.n_out for r in active)
        done, _ = engine.decode(k, float("inf")) 
        for r in active:
            r.n_out += done
        retire()
    assert engine.batch == 0
    for r in reqs:
        expected = reference(engine.model, engine.prompt_ids(r.rid, r.prompt_len), r.out_len)
        assert engine.outputs[r.rid] == expected, f"request {r.rid} diverged"


def test_decode_stops_at_the_step_limit_and_at_the_deadline(engine):
    import time

    engine.reset()
    engine.prefill([Request(0, 0, 8, 10**6), Request(1, 0, 8, 10**6)])
    steps, elapsed = engine.decode(5, float("inf"))
    assert steps == 5 and elapsed > 0
    steps, _ = engine.decode(10**6, time.perf_counter() + 0.05)
    assert 1 <= steps < 10**6
    steps, _ = engine.decode(10**6, time.perf_counter() - 1.0)
    assert steps == 1  # a step that has started always completes
    engine.reset()


def test_real_backend_serves_a_trace_and_always_releases_the_gpu(engine):
    cfg = Config(model="tiny-random", episode_s=3.0, interval_s=0.25)
    nv = FakeNvml()
    gpu = NvmlGpu(nvml=nv)
    profile = measure_profile(engine, gpu, cfg, warmup_s=0.3, settle_s=0.05, window_s=0.2, say=lambda *_: None)
    assert profile.caps[-1] is None and len(profile.caps) == cfg.n_levels
    assert profile.caps[0] == pytest.approx(0.5 * 1900)
    assert profile.meta["sustained_tokens_per_s"] > 0 and profile.request_overhead_s >= 0.0
    assert nv.locked is None  # profiling hands the GPU back

    env = LLMServingEnv(RealBackend(engine, gpu, profile.caps), profile, cfg, drain=True)
    out = run_episode(env, FixedPolicy(1), trace_seed=5)
    m = out["metrics"]
    assert m["completed"] == m["requests"] > 0
    assert m["energy_j"] > 0 and m["duration_s"] >= cfg.episode_s
    assert ("lock", 210, int(profile.caps[1])) in nv.calls
    assert all(h["clock_mhz"] is not None for h in out["history"])
    env.close()
    assert nv.locked is None


def test_profiling_pins_the_clock_when_a_range_lock_runs_below_the_cap(engine):
    cfg = Config(model="tiny-random")
    nv = FakeNvml(range_runs_low=True)
    gpu = NvmlGpu(nvml=nv)
    said = []
    profile = measure_profile(engine, gpu, cfg, warmup_s=0.3, settle_s=0.05, window_s=0.2, say=said.append)
    assert gpu.lock_mode == "exact" and profile.meta["lock_mode"] == "exact"
    assert any("pinning the clock" in line for line in said)
    assert not profile.meta["warnings"]
    assert profile.meta["clock_mhz"][0] == pytest.approx(profile.caps[0])
    assert profile.switch_s >= 0.0 and nv.locked is None
