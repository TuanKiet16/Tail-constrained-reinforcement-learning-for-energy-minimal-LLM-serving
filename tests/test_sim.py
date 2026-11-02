import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from llm_energy_rl.agents import FixedPolicy, QLearningAgent
from llm_energy_rl.backends import SimBackend
from llm_energy_rl.config import Config, WorkloadConfig, load_config
from llm_energy_rl.env import OBS_NAMES, LLMServingEnv, Scales
from llm_energy_rl.experiment import evaluate, make_sim_env, run_episode, train_qlearning
from llm_energy_rl.profile import HardwareProfile, synthetic_profile
from llm_energy_rl.server import Server
from llm_energy_rl.workload import Request, make_trace, mean_lengths


@pytest.fixture
def cfg():
    return Config(episode_s=60.0)


@pytest.fixture
def profile(cfg):
    return synthetic_profile(cfg.cap_fractions)


def test_trace_is_reproducible_and_well_formed():
    wl = WorkloadConfig()
    a, _ = make_trace(wl, 300.0, 4.0, seed=3)
    b, _ = make_trace(wl, 300.0, 4.0, seed=3)
    c, _ = make_trace(wl, 300.0, 4.0, seed=4)
    assert [(r.arrival, r.prompt_len, r.out_len) for r in a] == [(r.arrival, r.prompt_len, r.out_len) for r in b]
    assert [r.arrival for r in a] != [r.arrival for r in c]
    assert all(x.arrival <= y.arrival for x, y in zip(a, a[1:]))
    assert all(0 <= r.arrival < 300.0 for r in a)
    assert all(wl.prompt_min <= r.prompt_len <= wl.prompt_max for r in a)
    assert all(wl.output_min <= r.out_len <= wl.output_max for r in a)


def test_trace_rate_matches_the_regime_mix():
    wl = WorkloadConfig()
    dwell = np.array([r["mean_dwell_s"] for r in wl.regimes])
    load = np.array([r["load"] for r in wl.regimes])
    expected = float((dwell * load).sum() / dwell.sum()) * 5.0
    counts = [len(make_trace(wl, 4000.0, 5.0, seed=s)[0]) / 4000.0 for s in range(8)]
    assert np.mean(counts) == pytest.approx(expected, rel=0.1)


def test_idle_gpu_draws_idle_power(profile, cfg):
    quiet = Config(episode_s=30.0, workload=WorkloadConfig(regimes=[{"name": "off", "load": 0.0, "mean_dwell_s": 10.0}]))
    env = LLMServingEnv(SimBackend(profile), profile, quiet, drain=True)
    m = run_episode(env, FixedPolicy(profile.top), trace_seed=1)["metrics"]
    assert m["requests"] == 0
    assert m["energy_j"] == pytest.approx(profile.idle_w[profile.top] * 30.0)


def test_single_request_energy_and_latency_by_hand(profile):
    """One request on an idle server: every number can be computed from the profile."""
    backend, level = SimBackend(profile), 2
    backend.set_level(level)
    assert backend.now() == pytest.approx(profile.switch_s) 
    t0, e0 = backend.now(), backend.energy_j()
    server = Server(backend, max_batch=8)
    req = Request(rid=0, arrival=1.0, prompt_len=100, out_len=11)
    server.reset([req], t0)
    server.run_until(t0 + 5.0)
    prefill = profile.prefill_time(level, 100)
    step = profile.decode_time(level, 1)
    assert req.first_token_t == pytest.approx(t0 + 1.0 + prefill)
    assert req.finish_t == pytest.approx(t0 + 1.0 + prefill + 10 * step)
    busy = profile.prefill_w[level] * prefill + profile.decode_power(level, 1) * 10 * step
    idle = profile.idle_w[level] * (5.0 - prefill - 10 * step)
    assert backend.energy_j() - e0 == pytest.approx(busy + idle)


def test_every_request_is_served_in_full_and_batch_limit_holds(profile, cfg):
    peak = {"batch": 0}

    class Watch(SimBackend):
        def prefill(self, requests):
            super().prefill(requests)
            peak["batch"] = max(peak["batch"], self.batch)

    env = LLMServingEnv(Watch(profile, noise=0.03, seed=0), profile, cfg, drain=True)
    out = run_episode(env, FixedPolicy(0), trace_seed=11)
    assert env.server.all_done
    assert len(env.trace) > 20
    for r in env.trace:
        assert r.n_out == r.out_len
        assert r.first_token_t >= env.t0 + r.arrival
        assert r.finish_t >= r.first_token_t
    assert 1 <= peak["batch"] <= cfg.max_batch
    assert out["metrics"]["tokens"] == sum(r.out_len for r in env.trace)
    assert out["metrics"]["completed"] == len(env.trace)


def test_environment_follows_the_gymnasium_api(profile, cfg):
    env = make_sim_env(profile, cfg)
    check_env(env, skip_render_check=True)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (len(OBS_NAMES),)
    for _ in range(200):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        assert env.observation_space.contains(obs)
        assert reward <= 0.0 and not terminated
        if truncated:
            break
    assert truncated and "metrics" in info


def test_reward_is_energy_plus_weighted_latency_pressure(profile, cfg):
    env = make_sim_env(profile, cfg)
    env.reset(seed=2)
    scales = Scales.resolve(profile, cfg)
    seen = 0.0
    for _ in range(60):
        _, reward, _, _, info = env.step(0)
        expected = -info["energy_j"] / (scales.p_ref * cfg.interval_s) - cfg.slo_weight * info["violation"]
        assert reward == pytest.approx(expected)
        seen = max(seen, info["violation"])
    assert seen > 0.0 


def test_headroom_makes_the_penalty_start_earlier(profile):
    def total_violation(headroom):
        env = make_sim_env(profile, Config(episode_s=60.0, headroom=headroom, sim_noise=0.0))
        env.reset(options={"trace_seed": 3})
        return sum(env.step(1)[4]["violation"] for _ in range(60))

    assert total_violation(0.2) > total_violation(0.0)


def test_lower_cap_trades_latency_for_energy(profile, cfg):
    env = make_sim_env(profile, cfg, drain=True)
    result = evaluate(env, {"default": FixedPolicy(profile.top, "default"), "lowest": FixedPolicy(0, "lowest")}, episodes=2)
    low, base = result["policies"]["lowest"], result["policies"]["default"]
    assert low["energy_saving_pct"] > 10.0
    assert low["mean"]["tpot_mean_s"] > base["mean"]["tpot_mean_s"]
    assert low["mean"]["slo_both"] < base["mean"]["slo_both"]
    assert base["energy_saving_pct"] == pytest.approx(0.0)
    # identical work in both arms
    assert low["mean"]["tokens"] == base["mean"]["tokens"]


def test_qlearning_learns_to_save_energy_without_missing_targets(profile):
    cfg = Config(episode_s=120.0)
    agent = train_qlearning(make_sim_env(profile, cfg, seed=0), steps=100_000, seed=0)
    result = evaluate(make_sim_env(profile, cfg, seed=1), {"default": FixedPolicy(profile.top, "default"), "qlearning": agent}, episodes=6)
    rl, base = result["policies"]["qlearning"], result["policies"]["default"]
    assert rl["energy_saving_pct"] > 8.0
    assert rl["mean"]["slo_both"] > base["mean"]["slo_both"] - 0.02
    assert rl["mean"]["return"] > base["mean"]["return"]


def test_qlearning_table_survives_a_round_trip(tmp_path, profile, cfg):
    agent = train_qlearning(make_sim_env(profile, cfg), steps=2_000, seed=0)
    agent.save(tmp_path / "q.json")
    again = QLearningAgent.load(tmp_path / "q.json")
    assert np.array_equal(agent.q, again.q)
    obs = np.linspace(0, 1, len(OBS_NAMES), dtype=np.float32)
    assert agent.act(obs) == again.act(obs)


def test_unvisited_state_falls_back_to_default_clocks(profile):
    agent = QLearningAgent(profile.n_levels)
    assert agent.act(np.zeros(len(OBS_NAMES), dtype=np.float32)) == profile.top


def test_profile_round_trip_and_interpolation(tmp_path, profile):
    profile.save(tmp_path / "p.json")
    again = HardwareProfile.load(tmp_path / "p.json")
    assert again.caps == profile.caps and again.labels()[-1] == "default"
    between = again.decode_time(0, 3)
    assert again.decode_time(0, 2) < between < again.decode_time(0, 4)


def test_per_request_overhead_slows_admission_and_lowers_capacity(profile, cfg):
    plain = Scales.resolve(profile, cfg).capacity_rps
    base = profile.prefill_time(0, 100, 2)
    profile.request_overhead_s = 0.01
    assert profile.prefill_time(0, 100, 2) == pytest.approx(base + 0.02)
    assert Scales.resolve(profile, cfg).capacity_rps < plain


def test_config_file_overrides_and_rejects_typos(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("max_batch: 4\nworkload:\n  output_median: 32\n", encoding="utf-8")
    cfg = load_config(path, episode_s=10.0)
    assert (cfg.max_batch, cfg.episode_s, cfg.workload.output_median) == (4, 10.0, 32)
    path.write_text("max_batchh: 4\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(path)


def test_mean_lengths_respect_the_clipping():
    wl = WorkloadConfig(prompt_min=50, prompt_max=60, output_min=5, output_max=6)
    prompt, out = mean_lengths(wl)
    assert 50 <= prompt <= 60 and 5 <= out <= 6
