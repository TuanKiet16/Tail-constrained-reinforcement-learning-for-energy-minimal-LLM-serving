import json

import pytest

pytest.importorskip("torch")
pytest.importorskip("stable_baselines3")

from llm_energy_rl import pipeline  # noqa: E402
from llm_energy_rl.__main__ import main  # noqa: E402
from llm_energy_rl.agents import load_policy  # noqa: E402
from llm_energy_rl.config import Config  # noqa: E402
from llm_energy_rl.experiment import make_sim_env, train_sb3  # noqa: E402
from llm_energy_rl.fake_nvml import FakeNvml  # noqa: E402
from llm_energy_rl.profile import synthetic_profile  # noqa: E402


def test_demo_writes_policies_results_and_figures(tmp_path):
    code = main(["demo", "--out", str(tmp_path), "--episodes", "2", "--episode-s", "40", "--quick",
                 "--algos", "qlearning", "dqn", "ppo"])
    assert code == 0
    for name in ("profile.json", "qlearning.json", "dqn.zip", "ppo.zip", "eval_sim.json",
                 "curve_qlearning_sim.csv", "curve_dqn_sim.csv", "curve_ppo_sim.csv"):
        assert (tmp_path / name).exists(), name
    for name in ("summary.md", "energy.png", "latency.png", "learning.png", "profile.png",
                 "timeline_qlearning.png", "timeline_dqn.png", "timeline_ppo.png"):
        assert (tmp_path / "report_sim" / name).stat().st_size > 0, name
    result = json.loads((tmp_path / "eval_sim.json").read_text())
    assert set(result["policies"]) == {"default", "qlearning", "dqn", "ppo"}
    assert result["policies"]["default"]["energy_saving_pct"] == pytest.approx(0.0)
    summary = (tmp_path / "report_sim" / "summary.md").read_text(encoding="utf-8")
    assert "placeholder" in summary and "No RL (driver default)" in summary
    # the report can be rebuilt from the saved files alone
    assert main(["report", "--dir", str(tmp_path), "--tag", "sim"]) == 0


def test_saved_policies_load_and_act(tmp_path):
    cfg = Config(episode_s=20.0)
    profile = synthetic_profile(cfg.cap_fractions)
    for algo in pipeline.ALGOS:
        pipeline.train(algo, pipeline.sim_envs(algo, profile, cfg, 0), 2_500, tmp_path, say=lambda *_: None)
        policy = load_policy(f"{algo}:{pipeline.policy_file(tmp_path, algo)}", profile.n_levels)
        obs, _ = make_sim_env(profile, cfg).reset(seed=0)
        assert 0 <= policy.act(obs) < profile.n_levels


def test_ppo_trained_on_four_copies_continues_on_one(tmp_path):
    cfg = Config(episode_s=20.0)
    profile = synthetic_profile(cfg.cap_fractions)
    model = train_sb3("ppo", pipeline.sim_envs("ppo", profile, cfg, 0), 2_100)
    before = model.num_timesteps
    single = make_sim_env(profile, cfg)
    model = train_sb3("ppo", single, 600, model=model)
    assert model.n_envs == 1 and model.num_timesteps > before
    assert single.reward_scale == 1.0  # the training-time scaling is undone


def test_dry_run_exercises_the_gpu_path_and_frees_the_gpu(tmp_path, monkeypatch):
    fakes = []

    def fake_nvml(*args, **kwargs):
        fakes.append(FakeNvml(*args, **kwargs))
        return fakes[-1]

    monkeypatch.setattr("llm_energy_rl.fake_nvml.FakeNvml", fake_nvml)
    code = main(["run", "--dry-run", "--out", str(tmp_path), "--algos", "qlearning", "dqn", "--episodes", "1",
                 "--episode-s", "6", "--real-steps", "40"])
    assert code == 0
    run = tmp_path / "tiny-random"
    result = json.loads((run / "eval_real.json").read_text())
    assert set(result["policies"]) == {"default", "qlearning", "dqn"}
    for p in result["policies"].values():
        assert p["mean"]["completed"] == p["mean"]["requests"] > 0
    assert (run / "curve_qlearning_real.csv").exists() and (run / "curve_dqn_real.csv").exists()
    assert (run / "report_real" / "energy.png").exists() and (tmp_path / "models.png").exists()
    assert "dry run" in (run / "report_real" / "summary.md").read_text(encoding="utf-8")
    assert fakes and all(f.locked is None for f in fakes)
    assert any(c[0] == "lock" for c in fakes[-1].calls)


def test_run_reports_missing_admin_rights_instead_of_crashing(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("llm_energy_rl.fake_nvml.FakeNvml", lambda *a, **k: FakeNvml(permission=False))
    code = main(["run", "--dry-run", "--out", str(tmp_path), "--algos", "qlearning", "--episodes", "1"])
    assert code == 2
    assert "administrator" in capsys.readouterr().out
