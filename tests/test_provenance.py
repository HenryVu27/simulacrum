"""Results pooled across machines must share a recording and say how they were made (docs/gpu.md, E7, E7b)."""

import json
import shutil
import subprocess

import numpy as np
import pytest

from umwelt import data, experiment, voe


def test_a_job_that_must_not_record_refuses(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "DATA_DIR", tmp_path)
    monkeypatch.setenv("UMWELT_NO_RECORD", "1")
    with pytest.raises(FileNotFoundError, match="UMWELT_NO_RECORD"):
        data.load("lawful", 4, 8, seed=0)
    assert not list(tmp_path.iterdir())  # nothing recorded


def test_recording_hash_matches_the_shell(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "DATA_DIR", tmp_path)
    rec = tmp_path / "lawful_s0_e4_t8"
    rec.mkdir()
    for name in ("gt_agent_pos", "action", "gt_agent_th", "obs_vision"):  # underscores: the order trap
        np.save(rec / f"{name}.npy", np.arange(10, dtype=np.float32) * len(name))
    tool = shutil.which("sha256sum") or "shasum -a 256"
    shell = subprocess.run(f"{tool} ./*.npy | LC_ALL=C sort -k2 | {tool} | cut -c1-16", shell=True, cwd=rec,
                           capture_output=True, text=True, check=True).stdout.strip()
    assert data.recording_hash("lawful", 4, 8, 0) == shell


def test_a_tag_refuses_minds_made_with_other_settings(tmp_path):
    prov = {"backend": "gpu", "matmul_precision": "highest", "xla_flags": "--xla_gpu_deterministic_ops=true",
            "recording_hash": "e9b74db4218d7e27", "stimuli": "0123456789abcdef", "voe_draws": 200, "voe_boot": 2000}
    done = tmp_path / "human.json"
    done.write_text(json.dumps({"provenance": prov}))
    experiment.check_same_settings(done, dict(prov))  # same settings: reused
    with pytest.raises(RuntimeError, match="other settings"):
        experiment.check_same_settings(done, prov | {"matmul_precision": "None"})
    with pytest.raises(RuntimeError, match="recording_hash"):
        experiment.check_same_settings(done, prov | {"recording_hash": "306a35611bc61e58"})
    with pytest.raises(RuntimeError, match="stimuli"):  # scored on other surprise-test movies
        experiment.check_same_settings(done, prov | {"stimuli": "fedcba9876543210"})
    with pytest.raises(RuntimeError, match="voe_draws"):  # surprise scored over another number of noise draws
        experiment.check_same_settings(done, prov | {"voe_draws": 1})
    with pytest.raises(RuntimeError, match="voe_boot"):
        experiment.check_same_settings(done, prov | {"voe_boot": 500})
    before = tmp_path / "before_stimulus_hashes.json"
    before.write_text(json.dumps({"provenance": {k: v for k, v in prov.items() if k != "stimuli"}}))
    with pytest.raises(RuntimeError, match="stimuli"):  # movies unknown: not reused
        experiment.check_same_settings(before, prov)
    old = tmp_path / "old.json"
    old.write_text(json.dumps({"test_error": 0.2}))
    experiment.check_same_settings(old, prov)  # made before provenance existed: warned, reused


def test_the_stimulus_hash_sees_every_movie_array():
    def suite(vision=0.0, crit=3):
        movie = {"obs_vision": np.full((4, 5, 2), vision, np.float16), "action": np.zeros((4, 5, 2), np.float32)}
        return {"permanence": (movie, np.array([crit])), "solidity": ({k: v + 1 for k, v in movie.items()}, np.array([4]))}
    h = voe.suite_hash(suite())
    assert len(h) == 16 and h == voe.suite_hash(dict(reversed(list(suite().items()))))  # order of the dict ignored
    assert voe.suite_hash(suite(vision=0.5)) != h
    assert voe.suite_hash(suite(crit=2)) != h


def test_the_movies_are_rendered_on_the_cpu(monkeypatch):
    seen = []

    class Stop(Exception):
        pass

    def spy(device):  # record the device, then stop before any rendering
        seen.append(device.platform)
        raise Stop

    monkeypatch.setattr(voe.jax, "default_device", spy)
    with pytest.raises(Stop):
        voe.build_suite()
    assert seen == ["cpu"]


def test_provenance_names_every_setting(monkeypatch):
    """Every setting a tag checks is recorded, with the values the scores are made with."""
    monkeypatch.setattr(experiment, "recording_hash", lambda *a: "0" * 16)
    prov = experiment.provenance("lawful", 4, 0, stimuli="0123456789abcdef")
    assert set(experiment.SETTINGS) <= set(prov)
    assert prov["voe_draws"] == voe.NOISE_DRAWS and prov["voe_boot"] == voe.N_BOOT
