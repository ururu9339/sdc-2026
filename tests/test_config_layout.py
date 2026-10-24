import subprocess
import sys
from pathlib import Path

from src.training import config as C


def _root(tmp_path, *, manifest=True, repo_log=False):
    root = tmp_path / "repo"
    (root / "data").mkdir(parents=True)
    if repo_log:
        (root / "data" / "preview_log.jsonl").write_text("")
    if manifest:
        (root / "serving").mkdir()
        (root / "serving" / "manifest.json").write_text("{}")
    return root


def test_env_data_dir_wins_and_models_default_to_the_repo(tmp_path):
    root = _root(tmp_path)
    lay = C.resolve_layout({"SDC_DATA_DIR": str(tmp_path / "other")}, root=root)
    assert lay["mode"] == "env" and lay["data_dir"] == tmp_path / "other"
    assert lay["models_dir"] == root / "src" / "models"


def test_bundle_is_used_when_no_data_dir_and_no_repo_logs(tmp_path):
    root = _root(tmp_path)
    lay = C.resolve_layout({}, root=root)
    assert lay["mode"] == "bundle"
    assert lay["data_dir"] == root / "serving" and lay["models_dir"] == root / "serving" / "models"


def test_repo_logs_keep_the_repo_mode(tmp_path):
    root = _root(tmp_path, repo_log=True)
    lay = C.resolve_layout({}, root=root)
    assert lay["mode"] == "repo" and lay["data_dir"] == root / "data"


def test_no_manifest_means_repo_mode(tmp_path):
    root = _root(tmp_path, manifest=False)
    assert C.resolve_layout({}, root=root)["mode"] == "repo"


def test_models_env_wins_in_bundle_mode(tmp_path):
    root = _root(tmp_path)
    lay = C.resolve_layout({"SDC_MODELS_DIR": str(tmp_path / "m")}, root=root)
    assert lay["mode"] == "bundle" and lay["models_dir"] == tmp_path / "m"


def test_a_fresh_process_honours_sdc_data_dir(tmp_path):
    env = {"SDC_DATA_DIR": str(tmp_path / "d"), "PATH": "/usr/bin:/bin"}
    code = "from src.training import config as C; print(C.DATA_LAYOUT, C.DATA_DIR, C.MODELS_DIR)"
    out = subprocess.run([sys.executable, "-c", code], cwd=str(C.ROOT), env=env, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    mode, data, models = out.stdout.split()
    assert mode == "env" and Path(data) == tmp_path / "d" and Path(models) == C.ROOT / "src" / "models"
