import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_DIR = ROOT / "workflows"
NODES_INIT_PATH = ROOT / "nodes" / "__init__.py"
LOAD_MODEL_PATH = ROOT / "nodes" / "load_model.py"


EXPECTED_WORKFLOWS = {
    "simple.json",
    "advanced.json",
    "advanced_3d.json",
    "advanced_3d_multiview.json",
    "bas_relief.json",
    "video_multiview_depth.json",
    "da3_streaming.json",
}

PHASE1_STANDARD_NODE_TYPES = {
    "simple.json": {
        "LoadImage",
        "DepthAnything_V3",
        "PreviewImage",
        "DownloadAndLoadDepthAnythingV3Model",
    },
    "advanced.json": {
        "LoadImage",
        "DownloadAndLoadDepthAnythingV3Model",
        "DepthAnything_V3",
        "PreviewImage",
        "PreviewAny",
        "MaskPreview",
    },
}

FORBIDDEN_ABSOLUTE_PATHS = {
    "/home/shadeform/",
}


def load_workflow(name: str) -> dict:
    return json.loads((WORKFLOW_DIR / name).read_text(encoding="utf-8"))


def workflow_text(name: str) -> str:
    return (WORKFLOW_DIR / name).read_text(encoding="utf-8")


def workflow_node_types(name: str) -> set[str]:
    data = load_workflow(name)
    return {node["type"] for node in data["nodes"]}


def test_expected_workflow_files_are_present():
    actual = {path.name for path in WORKFLOW_DIR.glob("*.json")}
    assert actual == EXPECTED_WORKFLOWS


def test_slice1_standard_workflows_match_original_node_types():
    for name, expected_node_types in PHASE1_STANDARD_NODE_TYPES.items():
        assert workflow_node_types(name) == expected_node_types


def test_slice1_loader_nodes_are_exported():
    text = NODES_INIT_PATH.read_text(encoding="utf-8")
    assert "DA3_DownloadModel" in text
    assert "DownloadAndLoadDepthAnythingV3Model" in text


def test_slice1_loader_behavior_supports_known_models_and_downloads():
    text = LOAD_MODEL_PATH.read_text(encoding="utf-8")
    assert "MODEL_REPOS" in text
    assert "snapshot_download" in text
    assert "class DA3_DownloadModel" in text


def test_no_stale_absolute_paths_remain():
    for name in EXPECTED_WORKFLOWS:
        text = workflow_text(name)
        for marker in FORBIDDEN_ABSOLUTE_PATHS:
            assert marker not in text, f"{name} still contains stale absolute path {marker!r}"
