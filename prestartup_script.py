import logging
import shutil
from pathlib import Path

log = logging.getLogger("depthanythingv3")

SCRIPT_DIR = Path(__file__).resolve().parent
COMFYUI_DIR = SCRIPT_DIR.parent.parent

# Copy pointcloud VTK viewer from comfy-3d-viewers when the helper package exists.
try:
    from comfy_3d_viewers import copy_viewer
except ImportError:
    copy_viewer = None

if copy_viewer is not None:
    copy_viewer("pointcloud_vtk", SCRIPT_DIR / "web")
else:
    log.warning("comfy_3d_viewers is unavailable; skipping pointcloud viewer copy")

# Copy dynamic widgets JS
try:
    from comfy_dynamic_widgets import get_js_path
    src = Path(get_js_path())
    if src.exists():
        dst = SCRIPT_DIR / "web" / "js" / "dynamic_widgets.js"
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists() or src.stat().st_mtime > dst.stat().st_mtime:
            shutil.copy2(src, dst)
except ImportError:
    pass

# Copy assets
src_dir = SCRIPT_DIR / "assets"
dst_dir = COMFYUI_DIR / "input"
copied_files = []
if src_dir.exists():
    for src_file in src_dir.rglob("*"):
        if not src_file.is_file():
            continue
        rel_path = src_file.relative_to(src_dir)
        dst_file = dst_dir / rel_path
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        if not dst_file.exists() or src_file.stat().st_mtime > dst_file.stat().st_mtime:
            shutil.copy2(src_file, dst_file)
        copied_files.append(src_file.name)

log.info(f"Copied {len(copied_files)} asset(s) to {dst_dir}: {copied_files}")
