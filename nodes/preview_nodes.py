"""
Preview nodes for Point Clouds and Gaussian Splats
"""
import logging
from comfy_api.latest import io

log = logging.getLogger("depthanythingv3")


class DA3_PreviewPointCloud(io.ComfyNode):
    """Preview point cloud PLY data in the browser using VTK.js"""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="DA3_PreviewPointCloud",
            display_name="DA3 Preview Point Cloud",
            category="DepthAnythingV3",
            is_output_node=True,
            description="""Preview point cloud PLY data in 3D using VTK.js (scientific visualization).

Inputs:
- ply: PLY payload (typically from DA3 To Point Cloud or DA3 Multiview Point Cloud node)
- color_mode:
  - RGB: Show original texture colors from PLY data
  - View ID: Color points by source view (requires view_id in PLY)

Features:
- VTK.js rendering engine
- Trackball camera controls
- Axis orientation widget
- Adjustable point size
- Toggle between RGB and view-based coloring
- Max 2M points

Controls:
- Left Mouse: Rotate view
- Right Mouse: Pan camera
- Mouse Wheel: Zoom in/out
- Slider: Adjust point size""",
            inputs=[
                io.Ply.Input("ply", optional=True),
                io.Combo.Input("color_mode", options=["RGB", "View ID"], default="RGB", optional=True),
            ],
            outputs=[],
        )

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        """Force re-execution when color_mode changes."""
        color_mode = kwargs.get('color_mode', 'RGB')
        ply = kwargs.get('ply', None)
        ply_id = id(ply) if ply is not None else 'none'
        return f"{ply_id}_{color_mode}"

    @classmethod
    def execute(cls, ply=None, color_mode="RGB"):
        """Preview the point cloud using VTK.js."""
        from pathlib import Path

        log.info(f"preview() called with color_mode='{color_mode}'")

        if ply is None:
            return io.NodeOutput(ui={"file_path": [""]})

        import folder_paths
        temp_dir = folder_paths.get_temp_directory()

        if color_mode == "RGB":
            temp_path = Path(temp_dir) / "comfyui_preview_pointcloud.ply"
            ply.save_to(str(temp_path))
            log.info(f"RGB mode: wrote preview PLY to {temp_path}")
            return io.NodeOutput(ui={"file_path": [str(temp_path)]})

        # For View ID mode, recolor using PLY's structured data
        log.info("Attempting View ID mode")
        if ply.view_id is None or ply.is_gaussian:
            # No view_id data or gaussian PLY — fall back to RGB
            temp_path = Path(temp_dir) / "comfyui_preview_pointcloud.ply"
            ply.save_to(str(temp_path))
            log.info("No view_id data available, falling back to RGB mode")
            return io.NodeOutput(ui={"file_path": [str(temp_path)]})

        log.info(f"Recoloring {len(ply.view_id)} points by view_id")
        colors = _color_by_view_id(ply.view_id)

        from comfy_api.latest._util.ply_types import PLY
        recolored = PLY(
            points=ply.points,
            colors=colors,
            confidence=ply.confidence,
            view_id=ply.view_id,
        )

        temp_path = Path(temp_dir) / "comfyui_preview_pointcloud.ply"
        recolored.save_to(str(temp_path))
        log.info(f"View ID mode: wrote recolored preview PLY to {temp_path}")

        return io.NodeOutput(ui={"file_path": [str(temp_path)]})


def _color_by_view_id(view_id):
    """Generate colors based on view ID using a color palette."""
    import numpy as np

    color_palette = np.array([
        [1.0, 0.0, 0.0],  # Red
        [0.0, 0.0, 1.0],  # Blue
        [0.0, 1.0, 0.0],  # Green
        [1.0, 1.0, 0.0],  # Yellow
        [1.0, 0.0, 1.0],  # Magenta
        [0.0, 1.0, 1.0],  # Cyan
        [1.0, 0.5, 0.0],  # Orange
        [0.5, 0.0, 1.0],  # Purple
    ])

    num_views = len(color_palette)
    colors = np.zeros((len(view_id), 3), dtype=np.float32)

    for i in range(len(view_id)):
        view_idx = int(view_id[i]) % num_views
        colors[i] = color_palette[view_idx]

    return colors
