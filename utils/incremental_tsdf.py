"""Incremental colored TSDF meshing with Open3D's VoxelBlockGrid.

Expected per frame:
  image_rgb : (H, W, 3), uint8, RGB order
  points_cam: (H, W, 3), floating point, camera-coordinate XYZ
  moge_mask : (H, W) or (H, W, 1), bool/numeric validity mask
  pose      : (4, 4), camera-to-world by default

The VoxelBlockGrid persists across calls to integrate_frame().
"""








from __future__ import annotations
import numpy as np
import open3d as o3d
import open3d.core as o3c






class IncrementalTSDFMesher:



    def __init__(
        self,
        intrinsic: np.ndarray,
        voxel_size: float = 0.01,
        depth_min: float = 0.05,
        depth_max: float = 10.0,
        block_resolution: int = 16,
        block_count: int = 50_000,
        device: str | None = None,
        pose_is_camera_to_world: bool = True) -> None:

        if device is None:
            device = "CUDA:0" if o3c.cuda.is_available() else "CPU:0"

        self.device = o3c.Device(device)
        self.depth_min = float(depth_min)
        self.depth_max = float(depth_max)
        self.pose_is_camera_to_world = pose_is_camera_to_world
        self.frame_count = 0

        intrinsic = np.asarray(intrinsic, dtype=np.float64)
        if intrinsic.shape != (3, 3):
            raise ValueError(f"intrinsic must have shape (3, 3), got {intrinsic.shape}")
        if not np.isfinite(intrinsic).all():
            raise ValueError("intrinsic contains NaN or Inf")
        if intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0:
            raise ValueError("fx and fy must be positive")

        # The official Open3D example keeps calibration tensors on CPU while
        # the images and VoxelBlockGrid are placed on the selected device.
        self.K_np = intrinsic
        self.K = o3c.Tensor(intrinsic, dtype=o3c.Dtype.Float64)

        self.vbg = o3d.t.geometry.VoxelBlockGrid(
            attr_names=("tsdf", "weight", "color"),
            attr_dtypes=(o3c.float32, o3c.float32, o3c.float32),
            attr_channels=((1,), (1,), (3,)),
            voxel_size=float(voxel_size),
            block_resolution=int(block_resolution),
            block_count=int(block_count),
            device=self.device)

        print("\n[TSDF initialization]")
        print(f"  Open3D version : {o3d.__version__}")
        print(f"  CUDA available : {o3c.cuda.is_available()}")
        print(f"  device         : {self.device}")
        print(f"  voxel size     : {voxel_size}")
        print(f"  depth range    : [{self.depth_min}, {self.depth_max}]")
        print(f"  pose input     : {'camera-to-world' if pose_is_camera_to_world else 'world-to-camera'}")
        print("  intrinsic K:\n", self.K_np)







    @staticmethod
    def _normalize_mask(mask: np.ndarray | None, height: int, width: int) -> np.ndarray:
        if mask is None:
            return np.ones((height, width), dtype=bool)

        mask = np.asarray(mask)
        if mask.shape == (height, width, 1):
            mask = mask[..., 0]
        if mask.shape != (height, width):
            raise ValueError(
                f"moge_mask must have shape {(height, width)} or "
                f"{(height, width, 1)}, got {mask.shape}"
            )
        return mask.astype(bool)






    def _check_point_projection(self, points_cam: np.ndarray, valid: np.ndarray) -> None:
        """Diagnostic: verify XYZ agrees with K and its pixel location."""
        ys, xs = np.nonzero(valid)
        if len(xs) == 0:
            print("  projection test: skipped (no valid points)")
            return

        # Limit diagnostic work without changing integration data.
        step = max(1, len(xs) // 10_000)
        ys, xs = ys[::step], xs[::step]
        xyz = points_cam[ys, xs]
        z = xyz[:, 2]
        fx, fy = self.K_np[0, 0], self.K_np[1, 1]
        cx, cy = self.K_np[0, 2], self.K_np[1, 2]
        projected_u = fx * xyz[:, 0] / z + cx
        projected_v = fy * xyz[:, 1] / z + cy
        error = np.sqrt((projected_u - xs) ** 2 + (projected_v - ys) ** 2)
        print(
            "  XYZ/K reprojection error [px]: "
            f"median={np.median(error):.3f}, p95={np.percentile(error, 95):.3f}"
        )
        if np.median(error) > 2.0:
            print("  WARNING: XYZ, image pixels, and K may use different conventions.")






    def integrate_frame(
        self,
        image_rgb: np.ndarray,
        points_cam: np.ndarray,
        moge_mask: np.ndarray | None,
        pose: np.ndarray,
    ) -> None:
        """Integrate one frame into the persistent VoxelBlockGrid."""
        image_rgb = np.asarray(image_rgb)
        points_cam = np.asarray(points_cam)
        pose = np.asarray(pose, dtype=np.float64)

        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError(f"image_rgb must have shape (H, W, 3), got {image_rgb.shape}")
        height, width = image_rgb.shape[:2]
        if points_cam.shape != (height, width, 3):
            raise ValueError(
                f"points_cam must have shape {(height, width, 3)}, got {points_cam.shape}"
            )
        if pose.shape != (4, 4):
            raise ValueError(f"pose must have shape (4, 4), got {pose.shape}")
        if not np.isfinite(pose).all():
            raise ValueError("pose contains NaN or Inf")

        if image_rgb.dtype != np.uint8:
            print(f"  WARNING: converting image from {image_rgb.dtype} to uint8.")
            if np.issubdtype(image_rgb.dtype, np.floating) and image_rgb.max() <= 1.0:
                image_rgb = image_rgb * 255.0
            image_rgb = np.clip(image_rgb, 0, 255).astype(np.uint8)

        points_cam = points_cam.astype(np.float32, copy=False)
        mask = self._normalize_mask(moge_mask, height, width)

        # Open3D projective TSDF integration expects camera-axis Z-depth.
        depth_np = points_cam[..., 2].copy()
        valid = (
            mask
            & np.isfinite(points_cam).all(axis=-1)
            & (depth_np > self.depth_min)
            & (depth_np < self.depth_max)
        )
        depth_np[~valid] = 0.0

        color_np = np.ascontiguousarray(image_rgb.astype(np.float32) / 255.0)     
        depth_np = np.ascontiguousarray(depth_np, dtype=np.float32)

        T_cw = np.linalg.inv(pose) if self.pose_is_camera_to_world else pose.copy()

        self.frame_count += 1
        valid_depths = depth_np[valid]
        print(f"\n[Frame {self.frame_count}]")
        print(f"  image          : shape={image_rgb.shape}, dtype={image_rgb.dtype}, RGB assumed")
        print(f"  points         : shape={points_cam.shape}, dtype={points_cam.dtype}")
        print(f"  MoGe mask      : shape={mask.shape}, valid={mask.mean() * 100:.2f}%")
        print(f"  final depth    : valid={valid.mean() * 100:.2f}% ({valid.sum()}/{valid.size})")
        if valid_depths.size:
            print(
                "  valid Z stats  : "
                f"min={valid_depths.min():.5f}, "
                f"median={np.median(valid_depths):.5f}, "
                f"max={valid_depths.max():.5f}"
            )
        else:
            raise ValueError("No valid depths remain after mask and depth-range filtering")

        rotation = T_cw[:3, :3]
        print(f"  input pose t   : {pose[:3, 3]}")
        print(f"  T_cw t passed  : {T_cw[:3, 3]}")
        print(f"  det(T_cw.R)    : {np.linalg.det(rotation):.6f} (should be near +1)")
        print(f"  last pose row  : {T_cw[3]}")
        self._check_point_projection(points_cam, valid)

        depth = o3d.t.geometry.Image(o3c.Tensor(depth_np, dtype=o3c.Dtype.Float32, device=self.device))
        color = o3d.t.geometry.Image(
            o3c.Tensor(
                color_np,
                dtype=o3c.Dtype.Float32,
                device=self.device))
        extrinsic = o3c.Tensor(T_cw, dtype=o3c.Dtype.Float64)

        block_coords = self.vbg.compute_unique_block_coordinates(
            depth,
            self.K,
            extrinsic,
            depth_scale=1.0,
            depth_max=self.depth_max)
        print(f"  affected blocks: {block_coords.shape[0]}")

        self.vbg.integrate(
            block_coords,
            depth,
            color,
            self.K,
            self.K,
            extrinsic,
            depth_scale=1.0,
            depth_max=self.depth_max,
        )

        active_blocks = self.vbg.hashmap().size()
        print(f"  active blocks  : {active_blocks}")

        print(
        f"  O3D inputs     : "
        f"depth={depth_np.dtype}, "
        f"color={color_np.dtype}, "
        f"color range=[{color_np.min():.1f}, {color_np.max():.1f}]")





    def extract_mesh(
        self,
        weight_threshold: float = 1.0,
        extract_on_cpu: bool = True,
    ):
        """Extract the current legacy mesh without modifying the GPU VBG."""

        print(
            f"\n[Mesh extraction after "
            f"{self.frame_count} integrated frame(s)]"
        )

        # Keep self.vbg on GPU for future integrations, but perform
        # memory-heavy Marching Cubes on a temporary CPU copy.
        if extract_on_cpu:
            print("  copying VBG to CPU for mesh extraction...")

            extraction_vbg = self.vbg.to(
                o3c.Device("CPU:0"),
                copy=True,
            )
        else:
            extraction_vbg = self.vbg

        mesh_tensor = extraction_vbg.extract_triangle_mesh(
            weight_threshold=weight_threshold
        )

        vertex_count = int(mesh_tensor.vertex.positions.shape[0])
        triangle_count = int(mesh_tensor.triangle.indices.shape[0])

        print(f"  extraction device: {'CPU' if extract_on_cpu else self.device}")
        print(f"  weight threshold : {weight_threshold}")
        print(f"  mesh vertices    : {vertex_count}")
        print(f"  mesh triangles   : {triangle_count}")

        if vertex_count == 0:
            print(
                "  WARNING: mesh is empty; try "
                "weight_threshold=0.0."
            )

        # Your VisualizerWithKeyCallback requires a legacy mesh.
        mesh = mesh_tensor.to_legacy()

        if vertex_count > 0:
            mesh.compute_vertex_normals()

        # Delete only the temporary CPU copy.
        # self.vbg remains on GPU and unchanged.
        if extract_on_cpu:
            del extraction_vbg

        return mesh