import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy import ndimage
from typing import Dict, List, Tuple, Optional, Set
import os
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from dataclasses import dataclass, field


@dataclass
class OccupancyMapConfig:
    map_size_meters: float = 200.0
    resolution: float = 0.5

    image_width: int = 512
    image_height: int = 512
    fov_degrees: float = 90.0

    depth_scale: float = 86.0
    min_valid_depth: float = 0.5
    max_valid_depth: float = 50.0

    convert_to_height: bool = True

    ray_sample_interval: float = 0.5

    boundary_semantic_ids: List[int] = field(default_factory=lambda: [253, 161, 91, 126])

    frontier_min_size: int = 3

    max_exploration_distance: float = 5.0

    @property
    def map_size_grids(self) -> int:
        return int(self.map_size_meters / self.resolution)


class OccupancyMap2_5D:

    CH_MAX_HEIGHT = 0
    CH_MIN_HEIGHT = 1
    CH_EXPLORED = 2
    CH_OCCUPIED = 3
    CH_BOUNDARY = 4
    NUM_CHANNELS = 5

    def __init__(self, config: OccupancyMapConfig = None):
        self.config = config or OccupancyMapConfig()

        self.map_size = self.config.map_size_grids
        self.resolution = self.config.resolution
        self.map_center = self.map_size // 2

        self._init_map()

        self._init_camera_intrinsics()

        self.trajectory: List[np.ndarray] = []

        self._frontier_cache: Optional[np.ndarray] = None
        self._frontier_cache_valid: bool = False

        self.total_updates = 0
        self.total_points_processed = 0
        self.total_rays_cast = 0

    def _init_map(self):
        self.map = np.zeros((self.NUM_CHANNELS, self.map_size, self.map_size), dtype=np.float32)

        self.map[self.CH_MAX_HEIGHT, :, :] = -np.inf
        self.map[self.CH_MIN_HEIGHT, :, :] = np.inf
        self.map[self.CH_EXPLORED, :, :] = 0
        self.map[self.CH_OCCUPIED, :, :] = 0
        self.map[self.CH_BOUNDARY, :, :] = 0

    def _init_camera_intrinsics(self):
        fov_rad = np.radians(self.config.fov_degrees)
        self.fx = self.config.image_width / (2 * np.tan(fov_rad / 2))
        self.fy = self.config.image_height / (2 * np.tan(fov_rad / 2))
        self.cx = self.config.image_width / 2
        self.cy = self.config.image_height / 2

        u = np.arange(self.config.image_width)
        v = np.arange(self.config.image_height)
        self.u_grid, self.v_grid = np.meshgrid(u, v)

    def _compute_camera_extrinsics(self, pose: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        pos = pose[:3]
        quat = pose[3:7]

        R_w2b = R.from_quat(quat).as_matrix()
        R_c2w = R_w2b

        T_c2w = np.array(pos)

        return R_c2w, T_c2w

    def _depth_to_pointcloud(
            self,
            depth_img: np.ndarray,
            pose: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        depth = depth_img.squeeze()

        depth_meters = depth * self.config.depth_scale

        valid_mask = (depth_meters > self.config.min_valid_depth) & \
                     (depth_meters < self.config.max_valid_depth)

        _, T_c2w = self._compute_camera_extrinsics(pose)
        camera_position = T_c2w.copy()

        if not np.any(valid_mask):
            return np.empty((0, 3), dtype=np.float32), camera_position

        v_valid = self.v_grid[valid_mask]
        u_valid = self.u_grid[valid_mask]
        d_valid = depth_meters[valid_mask]

        x_cam = (u_valid - self.cx) * d_valid / self.fx
        y_cam = (v_valid - self.cy) * d_valid / self.fy
        z_cam = d_valid

        points_body = np.stack([z_cam, x_cam, y_cam], axis=1)

        R_c2w, T_c2w = self._compute_camera_extrinsics(pose)
        points_world = (R_c2w @ points_body.T).T + T_c2w

        return points_world.astype(np.float32), camera_position

    def _world_to_grid(self, points_world: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        grid_i = (points_world[:, 0] / self.resolution + self.map_center).astype(np.int32)
        grid_j = (points_world[:, 1] / self.resolution + self.map_center).astype(np.int32)

        valid_mask = (grid_i >= 0) & (grid_i < self.map_size) & \
                     (grid_j >= 0) & (grid_j < self.map_size)

        valid_indices = np.where(valid_mask)[0]

        return grid_i[valid_mask], grid_j[valid_mask], valid_indices

    def _world_to_grid_single(self, world_pos: np.ndarray) -> Tuple[int, int]:
        grid_i = int(world_pos[0] / self.resolution + self.map_center)
        grid_j = int(world_pos[1] / self.resolution + self.map_center)
        return grid_i, grid_j

    def _grid_to_world(self, grid_i: int, grid_j: int) -> np.ndarray:
        world_x = (grid_i - self.map_center) * self.resolution
        world_y = (grid_j - self.map_center) * self.resolution
        return np.array([world_x, world_y])

    def _points_to_heights(self, points_world: np.ndarray) -> np.ndarray:
        if self.config.convert_to_height:
            return -points_world[:, 2]
        else:
            return points_world[:, 2]

    def _cast_rays_dense(
            self,
            camera_position: np.ndarray,
            obstacle_points: np.ndarray,
            heights: np.ndarray
    ) -> Dict:
        if len(obstacle_points) == 0:
            return {'rays_cast': 0, 'cells_explored': 0, 'obstacles_marked': 0}

        stats = {'rays_cast': 0, 'cells_explored': 0, 'obstacles_marked': 0}

        cam_gi, cam_gj = self._world_to_grid_single(camera_position[:2])

        explored_cells = set()
        obstacle_cells = {}

        for idx in range(len(obstacle_points)):
            point = obstacle_points[idx]
            height = heights[idx]

            obs_gi, obs_gj = self._world_to_grid_single(point[:2])

            if not (0 <= obs_gi < self.map_size and 0 <= obs_gj < self.map_size):
                continue

            dx = point[0] - camera_position[0]
            dy = point[1] - camera_position[1]
            distance = np.sqrt(dx * dx + dy * dy)

            if distance < self.config.ray_sample_interval:
                explored_cells.add((obs_gi, obs_gj))
                if (obs_gi, obs_gj) not in obstacle_cells:
                    obstacle_cells[(obs_gi, obs_gj)] = {'heights': []}
                obstacle_cells[(obs_gi, obs_gj)]['heights'].append(height)
                continue

            num_samples = int(distance / self.config.ray_sample_interval) + 1
            t_values = np.linspace(0, 1, num_samples)

            for t in t_values:
                sample_x = camera_position[0] + t * dx
                sample_y = camera_position[1] + t * dy
                sample_gi = int(sample_x / self.resolution + self.map_center)
                sample_gj = int(sample_y / self.resolution + self.map_center)

                if 0 <= sample_gi < self.map_size and 0 <= sample_gj < self.map_size:
                    explored_cells.add((sample_gi, sample_gj))

            if (obs_gi, obs_gj) not in obstacle_cells:
                obstacle_cells[(obs_gi, obs_gj)] = {'heights': []}
            obstacle_cells[(obs_gi, obs_gj)]['heights'].append(height)

            stats['rays_cast'] += 1

        for (gi, gj) in explored_cells:
            self.map[self.CH_EXPLORED, gi, gj] = 1
        stats['cells_explored'] = len(explored_cells)

        for (gi, gj), data in obstacle_cells.items():
            heights_arr = np.array(data['heights'])
            max_h = np.max(heights_arr)
            min_h = np.min(heights_arr)

            current_max = self.map[self.CH_MAX_HEIGHT, gi, gj]
            current_min = self.map[self.CH_MIN_HEIGHT, gi, gj]
            self.map[self.CH_MAX_HEIGHT, gi, gj] = max(current_max, max_h)
            self.map[self.CH_MIN_HEIGHT, gi, gj] = min(current_min, min_h)

            self.map[self.CH_OCCUPIED, gi, gj] = 1

        stats['obstacles_marked'] = len(obstacle_cells)

        return stats

    def _extract_frontiers(self) -> np.ndarray:
        explored_free = (self.map[self.CH_EXPLORED] == 1) & (self.map[self.CH_OCCUPIED] == 0)

        unexplored = self.map[self.CH_EXPLORED] == 0

        kernel = np.ones((3, 3), dtype=np.float32)
        unexplored_dilated = ndimage.binary_dilation(unexplored, kernel)

        frontier_raw = explored_free & unexplored_dilated

        if self.config.frontier_min_size > 1:
            labeled, num_features = ndimage.label(frontier_raw)
            for i in range(1, num_features + 1):
                if np.sum(labeled == i) < self.config.frontier_min_size:
                    frontier_raw[labeled == i] = False

        return frontier_raw

    def get_frontiers(self, force_update: bool = False) -> np.ndarray:
        if not self._frontier_cache_valid or force_update:
            self._frontier_cache = self._extract_frontiers()
            self._frontier_cache_valid = True

        return self._frontier_cache

    def get_frontier_points(self) -> np.ndarray:
        frontier_map = self.get_frontiers()
        frontier_indices = np.where(frontier_map)

        if len(frontier_indices[0]) == 0:
            return np.empty((0, 2))

        world_x = (frontier_indices[0] - self.map_center) * self.resolution
        world_y = (frontier_indices[1] - self.map_center) * self.resolution

        return np.stack([world_x, world_y], axis=1)

    def get_frontier_clusters(self, min_cluster_size: int = 5) -> List[np.ndarray]:
        frontier_map = self.get_frontiers()
        labeled, num_features = ndimage.label(frontier_map)

        clusters = []
        for i in range(1, num_features + 1):
            cluster_mask = labeled == i
            if np.sum(cluster_mask) >= min_cluster_size:
                indices = np.where(cluster_mask)
                center_gi = np.mean(indices[0])
                center_gj = np.mean(indices[1])

                world_x = (center_gi - self.map_center) * self.resolution
                world_y = (center_gj - self.map_center) * self.resolution

                clusters.append(np.array([world_x, world_y]))

        return clusters

    def update(
            self,
            viewpoint_dep_imgs: Dict[str, np.ndarray],
            viewpoint_poses: Dict[str, np.ndarray],
            image_order: List[str],
            current_position: np.ndarray,
            viewpoint_rgb_imgs: Dict[str, np.ndarray] = None
    ) -> Dict:
        total_points = 0
        valid_points = 0
        total_rays = 0
        total_explored = 0
        total_obstacles = 0

        self._frontier_cache_valid = False

        for view_name in image_order:
            if view_name not in viewpoint_dep_imgs or view_name not in viewpoint_poses:
                continue

            depth_img = viewpoint_dep_imgs[view_name]
            pose = viewpoint_poses[view_name]

            if not isinstance(pose, np.ndarray):
                pose = np.array(pose)

            points_world, camera_position = self._depth_to_pointcloud(
                depth_img, pose
            )

            total_points += self.config.image_width * self.config.image_height
            valid_points += len(points_world)

            if len(points_world) == 0:
                continue

            heights = self._points_to_heights(points_world)

            ray_stats = self._cast_rays_dense(
                camera_position, points_world, heights
            )

            total_rays += ray_stats['rays_cast']
            total_explored += ray_stats['cells_explored']
            total_obstacles += ray_stats['obstacles_marked']

        self.trajectory.append(current_position.copy())

        self.total_updates += 1
        self.total_points_processed += valid_points
        self.total_rays_cast += total_rays

        explored_mask = self.map[self.CH_EXPLORED] > 0
        occupied_mask = self.map[self.CH_OCCUPIED] > 0
        boundary_mask = self.map[self.CH_BOUNDARY] > 0
        frontier_mask = self.get_frontiers()

        total_cells = self.map_size * self.map_size

        return {
            'total_points': total_points,
            'valid_points': valid_points,
            'rays_cast': total_rays,
            'cells_explored': total_explored,
            'obstacles_marked': total_obstacles,
            'exploration_coverage': float(np.sum(explored_mask) / total_cells),
            'occupied_cells': int(np.sum(occupied_mask)),
            'boundary_cells': int(np.sum(boundary_mask)),
            'frontier_cells': int(np.sum(frontier_mask)),
            'trajectory_length': len(self.trajectory),
            'total_updates': self.total_updates
        }

    def query_height_at(self, world_xy: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
        grid_i = int(world_xy[0] / self.resolution + self.map_center)
        grid_j = int(world_xy[1] / self.resolution + self.map_center)

        if not (0 <= grid_i < self.map_size and 0 <= grid_j < self.map_size):
            return None, None

        if self.map[self.CH_EXPLORED, grid_i, grid_j] == 0:
            return None, None

        if self.map[self.CH_OCCUPIED, grid_i, grid_j] == 0:
            return 0.0, 0.0

        max_h = self.map[self.CH_MAX_HEIGHT, grid_i, grid_j]
        min_h = self.map[self.CH_MIN_HEIGHT, grid_i, grid_j]

        return float(max_h), float(min_h)

    def is_explored(self, world_xy: np.ndarray) -> bool:
        grid_i = int(world_xy[0] / self.resolution + self.map_center)
        grid_j = int(world_xy[1] / self.resolution + self.map_center)

        if not (0 <= grid_i < self.map_size and 0 <= grid_j < self.map_size):
            return False

        return self.map[self.CH_EXPLORED, grid_i, grid_j] > 0

    def is_occupied(self, world_xy: np.ndarray) -> bool:
        grid_i = int(world_xy[0] / self.resolution + self.map_center)
        grid_j = int(world_xy[1] / self.resolution + self.map_center)

        if not (0 <= grid_i < self.map_size and 0 <= grid_j < self.map_size):
            return False

        return self.map[self.CH_OCCUPIED, grid_i, grid_j] > 0

    def is_boundary(self, world_xy: np.ndarray) -> bool:
        grid_i = int(world_xy[0] / self.resolution + self.map_center)
        grid_j = int(world_xy[1] / self.resolution + self.map_center)

        if not (0 <= grid_i < self.map_size and 0 <= grid_j < self.map_size):
            return False

        return self.map[self.CH_BOUNDARY, grid_i, grid_j] > 0

    def is_traversable(
            self,
            world_xy: np.ndarray,
            uav_height: float,
            safety_margin: float = 2.0
    ) -> bool:
        grid_i = int(world_xy[0] / self.resolution + self.map_center)
        grid_j = int(world_xy[1] / self.resolution + self.map_center)

        if not (0 <= grid_i < self.map_size and 0 <= grid_j < self.map_size):
            return False

        if self.map[self.CH_EXPLORED, grid_i, grid_j] == 0:
            return True

        if self.map[self.CH_BOUNDARY, grid_i, grid_j] > 0:
            return False

        if self.map[self.CH_OCCUPIED, grid_i, grid_j] == 0:
            return True

        max_h = self.map[self.CH_MAX_HEIGHT, grid_i, grid_j]
        min_h = self.map[self.CH_MIN_HEIGHT, grid_i, grid_j]

        if uav_height > max_h + safety_margin:
            return True
        if uav_height < min_h - safety_margin:
            return True

        return False

    def get_height_map(self) -> np.ndarray:
        height_map = self.map[self.CH_MAX_HEIGHT, :, :].copy()
        unoccupied = self.map[self.CH_OCCUPIED, :, :] == 0
        height_map[unoccupied] = np.nan
        height_map[np.isinf(height_map)] = np.nan
        return height_map

    def get_explored_mask(self) -> np.ndarray:
        return self.map[self.CH_EXPLORED, :, :] > 0

    def get_occupied_mask(self) -> np.ndarray:
        return self.map[self.CH_OCCUPIED, :, :] > 0

    def get_boundary_mask(self) -> np.ndarray:
        return self.map[self.CH_BOUNDARY, :, :] > 0

    def get_trajectory_array(self) -> np.ndarray:
        if len(self.trajectory) == 0:
            return np.empty((0, 3))
        return np.array(self.trajectory)

    def world_to_grid_coords(self, world_pos: np.ndarray) -> Tuple[float, float]:
        grid_i = world_pos[0] / self.resolution + self.map_center
        grid_j = world_pos[1] / self.resolution + self.map_center
        return grid_i, grid_j

    def _get_viewpoint_direction(self, pose: np.ndarray) -> np.ndarray:
        quat = pose[3:7]
        rot = R.from_quat(quat)

        forward_body = np.array([1.0, 0.0, 0.0])
        forward_world = rot.apply(forward_body)

        direction_2d = forward_world[:2]
        norm = np.linalg.norm(direction_2d)
        if norm > 1e-6:
            direction_2d = direction_2d / norm
        else:
            direction_2d = np.array([1.0, 0.0])

        return direction_2d

    def _compute_direction_alignment(
            self,
            current_position: np.ndarray,
            frontier_center: np.ndarray,
            viewpoint_direction: np.ndarray
    ) -> float:
        to_frontier = frontier_center - current_position[:2]
        norm = np.linalg.norm(to_frontier)
        if norm < 1e-6:
            return 0.0

        to_frontier_normalized = to_frontier / norm

        cos_sim = np.dot(to_frontier_normalized, viewpoint_direction)

        alignment = (cos_sim + 1.0) / 2.0

        return float(alignment)

    def _is_safe_at_height(
            self,
            grid_i: int,
            grid_j: int,
            uav_height: float,
            safety_margin: float = 2.0
    ) -> bool:
        if not (0 <= grid_i < self.map_size and 0 <= grid_j < self.map_size):
            return False

        if self.map[self.CH_EXPLORED, grid_i, grid_j] == 0:
            return True

        if self.map[self.CH_BOUNDARY, grid_i, grid_j] > 0:
            return False

        if self.map[self.CH_OCCUPIED, grid_i, grid_j] == 0:
            return True

        max_h = self.map[self.CH_MAX_HEIGHT, grid_i, grid_j]

        if uav_height > max_h + safety_margin:
            return True

        return False

    def _find_nearest_safe_point(
            self,
            world_xy: np.ndarray,
            uav_height: float,
            search_radius: int = 3,
            safety_margin: float = 2.0
    ) -> Tuple[np.ndarray, bool]:
        center_gi, center_gj = self._world_to_grid_single(world_xy)

        if self._is_safe_at_height(center_gi, center_gj, uav_height, safety_margin):
            return world_xy.copy(), True

        best_point = None
        best_distance_sq = float('inf')

        for di in range(-search_radius, search_radius + 1):
            for dj in range(-search_radius, search_radius + 1):
                if di == 0 and dj == 0:
                    continue

                gi = center_gi + di
                gj = center_gj + dj

                if self._is_safe_at_height(gi, gj, uav_height, safety_margin):
                    dist_sq = di * di + dj * dj
                    if dist_sq < best_distance_sq:
                        best_distance_sq = dist_sq
                        best_point = self._grid_to_world(gi, gj)

        if best_point is not None:
            return best_point, True
        else:
            return world_xy.copy(), False

    def _truncate_and_check_waypoint(
            self,
            current_position: np.ndarray,
            frontier_center: np.ndarray,
            max_distance: float,
            uav_height: float,
            safety_margin: float = 2.0,
            collision_search_radius: int = 3
    ) -> Dict:
        current_2d = current_position[:2]
        direction = frontier_center - current_2d
        original_distance = np.linalg.norm(direction)

        result = {
            'waypoint': frontier_center.copy(),
            'distance': original_distance,
            'is_truncated': False,
            'is_adjusted': False,
            'original_distance': original_distance
        }

        if original_distance < 1e-6:
            result['distance'] = 0.0
            return result

        direction_normalized = direction / original_distance

        if original_distance > max_distance:
            truncated_point = current_2d + direction_normalized * max_distance
            result['is_truncated'] = True
            result['distance'] = max_distance
        else:
            truncated_point = frontier_center.copy()
            result['distance'] = original_distance

        grid_i, grid_j = self._world_to_grid_single(truncated_point)

        if self._is_safe_at_height(grid_i, grid_j, uav_height, safety_margin):
            result['waypoint'] = truncated_point
        else:
            safe_point, found = self._find_nearest_safe_point(
                world_xy=truncated_point,
                uav_height=uav_height,
                search_radius=collision_search_radius,
                safety_margin=safety_margin
            )

            if found:
                result['waypoint'] = safe_point
                result['is_adjusted'] = True
                result['distance'] = float(np.linalg.norm(safe_point - current_2d))
            else:
                result['waypoint'] = truncated_point
                result['is_adjusted'] = False

        return result

    def get_semantically_guided_frontiers(
            self,
            scene_graph,
            current_landmark: Optional[str] = None,
            previous_anchor_label: Optional[str] = None,
            semantic_radius: float = 30.0,
            fallback_semantics: List[str] = None,
            top_k: int = 3,
            min_cluster_size: int = 5,
            current_step_detections: Optional[Dict[str, List[str]]] = None,
            viewpoint_poses: Optional[Dict[str, np.ndarray]] = None,
            current_position: Optional[np.ndarray] = None,
            compute_safe_waypoints: bool = True,
            max_exploration_distance: float = 5,
            safety_margin: float = 2.0,
            collision_search_radius: int = 3
    ) -> List[Dict]:
        if fallback_semantics is None:
            fallback_semantics = [
                "road", "street", "building", "intersection", "crossroad",
                "sidewalk", "path", "avenue", "boulevard", "highway"
            ]

        if max_exploration_distance is None:
            max_exploration_distance = self.config.max_exploration_distance

        frontier_clusters = self.get_frontier_clusters(min_cluster_size=min_cluster_size)

        if len(frontier_clusters) == 0:
            return []

        viewpoint_directions = {}
        if viewpoint_poses is not None:
            for view_name, pose in viewpoint_poses.items():
                if not isinstance(pose, np.ndarray):
                    pose = np.array(pose)
                viewpoint_directions[view_name] = self._get_viewpoint_direction(pose)

        valuable_viewpoints = []
        if current_step_detections is not None and current_landmark is not None:
            current_landmark_lower = current_landmark.lower().strip()
            for view_name, detections in current_step_detections.items():
                for det in detections:
                    if self._semantic_match(det.lower().strip(), current_landmark_lower):
                        valuable_viewpoints.append(view_name)
                        break

        uav_height = 10.0
        if current_position is not None and len(current_position) >= 3:
            uav_height = -current_position[2]

        scored_frontiers = []

        for cluster_center in frontier_clusters:
            query_point_3d = np.array([cluster_center[0], cluster_center[1], 0.0])

            realtime_match = self._check_realtime_detection_match(
                frontier_center=cluster_center,
                current_position=current_position,
                current_landmark=current_landmark,
                current_step_detections=current_step_detections,
                viewpoint_directions=viewpoint_directions,
                valuable_viewpoints=valuable_viewpoints
            )

            if realtime_match['matched']:
                score_info = {
                    'score': realtime_match['score'],
                    'reason': realtime_match['reason'],
                    'match_type': 'realtime_mllm_detection',
                    'matched_node_ids': [],
                    'best_distance': realtime_match.get('alignment_distance', 0),
                    'matched_viewpoint': realtime_match['viewpoint']
                }
            else:
                nearby_nodes = []
                if scene_graph is not None:
                    nearby_nodes = scene_graph.get_nearby_nodes(query_point_3d, radius=semantic_radius)

                score_info = self._compute_semantic_score(
                    frontier_center=cluster_center,
                    nearby_nodes=nearby_nodes,
                    current_landmark=current_landmark,
                    previous_anchor_label=previous_anchor_label,
                    fallback_semantics=fallback_semantics,
                    semantic_radius=semantic_radius
                )
                score_info['matched_viewpoint'] = None

            frontier_result = {
                "center": cluster_center.tolist(),
                "score": score_info["score"],
                "reason": score_info["reason"],
                "nearby_nodes": score_info["matched_node_ids"],
                "match_type": score_info["match_type"],
                "best_match_distance": score_info.get("best_distance", semantic_radius),
                "matched_viewpoint": score_info.get("matched_viewpoint")
            }

            if compute_safe_waypoints and current_position is not None:
                waypoint_result = self._truncate_and_check_waypoint(
                    current_position=current_position,
                    frontier_center=cluster_center,
                    max_distance=max_exploration_distance,
                    uav_height=uav_height,
                    safety_margin=safety_margin,
                    collision_search_radius=collision_search_radius
                )
                frontier_result["safe_waypoint"] = waypoint_result["waypoint"].tolist()
                frontier_result["safe_distance"] = waypoint_result["distance"]
                frontier_result["is_truncated"] = waypoint_result["is_truncated"]
                frontier_result["is_adjusted"] = waypoint_result["is_adjusted"]
                frontier_result["original_distance"] = waypoint_result["original_distance"]
            else:
                if current_position is not None:
                    original_distance = float(np.linalg.norm(cluster_center - current_position[:2]))
                else:
                    original_distance = 0.0
                frontier_result["safe_waypoint"] = cluster_center.tolist()
                frontier_result["safe_distance"] = original_distance
                frontier_result["is_truncated"] = False
                frontier_result["is_adjusted"] = False
                frontier_result["original_distance"] = original_distance

            frontier_result["is_obstacle_blocked"] = frontier_result["is_adjusted"]

            scored_frontiers.append(frontier_result)

        scored_frontiers.sort(key=lambda x: x["score"], reverse=True)

        return scored_frontiers[:top_k]

    def _check_realtime_detection_match(
            self,
            frontier_center: np.ndarray,
            current_position: Optional[np.ndarray],
            current_landmark: Optional[str],
            current_step_detections: Optional[Dict[str, List[str]]],
            viewpoint_directions: Dict[str, np.ndarray],
            valuable_viewpoints: List[str]
    ) -> Dict:
        result = {
            'matched': False,
            'score': 0.0,
            'reason': '',
            'viewpoint': None,
            'alignment_distance': 0.0
        }

        if not valuable_viewpoints or current_position is None:
            return result

        if len(viewpoint_directions) == 0:
            return result

        best_alignment = 0.0
        best_viewpoint = None

        for view_name in valuable_viewpoints:
            if view_name not in viewpoint_directions:
                continue

            view_direction = viewpoint_directions[view_name]
            alignment = self._compute_direction_alignment(
                current_position=current_position,
                frontier_center=frontier_center,
                viewpoint_direction=view_direction
            )

            if alignment > best_alignment:
                best_alignment = alignment
                best_viewpoint = view_name

        ALIGNMENT_THRESHOLD = 0.5
        WEIGHT_REALTIME = 0.95

        if best_alignment > ALIGNMENT_THRESHOLD and best_viewpoint is not None:
            detected_objects = current_step_detections.get(best_viewpoint, [])
            detected_str = ", ".join(detected_objects[:3])

            result['matched'] = True
            result['score'] = WEIGHT_REALTIME * best_alignment
            result['viewpoint'] = best_viewpoint
            result['alignment_distance'] = best_alignment
            result['reason'] = (
                f"MLLM detected target landmark \"{current_landmark}\" in {best_viewpoint} view, "
                f"this frontier aligns with view direction (alignment: {best_alignment:.2f}). "
                f"Detected objects: [{detected_str}]"
            )

        return result

    def _compute_semantic_score(
            self,
            frontier_center: np.ndarray,
            nearby_nodes: List,
            current_landmark: Optional[str],
            previous_anchor_label: Optional[str],
            fallback_semantics: List[str],
            semantic_radius: float
    ) -> Dict:
        WEIGHT_TARGET = 1.0
        WEIGHT_ANCHOR = 0.7
        WEIGHT_FALLBACK = 0.4
        WEIGHT_EXPLORATION = 0.1

        best_score = WEIGHT_EXPLORATION
        best_reason = "No relevant semantics found, suitable for exploring unknown areas"
        match_type = "exploration"
        matched_node_ids = []
        best_distance = semantic_radius

        if len(nearby_nodes) == 0:
            return {
                "score": best_score,
                "reason": best_reason,
                "match_type": match_type,
                "matched_node_ids": matched_node_ids,
                "best_distance": best_distance}

        for node in nearby_nodes:
            node_label = node.label.lower().strip()
            node_center = node.center_3d[:2]

            distance = np.linalg.norm(frontier_center - node_center)

            distance_factor = np.exp(-distance / semantic_radius)

            if current_landmark and self._semantic_match(node_label, current_landmark):
                score = WEIGHT_TARGET * distance_factor * node.confidence
                if score > best_score:
                    best_score = score
                    best_distance = distance
                    match_type = "target_landmark"
                    matched_node_ids = [node.node_id]
                    best_reason = (f"Target landmark \"{node.label}\" detected nearby "
                                   f"(distance: {distance:.1f}m, confidence: {node.confidence:.2f})")

            elif previous_anchor_label and self._semantic_match(node_label, previous_anchor_label):
                score = WEIGHT_ANCHOR * distance_factor * node.confidence
                if score > best_score:
                    best_score = score
                    best_distance = distance
                    match_type = "previous_anchor"
                    matched_node_ids = [node.node_id]
                    best_reason = (f"Previous subtask anchor \"{node.label}\" nearby "
                                   f"(distance: {distance:.1f}m)")

            else:
                for fallback in fallback_semantics:
                    if self._semantic_match(node_label, fallback):
                        score = WEIGHT_FALLBACK * distance_factor * node.confidence
                        if score > best_score:
                            best_score = score
                            best_distance = distance
                            match_type = "fallback_semantic"
                            matched_node_ids = [node.node_id]
                            best_reason = (f"\"{node.label}\" available for navigation reference "
                                           f"(distance: {distance:.1f}m)")
                        break

        return {
            "score": best_score,
            "reason": best_reason,
            "match_type": match_type,
            "matched_node_ids": matched_node_ids,
            "best_distance": best_distance
        }

    def _semantic_match(self, node_label: str, target: str) -> bool:
        target_lower = target.lower().strip()

        if target_lower in node_label or node_label in target_lower:
            return True

        target_words = set(target_lower.replace("_", " ").replace("-", " ").split())
        label_words = set(node_label.replace("_", " ").replace("-", " ").split())

        if target_words & label_words:
            return True

        return False

    def _fallback_frontier_ranking(self, frontier_clusters: List[np.ndarray], top_k: int) -> List[Dict]:
        if len(self.trajectory) == 0:
            results = []
            for i, center in enumerate(frontier_clusters[:top_k]):
                results.append({
                    "center": center.tolist(),
                    "score": 0.1,
                    "reason": "探索未知区域",
                    "nearby_nodes": [],
                    "match_type": "exploration",
                    "best_match_distance": float('inf'),
                    "safe_waypoint": center.tolist(),
                    "safe_distance": 0.0,
                    "is_truncated": False,
                    "is_adjusted": False,
                    "is_obstacle_blocked": False,
                    "matched_viewpoint": None
                })
            return results

        current_pos = self.trajectory[-1][:2]
        distances = [np.linalg.norm(center - current_pos) for center in frontier_clusters]
        sorted_indices = np.argsort(distances)

        results = []
        for idx in sorted_indices[:top_k]:
            center = frontier_clusters[idx]
            dist = distances[idx]
            results.append({
                "center": center.tolist(),
                "score": 0.1 * np.exp(-dist / 50.0),
                "reason": f"{dist:.1f}m",
                "nearby_nodes": [],
                "match_type": "exploration",
                "best_match_distance": float('inf'),
                "safe_waypoint": center.tolist(),
                "safe_distance": 0.0,
                "is_truncated": False,
                "is_adjusted": False,
                "is_obstacle_blocked": False,
                "matched_viewpoint": None
            })

        return results

    def format_frontiers_for_llm(self, guided_frontiers: List[Dict]) -> str:
        if not guided_frontiers:
            return "No exploration frontiers available currently."

        lines = ["### Recommended Exploration Directions (Semantically Guided Frontiers)\n"]

        for i, frontier in enumerate(guided_frontiers, 1):
            center = frontier["center"]
            safe_waypoint = frontier.get("safe_waypoint", center)
            score = frontier["score"]
            reason = frontier["reason"]
            match_type = frontier["match_type"]
            matched_viewpoint = frontier.get("matched_viewpoint")
            safe_distance = frontier.get("safe_distance", 0)
            is_truncated = frontier.get("is_truncated", False)
            is_adjusted = frontier.get("is_adjusted", False)

            type_desc = {
                "realtime_mllm_detection": "🎯 Realtime Detection",
                "target_landmark": "🎯 Target Landmark",
                "previous_anchor": "📍 Previous Anchor",
                "fallback_semantic": "🏙️ Urban Semantic",
                "exploration": "🔍 Unknown Exploration"
            }.get(match_type, "Exploration")

            if is_truncated and is_adjusted:
                waypoint_note = f"(truncated + adjusted, distance {safe_distance:.1f}m)"
            elif is_truncated:
                waypoint_note = f"(truncated to {safe_distance:.1f}m)"
            elif is_adjusted:
                waypoint_note = f"(adjusted for obstacle avoidance, distance {safe_distance:.1f}m)"
            else:
                waypoint_note = f"(direct, distance {safe_distance:.1f}m)"

            viewpoint_note = f" [from {matched_viewpoint} view]" if matched_viewpoint else ""

            lines.append(
                f"**Frontier {i}** [{type_desc}]{viewpoint_note}\n"
                f"  - Target Position: ({safe_waypoint[0]:.1f}, {safe_waypoint[1]:.1f}) {waypoint_note}\n"
                f"  - Relevance Score: {score:.2f}\n"
                f"  - Reason: {reason}\n"
            )

        lines.append(
            "\n**Suggestion**: Prioritize exploring towards the frontier with the highest score. "
            "Target positions have been adjusted for distance limits and obstacle avoidance.\n"
        )
        return "\n".join(lines)

    def visualize_unified(
            self,
            save_path: Optional[str] = None,
            show_trajectory: bool = True,
            show_current_position: bool = True,
            show_frontiers: bool = True,
            guided_frontiers: Optional[List[Dict]] = None,
            title: Optional[str] = None,
            figsize: Tuple[int, int] = (16, 14),
            dpi: int = 150,
            viewpoint_poses: Optional[Dict[str, np.ndarray]] = None,
            current_position: Optional[np.ndarray] = None,
            show_viewpoint_rays: bool = True,
            viewpoint_ray_length: float = 20.0,
            scene_graph=None,
            show_scene_graph_nodes: bool = True,
            scene_graph_node_radius: float = 50.0,
            current_landmark: Optional[str] = None,
            previous_anchor_label: Optional[str] = None
    ) -> plt.Figure:
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)

        explored_mask = self.get_explored_mask()
        occupied_mask = self.get_occupied_mask()
        boundary_mask = self.get_boundary_mask()
        frontier_mask = self.get_frontiers() if show_frontiers else np.zeros_like(explored_mask)
        height_map = self.get_height_map()

        rgba_image = np.zeros((self.map_size, self.map_size, 4), dtype=np.float32)

        unexplored = ~explored_mask
        rgba_image[unexplored] = [0.3, 0.3, 0.3, 1.0]

        explored_free = explored_mask & ~occupied_mask
        rgba_image[explored_free] = [0.8, 0.95, 0.8, 1.0]

        valid_heights = height_map[~np.isnan(height_map)]
        if len(valid_heights) > 0:
            h_min = np.percentile(valid_heights, 5)
            h_max = np.percentile(valid_heights, 95)
            if h_max <= h_min:
                h_max = h_min + 1

            height_normalized = np.zeros_like(height_map)
            obstacle_cells = occupied_mask & ~boundary_mask
            height_normalized[obstacle_cells] = np.clip(
                (height_map[obstacle_cells] - h_min) / (h_max - h_min), 0, 1)

            cmap = plt.cm.terrain
            for i in range(self.map_size):
                for j in range(self.map_size):
                    if obstacle_cells[i, j]:
                        color = cmap(height_normalized[i, j])
                        rgba_image[i, j] = [color[0], color[1], color[2], 1.0]

        rgba_image[boundary_mask] = [0.9, 0.2, 0.2, 1.0]

        if show_frontiers:
            rgba_image[frontier_mask] = [1.0, 0.8, 0.0, 1.0]

        ax.imshow(
            np.transpose(rgba_image, (1, 0, 2)),
            origin='lower',
            aspect='equal'
        )

        if show_scene_graph_nodes and scene_graph is not None:
            self._draw_scene_graph_nodes(
                ax=ax,
                scene_graph=scene_graph,
                current_position=current_position,
                radius=None,
                current_landmark=current_landmark,
                previous_anchor_label=previous_anchor_label
            )

        if show_viewpoint_rays and viewpoint_poses is not None and current_position is not None:
            self._draw_viewpoint_rays(
                ax=ax,
                viewpoint_poses=viewpoint_poses,
                current_position=current_position,
                ray_length=viewpoint_ray_length,
                guided_frontiers=guided_frontiers
            )

        if show_trajectory and len(self.trajectory) > 0:
            traj = self.get_trajectory_array()
            traj_grid = np.array([self.world_to_grid_coords(p) for p in traj])

            ax.plot(traj_grid[:, 0], traj_grid[:, 1],
                    'b-', linewidth=2.5, label='Trajectory', alpha=0.9, zorder=10)

            ax.scatter(traj_grid[0, 0], traj_grid[0, 1],
                       c='lime', s=200, marker='o', edgecolors='darkgreen',
                       linewidths=2, label='Start', zorder=11)

        if show_current_position and len(self.trajectory) > 0:
            current = self.trajectory[-1]
            curr_grid = self.world_to_grid_coords(current)
            ax.scatter(curr_grid[0], curr_grid[1],
                       c='red', s=300, marker='*', edgecolors='darkred',
                       linewidths=1.5, label='Current', zorder=12)

        if guided_frontiers and len(guided_frontiers) > 0:
            self._draw_guided_frontiers(ax, guided_frontiers)
        elif show_frontiers:
            clusters = self.get_frontier_clusters(min_cluster_size=5)
            for i, cluster_center in enumerate(clusters[:10]):
                grid_coords = self.world_to_grid_coords(cluster_center)
                ax.scatter(grid_coords[0], grid_coords[1],
                           c='orange', s=150, marker='D', edgecolors='darkorange',
                           linewidths=2, zorder=9)
                ax.annotate(f'F{i + 1}', (grid_coords[0] + 2, grid_coords[1] + 2),
                            fontsize=8, color='darkorange', weight='bold')

        legend_elements = self._create_legend_elements(
            show_trajectory=show_trajectory,
            guided_frontiers=guided_frontiers,
            show_viewpoint_rays=show_viewpoint_rays,
            viewpoint_poses=viewpoint_poses,
            show_scene_graph_nodes=show_scene_graph_nodes,
            scene_graph=scene_graph
        )
        ax.legend(handles=legend_elements, loc='upper right', fontsize=8,
                  framealpha=0.9, ncol=2)

        if len(valid_heights) > 0:
            sm = plt.cm.ScalarMappable(cmap='terrain',
                                       norm=plt.Normalize(vmin=h_min, vmax=h_max))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
            cbar.set_label('Obstacle Height (m)', fontsize=10)

        ax.set_xlabel('X (grid)', fontsize=11)
        ax.set_ylabel('Y (grid)', fontsize=11)

        stats = self.get_stats()
        if title:
            main_title = title
        else:
            main_title = '2.5D Occupancy Map'

        stats_text = (f"Coverage: {stats['exploration_coverage'] * 100:.1f}% | "
                      f"Obstacles: {stats['occupied_cells']} | "
                      f"Boundary: {stats['boundary_cells']} | "
                      f"Frontiers: {stats['frontier_cells']} | "
                      f"Traj: {stats['trajectory_length']} pts")

        if guided_frontiers and len(guided_frontiers) > 0:
            top_frontier = guided_frontiers[0]
            vp_info = f" [{top_frontier.get('matched_viewpoint', 'N/A')}]" if top_frontier.get('matched_viewpoint') else ""
            semantic_text = f" | Top: {top_frontier['match_type']}{vp_info} (score={top_frontier['score']:.2f})"
            stats_text += semantic_text

        if show_scene_graph_nodes and scene_graph is not None:
            total_nodes = len(scene_graph.nodes) if hasattr(scene_graph, 'nodes') else 0
            stats_text += f" | SceneGraph: {total_nodes} nodes"

        ax.set_title(f"{main_title}\n{stats_text}", fontsize=12, fontweight='bold')

        ax.grid(True, alpha=0.2, linestyle='--')

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')

        return fig

    def _draw_scene_graph_nodes(
            self,
            ax,
            scene_graph,
            current_position: Optional[np.ndarray],
            radius: float = None,
            current_landmark: Optional[str] = None,
            previous_anchor_label: Optional[str] = None
    ):
        nodes_dict = None
        if hasattr(scene_graph, 'nodes') and scene_graph.nodes:
            nodes_dict = scene_graph.nodes
        elif hasattr(scene_graph, 'get_all_nodes'):
            nodes_dict = scene_graph.get_all_nodes()

        if nodes_dict is None or len(nodes_dict) == 0:
            return

        print(f"[OccMap] Drawing {len(nodes_dict)} scene graph nodes")

        node_color = '#FF8C00'
        highlight_target_color = '#FF1493'
        highlight_anchor_color = '#00CED1'

        drawn_count = 0

        for node_id, node in nodes_dict.items():
            node_pos_3d = None
            if hasattr(node, 'center_3d') and node.center_3d is not None:
                node_pos_3d = np.array(node.center_3d)
            elif hasattr(node, 'position') and node.position is not None:
                node_pos_3d = np.array(node.position)
            elif hasattr(node, 'center') and node.center is not None:
                node_pos_3d = np.array(node.center)
            elif isinstance(node, dict):
                node_pos_3d = node.get('center_3d') or node.get('position') or node.get('center')
                if node_pos_3d is not None:
                    node_pos_3d = np.array(node_pos_3d)

            if node_pos_3d is None:
                print(f"[OccMap] Warning: Node {node_id} has no position")
                continue

            node_pos_2d = np.array([node_pos_3d[0], node_pos_3d[1]])

            if isinstance(node, dict):
                label = node.get('label', str(node_id))
                confidence = node.get('confidence', 0.5)
            else:
                label = getattr(node, 'label', str(node_id))
                confidence = getattr(node, 'confidence', 0.5)

            if current_position is not None and radius is not None:
                current_pos_2d = np.array([current_position[0], current_position[1]])
                distance = np.linalg.norm(node_pos_2d - current_pos_2d)
                if distance > radius:
                    continue

            grid_coords = self.world_to_grid_coords(node_pos_2d)

            is_target = current_landmark and label.lower() in current_landmark.lower()
            is_anchor = previous_anchor_label and label.lower() in previous_anchor_label.lower()

            if is_target:
                color = highlight_target_color
                edge_color = 'white'
                size = 150
                zorder = 15
            elif is_anchor:
                color = highlight_anchor_color
                edge_color = 'white'
                size = 130
                zorder = 14
            else:
                color = node_color
                edge_color = 'white'
                size = 100
                zorder = 10

            ax.scatter(
                grid_coords[0], grid_coords[1],
                c=color, s=size, marker='o',
                edgecolors=edge_color, linewidths=2,
                alpha=0.9, zorder=zorder
            )

            display_label = label[:15] + '...' if len(label) > 15 else label
            label_text = f"{display_label}\nconf:{confidence:.2f}\n({node_pos_2d[0]:.1f},{node_pos_2d[1]:.1f})"

            if is_target:
                fontsize = 7
                fontweight = 'bold'
                text_color = 'white'
                bbox_color = highlight_target_color
            elif is_anchor:
                fontsize = 7
                fontweight = 'bold'
                text_color = 'white'
                bbox_color = highlight_anchor_color
            else:
                fontsize = 6
                fontweight = 'normal'
                text_color = 'black'
                bbox_color = 'white'

            ax.annotate(
                label_text,
                (grid_coords[0], grid_coords[1]),
                xytext=(10, 10),
                textcoords='offset points',
                fontsize=fontsize,
                color=text_color,
                weight=fontweight,
                bbox=dict(
                    boxstyle='round,pad=0.3',
                    facecolor=bbox_color,
                    alpha=0.85,
                    edgecolor='gray',
                    linewidth=0.5
                ),
                zorder=zorder + 1
            )

            drawn_count += 1

        print(f"[OccMap] Actually drawn {drawn_count} nodes")

        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor=node_color,
                   markersize=8, label='SG: Node'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=highlight_target_color,
                   markersize=10, label='SG: Target Match'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=highlight_anchor_color,
                   markersize=10, label='SG: Anchor Match'),
        ]

        return legend_elements

    def _draw_viewpoint_rays(
            self,
            ax,
            viewpoint_poses: Dict[str, np.ndarray],
            current_position: np.ndarray,
            ray_length: float,
            guided_frontiers: Optional[List[Dict]]
    ):
        viewpoint_colors = {
            'left': '#E74C3C',
            'slightly_left': '#E67E22',
            'front': '#2ECC71',
            'slightly_right': '#3498DB',
            'right': '#9B59B6'
        }

        matched_viewpoints = set()
        if guided_frontiers:
            for f in guided_frontiers:
                vp = f.get('matched_viewpoint')
                if vp:
                    matched_viewpoints.add(vp)

        current_grid = self.world_to_grid_coords(current_position[:2])

        for view_name, pose in viewpoint_poses.items():
            if not isinstance(pose, np.ndarray):
                pose = np.array(pose)

            direction = self._get_viewpoint_direction(pose)
            color = viewpoint_colors.get(view_name, '#95A5A6')

            end_world = current_position[:2] + direction * ray_length
            end_grid = self.world_to_grid_coords(end_world)

            linewidth = 3 if view_name in matched_viewpoints else 1.5
            linestyle = '-' if view_name in matched_viewpoints else '--'
            alpha = 0.9 if view_name in matched_viewpoints else 0.5

            ax.annotate(
                '',
                xy=(end_grid[0], end_grid[1]),
                xytext=(current_grid[0], current_grid[1]),
                arrowprops=dict(
                    arrowstyle='-|>',
                    color=color,
                    lw=linewidth,
                    ls=linestyle,
                    alpha=alpha
                ),
                zorder=8
            )

            label_pos = (
                current_grid[0] + direction[0] * (ray_length / self.resolution) * 0.7,
                current_grid[1] + direction[1] * (ray_length / self.resolution) * 0.7
            )
            ax.text(
                label_pos[0], label_pos[1],
                view_name.replace('_', '\n'),
                fontsize=7,
                color=color,
                ha='center',
                va='center',
                weight='bold' if view_name in matched_viewpoints else 'normal',
                alpha=0.8,
                zorder=9
            )

    def _draw_guided_frontiers(self, ax, guided_frontiers: List[Dict]):
        match_type_style = {
            "realtime_mllm_detection": {"color": "#FF4500", "marker": "★", "label": "Realtime"},
            "target_landmark": {"color": "#FF1493", "marker": "★", "label": "Target"},
            "previous_anchor": {"color": "#00CED1", "marker": "◆", "label": "Anchor"},
            "fallback_semantic": {"color": "#9370DB", "marker": "●", "label": "Semantic"},
            "exploration": {"color": "#FFA500", "marker": "◇", "label": "Explore"}
        }

        for i, frontier in enumerate(guided_frontiers):
            center = frontier["center"]
            safe_waypoint = frontier.get("safe_waypoint", center)
            score = frontier["score"]
            match_type = frontier["match_type"]
            matched_viewpoint = frontier.get("matched_viewpoint")

            style = match_type_style.get(match_type, match_type_style["exploration"])

            center_grid = self.world_to_grid_coords(np.array(center))
            ax.scatter(center_grid[0], center_grid[1],
                       c=style["color"], s=100, marker='o',
                       edgecolors='white', linewidths=1, alpha=0.4, zorder=13)

            waypoint_grid = self.world_to_grid_coords(np.array(safe_waypoint))
            circle_size = 200 + score * 300

            ax.scatter(waypoint_grid[0], waypoint_grid[1],
                       c=style["color"], s=circle_size, marker='o',
                       edgecolors='white', linewidths=3, alpha=0.8, zorder=15)

            if center != safe_waypoint:
                ax.plot([center_grid[0], waypoint_grid[0]],
                        [center_grid[1], waypoint_grid[1]],
                        color=style["color"], linestyle=':', linewidth=1.5, alpha=0.6, zorder=14)

            ax.annotate(
                f'{i + 1}',
                (waypoint_grid[0], waypoint_grid[1]),
                fontsize=12, fontweight='bold', color='white',
                ha='center', va='center', zorder=16
            )

            viewpoint_str = f"\n[{matched_viewpoint}]" if matched_viewpoint else ""
            label_text = f"F{i + 1}: {score:.2f}{viewpoint_str}"
            ax.annotate(
                label_text,
                (waypoint_grid[0] + 5, waypoint_grid[1] + 5),
                fontsize=8, color=style["color"], weight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor=style["color"]),
                zorder=17
            )

    def _create_legend_elements(
            self,
            show_trajectory: bool,
            guided_frontiers: Optional[List[Dict]],
            show_viewpoint_rays: bool,
            viewpoint_poses: Optional[Dict[str, np.ndarray]],
            show_scene_graph_nodes: bool,
            scene_graph
    ) -> List:
        from matplotlib.lines import Line2D

        legend_elements = [
            Patch(facecolor=(0.3, 0.3, 0.3), label='Unexplored'),
            Patch(facecolor=(0.8, 0.95, 0.8), label='Explored (Free)'),
            Patch(facecolor=(0.6, 0.7, 0.5), label='Obstacles (Height)'),
            Patch(facecolor=(0.9, 0.2, 0.2), label='City Boundary'),
            Patch(facecolor=(1.0, 0.8, 0.0), label='Frontier'),
        ]

        if show_trajectory and len(self.trajectory) > 0:
            legend_elements.extend([
                Line2D([0], [0], color='blue', linewidth=2, label='Trajectory'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='lime',
                       markersize=10, label='Start'),
                Line2D([0], [0], marker='*', color='w', markerfacecolor='red',
                       markersize=15, label='Current'),
            ])

        if guided_frontiers and len(guided_frontiers) > 0:
            legend_elements.extend([
                Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF4500',
                       markersize=12, markeredgecolor='white', markeredgewidth=2, label='Realtime MLLM'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF1493',
                       markersize=12, markeredgecolor='white', markeredgewidth=2, label='Target Landmark'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='#00CED1',
                       markersize=12, markeredgecolor='white', markeredgewidth=2, label='Previous Anchor'),
            ])

        if show_viewpoint_rays and viewpoint_poses is not None:
            viewpoint_colors = {
                'left': '#E74C3C', 'slightly_left': '#E67E22', 'front': '#2ECC71',
                'slightly_right': '#3498DB', 'right': '#9B59B6'
            }
            for vp_name, vp_color in viewpoint_colors.items():
                if vp_name in viewpoint_poses:
                    legend_elements.append(
                        Line2D([0], [0], color=vp_color, linewidth=2, linestyle='--',
                               label=f'View: {vp_name}')
                    )

        if show_scene_graph_nodes and scene_graph is not None:
            legend_elements.extend([
                Line2D([0], [0], marker='s', color='w', markerfacecolor='#1E90FF',
                       markersize=8, label='SG: Landmark'),
                Line2D([0], [0], marker='^', color='w', markerfacecolor='#32CD32',
                       markersize=8, label='SG: Road'),
                Line2D([0], [0], marker='p', color='w', markerfacecolor='#FF6347',
                       markersize=8, label='SG: Building'),
                Line2D([0], [0], marker='h', color='w', markerfacecolor='#FFD700',
                       markersize=8, label='SG: Intersection'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF1493',
                       markersize=10, markeredgecolor='#FF1493', markeredgewidth=2,
                       label='SG: Target Match'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='#00CED1',
                       markersize=10, markeredgecolor='#00CED1', markeredgewidth=2,
                       label='SG: Anchor Match'),
            ])

        return legend_elements

    def save_matplotlib_viz_to_nav_data(
            self,
            scene_id: str,
            episode_id: str,
            step_counter: int,
            current_position: Optional[np.ndarray] = None,
            show_trajectory: bool = True,
            show_frontiers: bool = True,
            guided_frontiers: Optional[List[Dict]] = None,
            output_base: str = './output/nav_data',
            viewpoint_poses: Optional[Dict[str, np.ndarray]] = None,
            show_viewpoint_rays: bool = True,
            scene_graph=None,
            show_scene_graph_nodes: bool = True,
            scene_graph_node_radius: float = 50.0,
            current_landmark: Optional[str] = None,
            previous_anchor_label: Optional[str] = None
    ):
        save_dir = os.path.join(output_base, str(scene_id), str(episode_id), 'occupancy_map')
        os.makedirs(save_dir, exist_ok=True)

        img_path = os.path.join(save_dir, f'step_{step_counter:04d}.png')
        npy_path = os.path.join(save_dir, f'step_{step_counter:04d}_map.npy')

        title = f'Step {step_counter} | Scene {scene_id} | Episode {episode_id}'

        fig = self.visualize_unified(
            save_path=img_path,
            show_trajectory=show_trajectory,
            show_current_position=True,
            show_frontiers=show_frontiers,
            guided_frontiers=guided_frontiers,
            title=title,
            figsize=(14, 12),
            dpi=120,
            viewpoint_poses=viewpoint_poses,
            current_position=current_position,
            show_viewpoint_rays=show_viewpoint_rays,
            scene_graph=scene_graph,
            show_scene_graph_nodes=show_scene_graph_nodes,
            scene_graph_node_radius=scene_graph_node_radius,
            current_landmark=current_landmark,
            previous_anchor_label=previous_anchor_label
        )
        plt.close(fig)

        np.save(npy_path, self.map)

        frontier_path = os.path.join(save_dir, f'step_{step_counter:04d}_frontiers.npy')
        np.save(frontier_path, self.get_frontiers())

        if guided_frontiers:
            import json
            guided_path = os.path.join(save_dir, f'step_{step_counter:04d}_guided_frontiers.json')
            with open(guided_path, 'w', encoding='utf-8') as f:
                json.dump(guided_frontiers, f, indent=2, ensure_ascii=False)

    def update_and_visualize(
            self,
            viewpoint_dep_imgs: Dict[str, np.ndarray],
            viewpoint_poses: Dict[str, np.ndarray],
            image_order: List[str],
            current_position: np.ndarray,
            scene_id: str,
            episode_id: str,
            step_counter: int,
            scene_graph=None,
            current_landmark: Optional[str] = None,
            previous_anchor_label: Optional[str] = None,
            output_base: str = './output/nav_data',
            show_trajectory: bool = True,
            show_frontiers: bool = True,
            print_stats: bool = True,
            save_viz: bool = True,
            semantic_radius: float = 30.0,
            top_k_frontiers: int = 3,
            current_step_detections: Optional[Dict[str, List[str]]] = None,
            max_exploration_distance: float = 5,
            show_viewpoint_rays: bool = True,
            show_scene_graph_nodes: bool = True,
            scene_graph_node_radius: float = 50.0
    ) -> Dict:
        update_stats = self.update(
            viewpoint_dep_imgs=viewpoint_dep_imgs,
            viewpoint_poses=viewpoint_poses,
            image_order=image_order,
            current_position=current_position,
            viewpoint_rgb_imgs=None
        )

        if print_stats:
            sg_nodes = len(scene_graph.nodes) if scene_graph and hasattr(scene_graph, 'nodes') else 0
            print(f"[OccMap] Step {step_counter}: "
                  f"Coverage={update_stats['exploration_coverage'] * 100:.1f}%, "
                  f"Obstacles={update_stats['occupied_cells']}, "
                  f"Boundary={update_stats['boundary_cells']}, "
                  f"Frontiers={update_stats['frontier_cells']}, "
                  f"SceneGraph={sg_nodes} nodes")

        guided_frontiers = self.get_semantically_guided_frontiers(
            scene_graph=scene_graph,
            current_landmark=current_landmark,
            previous_anchor_label=previous_anchor_label,
            semantic_radius=semantic_radius,
            top_k=top_k_frontiers,
            min_cluster_size=5,
            current_step_detections=current_step_detections,
            viewpoint_poses=viewpoint_poses,
            current_position=current_position,
            compute_safe_waypoints=True,
            max_exploration_distance=max_exploration_distance
        )

        if save_viz:
            self.save_matplotlib_viz_to_nav_data(
                scene_id=scene_id,
                episode_id=episode_id,
                step_counter=step_counter,
                current_position=current_position,
                show_trajectory=show_trajectory,
                show_frontiers=show_frontiers,
                guided_frontiers=guided_frontiers,
                output_base=output_base,
                viewpoint_poses=viewpoint_poses,
                show_viewpoint_rays=show_viewpoint_rays,
                scene_graph=scene_graph,
                show_scene_graph_nodes=show_scene_graph_nodes,
                scene_graph_node_radius=scene_graph_node_radius,
                current_landmark=current_landmark,
                previous_anchor_label=previous_anchor_label
            )

        frontier_prompt = self.format_frontiers_for_llm(guided_frontiers)

        return {
            'update_stats': update_stats,
            'guided_frontiers': guided_frontiers,
            'frontier_prompt': frontier_prompt
        }

    def reset(self):
        self._init_map()
        self.trajectory = []
        self.total_updates = 0
        self.total_points_processed = 0
        self.total_rays_cast = 0
        self._frontier_cache = None
        self._frontier_cache_valid = False

    def get_stats(self) -> Dict:
        explored_mask = self.get_explored_mask()
        occupied_mask = self.get_occupied_mask()
        boundary_mask = self.get_boundary_mask()
        frontier_mask = self.get_frontiers()
        height_map = self.get_height_map()
        valid_heights = height_map[~np.isnan(height_map)]

        stats = {
            'map_size': self.map_size,
            'resolution': self.resolution,
            'total_updates': self.total_updates,
            'total_points_processed': self.total_points_processed,
            'total_rays_cast': self.total_rays_cast,
            'explored_cells': int(np.sum(explored_mask)),
            'exploration_coverage': float(np.sum(explored_mask) / (self.map_size ** 2)),
            'occupied_cells': int(np.sum(occupied_mask)),
            'boundary_cells': int(np.sum(boundary_mask)),
            'frontier_cells': int(np.sum(frontier_mask)),
            'trajectory_length': len(self.trajectory)
        }

        if len(valid_heights) > 0:
            stats.update({
                'height_min': float(np.min(valid_heights)),
                'height_max': float(np.max(valid_heights)),
                'height_mean': float(np.mean(valid_heights)),
                'height_std': float(np.std(valid_heights))
            })

        return stats