import numpy as np
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import json
from utils.logger import logger

@dataclass
class ObjectNode:
    node_id: str
    label: str
    confidence: float
    label_variants: Dict[str, float] = field(default_factory=dict)
    center_3d: np.ndarray = field(default_factory=lambda: np.zeros(3))
    bbox_3d: Optional[Dict] = None
    height_above_ground: float = 0.0
    pointcloud: Optional[np.ndarray] = None
    first_seen_step: int = 0
    last_seen_step: int = 0
    observation_count: int = 1
    viewpoints: List[str] = field(default_factory=list)
    subtask_id: int = -1
    is_anchor: bool = False
    anchor_for_subtask: int = -1

    def __post_init__(self):
        if not isinstance(self.center_3d, np.ndarray):
            self.center_3d = np.array(self.center_3d, dtype=np.float32)

        if self.height_above_ground == 0.0:
            self.height_above_ground = -self.center_3d[2]

        if not self.label_variants:
            self.label_variants = {self.label: self.confidence}

    @property
    def primary_label(self) -> str:
        if not self.label_variants:
            return self.label
        return max(self.label_variants, key=self.label_variants.get)

    def _update_primary_label(self):
        if self.label_variants:
            self.label = max(self.label_variants, key=self.label_variants.get)
            self.confidence = self.label_variants[self.label]

    def update_from_detection(self, detection: Dict, current_step: int):
        new_label = detection.get('label', self.label)
        new_conf = detection.get('score', detection.get('confidence', 0.5))

        if new_label in self.label_variants:
            self.label_variants[new_label] = max(self.label_variants[new_label], new_conf)
        else:
            self.label_variants[new_label] = new_conf

        self._update_primary_label()

        new_center = np.array(detection.get('coord', self.center_3d))
        weight = self.observation_count / (self.observation_count + 1)
        self.center_3d = weight * self.center_3d + (1 - weight) * new_center

        if 'bbox_3d' in detection and detection['bbox_3d']:
            self._merge_bbox(detection['bbox_3d'])

        self.last_seen_step = current_step
        self.observation_count += 1

        if 'viewpoint' in detection and detection['viewpoint'] not in self.viewpoints:
            self.viewpoints.append(detection['viewpoint'])

        if 'height_above_ground' in detection:
            new_height = detection['height_above_ground']
            self.height_above_ground = weight * self.height_above_ground + (1 - weight) * new_height
        else:
            self.height_above_ground = -self.center_3d[2]

    def _merge_bbox(self, new_bbox: Dict):
        if self.bbox_3d is None:
            self.bbox_3d = new_bbox.copy()
            return

        old_min = np.array(self.bbox_3d['min_bound'])
        old_max = np.array(self.bbox_3d['max_bound'])
        new_min = np.array(new_bbox['min_bound'])
        new_max = np.array(new_bbox['max_bound'])

        merged_min = np.minimum(old_min, new_min)
        merged_max = np.maximum(old_max, new_max)

        self.bbox_3d = {
            'min_bound': merged_min.tolist(),
            'max_bound': merged_max.tolist(),
            'center': ((merged_min + merged_max) / 2).tolist(),
            'size': (merged_max - merged_min).tolist()
        }

    def to_dict(self) -> Dict:
        return {
            "node_id": self.node_id,
            "label": self.label,
            "primary_label": self.primary_label,
            "label_variants": self.label_variants,
            "confidence": round(self.confidence, 3),
            "center_3d": [round(c, 2) for c in self.center_3d.tolist()],
            "height_above_ground": round(self.height_above_ground, 2),
            "bbox_3d": self.bbox_3d,
            "first_seen_step": self.first_seen_step,
            "last_seen_step": self.last_seen_step,
            "observation_count": self.observation_count,
            "viewpoints": self.viewpoints,
            "subtask_id": self.subtask_id,
            "is_anchor": self.is_anchor,
            "anchor_for_subtask": self.anchor_for_subtask
        }

class CitySceneGraph:
    def __init__(
            self,
            merge_distance_threshold: float = 5.0,
            edge_distance_threshold: float = 30.0,
            vertical_threshold: float = 3.0,
            semantic_match_strict: bool = False
    ):
        self.merge_distance_threshold = merge_distance_threshold
        self.edge_distance_threshold = edge_distance_threshold
        self.vertical_threshold = vertical_threshold
        self.semantic_match_strict = semantic_match_strict
        self.nodes: Dict[str, ObjectNode] = {}
        self.node_counter: int = 0
        self.current_subtask_id: int = 0
        self.completed_anchors: List[str] = []
        self.agent_trajectory: List[np.ndarray] = []
        self.relation_snapshots: List[Dict] = []

        logger.info(f"🗺️ CitySceneGraph Initialization Completed | Merge Threshold={merge_distance_threshold}m, Strict Semantic Match={'Yes' if semantic_match_strict else 'No (Compatibility Mode)'}")

    def reset(self):
        self.nodes.clear()
        self.node_counter = 0
        self.current_subtask_id = 0
        self.completed_anchors.clear()
        self.agent_trajectory.clear()
        logger.info("🔄 Scene graph has been reset")

    @staticmethod
    def _extract_core_noun(label: str) -> str:
        words = label.lower().strip().split()
        return words[-1] if words else ""

    def _is_semantically_compatible(self, label1: str, label2: str) -> bool:
        if label1.lower().strip() == label2.lower().strip():
            return True

        noun1 = self._extract_core_noun(label1)
        noun2 = self._extract_core_noun(label2)

        return noun1 == noun2 and noun1 != ""

    def _normalize_angle(self, angle_rad: float) -> float:
        while angle_rad > np.pi:
            angle_rad -= 2 * np.pi
        while angle_rad < -np.pi:
            angle_rad += 2 * np.pi
        return angle_rad

    def _yaw_to_direction_text(self, yaw_deg: float) -> str:
        yaw = yaw_deg % 360
        if yaw > 180:
            yaw -= 359.999

        if -22.5 <= yaw < 22.5:
            return "front"
        elif 22.5 <= yaw < 67.5:
            return "front-right"
        elif 67.5 <= yaw < 112.5:
            return "right"
        elif 112.5 <= yaw < 157.5:
            return "rear-right"
        elif yaw >= 157.5 or yaw < -157.5:
            return "rear"
        elif -157.5 <= yaw < -112.5:
            return "rear-left"
        elif -112.5 <= yaw < -67.5:
            return "left"
        elif -67.5 <= yaw < -22.5:
            return "front-left"
        else:
            return "nearby"

    def update_from_spatial_result(
            self,
            spatial_result: Dict,
            current_step: int,
            current_subtask_id: int,
            agent_position: np.ndarray
    ) -> Dict:
        self.current_subtask_id = current_subtask_id

        if not isinstance(agent_position, np.ndarray):
            agent_position = np.array(agent_position)
        self.agent_trajectory.append(agent_position.copy())

        detections = spatial_result.get("detections", [])

        new_nodes = []
        updated_nodes = []
        merged_labels = []

        for detection in detections:
            label = detection.get('label', 'unknown')
            coord = detection.get('coord', [0, 0, 0])

            if not coord or (isinstance(coord, list) and len(coord) < 3):
                continue

            matched_node_id, match_type = self._find_matching_node(detection)

            if matched_node_id:
                old_label = self.nodes[matched_node_id].label
                self.nodes[matched_node_id].update_from_detection(detection, current_step)
                updated_nodes.append(matched_node_id)

                if match_type == 'semantic_compatible' and old_label != label:
                    merged_labels.append(f"{label} → {matched_node_id}({old_label})")
                    logger.debug(f"  🔄 Label Fusion: '{label}' merged into node {matched_node_id} (Primary Label: '{self.nodes[matched_node_id].label}')")
                else:
                    logger.debug(f"  🔄 Updating Node: {matched_node_id} ({label})")
            else:
                new_node = self._create_node_from_detection(detection, current_step, current_subtask_id)
                self.nodes[new_node.node_id] = new_node
                new_nodes.append(new_node.node_id)
                logger.debug(f"  ✨ New Node Created: {new_node.node_id} ({label}) @ {coord}")

        stats = {
            "new_nodes": len(new_nodes),
            "updated_nodes": len(updated_nodes),
            "total_nodes": len(self.nodes),
            "merged_labels": len(merged_labels)
        }

        logger.info(f"📊 Scene Graph Update Step {current_step}: New Nodes={stats['new_nodes']}, Updated={stats['updated_nodes']}, Merged Labels={stats['merged_labels']} | Total: {stats['total_nodes']} nodes")

        return stats

    def _find_matching_node(self, detection: Dict) -> Tuple[Optional[str], str]:
        det_center = np.array(detection.get('coord', [0, 0, 0]))
        det_label = detection.get('label', 'unknown')

        best_match_id = None
        best_distance = float('inf')
        match_type = 'none'

        for node_id, node in self.nodes.items():
            distance = np.linalg.norm(node.center_3d[:2] - det_center[:2])

            if distance >= self.merge_distance_threshold:
                continue

            if distance >= best_distance:
                continue

            if self.semantic_match_strict:
                if node.label.lower().strip() == det_label.lower().strip():
                    best_distance = distance
                    best_match_id = node_id
                    match_type = 'exact'
            else:
                if self._is_semantically_compatible(node.label, det_label):
                    best_distance = distance
                    best_match_id = node_id
                    match_type = 'exact' if node.label.lower().strip() == det_label.lower().strip() else 'semantic_compatible'
                else:
                    for variant_label in node.label_variants.keys():
                        if self._is_semantically_compatible(variant_label, det_label):
                            best_distance = distance
                            best_match_id = node_id
                            match_type = 'semantic_compatible'
                            break

        return best_match_id, match_type

    def _create_node_from_detection(
            self,
            detection: Dict,
            current_step: int,
            current_subtask_id: int
    ) -> ObjectNode:
        self.node_counter += 1
        node_id = f"obj_{self.node_counter:04d}"

        center = np.array(detection.get('coord', [0, 0, 0]), dtype=np.float32)
        label = detection.get('label', 'unknown')
        confidence = detection.get('score', detection.get('confidence', 0.5))

        if 'height_above_ground' in detection:
            height_above_ground = detection['height_above_ground']
        else:
            height_above_ground = -center[2]

        node = ObjectNode(
            node_id=node_id,
            label=label,
            confidence=confidence,
            label_variants={label: confidence},
            center_3d=center,
            bbox_3d=detection.get('bbox_3d'),
            height_above_ground=height_above_ground,
            pointcloud=detection.get('pointcloud'),
            first_seen_step=current_step,
            last_seen_step=current_step,
            observation_count=1,
            viewpoints=[detection.get('viewpoint', 'unknown')],
            subtask_id=current_subtask_id
        )

        return node

    def compute_agent_to_nodes_relations(
            self,
            agent_position: np.ndarray,
            agent_yaw: float,
            subgraph_node_ids: Set[str]
    ) -> List[Dict]:
        if not isinstance(agent_position, np.ndarray):
            agent_position = np.array(agent_position)

        relations = []

        for node_id in subgraph_node_ids:
            if node_id not in self.nodes:
                continue

            node = self.nodes[node_id]

            diff = node.center_3d[:2] - agent_position[:2]
            distance = np.linalg.norm(diff)

            dx = diff[0]
            dy = diff[1]
            world_yaw = np.arctan2(dy, dx)

            relative_yaw = self._normalize_angle(world_yaw - agent_yaw)
            yaw_angle_deg = np.degrees(relative_yaw)

            relations.append({
                "node_id": node_id,
                "label": node.label,
                "center_3d": node.center_3d.tolist(),
                "confidence": node.confidence,
                "distance": round(distance, 2),
                "yaw_angle_deg": round(yaw_angle_deg, 1),
                "height_above_ground": round(node.height_above_ground, 2),
                "is_anchor": node.is_anchor,
                "observation_count": node.observation_count
            })

        relations.sort(key=lambda x: x['distance'])

        return relations

    def mark_node_as_anchor(self, node_id: str, subtask_id: int) -> bool:
        if node_id not in self.nodes:
            logger.warning(f"⚠️ Cannot mark anchor: Node {node_id} does not exist")
            return False

        node = self.nodes[node_id]
        node.is_anchor = True
        node.anchor_for_subtask = subtask_id

        if node_id not in self.completed_anchors:
            self.completed_anchors.append(node_id)

        logger.info(f"🎯 Node {node_id} ({node.label}) has been marked as the anchor for subtask {subtask_id}")
        return True

    def get_anchor_for_subtask(self, subtask_id: int) -> Optional[ObjectNode]:
        for node_id in self.completed_anchors:
            if node_id in self.nodes:
                node = self.nodes[node_id]
                if node.anchor_for_subtask == subtask_id:
                    return node
        return None

    def get_previous_anchor(self, current_subtask_id: int) -> Optional[ObjectNode]:
        if current_subtask_id <= 0:
            return None
        return self.get_anchor_for_subtask(current_subtask_id - 1)

    def extract_task_focused_subgraph(
            self,
            current_subtask_id: int,
            previous_anchor_id: Optional[str] = None,
            semantic_keywords: List[str] = None,
            max_semantic_nodes: int = 25
    ) -> Dict:
        subgraph_nodes: Dict[str, ObjectNode] = {}

        anchor_node = None
        current_task_nodes = []
        semantic_related_nodes = []

        for node_id, node in self.nodes.items():
            if node.subtask_id == current_subtask_id:
                subgraph_nodes[node_id] = node
                current_task_nodes.append(node_id)

        if previous_anchor_id and previous_anchor_id in self.nodes:
            anchor_node = self.nodes[previous_anchor_id]
            subgraph_nodes[previous_anchor_id] = anchor_node
        elif current_subtask_id > 0:
            prev_anchor = self.get_previous_anchor(current_subtask_id)
            if prev_anchor:
                anchor_node = prev_anchor
                subgraph_nodes[prev_anchor.node_id] = prev_anchor

        if semantic_keywords:
            semantic_candidates = []

            for node_id, node in self.nodes.items():
                if node_id in subgraph_nodes:
                    continue

                all_labels = [node.label] + list(node.label_variants.keys())
                matched = False

                for node_label in all_labels:
                    label_lower = node_label.lower()
                    for keyword in semantic_keywords:
                        if keyword.lower() in label_lower or label_lower in keyword.lower():
                            matched = True
                            break
                    if matched:
                        break

                if matched:
                    semantic_candidates.append((node_id, node))

            semantic_candidates.sort(key=lambda x: x[1].observation_count, reverse=True)

            for node_id, node in semantic_candidates[:max_semantic_nodes]:
                subgraph_nodes[node_id] = node
                semantic_related_nodes.append(node_id)

        subgraph = {
            "nodes": subgraph_nodes,
            "anchor_node": anchor_node,
            "anchor_node_id": anchor_node.node_id if anchor_node else None,
            "current_task_node_ids": current_task_nodes,
            "semantic_related_node_ids": semantic_related_nodes,
            "current_subtask_id": current_subtask_id,
            "stats": {
                "total_nodes": len(subgraph_nodes),
                "current_task_nodes": len(current_task_nodes),
                "semantic_nodes": len(semantic_related_nodes),
                "has_anchor": anchor_node is not None
            }
        }

        return subgraph

    def subgraph_to_llm_prompt(
            self,
            subgraph: Dict,
            current_subtask_instruction: str,
            agent_position: np.ndarray,
            agent_yaw: float = 0.0,
            include_spatial_relations: bool = True,
            max_objects_in_prompt: int = 30
    ) -> str:
        if not isinstance(agent_position, np.ndarray):
            agent_position = np.array(agent_position)

        lines = []

        lines.append("### Scene Graph Nodes (spatial relations centered on the drone)")
        lines.append("")

        nodes = subgraph.get("nodes", {})

        if not nodes:
            lines.append("- No relevant objects detected in current field of view.")
        else:
            subgraph_node_ids = set(nodes.keys())
            relations = self.compute_agent_to_nodes_relations(
                agent_position=agent_position,
                agent_yaw=agent_yaw,
                subgraph_node_ids=subgraph_node_ids
            )

            for rel in relations[:max_objects_in_prompt]:
                node = nodes.get(rel['node_id'])
                if not node:
                    continue

                anchor_mark = " [Anchor]" if rel['is_anchor'] else ""

                other_variants = [v for v in node.label_variants.keys() if v != node.label]
                if other_variants:
                    variants_str = f" (also called: {', '.join(other_variants[:2])})"
                else:
                    variants_str = ""

                direction_text = self._yaw_to_direction_text(rel['yaw_angle_deg'])

                lines.append(f"- **{rel['node_id']}**: \"{node.label}\"{variants_str}{anchor_mark}")
                lines.append(f"  - Confidence: {rel['confidence']:.2f}")
                lines.append(
                    f"  - Center: [{rel['center_3d'][0]:.1f}, {rel['center_3d'][1]:.1f}, {rel['center_3d'][2]:.1f}]")

                if node.bbox_3d:
                    min_b = node.bbox_3d['min_bound']
                    max_b = node.bbox_3d['max_bound']
                    lines.append(
                        f"  - BBox: [[{min_b[0]:.1f}, {min_b[1]:.1f}, {min_b[2]:.1f}], [{max_b[0]:.1f}, {max_b[1]:.1f}, {max_b[2]:.1f}]]")

                lines.append(f"  - Height above ground: {rel['height_above_ground']:.1f} m")
                lines.append(f"  - Distance: {rel['distance']:.1f} m")
                lines.append(f"  - Relative Yaw: {rel['yaw_angle_deg']:.1f}° ({direction_text})")
                lines.append("")

        prompt = "\n".join(lines)

        return prompt

    def visualize_scene_graph(
            self,
            save_path: str,
            highlight_subgraph: Dict = None,
            agent_yaw: float = None,
            show_labels: bool = True,
            show_trajectory: bool = True,
            figsize: Tuple[int, int] = (16, 12),
            height_range: Tuple[float, float] = (0, 70)
    ) -> None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

        highlight_node_ids = set()
        if highlight_subgraph:
            highlight_node_ids = set(highlight_subgraph.get("nodes", {}).keys())

        subtask_colors = plt.cm.tab10.colors

        agent_position = None
        if len(self.agent_trajectory) > 0:
            agent_position = self.agent_trajectory[-1]

        for node_id, node in self.nodes.items():
            pos = node.center_3d
            pos_x, pos_y = pos[0], pos[1]
            pos_z = -pos[2]
            pos_z = np.clip(pos_z, height_range[0], height_range[1])

            if node.is_anchor:
                color = 'red'
                size = 150
                marker = '*'
            elif node_id in highlight_node_ids:
                color = 'orange'
                size = 100
                marker = 'o'
            else:
                color_idx = node.subtask_id % len(subtask_colors)
                color = subtask_colors[color_idx] if node.subtask_id >= 0 else 'gray'
                size = 60
                marker = 'o'

            ax.scatter(pos_x, pos_y, pos_z, c=[color], s=size, marker=marker, alpha=0.9, zorder=10)

            if show_labels:
                variant_count = len(node.label_variants)
                label_text = f"{node_id}\n\"{node.label}\""
                if variant_count > 1:
                    label_text += f"\n(+{variant_count - 1} variants)"
                label_text += f"\n[{pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}]"
                label_text += f"\nconf={node.confidence:.2f}"

                ax.text(pos_x, pos_y, pos_z + 2, label_text, fontsize=6, ha='center', va='bottom',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='none'))

        if hasattr(self, 'relation_snapshots') and self.relation_snapshots:
            for snap in self.relation_snapshots:
                agent_pos = np.array(snap["agent_position"])
                agent_z_disp = np.clip(-agent_pos[2], height_range[0], height_range[1])
                for rel in snap["relations"]:
                    node_id = rel["node_id"]
                    if node_id not in self.nodes:
                        continue
                    node = self.nodes[node_id]
                    node_center = np.array(node.center_3d)
                    node_z_disp = np.clip(-node_center[2], height_range[0], height_range[1])
                    ax.plot(
                        [agent_pos[0], node_center[0]],
                        [agent_pos[1], node_center[1]],
                        [agent_z_disp, node_z_disp],
                        'c--', linewidth=0.8, alpha=0.4, zorder=2
                    )

        if len(self.agent_trajectory) > 0:
            traj = np.array(self.agent_trajectory)
            for i, pos in enumerate(traj):
                z_disp = np.clip(-pos[2], height_range[0], height_range[1])
                ax.scatter(pos[0], pos[1], z_disp, c='red', s=15, alpha=0.7, zorder=9)
                if i % max(1, len(traj) // 10) == 0 or i in [0, len(traj) - 1]:
                    ax.text(pos[0], pos[1], z_disp + 1, str(i), fontsize=5, ha='center', alpha=0.8)

        if show_trajectory and len(self.agent_trajectory) > 1:
            traj = np.array(self.agent_trajectory)
            traj_display = traj.copy()
            traj_display[:, 2] = -traj[:, 2]
            traj_display[:, 2] = np.clip(traj_display[:, 2], height_range[0], height_range[1])
            ax.plot(traj_display[:, 0], traj_display[:, 1], traj_display[:, 2],
                    'b-', linewidth=2, alpha=0.6, label='Agent Trajectory')

        if agent_position is not None:
            agent_z_display = np.clip(-agent_position[2], height_range[0], height_range[1])
            ax.scatter(agent_position[0], agent_position[1], agent_z_display,
                       c='red', s=200, marker='^', label='Agent', zorder=15, edgecolors='white', linewidths=2)
            if agent_yaw is not None:
                arrow_length = 10.0
                arrow_dx = arrow_length * np.cos(agent_yaw)
                arrow_dy = arrow_length * np.sin(agent_yaw)
                ax.quiver(
                    agent_position[0], agent_position[1], agent_z_display,
                    arrow_dx, arrow_dy, 0,
                    color='red', arrow_length_ratio=0.3, linewidth=2, alpha=0.8
                )
                yaw_deg = np.degrees(agent_yaw)
                ax.text(agent_position[0], agent_position[1], agent_z_display + 5,
                        f'YAW={yaw_deg:.1f}°', fontsize=10, ha='center', va='bottom',
                        color='red', fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='red'))

        ax.set_zlim(height_range[0], height_range[1])
        ax.set_xlabel('X (North)', fontsize=10)
        ax.set_ylabel('Y (East)', fontsize=10)
        ax.set_zlabel('Height (m)', fontsize=10)
        ax.set_title(f'Scene Graph | Nodes: {len(self.nodes)} | Steps: {len(self.agent_trajectory)}', fontsize=12)

        legend_elements = [
            plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='red', markersize=15, label='Anchor Node'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10,
                       label='Current Subgraph'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=8, label='Other Nodes'),
            plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='red', markersize=12, label='Agent (Current)'),
            plt.Line2D([0], [0], color='red', marker='o', markersize=6, linestyle='', label='Agent (Past)'),
            plt.Line2D([0], [0], color='cyan', linestyle='--', linewidth=2, label='Agent-Node Relation (All Steps)'),
        ]
        if show_trajectory:
            legend_elements.append(
                plt.Line2D([0], [0], color='blue', linewidth=2, label='Trajectory')
            )
        ax.legend(handles=legend_elements, loc='upper left', fontsize=8)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.debug(f"📊 Scene Graph visualization saved: {save_path}")

    def visualize_subgraph(
            self,
            subgraph: Dict,
            save_path: str,
            agent_position: np.ndarray = None,
            agent_yaw: float = None,
            figsize: Tuple[int, int] = (14, 10),
            height_range: Tuple[float, float] = (0, 70)
    ) -> None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

        nodes = subgraph.get("nodes", {})
        anchor_node_id = subgraph.get("anchor_node_id")
        current_task_ids = set(subgraph.get("current_task_node_ids", []))
        semantic_ids = set(subgraph.get("semantic_related_node_ids", []))

        for node_id, node in nodes.items():
            pos = node.center_3d
            pos_x, pos_y = pos[0], pos[1]
            pos_z = -pos[2]

            pos_z = np.clip(pos_z, height_range[0], height_range[1])

            if node_id == anchor_node_id:
                color = 'red'
                size = 200
                marker = '*'
            elif node_id in current_task_ids:
                color = 'green'
                size = 120
                marker = 'o'
            elif node_id in semantic_ids:
                color = 'purple'
                size = 80
                marker = 's'
            else:
                color = 'gray'
                size = 60
                marker = 'o'

            ax.scatter(pos_x, pos_y, pos_z, c=[color], s=size, marker=marker, alpha=0.9, zorder=10)

            variant_count = len(node.label_variants)
            label_text = f"{node_id}\n\"{node.label}\""
            if variant_count > 1:
                label_text += f"\n(+{variant_count - 1} variants)"
            label_text += f"\n[{pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}]"
            label_text += f"\nconf={node.confidence:.2f}"

            ax.text(pos_x, pos_y, pos_z + 2, label_text, fontsize=7, ha='center', va='bottom',
                    fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='none'))

        if agent_position is not None:
            if not isinstance(agent_position, np.ndarray):
                agent_position = np.array(agent_position)

            agent_z_display = np.clip(-agent_position[2], height_range[0], height_range[1])

            relation_map = {}
            if agent_yaw is not None:
                subgraph_node_ids = set(nodes.keys())
                relations = self.compute_agent_to_nodes_relations(
                    agent_position=agent_position,
                    agent_yaw=agent_yaw,
                    subgraph_node_ids=subgraph_node_ids
                )
                relation_map = {rel['node_id']: rel for rel in relations}

            for node_id, node in nodes.items():
                node_z_display = np.clip(-node.center_3d[2], height_range[0], height_range[1])

                ax.plot(
                    [agent_position[0], node.center_3d[0]],
                    [agent_position[1], node.center_3d[1]],
                    [agent_z_display, node_z_display],
                    'c--', linewidth=1.5, alpha=0.6, zorder=5
                )

                if node_id in relation_map:
                    rel = relation_map[node_id]
                    mid_x = (agent_position[0] + node.center_3d[0]) / 2
                    mid_y = (agent_position[1] + node.center_3d[1]) / 2
                    mid_z = (agent_z_display + node_z_display) / 2

                    direction_text = self._yaw_to_direction_text(rel['yaw_angle_deg'])
                    edge_label = f"{rel['distance']:.1f}m\n{rel['yaw_angle_deg']:.0f}°"

                    ax.text(mid_x, mid_y, mid_z, edge_label, fontsize=6, ha='center', va='center',
                            color='darkcyan', fontweight='bold',
                            bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.7, edgecolor='cyan',
                                      linewidth=0.5))

        if agent_position is not None:
            agent_z_display = np.clip(-agent_position[2], height_range[0], height_range[1])

            ax.scatter(agent_position[0], agent_position[1], agent_z_display,
                       c='red', s=200, marker='^', label='Agent', zorder=15, edgecolors='white', linewidths=2)

            if agent_yaw is not None:
                arrow_length = 8.0
                arrow_dx = arrow_length * np.cos(agent_yaw)
                arrow_dy = arrow_length * np.sin(agent_yaw)

                ax.quiver(
                    agent_position[0], agent_position[1], agent_z_display,
                    arrow_dx, arrow_dy, 0,
                    color='red', arrow_length_ratio=0.3, linewidth=2, alpha=0.8
                )

                yaw_deg = np.degrees(agent_yaw)
                ax.text(agent_position[0], agent_position[1], agent_z_display + 4,
                        f'YAW={yaw_deg:.1f}°', fontsize=10, ha='center', va='bottom',
                        color='red', fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='red'))

        ax.set_zlim(height_range[0], height_range[1])

        ax.set_xlabel('X (North)', fontsize=10)
        ax.set_ylabel('Y (East)', fontsize=10)
        ax.set_zlabel('Height (m)', fontsize=10)

        stats = subgraph.get("stats", {})
        ax.set_title(
            f'Task-Focused Subgraph | Subtask {subgraph.get("current_subtask_id", "?")} | Nodes: {stats.get("total_nodes", 0)}',
            fontsize=11
        )

        legend_elements = [
            plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='red',
                       markersize=15, label='Anchor (Previous)'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green',
                       markersize=12, label='Current Task Nodes'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='purple',
                       markersize=10, label='Semantic Related'),
            plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='red',
                       markersize=12, label='Agent'),
            plt.Line2D([0], [0], color='cyan', linestyle='--', linewidth=2,
                       label='Agent-Node Relation'),
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=8)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.debug(f"📊 Subgraph visualization saved: {save_path}")

    def get_nodes_by_subtask(self, subtask_id: int) -> List[ObjectNode]:
        return [node for node in self.nodes.values() if node.subtask_id == subtask_id]

    def get_nearby_nodes(
            self,
            position: np.ndarray,
            radius: float = 20.0
    ) -> List[ObjectNode]:
        if not isinstance(position, np.ndarray):
            position = np.array(position)

        nearby = []
        for node in self.nodes.values():
            dist = np.linalg.norm(node.center_3d - position)
            if dist <= radius:
                nearby.append(node)

        return sorted(nearby, key=lambda n: np.linalg.norm(n.center_3d - position))

    def find_node_by_label(self, label: str, exact_match: bool = True) -> List[ObjectNode]:
        results = []
        label_lower = label.lower().strip()

        for node in self.nodes.values():
            all_labels = [node.label] + list(node.label_variants.keys())

            for node_label in all_labels:
                if node_label.lower().strip() == label_lower:
                    results.append(node)
                    break
        return results

    def to_dict(self) -> Dict:
        return {
            "nodes": {nid: node.to_dict() for nid, node in self.nodes.items()},
            "completed_anchors": self.completed_anchors,
            "current_subtask_id": self.current_subtask_id,
            "agent_trajectory": [pos.tolist() for pos in self.agent_trajectory],
            "stats": {
                "total_nodes": len(self.nodes),
                "total_anchors": len(self.completed_anchors)
            }
        }

    def save_to_json(self, save_path: str):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        def convert_to_serializable(obj):
            if isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64, np.floating)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64, np.integer)):
                return int(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            return obj

        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(convert_to_serializable(self.to_dict()), f, indent=2, ensure_ascii=False)
        logger.info(f"💾 Scene graph saved: {save_path}")

    def process_step(
            self,
            spatial_result: Dict,
            current_step: int,
            current_subtask_id: int,
            curr_pose,
            subtask_instruction: str,
            semantic_keywords: List[str] = None,
            previous_anchor_id: Optional[str] = None,
            max_semantic_nodes: int = 25,
            max_objects_in_prompt: int = 30,
            viz_dir: Optional[str] = None,
            viz_interval: int = 1
    ) -> Dict:
        import airsim

        agent_position = np.array([
            curr_pose.position.x_val,
            curr_pose.position.y_val,
            curr_pose.position.z_val
        ])
        _, _, agent_yaw = airsim.to_eularian_angles(curr_pose.orientation)

        result = {
            "update_stats": {},
            "subgraph": {},
            "prompt": "",
            "viz_paths": {},
            "agent_position": agent_position,
            "agent_yaw": agent_yaw
        }

        update_stats = self.update_from_spatial_result(
            spatial_result=spatial_result,
            current_step=current_step,
            current_subtask_id=current_subtask_id,
            agent_position=agent_position
        )
        result["update_stats"] = update_stats

        subgraph = self.extract_task_focused_subgraph(
            current_subtask_id=current_subtask_id,
            previous_anchor_id=previous_anchor_id,
            semantic_keywords=semantic_keywords,
            max_semantic_nodes=max_semantic_nodes
        )
        result["subgraph"] = subgraph

        prompt = self.subgraph_to_llm_prompt(
            subgraph=subgraph,
            current_subtask_instruction=subtask_instruction,
            agent_position=agent_position,
            agent_yaw=agent_yaw,
            include_spatial_relations=True,
            max_objects_in_prompt=max_objects_in_prompt
        )
        result["prompt"] = prompt

        if viz_dir and (current_step % viz_interval == 0):
            os.makedirs(viz_dir, exist_ok=True)

            full_graph_path = os.path.join(viz_dir, f"full_graph_step_{current_step:04d}.png")
            self.visualize_scene_graph(
                save_path=full_graph_path,
                highlight_subgraph=subgraph,
                agent_yaw=agent_yaw,
                show_labels=True,
                show_trajectory=True
            )
            result["viz_paths"]["full_graph"] = full_graph_path

            subgraph_path = os.path.join(viz_dir, f"subgraph_step_{current_step:04d}.png")
            self.visualize_subgraph(
                subgraph=subgraph,
                save_path=subgraph_path,
                agent_position=agent_position,
                agent_yaw=agent_yaw
            )
            result["viz_paths"]["subgraph"] = subgraph_path

            subgraph_node_ids = set(subgraph["nodes"].keys())
            if subgraph_node_ids:
                snapshot = {
                    "step": current_step,
                    "subtask_id": current_subtask_id,
                    "agent_position": agent_position.copy().tolist(),
                    "agent_yaw": float(agent_yaw),
                    "relations": [
                        {
                            "node_id": rel["node_id"],
                            "center_3d": rel["center_3d"],
                            "distance": rel["distance"],
                            "yaw_angle_deg": rel["yaw_angle_deg"]
                        }
                        for rel in self.compute_agent_to_nodes_relations(
                            agent_position=agent_position,
                            agent_yaw=agent_yaw,
                            subgraph_node_ids=subgraph_node_ids
                        )
                    ]
                }
                self.relation_snapshots.append(snapshot)

        subgraph_node_ids = set(subgraph.get("nodes", {}).keys())
        if subgraph_node_ids:
            relations = self.compute_agent_to_nodes_relations(
                agent_position=agent_position,
                agent_yaw=agent_yaw,
                subgraph_node_ids=subgraph_node_ids
            )
            relation_map = {rel['node_id']: rel for rel in relations}
        else:
            relation_map = {}

        result["relation_map"] = relation_map

        return result

    def finalize_episode(
            self,
            viz_dir: Optional[str] = None,
            agent_yaw: Optional[float] = None
    ) -> Dict:
        result = {"json_path": None, "viz_path": None}

        if viz_dir:
            os.makedirs(viz_dir, exist_ok=True)

            json_path = os.path.join(viz_dir, "scene_graph_final.json")
            self.save_to_json(json_path)
            result["json_path"] = json_path

            viz_path = os.path.join(viz_dir, "full_graph_final.png")
            self.visualize_scene_graph(
                save_path=viz_path,
                highlight_subgraph=None,
                agent_yaw=agent_yaw,
                show_labels=True,
                show_trajectory=True)
            result["viz_path"] = viz_path
            logger.info(f"💾 Scene graph saved to: {viz_dir}")

        return result