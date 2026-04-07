import json
import torch
import airsim
import copy
import numpy as np
import cv2
import os
from tqdm import tqdm
from src.llm.query_llm import OpenAI_LLM_v2, OpenAI_LLM_v3, Ollama_LLM
from airsim_plugin.AirVLNSimulatorClientTool_sem import AirVLNSimulatorClientTool
from utils.logger import logger
from utils.utils import calculate_movement_steps
from utils.env_utils import getPoseAfterMakeActions, get_pano_observations

from Spatial_Reasoning.unified_navigation_visualizer064 import UnifiedNavigationVisualizer
from Spatial_Reasoning.Spatia_reasoning_067 import SpatialReasoningEnhancer
from Spatial_Reasoning.navigation_core065 import NavigationOrchestrator

from Spatial_Reasoning.map.occupancymap2_5d_065 import OccupancyMap2_5D, OccupancyMapConfig

from Spatial_Reasoning.map.scene_graph067 import CitySceneGraph


def CityNavAgent(scene_id, split, data_dir="/home/zhouhuan/project/CityNavAgent/CityNavAgent-main/Spatial_Reasoning",
                 max_step_size=200, record=False, enable_viz=True, viz_config=None):
    logger.info(f"🚀 Starting CityNavAgent | Scene ID: {scene_id} | Dataset Split: {split}")
    env_id = scene_id

    if record:
        log_dir = f"/project/CityNavAgent/CityNavAgent-main/Spatial_Reasoning/output/nav_data/{env_id}"
        os.makedirs(log_dir, exist_ok=True)
        log_file_path = os.path.join(log_dir, f"navigation_env{env_id}.log")
        logger.add_filehandler(log_file_path)
        logger.info(f"📝 Log file will be saved to: {log_file_path}")

        trajectory_save_path = f"/project/CityNavAgent/CityNavAgent-main/Spatial_Reasoning/output/nav_data/{env_id}/trajectory_viz"
        os.makedirs(trajectory_save_path, exist_ok=True)
        logger.info(f"📝 Trajectory file will be saved to: {trajectory_save_path}")

    data_root = os.path.join(data_dir, f"gt_by_env/{env_id}/{split}_landmk_subtasks_{env_id}_1.json")

    with open(data_root, 'r') as f:
        navi_tasks = json.load(f)['episodes']
    #nav_evaluator = CityNavEvaluator()

    llm = OpenAI_LLM_v2(
        max_tokens=20000,
        model_name="********",
        api_key="***************",
        client_type="openai",
        cache_name="navigation",
        finish_reasons=["stop", "length"],
    )

    machines_info = [{'MACHINE_IP': '127.0.0.1', 'SOCKET_PORT': 30000, 'MAX_SCENE_NUM': 1, 'open_scenes': [scene_id]}]
    tool = AirVLNSimulatorClientTool(machines_info=machines_info)
    tool.run_call()
    logger.info("✅ AirSim environment started successfully")

    spatial_reasoner = SpatialReasoningEnhancer(
        rec_model_path=None,
        device="cuda" if torch.cuda.is_available() else "cpu",
        image_width=512,
        image_height=512,
        fov_degrees=90,
        max_valid_depth=80,
        min_valid_depth=1.2)

    nav_orchestrator = NavigationOrchestrator(
        llm_client=llm,
        spatial_enhancer=spatial_reasoner)

    occupancy_config = OccupancyMapConfig(
        map_size_meters=250.0,
        resolution=3.5,
        image_width=512,
        image_height=512,
        fov_degrees=90.0,
        depth_scale=155,
        min_valid_depth=1.2,
        max_valid_depth=80.0,
        ray_sample_interval=3,
        boundary_semantic_ids=[253, 161, 91, 126],
        frontier_min_size=3)

    predict_routes = []
    for i in tqdm([7, 14]):
        navi_task = navi_tasks[i]
        try:
            episode_id = navi_task['episode_id']
            prefixed_episode_id = f"{i}_{episode_id}"
            logger.info(f"📍 Starting task {i}/{len(navi_tasks)} | Task ID: {episode_id}" + "=" * 80)

            landmarks = navi_task["instruction"]["landmarks"]
            instruction = navi_task["instruction"]['instruction_text']
            logger.info(f"🗺️ Instruction: {instruction} | Landmarks: {landmarks}")

            occupancy_map = OccupancyMap2_5D(config=occupancy_config)
            occupancy_map.reset()

            scene_graph = CitySceneGraph(
                merge_distance_threshold=8.0,
                edge_distance_threshold=18.0,
                vertical_threshold=3.0,
                semantic_match_strict=False
            )

            scene_graph_viz_dir = os.path.join(
                "/project/CityNavAgent/CityNavAgent-main/Spatial_Reasoning/output/nav_data", str(env_id),
                prefixed_episode_id, "scene_graph")
            os.makedirs(scene_graph_viz_dir, exist_ok=True)

            visualizer = None
            data_base_dir = None

            if record:
                data_base_dir = f"/project/CityNavAgent/CityNavAgent-main/Spatial_Reasoning/output/nav_data/{env_id}/{prefixed_episode_id}"

            if enable_viz:
                viz_output_dir = os.path.join(
                    "/project/CityNavAgent/CityNavAgent-main/Spatial_Reasoning/output/nav_data",
                    str(env_id), prefixed_episode_id,
                    "visualizer") if record else f'./viz_output/{env_id}/{prefixed_episode_id}'
                os.makedirs(viz_output_dir, exist_ok=True)

                visualizer = UnifiedNavigationVisualizer(
                    scene_id=str(scene_id),
                    episode_id=episode_id,
                    output_dir=viz_output_dir,
                    enable_realtime=True,
                    save_static_frames=record,
                    save_interval=1,
                    window_title=f"Drone Navigation - Scene {scene_id} Episode {episode_id}")
                logger.info(f"✅ Unified visualizer initialized | Output directory: {viz_output_dir}")

            start_pose_list = navi_task["start_position"] + navi_task["start_rotation"][1:] + [
                navi_task["start_rotation"][0]]
            curr_pose = convert_airsim_pose(start_pose_list)
            tool.setPoses([[curr_pose]])

            # Initial rotation actions
            for _ in range(4):
                new_pose = getPoseAfterMakeActions(curr_pose, [4])
                curr_pose = new_pose
                tool.setPoses([[curr_pose]])

            nav_state = nav_orchestrator.initialize_task(
                episode_data=navi_task)

            next_landmark_idx = 0
            step_counter = 0
            step_size = 0

            hist_step_size = []
            data_dict = {
                "episode_id": episode_id,
                "instruction": instruction,
                "gt_traj": [pose[:3] for pose in navi_task['reference_path']],
                "gt_traj_node": [pose for pose in navi_task['reference_node_path']],
                "pred_traj": [],
                "pred_traj_explore": [
                    list(curr_pose.position) + list(airsim.to_eularian_angles(curr_pose.orientation))],
                "pred_traj_memory": [],
                "success": False}

            map_stats = {
                'total_obstacle_points': 0,
                'total_free_rays': 0,
                'update_count': 0}

            while step_size < max_step_size:
                current_subtask = nav_state.instruction_progress.get_current_subtask_info()
                completed_count = nav_state.instruction_progress.get_completed_count()
                total_count = nav_state.instruction_progress.get_total_tasks()
                skipped_count = nav_state.instruction_progress.get_skipped_count()

                if current_subtask:
                    logger.info(
                        f"🧭 Step {step_counter} | Subtask Progress {completed_count}/{total_count} (Skipped:{skipped_count})" + "-" * 60)
                    logger.info(f"   Current Subtask: [{current_subtask.subtask_id}] {current_subtask.subtask}")
                    logger.info(f"   Type: {current_subtask.task_type} | Landmark: {current_subtask.landmark}")
                    if current_subtask.execution_rationale:
                        logger.info(f"   Execution Rationale: {current_subtask.execution_rationale}...")
                else:
                    logger.info(f"🧭 Step {step_counter} | All subtasks completed")

                try:
                    pano_obs, pano_pose = get_pano_observations(curr_pose, tool, scene_id=scene_id)
                    pano_obs_imgs = [pano_obs[6][0], pano_obs[7][0], pano_obs[0][0], pano_obs[1][0], pano_obs[2][0]]
                    pano_obs_deps = [pano_obs[6][1], pano_obs[7][1], pano_obs[0][1], pano_obs[1][1], pano_obs[2][1]]
                    pano_obs_sems = [pano_obs[6][2], pano_obs[7][2], pano_obs[0][2], pano_obs[1][2], pano_obs[2][2]]
                    pano_obs_poses = [pano_pose[6], pano_pose[7], pano_pose[0], pano_pose[1], pano_pose[2]]
                    if record:
                        save_navigation_data(
                            base_dir=data_base_dir,
                            step_counter=step_counter,
                            curr_pose=curr_pose,
                            instruction=instruction,
                            next_landmark_idx=next_landmark_idx,
                            landmarks=landmarks,
                            step_size=step_size,
                            max_step_size=max_step_size,
                            pano_obs=pano_obs,
                            pano_pose=pano_pose)
                except Exception as e:
                    logger.error(f"❌ Failed to obtain observation: {e}")
                    break

                image_order = ["left", "slightly_left", "front", "slightly_right", "right"]
                rgb_dict = {k: pano_obs_imgs[i] for i, k in enumerate(image_order)}
                dep_dict = {k: pano_obs_deps[i] for i, k in enumerate(image_order)}
                seg_dict = {k: pano_obs_sems[i] for i, k in enumerate(image_order)}
                pose_dict = {k: pano_obs_poses[i] for i, k in enumerate(image_order)}

                focused_landmarks = nav_orchestrator.get_detection_landmarks()
                logger.info(f"🔍 Detecting target landmarks: {focused_landmarks}")

                spatial_result = spatial_reasoner.perform_mllm_guided_spatial_reasoning(
                    viewpoint_rgb_imgs=rgb_dict,
                    viewpoint_dep_imgs=dep_dict,
                    viewpoint_poses=pose_dict,
                    instruction=instruction,
                    original_landmarks=focused_landmarks,
                    all_landmarks=landmarks,
                    llm_client=llm,
                    image_order=["left", "slightly_left", "front", "slightly_right", "right"],
                    box_threshold=0.35,
                    text_threshold=0.25,
                    enable_sam=True,
                    downsample_step=15,
                    save_masks=False
                )

                spatial_text_context = spatial_result["text_context"]
                visual_descriptions = spatial_result["visual_descriptions"]
                per_view_landmarks = spatial_result["per_view_landmarks"]
                detections = spatial_result["detections"]
                depth_stats = spatial_result["depth_stats"]

                current_subtask_id = current_subtask.subtask_id if current_subtask else 0
                subtask_instruction = current_subtask.subtask if current_subtask else instruction

                sg_result = scene_graph.process_step(
                    spatial_result=spatial_result,
                    current_step=step_counter,
                    current_subtask_id=current_subtask_id,
                    curr_pose=curr_pose,
                    subtask_instruction=subtask_instruction,
                    semantic_keywords=focused_landmarks,
                    max_semantic_nodes=30,
                    max_objects_in_prompt=30,
                    viz_dir=scene_graph_viz_dir)

                scene_graph_prompt = sg_result["prompt"]
                subgraph = sg_result["subgraph"]
                relation_map = sg_result.get("relation_map", {})
                current_pos_np = sg_result["agent_position"]
                agent_yaw = sg_result["agent_yaw"]
                logger.info(f"📝 Scene graph prompt:\n{scene_graph_prompt}")

                subgraph_node_count = len(sg_result["subgraph"]["nodes"])
                logger.info(
                    f"📊 Step {step_counter} | Subgraph nodes: {subgraph_node_count} "
                    f"(Current task: {len(sg_result['subgraph'].get('current_task_node_ids', []))}, "
                    f"Semantic related: {len(sg_result['subgraph'].get('semantic_related_node_ids', []))}, "
                    f"Has anchor: {'Yes' if sg_result['subgraph'].get('anchor_node') else 'No'})")

                previous_anchor = scene_graph.get_previous_anchor(
                    current_subtask.subtask_id) if current_subtask else None
                result = occupancy_map.update_and_visualize(
                    viewpoint_dep_imgs=dep_dict,
                    viewpoint_poses=pose_dict,
                    image_order=image_order,
                    current_position=current_pos_np,
                    scene_id=str(scene_id),
                    episode_id=str(prefixed_episode_id),
                    step_counter=step_counter,
                    viewpoint_sem_imgs=seg_dict,
                    scene_graph=scene_graph,
                    current_landmark=focused_landmarks[0] if focused_landmarks else None,
                    previous_anchor_label=previous_anchor.label if previous_anchor else None,
                    output_base='/home/zhouhuan/project/CityNavAgent/CityNavAgent-main/Spatial_Reasoning/output/nav_data',
                    current_step_detections=per_view_landmarks,
                    max_exploration_distance=3.0,
                    show_viewpoint_rays=True)

                logger.info(f"🧭 Semantic-guided frontiers:\n{result['frontier_prompt']}")
                frontier_prompt = None

                step_result = nav_orchestrator.process_step(
                    scene_graph_prompt=scene_graph_prompt,
                    frontier_prompt=frontier_prompt,
                    viewpoint_poses=pose_dict,
                    instruction=instruction,
                    landmarks=landmarks,
                    scene_graph=scene_graph,
                    occupancy_map=occupancy_map,
                    current_step=step_counter)

                action = step_result['action']
                parameters = step_result['parameters']
                is_terminal = step_result['is_terminal']
                task_completed = step_result.get('task_completed', False)
                current_phase = step_result.get('current_phase', 'Search')
                selected_path = step_result.get('selected_path', 'B')

                if visualizer:
                    visualizer.update_preliminary(
                        step_counter=step_counter,
                        viewpoint_images=rgb_dict,
                        spatial_result=spatial_result,
                        viewpoint_poses=pose_dict,
                        curr_pose=curr_pose,
                        instruction=instruction,
                        landmarks=landmarks,
                        next_landmark_idx=next_landmark_idx,
                        current_subtask=current_subtask,
                        step_result=step_result,
                        subgraph=subgraph,
                        relation_map=relation_map,
                        agent_position=current_pos_np,
                        agent_yaw=agent_yaw
                    )

                logger.info(f"📊 Decision Results:")
                logger.info(f"   Action: {action} | Parameters: {parameters}")
                logger.info(f"   Current Phase: {current_phase} | Selected Path: {selected_path}")
                logger.info(f"   LLM Judged Completed: {task_completed}")
                logger.info(f"   Consecutive Failures: {step_result.get('consecutive_failures', 0)}/3")

                if step_result.get('task_just_completed'):
                    logger.info(f"   ✅ Subtask {step_result.get('completed_subtask_id')} completed this step")
                if step_result.get('task_just_skipped'):
                    logger.info(f"   ⏭️ Subtask {step_result.get('skipped_subtask_id')} skipped this step")

                logger.info(f"📊 Instruction Progress: {step_result.get('instruction_progress', {})}")

                new_pose = step_result.get('new_pose')

                sz, mid_coords = calculate_movement_steps(curr_pose, new_pose)
                logger.info(
                    f"➡️ Movement command: Forward {sz} steps | Target position: ({new_pose.position.x_val:.2f}, {new_pose.position.y_val:.2f}, {new_pose.position.z_val:.2f})")

                data_dict['pred_traj'].extend([c[:3] for c in mid_coords])
                data_dict['pred_traj_explore'].extend(mid_coords)

                tool.setPoses([[new_pose]])
                curr_pose = new_pose
                step_size += sz

                if is_terminal:
                    logger.info("🏆 Navigation orchestrator determined task completion!")
                    break

                if nav_state.instruction_progress.is_all_completed():
                    logger.info("🏆 All subtasks completed!")
                    break

                hist_step_size.append(sz)
                if len(hist_step_size) >= 5:
                    recent_movement = sum(hist_step_size[-4:])
                    if recent_movement == 0:
                        logger.info("⚠️ No effective movement for 4 consecutive steps, stuck, terminating navigation")
                        break
                step_counter += 1

            if scene_graph_viz_dir:
                scene_graph.save_to_json(os.path.join(scene_graph_viz_dir, "scene_graph_final.json"))
                scene_graph.visualize_scene_graph(save_path=os.path.join(scene_graph_viz_dir, "full_graph_final.png"),
                                                  highlight_subgraph=None, show_labels=True, show_trajectory=True)

            if visualizer:
                visualizer.close()
                logger.info("✅ Visualizer closed")

            stop_pos = np.array([curr_pose.position.x_val, curr_pose.position.y_val, curr_pose.position.z_val])
            target_pos = np.array(navi_task["goals"][0]['position'])
            ne = np.linalg.norm(target_pos - stop_pos)
            data_dict.update({"final_ne": float(ne), "total_steps": step_size})
            if ne < 20:
                data_dict["success"] = True
                logger.info(f"✅ Task success | NE: {ne:.2f} | Steps: {step_size}")
            else:
                data_dict["success"] = False
                logger.info(f"❌ Task failed | NE: {ne:.2f} | Steps: {step_size}")

            nav_evaluator.update(data_dict)
            predict_routes.append(data_dict)

            if record:
                visualize_single_task_trajectory(data_dict, trajectory_save_path, task_index=i)

                single_task_data = data_dict.copy()
                single_task_data['final_pred_traj'] = single_task_data['pred_traj_explore']
                single_task_save_dir = f"/project/CityNavAgent/CityNavAgent-main/Spatial_Reasoning/output/nav_data/{env_id}/{prefixed_episode_id}"
                os.makedirs(single_task_save_dir, exist_ok=True)
                single_task_save_path = os.path.join(single_task_save_dir, f"trajectory_{prefixed_episode_id}.json")
                with open(single_task_save_path, 'w') as f:
                    json.dump(single_task_data, f, indent=4)
                logger.info(f"📁 Single task trajectory saved to: {single_task_save_path}")

                trajectory_json_path = f"/project/CityNavAgent/CityNavAgent-main/Spatial_Reasoning/output/nav_data/{env_id}/output_data_{env_id}.json"

                existing_routes = []
                if os.path.exists(trajectory_json_path):
                    try:
                        with open(trajectory_json_path, 'r') as f:
                            existing_routes = json.load(f)
                        if not isinstance(existing_routes, list):
                            existing_routes = []
                    except (json.JSONDecodeError, Exception) as e:
                        logger.warning(f"⚠️ Failed to read existing trajectory file: {e}, creating new file")
                        existing_routes = []

                current_task_data = copy.deepcopy(data_dict)
                current_task_data['final_pred_traj'] = current_task_data['pred_traj_explore']

                current_episode_id = current_task_data.get('episode_id')
                replaced = False
                for idx, route in enumerate(existing_routes):
                    if route.get('episode_id') == current_episode_id:
                        existing_routes[idx] = current_task_data
                        replaced = True
                        logger.info(f"🔄 Updated record for task {current_episode_id}")
                        break

                if not replaced:
                    existing_routes.append(current_task_data)

                with open(trajectory_json_path, 'w') as f:
                    json.dump(existing_routes, f, indent=4)
                logger.info(f"📁 Global trajectory updated: {trajectory_json_path} (Total {len(existing_routes)} tasks)")

            nav_evaluator.log_metrics()
            logger.info("🔚 All tasks execution finished")

        except Exception as e:
            logger.error(f"⚠️ Task {i} error, skipping. Error: {e}")
            continue


def visualize_single_task_trajectory(data_dict, save_path=None, task_index=None):
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    episode_id = data_dict.get('episode_id', 'unknown')
    filename = f"{task_index}_{episode_id}.png"
    full_save_path = os.path.join(save_path, filename)

    def safe_extract_trajectory(traj_data):
        if traj_data is None or len(traj_data) == 0:
            return np.array([])
        traj_array = np.array(traj_data)
        if traj_array.ndim == 1:
            if len(traj_array) >= 3 and len(traj_array) % 3 == 0:
                traj_array = traj_array.reshape(-1, 3)
            elif len(traj_array) >= 4 and len(traj_array) % 4 == 0:
                traj_array = traj_array.reshape(-1, 4)[:, :3]
            else:
                return np.array([])
        if traj_array.ndim == 2 and traj_array.shape[1] >= 3:
            return traj_array[:, :3]
        return np.array([])

    gt_traj = safe_extract_trajectory(data_dict.get('gt_traj_node', []))
    pred_traj = safe_extract_trajectory(data_dict.get('pred_traj_explore', []))

    if len(gt_traj) == 0 and len(pred_traj) == 0:
        print("⚠️ No valid trajectory data, skipping visualization")
        return None

    fig = plt.figure(figsize=(18, 14), facecolor='white')
    ax = fig.add_subplot(111, projection='3d')

    ax.set_facecolor('#f8f9fa')
    ax.xaxis.pane.fill = True
    ax.yaxis.pane.fill = True
    ax.zaxis.pane.fill = True
    ax.xaxis.pane.set_facecolor('#e9ecef')
    ax.yaxis.pane.set_facecolor('#e9ecef')
    ax.zaxis.pane.set_facecolor('#dee2e6')

    all_points = []
    if len(gt_traj) > 0:
        all_points.append(gt_traj)
    if len(pred_traj) > 0:
        all_points.append(pred_traj)

    if all_points:
        all_coords = np.vstack(all_points)
        x_min, x_max = all_coords[:, 0].min(), all_coords[:, 0].max()
        y_min, y_max = all_coords[:, 1].min(), all_coords[:, 1].max()
        z_min, z_max = all_coords[:, 2].min(), all_coords[:, 2].max()

        x_range = x_max - x_min
        y_range = y_max - y_min
        z_range = z_max - z_min

        xy_max_range = max(x_range, y_range, 10)

        x_mid = (x_max + x_min) / 2
        y_mid = (y_max + y_min) / 2
        z_mid = (z_max + z_min) / 2

    if len(gt_traj) > 0:
        ax.plot(gt_traj[:, 0], gt_traj[:, 1], gt_traj[:, 2],
                color='#2ecc71', linewidth=4, alpha=0.9, label='Ground Truth')

        ax.scatter(gt_traj[:, 0], gt_traj[:, 1], gt_traj[:, 2],
                   color='#2ecc71', s=30, alpha=0.6)

        ax.scatter(gt_traj[0, 0], gt_traj[0, 1], gt_traj[0, 2],
                   color='#27ae60', s=300, marker='o', edgecolors='white',
                   linewidths=3, zorder=10)

        ax.text(gt_traj[0, 0], gt_traj[0, 1], gt_traj[0, 2] - xy_max_range * 0.05,
                f'GT Start\n({gt_traj[0, 0]:.1f}, {gt_traj[0, 1]:.1f}, {gt_traj[0, 2]:.1f})',
                fontsize=10, fontweight='bold', color='#27ae60',
                ha='center', va='bottom',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='#27ae60', alpha=0.9))

        ax.scatter(gt_traj[-1, 0], gt_traj[-1, 1], gt_traj[-1, 2],
                   color='#e74c3c', s=350, marker='s', edgecolors='white',
                   linewidths=3, zorder=10)

        ax.text(gt_traj[-1, 0], gt_traj[-1, 1], gt_traj[-1, 2] + xy_max_range * 0.08,
                f'GT Goal\n({gt_traj[-1, 0]:.1f}, {gt_traj[-1, 1]:.1f}, {gt_traj[-1, 2]:.1f})',
                fontsize=11, fontweight='bold', color='#e74c3c',
                ha='center', va='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='#e74c3c', alpha=0.9))

    if len(pred_traj) > 0:
        ax.plot(pred_traj[:, 0], pred_traj[:, 1], pred_traj[:, 2],
                color='#3498db', linewidth=4, alpha=0.9, label='Predicted')

        step = max(1, len(pred_traj) // 20)
        ax.scatter(pred_traj[::step, 0], pred_traj[::step, 1], pred_traj[::step, 2],
                   color='#3498db', s=25, alpha=0.6)

        ax.scatter(pred_traj[0, 0], pred_traj[0, 1], pred_traj[0, 2],
                   color='#2980b9', s=250, marker='^', edgecolors='white',
                   linewidths=2, zorder=10)

        ax.text(pred_traj[0, 0] + xy_max_range * 0.05, pred_traj[0, 1], pred_traj[0, 2],
                f'Pred Start\n({pred_traj[0, 0]:.1f}, {pred_traj[0, 1]:.1f}, {pred_traj[0, 2]:.1f})',
                fontsize=9, fontweight='bold', color='#2980b9',
                ha='left', va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='#2980b9', alpha=0.9))

        ax.scatter(pred_traj[-1, 0], pred_traj[-1, 1], pred_traj[-1, 2],
                   color='#f39c12', s=400, marker='*', edgecolors='white',
                   linewidths=3, zorder=10)

        ax.text(pred_traj[-1, 0], pred_traj[-1, 1] + xy_max_range * 0.08, pred_traj[-1, 2],
                f'Pred End\n({pred_traj[-1, 0]:.1f}, {pred_traj[-1, 1]:.1f}, {pred_traj[-1, 2]:.1f})',
                fontsize=11, fontweight='bold', color='#f39c12',
                ha='center', va='bottom',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#fff3cd',
                          edgecolor='#f39c12', linewidth=2, alpha=0.95))

    if len(gt_traj) > 0 and len(pred_traj) > 0:
        final_error_3d = np.linalg.norm(gt_traj[-1] - pred_traj[-1])
        xy_error = np.linalg.norm(gt_traj[-1, :2] - pred_traj[-1, :2])
        z_error = abs(gt_traj[-1, 2] - pred_traj[-1, 2])
        x_error = abs(gt_traj[-1, 0] - pred_traj[-1, 0])
        y_error = abs(gt_traj[-1, 1] - pred_traj[-1, 1])

        ax.plot([gt_traj[-1, 0], pred_traj[-1, 0]],
                [gt_traj[-1, 1], pred_traj[-1, 1]],
                [gt_traj[-1, 2], pred_traj[-1, 2]],
                'r--', linewidth=3, alpha=0.8)

        mid_x = (gt_traj[-1, 0] + pred_traj[-1, 0]) / 2
        mid_y = (gt_traj[-1, 1] + pred_traj[-1, 1]) / 2
        mid_z = (gt_traj[-1, 2] + pred_traj[-1, 2]) / 2

        ax.text(mid_x, mid_y, mid_z - xy_max_range * 0.03,
                f'Error: {final_error_3d:.1f}m\n(XY:{xy_error:.1f}m, Z:{z_error:.1f}m)',
                fontsize=12, fontweight='bold', color='#c0392b',
                ha='center', va='bottom',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#ffebee',
                          edgecolor='#c0392b', linewidth=2, alpha=0.95))

    if all_points:
        half_range = xy_max_range / 2 * 1.3

        ax.set_xlim(x_mid - half_range, x_mid + half_range)
        ax.set_ylim(y_mid - half_range, y_mid + half_range)

        z_half = max(z_range / 2, 2) * 1.3
        ax.set_zlim(z_mid - z_half, z_mid + z_half)

        ax.set_box_aspect([1, 1, z_range / xy_max_range if xy_max_range > 0 else 1])

    ax.invert_zaxis()

    ax.set_xlabel('X [m] (+ Forward, - Backward)', fontsize=12, fontweight='bold', labelpad=15)
    ax.set_ylabel('Y [m] (+ Right, - Left)', fontsize=12, fontweight='bold', labelpad=15)
    ax.set_zlabel('Z [m] (- Up, + Down)', fontsize=12, fontweight='bold', labelpad=15)
    ax.tick_params(axis='both', labelsize=11)

    ax.view_init(elev=25, azim=-135)

    episode_id = data_dict.get("episode_id", "Unknown")
    ax.set_title(f'3D Trajectory Comparison - Episode: {episode_id}\n(AirSim NED Coordinate System)',
                 fontsize=16, fontweight='bold', pad=20)

    success_status = "✓ SUCCESS" if data_dict.get('success', False) else "✗ FAILED"
    status_color = '#27ae60' if data_dict.get('success', False) else '#e74c3c'
    fig.text(0.5, 0.94, success_status, fontsize=20, fontweight='bold',
             color=status_color, ha='center',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                       edgecolor=status_color, linewidth=2.5))

    ax.legend(loc='upper left', fontsize=12, framealpha=0.95)

    if len(gt_traj) > 0 and len(pred_traj) > 0:
        stats_text = f"""═══════════════════════════════
 TRAJECTORY STATISTICS (AirSim NED)
═══════════════════════════════
 GT Goal:    ({gt_traj[-1, 0]:>7.2f}, {gt_traj[-1, 1]:>7.2f}, {gt_traj[-1, 2]:>7.2f})
 Pred End:   ({pred_traj[-1, 0]:>7.2f}, {pred_traj[-1, 1]:>7.2f}, {pred_traj[-1, 2]:>7.2f})
───────────────────────────────
 3D Error:   {final_error_3d:>7.2f} m
 XY Error:   {xy_error:>7.2f} m
 Z Error:    {z_error:>7.2f} m
───────────────────────────────
 X Error:    {x_error:>7.2f} m (Forward/Back)
 Y Error:    {y_error:>7.2f} m (Left/Right)
───────────────────────────────
 GT Length:  {len(gt_traj):>4} points
 Pred Length:{len(pred_traj):>4} points
═══════════════════════════════"""

        fig.text(0.02, 0.02, stats_text, fontsize=10, fontfamily='monospace',
                 verticalalignment='bottom',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8f9fa',
                           edgecolor='#6c757d', linewidth=1.5, alpha=0.95))

    coord_info = """AirSim NED Coordinate:
X: + Forward  / - Backward
Y: + Right    / - Left
Z: + Down     / - Up (Higher)"""

    fig.text(0.98, 0.02, coord_info, fontsize=9, fontfamily='monospace',
             verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#e3f2fd',
                       edgecolor='#1976d2', linewidth=1.5, alpha=0.95))

    plt.tight_layout()
    plt.savefig(full_save_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    return save_path


def convert_airsim_pose(pose):
    assert len(pose) == 7, "The length of input pose must be 7"
    return airsim.Pose(
        position_val=airsim.Vector3r(pose[0], pose[1], pose[2]),
        orientation_val=airsim.Quaternionr(pose[3], pose[4], pose[5], pose[6]))


def save_navigation_data(base_dir, step_counter, curr_pose, instruction,
                         next_landmark_idx, landmarks, step_size, max_step_size,
                         pano_obs, pano_pose):
    view_to_index = {
        "left": 6,
        "slightly_left": 7,
        "front": 0,
        "slightly_right": 1,
        "right": 2,
    }

    rgb_dir = os.path.join(base_dir, "rgb")
    os.makedirs(rgb_dir, exist_ok=True)
    for view, idx in view_to_index.items():
        rgb_img = pano_obs[idx][0]
        filename = f"step_{step_counter:04d}_{view}.png"
        save_path = os.path.join(rgb_dir, filename)
        cv2.imwrite(save_path, rgb_img)


if __name__ == '__main__':
    env_id = 12
    split = "val_seen"
    save_demo = True
    CityNavAgent(env_id, split, max_step_size=75, record=save_demo)
