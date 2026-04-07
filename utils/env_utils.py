import math
import numba as nb
import airsim
import numpy as np
import copy

from airsim_plugin.airsim_settings import AirsimActions, AirsimActionSettings
from airsim_plugin.AirVLNSimulatorClientTool import AirVLNSimulatorClientTool

from src.common.param import args

from utils.logger import logger
from utils.shorest_path_sensor import ShortestPathSensor, EuclideanDistance3, EuclideanDistance1


class SimState:
    def __init__(self, index=-1,
                 step=0,
                 episode_info={},
                 pose=airsim.Pose(),
                 ):
        self.index = index
        self.step = step
        self.episode_info = copy.deepcopy(episode_info)

        self.pose = pose
        self.trajectory = []
        self.is_end = False

        self.SUCCESS_DISTANCE = 20

        self.DistanceToGoal = {
            '_metric': 0,
            '_previous_position': None,
        }
        self.Success = {
            '_metric': 0,
        }
        self.NDTW = {
            '_metric': 0,
            'locations': [],
            'gt_locations': (np.array(episode_info['reference_path'])[:, 0:3]).tolist(),
        }
        self.SDTW = {
            '_metric': 0,
        }
        self.PathLength = {
            '_metric': 0,
            '_previous_position': None,
        }
        self.OracleSuccess = {
            '_metric': 0,
        }
        self.StepsTaken = {
            '_metric': 0,
        }

        self.distance_data_pre_frame = {
            'distance_front_data': 0.0,
            'distance_up_data': 0.0,
            'distance_down_data': 0.0,
            'distance_left_data': 0.0,
            'distance_right_data': 0.0,
        }

        self.pre_carrot_idx = 0
        self.start_point_nearest_node_token = None
        self.end_point_nearest_node_token = None
        self.progress = 0.0
        self.waypoint = {}
        self.unique_path = None
        self.pre_action = AirsimActions.STOP
        self.is_collisioned = False


class ENV:
    def __init__(self, load_scenes: list):
        self.batch = None

        if args.run_type in ['collect', 'train'] and args.collect_type in ['dagger', 'SF']:
            self.shortest_path_sensor = ShortestPathSensor(args.nav_graph_path, args.token_dict_path, load_scenes)

            self.nav_graph_token_dict = {}
            for scen_id in load_scenes:
                self.nav_graph_token_dict[scen_id] = self.shortest_path_sensor.token_dicts[scen_id]

    def set_batch(self, batch):
        self.batch = copy.deepcopy(batch)
        return

    def get_obs_at(self, index: int, state):
        assert self.batch is not None, 'batch is None'
        item = self.batch[index]

        if args.run_type in ['collect', 'train'] and args.collect_type in ['TF']:
            if state.step >= len(item['actions']) or state.is_end:
                teacher_action = AirsimActions.STOP
                done = state.is_end
                progress = 1.0
            else:
                teacher_action = item['actions'][state.step]
                done = state.is_end
                progress = state.step / len(item['actions'])
        elif args.run_type in ['eval'] and args.collect_type in ['TF']:
            teacher_action = AirsimActions.STOP
            done = state.is_end
            if not done:
                progress = 0.0
            else:
                progress = 1.0

        elif args.run_type in ['collect', 'train'] and args.collect_type in ['dagger', 'SF']:
            if state.is_end:
                teacher_action = AirsimActions.STOP
            else:
                teacher_action, state = get_teacher_action_at(index, state, self.nav_graph_token_dict, self.shortest_path_sensor)

            done = state.is_end
            progress, state = get_progress_sensor_at(index, state, self.nav_graph_token_dict, self.shortest_path_sensor)
        elif args.run_type in ['eval'] and args.collect_type in ['dagger', 'SF']:
            teacher_action = AirsimActions.STOP
            done = state.is_end
            if not done:
                progress = 0.0
            else:
                progress = 1.0

        else:
            logger.error('wrong type')
            raise NotImplementedError

        return (teacher_action, done, progress), state


def get_teacher_action_at(index: int, state: SimState, nav_graph_token_dict: dict, shortest_path_sensor: ShortestPathSensor):
    prev_action = state.pre_action

    if state.step < len(state.episode_info['reference_path']) and \
        np.allclose(
            [state.pose.position.x_val, state.pose.position.y_val, state.pose.position.z_val],
            state.episode_info['reference_path'][state.step][0:3],
            atol=1e-3
        ):

        teacher_action = state.episode_info['actions'][state.step]

    else:
        scene_id = int(state.episode_info['scene_id'])

        source = [state.pose.position.x_val, state.pose.position.y_val, state.pose.position.z_val]
        source_nearest_node_token = cast_point_to_nearest_node_in_nav_graph_2(source, nav_graph_token_dict[scene_id])

        if args.dagger_mode == 'end':
            if state.end_point_nearest_node_token is not None:
                target_nearest_node_token = state.end_point_nearest_node_token
            else:
                target = state.episode_info['goals'][0]['position']
                target_nearest_node_token = cast_point_to_nearest_node_in_nav_graph_2(target, nav_graph_token_dict[scene_id])

                state.end_point_nearest_node_token = target_nearest_node_token

        elif args.dagger_mode == 'nearest':
            waypoint, state = find_waypoint_at(index, state)
            waypoint_token = cast_point_to_nearest_node_in_nav_graph_2(waypoint, nav_graph_token_dict[scene_id])

            target = waypoint
            target_nearest_node_token = waypoint_token

        else:
            raise NotImplementedError

        pose = state.pose


        while True:
            shortest_paths = shortest_path_sensor.get_shortest_paths(
                str(source_nearest_node_token),
                str(target_nearest_node_token),
                int(scene_id)
            )
            if len(shortest_paths) <= 1:
                if state.pre_carrot_idx < len(state.unique_path)-1:
                    waypoint, state = find_waypoint_at(index, state, force_to_find_next=True)
                    waypoint_token = cast_point_to_nearest_node_in_nav_graph_2(waypoint, nav_graph_token_dict[scene_id])

                    target = waypoint
                    target_nearest_node_token = waypoint_token
                else:
                    teacher_action = AirsimActions.STOP
                    return teacher_action, state
            else:
                break

        vs_token = shortest_path_sensor.get_vs_token(shortest_paths[1], scene_id)
        next_point = np.array(nav_graph_token_dict[scene_id][vs_token])


        #
        end_to_current_yaw = math.atan2(next_point[1] - pose.position.y_val, next_point[0] - pose.position.x_val) - airsim.to_eularian_angles(pose.orientation)[2]
        if abs(end_to_current_yaw) > math.radians(90):
            if abs(end_to_current_yaw) > math.radians(180):
                if end_to_current_yaw > 0:
                    end_to_current_yaw = math.radians(-360) + end_to_current_yaw
                else:
                    end_to_current_yaw = math.radians(360) + end_to_current_yaw

            if end_to_current_yaw > 0:
                teacher_action = AirsimActions.TURN_RIGHT
            else:
                teacher_action = AirsimActions.TURN_LEFT

            return teacher_action, state


        #
        new_poses = []
        action_list = [action for action in AirsimActions._known_actions.values()]
        for action in action_list:
            new_pose = getPoseAfterMakeAction(pose, action)

            if action in [AirsimActions.TURN_LEFT, AirsimActions.TURN_RIGHT]:
                new_pose = getPoseAfterMakeAction(new_pose, AirsimActions.MOVE_FORWARD)

            new_poses.append(new_pose)

        new_positions = []
        for new_pose in new_poses:
            new_positions.append(
                np.array([new_pose.position.x_val, new_pose.position.y_val, new_pose.position.z_val])
            )

        distance_list = []
        for position in new_positions:
            distance_list.append(
                EuclideanDistance3(position, next_point)
            )
        distance_list_argsort = np.array(distance_list).argsort()


        #
        teacher_action = AirsimActions.STOP
        for i in range(len(distance_list_argsort)):
            if distance_list_argsort[i] in [AirsimActions.STOP]:
                continue

            # 上下
            if action_list[distance_list_argsort[i]] == AirsimActions.GO_UP and prev_action == AirsimActions.GO_DOWN:
                continue
            elif action_list[distance_list_argsort[i]] == AirsimActions.GO_DOWN and prev_action == AirsimActions.GO_UP:
                continue

            # 左右
            elif action_list[distance_list_argsort[i]] == AirsimActions.MOVE_LEFT and prev_action == AirsimActions.MOVE_RIGHT:
                continue
            elif action_list[distance_list_argsort[i]] == AirsimActions.MOVE_RIGHT and prev_action == AirsimActions.MOVE_LEFT:
                continue

            # 左转 右转
            elif action_list[distance_list_argsort[i]] == AirsimActions.TURN_LEFT and prev_action == AirsimActions.TURN_RIGHT:
                continue
            elif action_list[distance_list_argsort[i]] == AirsimActions.TURN_RIGHT and prev_action == AirsimActions.TURN_LEFT:
                continue

            teacher_action = action_list[distance_list_argsort[i]]
            break

    return teacher_action, state


def find_waypoint_at(index: int, state: SimState, force_to_find_next=False):
    if state.step in state.waypoint.keys() and not force_to_find_next:
        carrot_pos = state.waypoint[state.step]

    else:
        curr_pos = [state.pose.position.x_val, state.pose.position.y_val, state.pose.position.z_val]
        pre_carrot_idx = state.pre_carrot_idx
        path = state.episode_info['reference_path']
        unique_path = state.unique_path

        if not force_to_find_next:
            carrot_pos, carrot_idx, unique_path = find_carrot_pos(curr_pos, pre_carrot_idx, path, unique_path)
        else:
            carrot_pos, carrot_idx, unique_path = find_carrot_pos(curr_pos, pre_carrot_idx+1, path, unique_path)

        state.pre_carrot_idx = carrot_idx
        state.waypoint[state.step] = carrot_pos
        state.unique_path = unique_path

    return carrot_pos, state

def find_carrot_pos(curr_pos, pre_carrot_idx, path, unique_path):
    look_ahead_scale = 10

    assert len(path) > 0, 'wrong path'

    if unique_path is None or len(unique_path) == 0:
        unique_path = []
        for point_index, point in enumerate(path):
            if len(unique_path) > 0 and point[0:3] in [unique_path[-1]]:
                continue
            else:
                unique_path.append(point[0:3])

    assert len(unique_path) > 0, 'wrong unique_path'

    if pre_carrot_idx >= len(unique_path)-1:
        carrot_pos = unique_path[-1][0:3]
        carrot_idx = len(unique_path)-1
        return carrot_pos, carrot_idx, unique_path

    compute_pre_carrot_idx = pre_carrot_idx
    min_distance_xyz = 1e10
    for unique_path_index in range(pre_carrot_idx, min(pre_carrot_idx+3, len(unique_path))):
        distance_xyz = EuclideanDistance3(np.array(curr_pos[0:3]), np.array(unique_path[unique_path_index][0:3]))
        if round(distance_xyz) < min_distance_xyz:
            compute_pre_carrot_idx = unique_path_index
            min_distance_xyz = distance_xyz

    carrot_idx = None
    for unique_path_index in range(compute_pre_carrot_idx, len(unique_path)):
        distance_xy = EuclideanDistance3(np.array(curr_pos[0:2]), np.array(unique_path[unique_path_index][0:2]))
        distance_z = EuclideanDistance1(np.array(curr_pos[2]), np.array(unique_path[unique_path_index][2]))
        if round(distance_xy) >= look_ahead_scale * AirsimActionSettings.FORWARD_STEP_SIZE or \
            round(distance_z) >= look_ahead_scale * AirsimActionSettings.UP_DOWN_STEP_SIZE:
            carrot_idx = unique_path_index
            break

    if carrot_idx is None:
        carrot_idx = len(unique_path) - 1

    carrot_pos = unique_path[carrot_idx][0:3]

    return carrot_pos, carrot_idx, unique_path

def get_progress_sensor_at(index: int, state: SimState, nav_graph_token_dict: dict, shortest_path_sensor: ShortestPathSensor):
    scene_id = int(state.episode_info['scene_id'])

    # 1
    if state.start_point_nearest_node_token is not None:
        start_point_token = state.start_point_nearest_node_token
    else:
        start_point = state.episode_info['start_position']
        start_point_token = cast_point_to_nearest_node_in_nav_graph_2(start_point, nav_graph_token_dict[scene_id])
        state.start_point_nearest_node_token = start_point_token

    # 2
    curr_pos = [state.pose.position.x_val, state.pose.position.y_val, state.pose.position.z_val]
    curr_pos_token = cast_point_to_nearest_node_in_nav_graph_2(curr_pos, nav_graph_token_dict[scene_id])

    # 3
    waypoint, state = find_waypoint_at(index, state)
    waypoint_token = cast_point_to_nearest_node_in_nav_graph_2(waypoint, nav_graph_token_dict[scene_id])

    # 4
    if state.end_point_nearest_node_token is not None:
        goal_token = state.end_point_nearest_node_token
    else:
        goal = state.episode_info['goals'][0]['position']
        goal_token = cast_point_to_nearest_node_in_nav_graph_2(goal, nav_graph_token_dict[scene_id])
        state.end_point_nearest_node_token = goal_token

    # 1
    curr_pos_2_waypoint_shortest_paths = shortest_path_sensor.get_shortest_paths(
        source=curr_pos_token,
        target=waypoint_token,
        scene_id=int(scene_id),
    )
    if len(curr_pos_2_waypoint_shortest_paths) <= 1:
        curr_pos_2_waypoint_steps = 0
    else:
        curr_pos_2_waypoint_steps = len(curr_pos_2_waypoint_shortest_paths) - 1

    # 2
    waypoint_2_goal_steps = len(state.unique_path)-1 - state.pre_carrot_idx

    # 3
    start_point_2_goal_steps = len(state.unique_path)-1


    #
    if start_point_2_goal_steps <= 0:
        logger.warning('start_point_2_goal_steps <= 0')
        progress = 1.0
    else:
        progress = 1.0 - (curr_pos_2_waypoint_steps + waypoint_2_goal_steps) / start_point_2_goal_steps
        if progress <= 0:
            progress = 0.0

    state.progress = progress
    return progress, state


def getPoseAfterMakeAction(pose: airsim.Pose, action):
    current_position = np.array([pose.position.x_val, pose.position.y_val, pose.position.z_val])
    current_rotation = np.array([
        pose.orientation.x_val, pose.orientation.y_val, pose.orientation.z_val, pose.orientation.w_val
    ])

    if action == AirsimActions.MOVE_FORWARD:
        pitch, roll, yaw = airsim.to_eularian_angles(airsim.Quaternionr(
            x_val=current_rotation[0],
            y_val=current_rotation[1],
            z_val=current_rotation[2],
            w_val=current_rotation[3]
        ))
        pitch = 0
        roll = 0

        unit_x = 1 * math.cos(pitch) * math.cos(yaw)
        unit_y = 1 * math.cos(pitch) * math.sin(yaw)
        unit_z = 1 * math.sin(pitch) * (-1)
        unit_vector = np.array([unit_x, unit_y, unit_z])
        assert unit_z == 0

        new_position = np.array(current_position) + unit_vector * AirsimActionSettings.FORWARD_STEP_SIZE
        new_rotation = current_rotation.copy()
    elif action == AirsimActions.TURN_LEFT:
        pitch, roll, yaw = airsim.to_eularian_angles(airsim.Quaternionr(
            x_val=current_rotation[0],
            y_val=current_rotation[1],
            z_val=current_rotation[2],
            w_val=current_rotation[3]
        ))
        pitch = 0
        roll = 0

        new_pitch = pitch
        new_roll = roll
        new_yaw = yaw - math.radians(AirsimActionSettings.TURN_ANGLE)
        if float(new_yaw * 180 / math.pi) < -180:
            new_yaw = math.radians(360) + new_yaw

        new_position = current_position.copy()
        new_rotation = airsim.to_quaternion(new_pitch, new_roll, new_yaw)
        new_rotation = [
            new_rotation.x_val, new_rotation.y_val, new_rotation.z_val, new_rotation.w_val
        ]
    elif action == AirsimActions.TURN_RIGHT:
        pitch, roll, yaw = airsim.to_eularian_angles(airsim.Quaternionr(
            x_val=current_rotation[0],
            y_val=current_rotation[1],
            z_val=current_rotation[2],
            w_val=current_rotation[3]
        ))
        pitch = 0
        roll = 0

        new_pitch = pitch
        new_roll = roll
        new_yaw = yaw + math.radians(AirsimActionSettings.TURN_ANGLE)
        if float(new_yaw * 180 / math.pi) > 180:
            new_yaw = math.radians(-360) + new_yaw

        new_position = current_position.copy()
        new_rotation = airsim.to_quaternion(new_pitch, new_roll, new_yaw)
        new_rotation = [
            new_rotation.x_val, new_rotation.y_val, new_rotation.z_val, new_rotation.w_val
        ]
    elif action == AirsimActions.GO_UP:
        pitch, roll, yaw = airsim.to_eularian_angles(airsim.Quaternionr(
            x_val=current_rotation[0],
            y_val=current_rotation[1],
            z_val=current_rotation[2],
            w_val=current_rotation[3]
        ))
        pitch = 0
        roll = 0

        unit_vector = np.array([0, 0, -1])

        new_position = np.array(current_position) + unit_vector * AirsimActionSettings.UP_DOWN_STEP_SIZE
        new_rotation = current_rotation.copy()
    elif action == AirsimActions.GO_DOWN:
        pitch, roll, yaw = airsim.to_eularian_angles(airsim.Quaternionr(
            x_val=current_rotation[0],
            y_val=current_rotation[1],
            z_val=current_rotation[2],
            w_val=current_rotation[3]
        ))
        pitch = 0
        roll = 0

        unit_vector = np.array([0, 0, -1])

        new_position = np.array(current_position) + unit_vector * AirsimActionSettings.UP_DOWN_STEP_SIZE * (-1)
        new_rotation = current_rotation.copy()
    elif action == AirsimActions.MOVE_LEFT:
        pitch, roll, yaw = airsim.to_eularian_angles(airsim.Quaternionr(
            x_val=current_rotation[0],
            y_val=current_rotation[1],
            z_val=current_rotation[2],
            w_val=current_rotation[3]
        ))
        pitch = 0
        roll = 0

        unit_x = 1.0 * math.cos(math.radians(float(yaw * 180 / math.pi) + 90))
        unit_y = 1.0 * math.sin(math.radians(float(yaw * 180 / math.pi) + 90))
        unit_vector = np.array([unit_x, unit_y, 0])

        new_position = np.array(current_position) + unit_vector * AirsimActionSettings.LEFT_RIGHT_STEP_SIZE * (-1)
        new_rotation = current_rotation.copy()
    elif action == AirsimActions.MOVE_RIGHT:
        pitch, roll, yaw = airsim.to_eularian_angles(airsim.Quaternionr(
            x_val=current_rotation[0],
            y_val=current_rotation[1],
            z_val=current_rotation[2],
            w_val=current_rotation[3]
        ))
        pitch = 0
        roll = 0

        unit_x = 1.0 * math.cos(math.radians(float(yaw * 180 / math.pi) + 90))
        unit_y = 1.0 * math.sin(math.radians(float(yaw * 180 / math.pi) + 90))
        unit_vector = np.array([unit_x, unit_y, 0])

        new_position = np.array(current_position) + unit_vector * AirsimActionSettings.LEFT_RIGHT_STEP_SIZE
        new_rotation = current_rotation.copy()
    else:
        new_position = current_position.copy()
        new_rotation = current_rotation.copy()

    new_pose = airsim.Pose(
        position_val=airsim.Vector3r(
            x_val=new_position[0],
            y_val=new_position[1],
            z_val=new_position[2]
        ),
        orientation_val=airsim.Quaternionr(
            x_val=new_rotation[0],
            y_val=new_rotation[1],
            z_val=new_rotation[2],
            w_val=new_rotation[3]
        )
    )
    return new_pose


def getPoseAfterMakeActions(
        pose: airsim.Pose,
        actions,
):
    current_pose = copy.deepcopy(pose)
    for action in actions:
        current_pose = getPoseAfterMakeAction(current_pose, action)

    return current_pose


def cast_point_to_nearest_node_in_nav_graph_2(pos, nav_graph_token_dict) -> str:
    pos = np.array(pos, np.float32)
    node_key_list = list(nav_graph_token_dict.keys())
    nav_points = np.array(list(nav_graph_token_dict.values()), np.float32)

    Found = False
    radius = 12
    while not Found:
        xyz_limit = np.array([
            pos[0] + radius, pos[0] - radius,
            pos[1] + radius, pos[1] - radius,
            pos[2] + radius, pos[2] - radius,
        ], np.float32)

        nav_points_limit = VerticesPreprocess(nav_points, xyz_limit)

        if len(nav_points_limit) > 0:
            Found = True
        else:
            radius = radius + 5

    distance_list = np.zeros(shape=(len(nav_points_limit),), dtype=np.float32)
    for idx in range(len(nav_points_limit)):
        distance_list[idx] = EuclideanDistance3(pos[0:3], nav_points_limit[idx][0:3])

    nearest_point = nav_points_limit[np.argsort(distance_list)[0]]

    nearest_node_token_idx = WhereIs3(nearest_point, nav_points)
    assert nearest_node_token_idx != -1, 'wrong nearest_node_token_idx'
    nearest_node_token = node_key_list[nearest_node_token_idx]

    return nearest_node_token

@nb.njit(nogil=True, cache=True)
def VerticesPreprocess(vertices: np.array, xyz_limit: np.array):
    x = vertices[:, 0]
    y = vertices[:, 1]
    z = vertices[:, 2]

    xx = x[np.logical_and(z > xyz_limit[5], z < xyz_limit[4])]
    yy = y[np.logical_and(z > xyz_limit[5], z < xyz_limit[4])]
    zz = z[np.logical_and(z > xyz_limit[5], z < xyz_limit[4])]

    xxx = xx[np.logical_and(xx > xyz_limit[1], xx < xyz_limit[0])]
    yyy = yy[np.logical_and(xx > xyz_limit[1], xx < xyz_limit[0])]
    zzz = zz[np.logical_and(xx > xyz_limit[1], xx < xyz_limit[0])]

    xxxx = xxx[np.logical_and(yyy > xyz_limit[3], yyy < xyz_limit[2])]
    yyyy = yyy[np.logical_and(yyy > xyz_limit[3], yyy < xyz_limit[2])]
    zzzz = zzz[np.logical_and(yyy > xyz_limit[3], yyy < xyz_limit[2])]

    dims = np.zeros((len(xxxx), 3), dtype=np.float32)

    dims[:, 0] = xxxx
    dims[:, 1] = yyyy
    dims[:, 2] = zzzz

    return dims

@nb.njit(nogil=True, cache=True)
def WhereIs3(ele, data):
    for idx in range(len(data)):
        if (ele[0] == data[idx][0]) and \
            (ele[1] == data[idx][1]) and \
            (ele[2] == data[idx][2]):
            return idx
    return -1


def cast_point_to_nearest_node_in_nav_graph(pos, nav_graph_token_dict) -> str:
    pos = np.array(pos)
    node_key_list = np.array(list(nav_graph_token_dict.keys()))
    distance_list = np.zeros(shape=(len(nav_graph_token_dict)), dtype=np.float32)
    distance_list += -1
    nav_points = np.array(list(nav_graph_token_dict.values()), np.float32).reshape((-1, 3))

    nearest_node_token = _cast_point_to_nearest_node_in_nav_graph_NUMBA(pos, node_key_list, distance_list, nav_points)

    return nearest_node_token

@nb.njit(nogil=True, cache=True)
def _cast_point_to_nearest_node_in_nav_graph_NUMBA(pos, node_key_list, distance_list, nav_points):
    for idx in range(len(node_key_list)):
        if abs(nav_points[idx][0] - pos[0]) <= 30 and \
            abs(nav_points[idx][1] - pos[1]) <= 30 and \
            abs(nav_points[idx][2] - pos[2]) <= 20:
            distance_list[idx] = EuclideanDistance3(pos[0:3], nav_points[idx][0:3])

    for idx in np.argsort(distance_list):
        if distance_list[idx] == -1:
            continue

        break

    nearest_node_token = node_key_list[idx]

    return nearest_node_token



def get_pano_observations(
        current_pose: airsim.Pose,
        tool: AirVLNSimulatorClientTool,
        scene_id=0
):
    # return pano rgb-d observation.
    # pano sequence:
    # front, slightly right, right, slightly back right, back,
    # slightly back left, left, slightly left
    # actions = [3,3,3]
    actions = [3]
    pano_obs = []
    pano_pose = []
    new_pose = current_pose
    for i in range(8):
        pano_pose.append(np.array(
            [new_pose.position.x_val, new_pose.position.y_val, new_pose.position.z_val,
             new_pose.orientation.x_val,new_pose.orientation.y_val,new_pose.orientation.z_val,new_pose.orientation.w_val]))
        # print(f"new pose:{new_pose}")
        if scene_id in [1, 7]:
            obs_responses = tool.getImageResponses(camera_id='front_0')
        else:
            obs_responses = tool.getImageResponses_v2(camera_id='front_0')
        pano_obs.append(obs_responses[0][0])
        new_pose = getPoseAfterMakeActions(new_pose, actions)
        tool.setPoses([[new_pose]])

    return pano_obs, pano_pose

# def get_pano_observations(
#         current_pose: airsim.Pose,
#         tool: AirVLNSimulatorClientTool,
#         scene_id=0
# ):
#     # return pano rgb-d observation.
#     # pano sequence:
#     # front, slightly right, right, slightly back right, back,
#     # slightly back left, left, slightly left
#     actions = [3, 3, 3]  # 假设3代表特定的旋转动作（如偏航角变化）
#     # 初始化全景观测列表：将存储8个视角的RGB-D图像数据
#     pano_obs = []
#     # 初始化位姿列表：将存储8个视角对应的无人机位姿
#     pano_pose = []
#     # 设置起始姿态为当前无人机的姿态
#     new_pose = current_pose
#
#     original_pose = copy.deepcopy(current_pose)  # 保存初始姿态
#
#     #logger.info(f"位姿: {current_pose}")
#     for i in range(8):
#         # 记录当前无人机的位姿信息（位置和方向）
#         # 创建一个包含7个元素的NumPy数组：[x, y, z, qx, qy, qz, qw]
#         # 其中(x,y,z)是位置，(qx,qy,qz,qw)是表示方向的四元数
#         pano_pose.append(np.array(
#             [new_pose.position.x_val, new_pose.position.y_val, new_pose.position.z_val,
#              new_pose.orientation.x_val,new_pose.orientation.y_val,new_pose.orientation.z_val,new_pose.orientation.w_val]))  #(x,y,z)+方向四元数
#
#         if scene_id in [1, 7]:    # 根据场景ID选择合适的图像获取方法
#             obs_responses = tool.getImageResponses(camera_id='front_0')
#         else:
#             #返回# [机器索引][场景索引][(RGB图像, 深度图像)]
#             obs_responses = tool.getImageResponses_v2(camera_id='front_0')
#         # 全景拍摄需要8个不同视角，每次循环获取的是当前视角的数据；结合原始的pano_obs_imgs = [pano_obs[6][0], pano_obs[7][0], pano_obs[0][0], pano_obs[1][0], pano_obs[2][0], pano_obs[4][0]]进行看
#         # 经过循环旋转相机，维度实际变为了[机器][场景][角度][(RGB图像, 深度图像)]
#         pano_obs.append(obs_responses[0][0]) #  用索引[0][0]获取第一台机器第一个场景的第一个相机数据
#
#         # 执行旋转动作，使无人机转向下一个视角位置
#         new_pose = getPoseAfterMakeActions(new_pose, actions)
#         # 在仿真环境中更新无人机的新姿态
#         tool.setPoses([[new_pose]])
#         #logger.info(f"新位姿: {new_pose}")
#     # 新增：转回初始方向
#     tool.setPoses([[original_pose]])
#     #logger.info(f"转会原始位姿: {original_pose}")
#
#     return pano_obs, pano_pose


# def get_front_observations(
#         current_pose: airsim.Pose,
#         tool: AirVLNSimulatorClientTool,
#         scene_id=0
# ):
#     """获取固定前向相机的观测数据（带错误处理）"""
#     try:
#         # 1. 获取图像响应
#         if scene_id in [1, 7]:
#             responses = tool.getImageResponses(camera_id='front_0')
#         else:
#             responses = tool.getImageResponses_v2(camera_id='front_0')
#
#         # 2. 详细验证响应数据结构
#         if (responses is None or
#                 len(responses) == 0 or
#                 len(responses[0]) == 0 or
#                 len(responses[0][0]) == 0):
#             logger.warning("获取到空图像响应")
#             return None, None
#
#         # 3. 提取相机数据 - 不再尝试解包元组
#         # responses[0][0][0] 是 (img_rgb, img_depth) 元组
#         camera_data = responses[0][0]
#
#         # 4. 验证相机数据类型
#         if not isinstance(camera_data, tuple) or len(camera_data) != 2:
#             logger.warning(
#                 f"相机数据结构错误: {type(camera_data)}, 长度: {len(camera_data) if hasattr(camera_data, '__len__') else '无'}")
#             return None, None
#
#         # 5. 获取相机位姿
#         camera_pose = [
#             current_pose.position.x_val,
#             current_pose.position.y_val,
#             current_pose.position.z_val,
#             current_pose.orientation.x_val,
#             current_pose.orientation.y_val,
#             current_pose.orientation.z_val,
#             current_pose.orientation.w_val
#         ]
#         # 6. 返回相机数据和位姿
#         return camera_data, camera_pose
#
#     except Exception as e:
#         logger.error(f"获取前向观测失败: {type(e).__name__}: {str(e)}")
#         import traceback
#         logger.debug(traceback.format_exc())
#         return None, None
#
#     except Exception as e:
#         # 7. 详细错误日志（包含具体错误位置）
#         logger.error(f"获取前向观测失败: {type(e).__name__}: {str(e)}")
#         # 记录堆栈跟踪便于调试
#         import traceback
#         logger.debug(traceback.format_exc())
#         return None, None



def get_front_observations(
        current_pose: airsim.Pose,
        tool: AirVLNSimulatorClientTool,
        scene_id=0
):
    # return pano rgb-d observation.
    # pano sequence:
    # front, slightly right, right, slightly back right, back,
    # slightly back left, left, slightly left
    pano_obs = []
    pano_pose = []

    if scene_id in [1, 7]:
        obs_responses = tool.getImageResponses(camera_id='front_0')
    else:
        obs_responses = tool.getImageResponses_v2(camera_id='front_0')
    pano_obs.append(obs_responses[0][0])

    return pano_obs, pano_pose


def evaluate_spatial_accuracy(detections, airsim_client: airsim.VehicleClient, logger=None):
    if logger is None:
        import logging
        logger = logging.getLogger(__name__)

    try:
        all_object_names = airsim_client.simListSceneObjects()
    except Exception as e:
        logger.warning(f"⚠️ 无法获取 AirSim 场景对象列表: {e}")
        return

    keyword_to_gt = {}
    for name in all_object_names:
        if any(prefix in name for prefix in ["Drone_", "BP_", "SK_", "SM_", "Camera_", "Player"]):
            continue
        try:
            pose = airsim_client.simGetObjectPose(name)
            if pose.position == airsim.Vector3r(0, 0, 0):
                continue
            center = np.array([pose.position.x_val, pose.position.y_val, pose.position.z_val])
            clean_name = ''.join([c if c.isalpha() else ' ' for c in name]).strip().lower()
            keywords = [kw for kw in clean_name.split() if len(kw) >= 3]
            for kw in keywords:
                keyword_to_gt.setdefault(kw, []).append({"name": name, "center": center})
        except Exception:
            continue

    # === 临时调试：打印关键词映射（只打一次）===
    if detections and not getattr(evaluate_spatial_accuracy, '_debug_logged', False):
        logger.info("🔍 真值关键词映射:")
        for kw, objs in keyword_to_gt.items():
            logger.info(f"  '{kw}' → {[obj['name'] for obj in objs]}")
        evaluate_spatial_accuracy._debug_logged = True
    # ==========================================

    for det in detections:
        label = det["label"].lower()
        pred_pos = np.array(det["coord"])
        label_words = set(label.split())  # 🔥 关键：拆分为独立词
        matched_gt_list = []

        for kw, gt_list in keyword_to_gt.items():
            if kw in label_words:  # 🔥 仅当 kw 是 label 的一个独立词
                matched_gt_list.extend(gt_list)

        if not matched_gt_list:
            logger.debug(f"🔍 无真值匹配: 预测 '{label}'")
            continue

        distances = [np.linalg.norm(pred_pos - gt["center"]) for gt in matched_gt_list]
        min_idx = int(np.argmin(distances))
        min_dist = distances[min_idx]
        matched_gt = matched_gt_list[min_idx]

        logger.info(
            f"🎯 空间推理验证 | Label: {label} | "
            f"预测: [{pred_pos[0]:.1f}, {pred_pos[1]:.1f}, {pred_pos[2]:.1f}] | "
            f"真值: {matched_gt['name']} @ [{matched_gt['center'][0]:.1f}, {matched_gt['center'][1]:.1f}, {matched_gt['center'][2]:.1f}] | "
            f"L2误差: {min_dist:.2f}m"
        )