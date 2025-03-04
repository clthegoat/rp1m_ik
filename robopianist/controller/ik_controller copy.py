from robopianist.utils import inverse_kinematics as ik
from robopianist.utils import qp_solver
import numpy as np
import dm_env
from mujoco_utils import mjcf_utils, physics_utils, spec_utils, types
from dm_control.mujoco.wrapper import mjbindings

LH_JOINTS_NAMES = [
    "lh_shadow_hand/lh_WRJ2",
    "lh_shadow_hand/lh_WRJ1",
    "lh_shadow_hand/lh_THJ4",
    "lh_shadow_hand/lh_THJ3",
    "lh_shadow_hand/lh_THJ2",
    "lh_shadow_hand/lh_FFJ4",
    "lh_shadow_hand/lh_FFJ3",
    # "lh_shadow_hand/lh_FFJ0",
    "lh_shadow_hand/lh_FFJ2", # tendon lh_FFJ0
    "lh_shadow_hand/lh_FFJ1", # tendon lh_FFJ0
    "lh_shadow_hand/lh_MFJ4",
    "lh_shadow_hand/lh_MFJ3",
    # "lh_shadow_hand/lh_MFJ0",
    "lh_shadow_hand/lh_MFJ2", # tendon lh_MFJ0
    "lh_shadow_hand/lh_MFJ1", # tendon lh_MFJ0
    "lh_shadow_hand/lh_RFJ4",
    "lh_shadow_hand/lh_RFJ3",
    # "lh_shadow_hand/lh_RFJ0",
    "lh_shadow_hand/lh_RFJ2", # tendon lh_RFJ0
    "lh_shadow_hand/lh_RFJ1", # tendon lh_RFJ0
    "lh_shadow_hand/lh_LFJ4",
    "lh_shadow_hand/lh_LFJ3",
    # "lh_shadow_hand/lh_LFJ0",
    "lh_shadow_hand/lh_LFJ2", # tendon lh_LFJ0
    "lh_shadow_hand/lh_LFJ1", # tendon lh_LFJ0
    "lh_shadow_hand/forearm_tx",
    "lh_shadow_hand/forearm_ty"
]

RH_JOINTS_NAMES = [
    "rh_shadow_hand/rh_WRJ2",
    "rh_shadow_hand/rh_WRJ1",
    "rh_shadow_hand/rh_THJ4",
    "rh_shadow_hand/rh_THJ3",
    "rh_shadow_hand/rh_THJ2",
    "rh_shadow_hand/rh_FFJ4",
    "rh_shadow_hand/rh_FFJ3",
    # "rh_shadow_hand/rh_FFJ0",
    "rh_shadow_hand/rh_FFJ2", # tendon rh_FFJ0
    "rh_shadow_hand/rh_FFJ1", # tendon rh_FFJ0
    "rh_shadow_hand/rh_MFJ4",
    "rh_shadow_hand/rh_MFJ3",
    # "rh_shadow_hand/rh_MFJ0",
    "rh_shadow_hand/rh_MFJ2", # tendon rh_MFJ0
    "rh_shadow_hand/rh_MFJ1", # tendon rh_MFJ0
    "rh_shadow_hand/rh_RFJ4",
    "rh_shadow_hand/rh_RFJ3",
    # "rh_shadow_hand/rh_RFJ0",
    "rh_shadow_hand/rh_RFJ2", # tendon rh_RFJ0
    "rh_shadow_hand/rh_RFJ1", # tendon rh_RFJ0
    "rh_shadow_hand/rh_LFJ4",
    "rh_shadow_hand/rh_LFJ3",
    # "rh_shadow_hand/rh_LFJ0",
    "rh_shadow_hand/rh_LFJ2", # tendon rh_LFJ0
    "rh_shadow_hand/rh_LFJ1", # tendon rh_LFJ0
    "rh_shadow_hand/forearm_tx",
    "rh_shadow_hand/forearm_ty"
]

ALL_JOINTS_NAMES = RH_JOINTS_NAMES + LH_JOINTS_NAMES

# For one hand
RH_FF_BASE_IDX = 92
RH_MF_BASE_IDX = 96
RH_RF_BASE_IDX = 100
RH_LF_BASE_IDX = 104
RH_THUMB_BASE_IDX = 108
LH_FF_BASE_IDX = 115
LH_MF_BASE_IDX = 119
LH_RF_BASE_IDX = 123
LH_LF_BASE_IDX = 127
LH_THUMB_BASE_IDX = 131

RH_FINGER_BASE_IDX = {'ff': RH_FF_BASE_IDX, 'mf': RH_MF_BASE_IDX, 'rf': RH_RF_BASE_IDX, 'lf': RH_LF_BASE_IDX, 'th': RH_THUMB_BASE_IDX}
LH_FINGER_BASE_IDX = {'ff': LH_FF_BASE_IDX, 'mf': LH_MF_BASE_IDX, 'rf': LH_RF_BASE_IDX, 'lf': LH_LF_BASE_IDX, 'th': LH_THUMB_BASE_IDX}

# Joints per finger
FINGER_JOINTS = {'ff': 4, 'mf': 4, 'rf': 4, 'lf': 4, 'th':5}

# Piano key indices for Environment B
WHITE_KEY_INDICES = [
    0, 2, 3, 5, 7, 8, 10, 12, 14, 15, 17, 19, 20, 22, 24, 26, 27, 29, 31, 32, 34, 36, 38, 39,
    41, 43, 44, 46, 48, 50, 51, 53, 55, 56, 58, 60, 62, 63, 65, 67, 68, 70, 72, 74, 75, 77, 79,
    80, 82, 84, 86, 87
]

BLACK_KEY_INDICES = [
    1, 4, 6, 9, 11, 13, 16, 18, 21, 23, 25, 28, 30, 33, 35, 37, 40, 42, 45, 47, 49, 52, 54, 57,
    59, 61, 64, 66, 69, 71, 73, 76, 78, 81, 83, 85
]


mjlib = mjbindings.mjlib

_INVALID_JOINT_NAMES_TYPE = (
    '`joint_names` must be either None, a list, a tuple, or a numpy array; '
    'got {}.')
_REQUIRE_TARGET_POS_OR_QUAT = (
    'At least one of `target_pos` or `target_quat` must be specified.')



def move_fingers_to_pos_qp(env: dm_env.Environment, 
                            hand_action: np.ndarray,
                            finger_names: list=['lf', 'rf', 'mf', 'ff', 'th'],
                            hand_side: str='none',
                            targeting_wrist: bool=False,
                            ):
    """Move the specified fingers to the
      specified position."""
    if hand_side == 'left':
        FINGER_BASE_IDX = LH_FINGER_BASE_IDX
        prefix = "lh_shadow_hand/"
    elif hand_side == 'right':
        FINGER_BASE_IDX = RH_FINGER_BASE_IDX
        prefix = "rh_shadow_hand/"
    else:
        FINGER_BASE_IDX = OH_FINGER_BASE_IDX
        prefix = "lh_shadow_hand/"
    # if targeting_wrist:
    #     site_names = [prefix+"wrist_site"]
    # else:
    #     site_names = []
    site_names = []
    site_names.extend([prefix+finger_name+"distal_site" for finger_name in finger_names])
    # site_names = [prefix+finger_name+"distal_site" for finger_name in finger_names]
    target_poses = []
    target_quats = []
    pos_weights = np.array([1e4]*len(site_names))
    # Reset the color of all keys
    # env.physics.bind(env.task.piano._key_geoms).rgba = (0.5, 0.5, 0.5, 1.0)
    for i in range(6):
        # if hand_action[7][i] != -1:
        #     env.physics.bind(env.task.piano._key_geoms[int(hand_action[7][i])]).rgba = (0.0, 1.0, 0.0, 1.0)
            # pos_weights[i-1] = 1.0
        if i == 0:
            # if not targeting_wrist:
            #     # Don't set target position for the wrist
            #     continue
            continue
        target_pos = (hand_action[0][i], hand_action[1][i], hand_action[2][i])
        # target_quat = (hand_action[3][i], hand_action[4][i], hand_action[5][i], hand_action[6][i])
        target_poses.append(target_pos)
        # target_quats.append(target_quat)
    # print(target_poses)
    # for key in key_indices:
    #     if key not in hand_action[7]:
    #         env.physics.bind(env.task.piano._key_geoms[key]).rgba = (1.0, 0.0, 0.0, 1.0)
        # if key in hand_action[7]:
        #     env.physics.bind(env.task.piano._key_geoms[key]).rgba = (0.5, 0.5, 0.0, 1.0)
    # Wrist, forearm and the dedicated finger joints are available for IK
    if hand_side == 'left':
        wrist_joint_names = [prefix+"lh_WRJ2",
                               prefix+"lh_WRJ1"]
        forearm_joint_names = [prefix+"forearm_tx",
                                prefix+'forearm_ty']
    elif hand_side == 'right':
        wrist_joint_names = [prefix+"rh_WRJ2",
                               prefix+"rh_WRJ1"]
        forearm_joint_names = [prefix+"forearm_tx",
                                prefix+'forearm_ty']
    else:
        wrist_joint_names = ["lh_shadow_hand/"+"lh_WRJ2",
                               "lh_shadow_hand/"+"lh_WRJ1"]
        forearm_joint_names = ["lh_shadow_hand/"+"forearm_tx",
                                "lh_shadow_hand/"+'forearm_ty']
    finger_joint_names = []
    # for finger_name in finger_names:
    #     finger_joint_names.extend([env.physics.model.id2name(FINGER_BASE_IDX[finger_name]+i, 'joint') for i in range(FINGER_JOINTS[finger_name])])
    # joint_names = forearm_joint_names + finger_joint_names + wrist_joint_names
    # joint_names = wrist_joint_names + finger_joint_names + forearm_joint_names
    if hand_side == 'left':
        # joint_names = [name for name in joint_names if name in LH_JOINTS_NAMES]
        joint_names = LH_JOINTS_NAMES
    elif hand_side == 'right':
        # joint_names = [name for name in joint_names if name in RH_JOINTS_NAMES]
        joint_names = RH_JOINTS_NAMES
    # Calculate the IK result
    # print(f"Joint names: {joint_names}")
    # print(f"finger_names: {finger_names}")
    # print(f"forearm_joint_names: {forearm_joint_names}")
    # print(f"finger_joint_names: {finger_joint_names}")
    # print(f"wrist_joint_names: {wrist_joint_names}")

    solver = qp_solver.IK_qpsolver(physics=env.physics,
                                    site_names=site_names,
                                    target_pos=target_poses,
                                    joint_names=joint_names,
                                    pos_weights=pos_weights,
                                    )
    qvel = solver.solve()
    return qvel, solver.dof_indices, target_poses

def move_fingers_to_pos(env: dm_env.Environment, 
                        hand_action: np.ndarray,
                        finger_names: list=['lf', 'rf', 'mf', 'ff', 'th'],
                        hand_side: str='none',
                        ):
    """Move the specified fingers to the
      specified position."""
    if hand_side == 'left':
        FINGER_BASE_IDX = LH_FINGER_BASE_IDX
        prefix = "lh_shadow_hand/"
    elif hand_side == 'right':
        FINGER_BASE_IDX = RH_FINGER_BASE_IDX
        prefix = "rh_shadow_hand/"
    else:
        FINGER_BASE_IDX = OH_FINGER_BASE_IDX
        prefix = "lh_shadow_hand/"
    site_names = [prefix+finger_name+"distal_site" for finger_name in finger_names]
    target_poses = []
    target_quats = []
    # Reset the color of all keys
    env.physics.bind(env.task.piano._key_geoms).rgba = (0.5, 0.5, 0.5, 1.0)
    for i in range(6):
        if hand_action[7][i] != -1:
            env.physics.bind(env.task.piano._key_geoms[int(hand_action[7][i])]).rgba = (0.0, 1.0, 0.0, 1.0)
        if i == 0:
            # Don't set target position for the wrist
            continue
        target_pos = (hand_action[0][i], hand_action[1][i], hand_action[2][i])
        target_quat = (hand_action[3][i], hand_action[4][i], hand_action[5][i], hand_action[6][i])
        target_poses.append(target_pos)
        target_quats.append(target_quat)
    # Change the color of the activated key
    activation = env.task.piano.activation
    # Find the index of True
    key_indices = np.where(activation == True)[0]
    for key in key_indices:
        if key not in hand_action[7]:
            env.physics.bind(env.task.piano._key_geoms[key]).rgba = (1.0, 0.0, 0.0, 1.0)
    # Wrist, forearm and the dedicated finger joints are available for IK
    if hand_side == 'left':
        wrist_joint_names = [prefix+"lh_WRJ2",
                               prefix+"lh_WRJ1"]
        forearm_joint_names = [prefix+"forearm_tx",
                                prefix+'forearm_ty',
                                prefix+'forearm_tz']
    elif hand_side == 'right':
        wrist_joint_names = [prefix+"rh_WRJ2",
                               prefix+"rh_WRJ1"]
        forearm_joint_names = [prefix+"forearm_tx",
                                prefix+'forearm_ty',
                                prefix+'forearm_tz']
    else:
        wrist_joint_names = ["lh_shadow_hand/"+"lh_WRJ2",
                               "lh_shadow_hand/"+"lh_WRJ1"]
        forearm_joint_names = ["lh_shadow_hand/"+"forearm_tx",
                                "lh_shadow_hand/"+'forearm_ty',
                                "lh_shadow_hand/"+'forearm_tz']
    finger_joint_names = []
    for finger_name in finger_names:
        finger_joint_names.extend([env.physics.model.id2name(FINGER_BASE_IDX[finger_name]+i, 'joint') for i in range(FINGER_JOINTS[finger_name])])
    joint_names = forearm_joint_names + finger_joint_names + wrist_joint_names
    # Calculate the IK result
    ik_result = ik.qpos_from_multiple_site_pos(physics=env.physics,
                                                site_names=site_names,
                                                target_pos=target_poses,
                                                # target_quat=target_quats,
                                                joint_names=joint_names,
                                                pos_weight=np.array([1.0]*len(site_names)),
                                                # rot_weight=0.5
                                                )
    return ik_result

def move_fingers_to_keys(env: dm_env.Environment, 
                         key_indices: list, 
                         offset_x: float=[0]*5,
                         offset_y: float=[0]*5,
                         finger_names: list=['lf', 'rf', 'mf', 'ff', 'th'],
                         ):
    """Move the specified fingers to the
      specified piano keys."""
    # print(env.physics.model.site_group)
    assert len(key_indices) == len(finger_names)
    site_names = ["lh_shadow_hand/"+finger_name+"distal_site" for finger_name in finger_names]
    target_poses = []
    for i, key_index in enumerate(key_indices):
        if key_index in WHITE_KEY_INDICES:
            prefix = "white_key_"
        elif key_index in BLACK_TWIN_KEY_INDICES or key_index in BLACK_TRIPLET_KEY_INDICES:
            prefix = "black_key_"
        else:
            raise ValueError(f"Invalid key index: {key_index}")
        target_key = mjcf_utils.safe_find(env.task.piano._mjcf_root, "body", f"{prefix}{key_index}")
        target_pos = target_key.pos
        target_pos = (target_pos[0]+offset_x[i], target_pos[1]+offset_y[i], target_pos[2])
        # target_key.add("site", type="sphere", pos=(0.1, 0.1, 0.1), rgba=(1, 0, 0, 1))
        key_geom = env.task.piano.keys[key_index].geom[0]
        # Uncomment method _update_key_color
        # env.physics.bind(key_geom).rgba = (1.0, 1.0, 1.0, 1.0)
        env.physics.bind(env.task.piano._key_geoms[key_index]).rgba = (0.0, 1.0, 0.0, 1.0)
        # print(key_geom.rgba)
        # print(mjcf_utils.safe_find_all(env.task.piano._mjcf_root, "site"))

        target_poses.append(target_pos)
    # Wrist, forearm and the dedicated finger joints are available for IK
    wrist_joint_names = ["lh_shadow_hand/"+"lh_WRJ2",
                            "lh_shadow_hand/"+"lh_WRJ1"]
    forearm_joint_names = ["lh_shadow_hand/"+env.task._hand.joints[-3].name,
                            "lh_shadow_hand/"+env.task._hand.joints[-2].name, 
                            "lh_shadow_hand/"+env.task._hand.joints[-1].name]
    finger_joint_names = []
    for finger_name in finger_names:
        finger_joint_names.extend([env.physics.model.id2name(FINGER_BASE_IDX[finger_name]+i, 'joint') for i in range(FINGER_JOINTS[finger_name])])
    joint_names = forearm_joint_names + finger_joint_names + wrist_joint_names
    # Calculate the IK result
    ik_result = ik.qpos_from_multiple_site_pos(physics=env.physics,
                                                site_names=site_names,
                                                target_pos=target_poses,
                                                joint_names=joint_names,
                                                pos_weight=np.array([1.0]*len(site_names)),
                                                )
    return ik_result

def move_finger_to_key(env: dm_env.Environment, key_index: int, finger_name):
    """Move the specified finger to the specified piano key."""
    if key_index in WHITE_KEY_INDICES:
        prefix = "white_key_"
    elif key_index in BLACK_TWIN_KEY_INDICES or key_index in BLACK_TRIPLET_KEY_INDICES:
        prefix = "black_key_"
    else:
        raise ValueError(f"Invalid key index: {key_index}")
    
    # Position of the piano joint as the target position
    target_pos = mjcf_utils.safe_find(env.task.piano._mjcf_root, "body", f"{prefix}{key_index}").pos

    # # For test
    # target_pos = (0.4, mjcf_utils.safe_find(env.task.piano._mjcf_root, "body", f"{prefix}{key_index}").pos[1], 0.13)
    # print(target_pos)
    # print(mjcf_utils.safe_find_all(env.task._hand._mjcf_root, "body"))
    # print(mjcf_utils.safe_find_all(env.task._hand._mjcf_root, "site"))
    # print(env.task._hand.root_body.pos) 
    # print(env.physics.named.data.qpos)
    # print(env.physics.model.id2name(89, 'joint'))

    # Wrist, forearm and the dedicated finger joints are available for IK
    wrist_joint_names = ["lh_shadow_hand/"+"lh_WRJ2",
                            "lh_shadow_hand/"+"lh_WRJ1"]
    forearm_joint_names = ["lh_shadow_hand/"+env.task._hand.joints[-2].name, 
                            "lh_shadow_hand/"+env.task._hand.joints[-1].name]
    finger_joint_names = [env.physics.model.id2name(FINGER_BASE_IDX[finger_name]+i, 'joint') for i in range(FINGER_JOINTS[finger_name])]
    joint_names = forearm_joint_names + finger_joint_names + wrist_joint_names
    
    # Calculate the IK result
    ik_result = ik.qpos_from_site_pose(physics=env.physics, 
                                       site_name="lh_shadow_hand/"+finger_name+"distal_site", 
                                       target_pos=target_pos,
                                       joint_names=joint_names,
                                    #    joint_names=("lh_shadow_hand/"+env.task._hand.joints[-2].name, 
                                    #                 "lh_shadow_hand/"+env.task._hand.joints[-1].name),
                                       )
    return ik_result
    
    
