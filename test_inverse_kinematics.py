#%%
import sys, os
sys.path.insert(0, os.path.abspath(".."))
os.environ['MUJOCO_GL'] = 'egl'

from dm_control.mujoco.wrapper import mjbindings

import subprocess
if subprocess.run("nvidia-smi").returncode:
    raise RuntimeError(
        "Cannot communicate with GPU. "
        "Make sure you are using a GPU Colab runtime. "
        "Go to the Runtime menu and select Choose runtime type."
    )
import collections
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dm_env
import zarr
import tqdm

from robopianist.suite.tasks import self_actuated_piano
from robopianist.suite.tasks import piano_with_shadow_hands
from robopianist.wrappers import PianoSoundVideoWrapper
from robopianist import music

import robopianist.wrappers as robopianist_wrappers
from robopianist.wrappers.residual import ResidualWrapper

from mujoco_utils import composer_utils, mjcf_utils, physics_utils, spec_utils, types

from dm_env_wrappers import CanonicalSpecWrapper
from dm_env_wrappers._src import base
import dm_env_wrappers as wrappers

from einops import rearrange
from tqdm.auto import trange

class ForearmPrimitiveCollisionWrapper(base.EnvironmentWrapper):
    def __init__(self, env):
        super().__init__(env)
        self._env = env

        # change the collision type of the forearm from mesh to capsule
        for geom in self._env.task.right_hand._mjcf_root.find_all('geom'):
            if (geom.dclass.dclass == "plastic_collision"
                and geom.mesh is not None
                and geom.mesh.name is not None
                and geom.mesh.name.endswith("forearm_collision")):
                geom.type = "capsule"
                geom.margin = 0.03
        for geom in self._env.task.left_hand._mjcf_root.find_all('geom'):
            if (geom.dclass.dclass == "plastic_collision"
                and geom.mesh is not None
                and geom.mesh.name is not None
                and geom.mesh.name.endswith("forearm_collision")):
                geom.type = "capsule"
                geom.margin = 0.03

    def reset(self):
        return self._env.reset()

    def step(self, action):
        time_step = self._env.step(action)
        return time_step

# create a simple data loader for behavior cloning training
class RP1M_DataLoader:
    def __init__(self, data_path, task_name, batch_size):
        self.batch_size = batch_size
        data_root = zarr.open(data_path, 'r')
        print(data_root.tree())
        task_data = data_root[f'{task_name}']
        # 'actions', 'goals', 'hand_fingertips', 'hand_joints', 'piano_states'
        # RoboPianist's observation includes [goal, piano_state, hand_state]
        observations = np.concatenate([
            task_data['goals'].astype(np.float32),
            task_data['piano_states'].astype(np.float32),
            task_data['hand_joints'].astype(np.float32)
        ], axis=-1)
        self.observations = rearrange(observations,
                                'num_traj ep_len state_dim -> (num_traj ep_len) state_dim')
        self.observations_episode = rearrange(observations,
                                'num_traj ep_len state_dim -> num_traj ep_len state_dim')
        actions = task_data['actions'][:].astype(np.float32)
        self.actions = rearrange(actions,
                                'num_traj ep_len action_dim -> (num_traj ep_len) action_dim')
        self.actions_episode = rearrange(actions,
                                'num_traj ep_len action_dim -> num_traj ep_len action_dim')
        fingertip_pos = task_data['hand_fingertips'][:].astype(np.float32)  # Added [:]
        self.fingertip_pos = rearrange(fingertip_pos,
                                'num_traj ep_len fingertip_dim -> (num_traj ep_len) fingertip_dim')
        self.fingertip_pos_episode = rearrange(fingertip_pos,
                                'num_traj ep_len fingertip_dim -> num_traj ep_len fingertip_dim')
        
        self.hand_joints = task_data['hand_joints'][:].astype(np.float32)
        self.hand_joints_episode = rearrange(self.hand_joints,
                                'num_traj ep_len hand_joint_dim -> num_traj ep_len hand_joint_dim')
    def sample(self):
        idx = np.random.randint(0, self.observations.shape[0], self.batch_size)
        return self.observations[idx], self.actions[idx], self.fingertip_pos[idx], self.hand_joints[idx]

    def sample_a_sequence(self, seq_len=10):
        idx = np.random.randint(0, self.observations.shape[0], seq_len)
        return self.observations[idx], self.actions[idx], self.fingertip_pos[idx], self.hand_joints[idx], seq_len
    
    def sample_a_episode(self):
        # idx = np.random.randint(0, self.observations_episode.shape[0])
        idx = 0
        return self.observations_episode[idx], self.actions_episode[idx], self.fingertip_pos_episode[idx], self.hand_joints_episode[idx]

# %%

if __name__ == "__main__":

    # set seed
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    data_path = '/fast/lechen/rp1m_data/rp1m_toy/rp1m_toy.zarr'
    task_name = 'RoboPianist-debug-TwinkleTwinkleRousseau-v0_0'

    dataloader = RP1M_DataLoader(data_path, task_name, batch_size=1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Get target positions from the dataloader
    obs, actions, fingertip_pos, hand_joints = dataloader.sample_a_episode()
    print(f"obs shape: {obs.shape}, actions shape: {actions.shape}, fingertip_pos shape: {fingertip_pos.shape}, hand_joints shape: {hand_joints.shape}")
    # import ipdb; ipdb.set_trace()
    # fingertip_pos = fingertip_pos[0:197, :]
    # actions = actions[0:197, :]
    # print(f"actions shape: {actions.shape}") # (197, 39)
    seq_len = fingertip_pos.shape[0]

    # take the first 15 for right hand and the last 15 for left hand
    right_hand_fingertip_pos = fingertip_pos[:, :15]
    right_wrist_pos = np.zeros((seq_len, 3))
    right_hand_fingertip_pos_reshape = np.zeros((seq_len, 3, 6))
    right_hand_fingertip_pos_reshape[:, 0:3, 0] = right_wrist_pos
    right_hand_fingertip_pos_reshape[:, 0:3, 1] = right_hand_fingertip_pos[:, 0:3]
    right_hand_fingertip_pos_reshape[:, 0:3, 2] = right_hand_fingertip_pos[:, 3:6]
    right_hand_fingertip_pos_reshape[:, 0:3, 3] = right_hand_fingertip_pos[:, 6:9]
    right_hand_fingertip_pos_reshape[:, 0:3, 4] = right_hand_fingertip_pos[:, 9:12]
    right_hand_fingertip_pos_reshape[:, 0:3, 5] = right_hand_fingertip_pos[:, 12:15]

    left_hand_fingertip_pos = fingertip_pos[:, 15:]  
    left_wrist_pos = np.zeros((seq_len, 3))
    left_hand_fingertip_pos_reshape = np.zeros((seq_len, 3, 6))
    left_hand_fingertip_pos_reshape[:, 0:3, 0] = left_wrist_pos
    left_hand_fingertip_pos_reshape[:, 0:3, 1] = left_hand_fingertip_pos[:, 0:3]
    left_hand_fingertip_pos_reshape[:, 0:3, 2] = left_hand_fingertip_pos[:, 3:6]
    left_hand_fingertip_pos_reshape[:, 0:3, 3] = left_hand_fingertip_pos[:, 6:9]
    left_hand_fingertip_pos_reshape[:, 0:3, 4] = left_hand_fingertip_pos[:, 9:12]
    left_hand_fingertip_pos_reshape[:, 0:3, 5] = left_hand_fingertip_pos[:, 12:15]
    
    left_hand_action_list = left_hand_fingertip_pos_reshape  # (197, 3, 6)
    right_hand_action_list = right_hand_fingertip_pos_reshape  # (197, 3, 6)
    
    task = piano_with_shadow_hands.PianoWithShadowHands(
        change_color_on_activation=True,
        midi=music.load("TwinkleTwinkleRousseau", stretch=1.25), # note that in the RP1M dataset, stretch 1.25 is applied.
        trim_silence=True,
        control_timestep=0.05,
        gravity_compensation=True,
        primitive_fingertip_collisions=False, # @YI
        reduced_action_space=True,
        n_steps_lookahead=0,
        disable_fingering_reward=True,
        disable_forearm_reward=False,
        disable_colorization=False,
        disable_hand_collisions=True, # @YI
        attachment_yaw=0.0,
    )

    # task = piano_with_shadow_hands.PianoWithShadowHands(
    #     change_color_on_activation=True,
    #     midi=music.load("TwinkleTwinkleRousseau", stretch=1.25), # note that in the RP1M dataset, stretch 1.25 is applied.
    #     trim_silence=True,
    #     control_timestep=0.05,
    #     gravity_compensation=True,
    #     primitive_fingertip_collisions=True,
    #     reduced_action_space=True,
    #     n_steps_lookahead=0,
    #     disable_fingering_reward=True,
    #     disable_forearm_reward=False,
    #     disable_colorization=False,
    #     disable_hand_collisions=False,
    #     attachment_yaw=0.0,
    # )

    env = composer_utils.Environment(
        task=task, strip_singleton_obs_buffer_dim=True, recompile_physics=False
    )

    env = PianoSoundVideoWrapper(
        env,
        record_every=1,
        camera_id="piano/back",
        record_dir=".",
    )

    # @ YI
    env = ForearmPrimitiveCollisionWrapper(env)

    env = ResidualWrapper(env, 
                        demonstrations_lh=left_hand_action_list,
                        demonstrations_rh=right_hand_action_list,
                        demo_ctrl_timestep=0.05,
                        enable_ik=True)

    env = wrappers.EpisodeStatisticsWrapper(
        environment=env, deque_size=1
    )
    env = robopianist_wrappers.MidiEvaluationWrapper(
        environment=env, deque_size=1
    )
    
    # env = wrappers.DmControlWrapper(env) 
    
    env = CanonicalSpecWrapper(env)

    step = 0
    timestep = env.reset()
    reward = 0
    
    timesteps = tqdm.tqdm(range(actions.shape[0]))
    for step in timesteps:
        action = actions[step]
        timestep = env.step(action)
        reward += timestep.reward
        timesteps.set_description(f"Reward: {reward:.2f}")
        if timestep.last():
            # break
            env.reset()

    print(env.get_musical_metrics())

    # mj_model = env.physics.model
    # name_list=[]
    # for i in range(mj_model.njnt):
    #     print(i, mj_model.joint(i).name)
