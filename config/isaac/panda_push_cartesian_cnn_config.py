import sys
sys.path.append("../isaac_projects/panda-isaac")
from panda_isaac.panda_push import PandaPushEnv
from panda_isaac.base_config import BaseConfig
sys.path.remove("../isaac_projects/panda-isaac")


STATE_HISTORY_LENGTH = 1
IMAGE_HISTORY_LENGTH = 1


class PushConfig(BaseConfig):
    class env(BaseConfig.env):
        seed = 42
        num_envs = 512
        num_observations = IMAGE_HISTORY_LENGTH * 3 * 84 * 84 + STATE_HISTORY_LENGTH * 15
        num_actions = 4
        num_state_obs = 18 * STATE_HISTORY_LENGTH
        max_episode_length = 100
    
    class asset(BaseConfig.asset):
        robot_urdf = "urdf/franka_description/robots/franka_panda_cam.urdf"
    
    class obs(BaseConfig.obs):
        type = "pixel"
        im_size = 84
        history_length = IMAGE_HISTORY_LENGTH
        state_history_length = STATE_HISTORY_LENGTH
        noise = True
    
    class cam(BaseConfig.cam):
        view = "ego"
        # view = "third"
        fov = 88
        w = 149
        h = 84
        # loc_r = [180, -45.0, 180.0]
        # loc_p = [0.11104 - 0.0592106 - 0.01, -0.0156, 0.015]
    
    class control(BaseConfig.control):
        decimal = 6
        controller = "ik"
        # controller = "osc"
        # common_speed = 2.17 * 1
        # common_speed = 1.5 * decimal
    
    class reward(BaseConfig.reward):
        type = "dense"
        contact_coef = -0.1
    
    class safety(BaseConfig.safety):
        brake_on_contact = False
        contact_force_th = 30.0


def goal_in_air_cl(_locals, _globals):
    if _locals["j"] > 1:
        import torch
        ep_infos = _locals["ep_infos"]
        success_rate = torch.mean(torch.tensor(ep_infos["is_success"]).float()).item()
        cur_goal_in_air_ratio = _locals["self"].env.goal_in_air
        if success_rate > 0.6:
            _locals["self"].env.set_goal_in_air_ratio(min(cur_goal_in_air_ratio + 0.1, 1.0))
        print("Goal in air ratio =", _locals["self"].env.goal_in_air)


def contact_force_th_cl(_locals, _globals):
    if _locals["j"] > 1:
        import torch
        ep_infos = _locals["ep_infos"]
        success_rate = torch.mean(torch.tensor(ep_infos["is_success"]).float()).item()
        cur_contact_force_th = _locals["self"].env.cfg.safety.contact_force_th
        if success_rate > 0.6:
            _locals["self"].env.set_contact_force_th(max(0.5 * cur_contact_force_th, 5.0))
        print("Contact force threshold =", _locals["self"].env.cfg.safety.contact_force_th)


config = dict(
    env_id="IsaacPandaPushCNN-v0",
    algo="ppo",
    name="1i1s_oldcam_bcexpert30_prvlg",
    # name="1i1s_oldcam_bc30_statenoise_camurdf_contact-0.1",
    # name="test_joint_decimal6_1024w_step64_dense",
    total_timesteps=int(2e7),
    entry_point=PandaPushEnv,
    env_config=PushConfig(),
    sim_device="cuda:0",
    policy_type=("policies.cnn", "CNNStatePolicy"),
    policy=dict(
        image_shape=(IMAGE_HISTORY_LENGTH, 3, 84, 84), 
        state_dim=15 * STATE_HISTORY_LENGTH, 
        action_dim=4,
        hidden_size=64,
        previ_dim=3,
        state_only_critic=False,
        # num_bin=21,
    ),
    # policy_type=("policies.cnn", "CNNStateHistoryPolicy"),
    # policy=dict(
    #     image_shape=(3, 84, 84),
    #     state_dim=15,
    #     action_dim=4,
    #     hidden_size=64,
    #     lstm_hidden_size=64,
    #     previ_dim=3,
    #     state_only_critic=True,
    # ),
    train=dict(
      n_steps=128,
      nminibatches=16,
      # learning_rate=1e-3,
      learning_rate=1.5e-4,
      # cliprange=0.1,
      n_imitation_epoch=30,
      dagger=False,
      aux_loss_coef=0.0,
      previlege_critic=True,
      use_wandb=False
    ),
    callback=[contact_force_th_cl],
)
