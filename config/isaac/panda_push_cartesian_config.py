import sys
sys.path.append("../isaac_projects/panda-isaac")
from panda_isaac.panda_push import PandaPushEnv
from panda_isaac.base_config import BaseConfig
sys.path.remove("../isaac_projects/panda-isaac")


STATE_HISTORY_LENGTH = 1


class PushConfig(BaseConfig):
    class env(BaseConfig.env):
        seed = 42
        num_envs = 1024
        num_observations = (3 + 15) * STATE_HISTORY_LENGTH
        num_actions = 4
        num_state_obs = 18 * STATE_HISTORY_LENGTH
        max_episode_length = 100
        init_goal_in_air = 0.0
    
    class obs(BaseConfig.obs):
        type = "state"
        state_history_length = STATE_HISTORY_LENGTH
    
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
    
    class domain_randomization(BaseConfig.domain_randomization):
        friction_range = [1.0, 1.5]


def goal_in_air_cl(_locals, _globals):
    if _locals["j"] > 1:
        import torch
        ep_infos = _locals["ep_infos"]
        success_rate = torch.mean(torch.tensor(ep_infos["is_success"]).float()).item()
        cur_goal_in_air_ratio = _locals["self"].env.goal_in_air
        if success_rate > 0.7:
            _locals["self"].env.set_goal_in_air_ratio(min(cur_goal_in_air_ratio + 0.1, 1.0))
        print("Goal in air ratio =", _locals["self"].env.goal_in_air)


def contact_force_th_cl(_locals, _globals):
    if _locals["j"] > 1:
        import torch
        ep_infos = _locals["ep_infos"]
        success_rate = torch.mean(torch.tensor(ep_infos["is_success"]).float()).item()
        cur_contact_force_th = _locals["self"].env.cfg.safety.contact_force_th
        if success_rate > 0.6:
            _locals["self"].env.set_contact_force_th(max(0.5 * cur_contact_force_th, 2.0))
        print("Contact force threshold =", _locals["self"].env.cfg.safety.contact_force_th)


config = dict(
    env_id="IsaacPandaPushState-v0",
    algo="ppo",
    name="push_seppolicy_randsym_noxpen_nogripperent_fric1-1.5",
    # name="test_joint_decimal6_1024w_step64_dense",
    total_timesteps=int(3e8),
    entry_point=PandaPushEnv,
    env_config=PushConfig(),
    # policy_type="mlp",
    policy_type=("policies.mlp", "PandaSepPolicy"),
    policy=dict(
        obs_dim=18,
        hidden_size=128,
        n_layers=3,
        # num_bin=21,
    ),
    train=dict(
      # n_steps=1024,
      n_steps=128,
      nminibatches=32,
      # learning_rate=1e-3,
      learning_rate=2.5e-4,
      # cliprange=0.1,
      use_wandb=False
    ),
    callback=[contact_force_th_cl],
)
