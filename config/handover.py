import numpy as np
import torch


class ObsParser(object):
    def __init__(self, robot_dim, obj_dim, goal_dim):
        self.robot_dim = robot_dim
        self.obj_nogoal_dim = obj_dim  # per object
        self.goal_dim = goal_dim  # per object
        self.obj_dim = self.obj_nogoal_dim + self.goal_dim

    def forward(self, obs: torch.Tensor):
        assert len(obs.shape) == 2
        robot_obs = torch.narrow(obs, dim=1, start=0, length=self.robot_dim)
        assert (obs.shape[-1] - self.robot_dim) % (self.obj_nogoal_dim + 2 * self.goal_dim) == 0
        num_object = int((obs.shape[-1] - self.robot_dim) / (self.obj_nogoal_dim + 2 * self.goal_dim))
        objects_obs = torch.narrow(obs, dim=1, start=self.robot_dim,
                                   length=self.obj_nogoal_dim * num_object)
        objects_obs = torch.reshape(objects_obs, (obs.shape[0], num_object, self.obj_nogoal_dim))
        desired_goals = torch.narrow(obs, dim=1, start=obs.shape[-1]-self.goal_dim * num_object,
                                     length=self.goal_dim * num_object)
        desired_goals = torch.reshape(desired_goals, (obs.shape[0], num_object, self.goal_dim))
        objects_obs = torch.cat([objects_obs, desired_goals], dim=-1)
        masks = torch.norm(objects_obs, dim=-1) < 1e-3
        return robot_obs, objects_obs, masks


def gap_distance_callback(_locals, _globals):
    if _locals["j"] > 0:
        ep_infos = _locals["ep_infos"]
        success_rate = np.mean([ep_info["is_success"] for ep_info in ep_infos])
        print("Success rate =", success_rate)
        cur_gap_distance = _locals["self"].env.get_attr("current_gap_distance")[0]
        if success_rate > 0.7:
            _locals["self"].env.env_method("change_gap_distance", cur_gap_distance + 0.1)
        print("Gap distance =", _locals["self"].env.get_attr("current_gap_distance")[0])


config = dict(
    env_id="PandaHandoverBimanual-v1",
    num_workers=128,
    algo="ppo",
    name="1b_os0.5_inhand0.5_sparse_gapcl_wopp_ho100_done",
    total_timesteps=int(5e8),
    create_env_kwargs=dict(
        obs_keys=["observation", "achieved_goal", "desired_goal"],
        done_when_success=True,
        reward_offset=0,
        reward_scale=1,
        kwargs=dict(
            os_rate=0.5,
            obj_in_hand_rate=0.5,
            reward_type="sparse",
        ),
    ),
    policy_type="attention_discrete",
    # obs_parser=dict(
    #     robot_dim=14,
    #     obj_dim=18,
    #     goal_dim=3,
    # ),
    obs_parser=ObsParser(robot_dim=14, obj_dim=18, goal_dim=3),
    policy=dict(
        hidden_size=64,
        num_bin=21,
        feature_extractor="self_attention",
        shared=False,
        n_critic_layers=1,
        n_actor_layers=1,
        kwargs=dict(
            n_attention_blocks=3,
            n_heads=1,
        ),
    ),
    train=dict(
        n_steps=2048,
        use_wandb=True,
    ),
    callback=[gap_distance_callback],
)