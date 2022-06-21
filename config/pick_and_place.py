import torch


class StackingObsParser(object):
    def __init__(self, robot_dim, obj_dim, goal_dim):
        self.robot_dim = robot_dim + 6
        self.arm_dim = robot_dim
        self.obj_dim = obj_dim
        self.goal_dim = goal_dim

    def forward(self, obs: torch.Tensor):
        assert isinstance(obs, torch.Tensor)
        assert len(obs.shape) == 2
        # robot_dim = env.get_attr("robot_dim")[0]
        # object_dim = env.get_attr("object_dim")[0]
        # goal_dim = env.get_attr("goal")[0].shape[0]
        robot_obs = torch.narrow(obs, dim=1, start=0, length=self.arm_dim)
        achieved_obs = torch.narrow(obs, dim=1, start=obs.shape[1] - 2 * self.goal_dim, length=3)
        goal_obs = torch.narrow(obs, dim=1, start=obs.shape[1] - self.goal_dim, length=3)
        robot_obs = torch.cat([robot_obs, achieved_obs, goal_obs], dim=-1)
        objects_obs = torch.narrow(obs, dim=1, start=self.arm_dim,
                                   length=obs.shape[1] - self.arm_dim - 2 * self.goal_dim)
        objects_obs = torch.reshape(objects_obs, (objects_obs.shape[0], -1, self.obj_dim))
        masks = torch.norm(objects_obs + 1, dim=-1) < 1e-3
        # print("robot obs", robot_obs, "objects obs", objects_obs, "masks", masks)
        # exit()
        return robot_obs, objects_obs, masks


config = dict(
    env_id="BulletPickAndPlace-v1",
    num_workers=64,
    algo="ppo",
    name="test",
    total_timesteps=int(2e8),
    create_env_kwargs=dict(
        obs_keys=["observation", "achieved_goal", "desired_goal"],
        done_when_success=True,
        reward_offset=0.0,
        reward_scale=1,
        kwargs=dict(
            n_object=6,
            reward_type="sparse",
        ),
    ),
    policy_type="attention_discrete",
    obs_parser=StackingObsParser(robot_dim=11, obj_dim=16, goal_dim=3+6),
    policy=dict(
        hidden_size=64,
        num_bin=21,
        feature_extractor="self_attention",
        shared=False,
        n_critic_layers=1,
        n_actor_layers=1,
        kwargs=dict(
            n_attention_blocks=2,
            n_heads=1,
        ),
    ),
    train=dict(
        n_steps=4096,
    )
)
