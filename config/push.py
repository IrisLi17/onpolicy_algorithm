config = dict(
    env_id="PandaPushReal-v0",
    num_workers=64,
    algo="ppo",
    name="test",
    total_timesteps=int(1e8),
    create_env_kwargs=dict(
        obs_keys=["observation", "achieved_goal", "desired_goal"],
        done_when_success=True,
        reward_scale=1,
        normalize=False,
        kwargs=dict(reward_type="sparse"),
    ),
    policy_type="mlp",
    policy=dict(
        hidden_size=64,
        # num_bin=21,
    ),
    train=dict(
      use_wandb=False
    ),
)