config = dict(
    env_id="A1-v0",
    num_workers=16,
    algo="ppo",
    log_dir="logs/unitree_a1/test",
    total_timesteps=int(1e7),
    create_env_kwargs=dict(
        obs_keys=["vision", "joints", "inertial"],
        reward_scale=1.0,
        info_keywords=(),
    ),
    policy_type=("policies.unitree_robot_policy", "MultiModalPolicy"),
    policy=dict(
        image_shape=(4, 64, 64),
        state_shape=4 * 18,
        action_dim=12,
    ),
    train=dict(
        nminibatches=16,
        noptepochs=4,
        n_steps=1024,
    ),
    use_wandb=True
)