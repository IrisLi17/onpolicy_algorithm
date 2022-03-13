config = dict(
    env_id="A1-v1",
    num_workers=16,
    algo="ppo",
    name="mlp",
    total_timesteps=int(1e7),
    create_env_kwargs=dict(
        obs_keys=["HistoricSensorWrapper(IMU)", "HistoricSensorWrapper(LastAction)",
                  "HistoricSensorWrapper(MotorAngle)"],
        reward_scale=1.0,
        scale_action=True,
        info_keywords=(),
        kwargs=dict(),
    ),
    policy_type="mlp",
    policy=dict(
        hidden_size=128,
    ),
    train=dict(
        nminibatches=16,
        noptepochs=4,
        n_steps=1024,
        learning_rate=1e-4,
        use_wandb=True,
    ),
)