import pybullet as p

config = dict(
    env_id="A1-v1",
    num_workers=16,
    algo="ppo",
    name="mm_rewardtest_normobsret",
    total_timesteps=int(1e7),
    create_env_kwargs=dict(
        obs_keys=["HistoricSensorWrapper(DepthFrontSensor)", "HistoricSensorWrapper(IMU)",
                  "HistoricSensorWrapper(LastAction)", "HistoricSensorWrapper(MotorAngle)"],
        reward_scale=1.0,
        scale_action=True,
        normalize=["obs", "ret"],
        info_keywords=("total_distance", "total_drift", "total_shake", "total_energy"),
        kwargs=dict(task="simple_forward",
                    enable_randomizer=True,
                    enable_terrain_random=False,
                    task_kwargs=dict(energy_weight=0.005, drift_weight=0.0, shake_weight=0.1,
                                     survive_weight=0.1, fall_reward=-20.0)),
    ),
    policy_type=("policies.unitree_robot_policy", "MultiModalPolicy"),
    policy=dict(
        image_shape=(4, 64, 64),
        state_shape=3 * 28,
        action_dim=12,
    ),
    train=dict(
        nminibatches=16,
        noptepochs=10,
        n_steps=1024,
        learning_rate=1e-4,
        use_wandb=True,
    ),
)
