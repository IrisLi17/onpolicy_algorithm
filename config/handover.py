config = dict(
    env_id="PandaTowerBimanualPPO-v1",
    num_workers=64,
    log_dir="logs/handover/test",
    total_timesteps=int(1e8),
    create_env_kwargs=dict(
        obs_keys=["observation", "achieved_goal", "desired_goal"],
        done_when_success=False,
        kwargs=dict(os_rate=1),
    ),
    obs_parser=dict(
        robot_dim=14,
        obj_dim=12,
        goal_dim=3,
    ),
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
)