from config.handover import ObsParser, gap_distance_callback


config = dict(
    env_id="PandaHandoverBimanual-v2",
    num_workers=128,
    algo="ppo",
    name="2b_os0.5_inhand0.0_sparse",
    total_timesteps=int(5e8),
    create_env_kwargs=dict(
        obs_keys=["observation", "achieved_goal", "desired_goal"],
        done_when_success=True,
        reward_offset=0,
        reward_scale=1,
        info_keywords=("is_success", "n_inplace"),
        kwargs=dict(
            os_rate=0.5,
            obj_in_hand_rate=0.0,
            reward_type="sparse",
            initial_gap_distance=0.5,
        ),
    ),
    policy_type="attention_discrete",
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
