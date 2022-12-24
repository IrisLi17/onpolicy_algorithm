from policies.mvp_hybrid_policy import HybridMlpStatePolicy
config = dict(
    env_id="BulletDrawerState-v1",
    num_workers=64,
    algo="ppo",
    name="debug",
    # log_dir="logs/BulletDrawer-v1/test",
    total_timesteps=int(5e7),
    create_env_kwargs=dict(
        kwargs=dict(reward_type="dense"),
    ),
    policy_class=HybridMlpStatePolicy,
    policy=dict(
        state_obs_dim=22, 
        n_primitive=3, 
        act_dim=4, 
        num_bin=64,
        hidden_dim=128, 
    ),
    train=dict(
        use_wandb=True,
        n_steps=1024,
    ),
    save_interval=20,
)
