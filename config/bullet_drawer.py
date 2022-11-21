config = dict(
    env_id="BulletDrawer-v1",
    num_workers=16,
    algo="ppo",
    name="test",
    log_dir="logs/BulletDrawer-v1/test",
    total_timesteps=int(1e6),
    create_env_kwargs=dict(
        ),
    policy=dict(
        mvp_feat_dim=768, 
        state_obs_dim=18, 
        n_primitive=4, 
        act_dim=4, 
        num_bin=32,
        hidden_dim=128, 
        proj_img_dim=128, 
        proj_state_dim=64,
    ),
    train=dict(
        use_wandb=False,
        n_steps=256,
    )
)