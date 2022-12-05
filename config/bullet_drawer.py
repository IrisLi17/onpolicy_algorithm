config = dict(
    env_id="BulletDrawer-v1",
    num_workers=64,
    algo="ppo",
    name="third_dense_fixdrawer_parammask",
    # log_dir="logs/BulletDrawer-v1/test",
    total_timesteps=int(5e7),
    create_env_kwargs=dict(
        kwargs=dict(reward_type="dense", view_mode="third"),
    ),
    policy=dict(
        mvp_feat_dim=768, 
        state_obs_dim=14, 
        n_primitive=3, 
        act_dim=4, 
        num_bin=64,
        hidden_dim=128, 
        proj_img_dim=128, 
        proj_state_dim=64,
        use_param_mask=True,
        use_privilege=False,
    ),
    train=dict(
        use_wandb=True,
        n_steps=1024,
    ),
    save_interval=20,
)
