from policies.mvp_hybrid_policy import HybridMlpGaussianPolicy
obj_task_ratio = 1.0
config = dict(
    env_id="BulletDrawer-v1",
    num_workers=64,
    algo="ppo",
    name="debug",
    # log_dir="logs/BulletDrawer-v1/test",
    total_timesteps=int(5e7),
    create_env_kwargs=dict(
        kwargs=dict(reward_type="sparse", view_mode="ego", obj_task_ratio=obj_task_ratio),
    ),
    policy_class=HybridMlpGaussianPolicy,
    policy=dict(
        mvp_feat_dim=768, 
        state_obs_dim=7, 
        n_primitive=3, 
        act_dim=4, 
        # num_bin=64,
        hidden_dim=128, 
        proj_img_dim=128, 
        proj_state_dim=64,
        # use_privilege=True,
    ),
    train=dict(
        use_wandb=False,
        n_steps=1024,
        gamma=0.95,
    ),
    warmup_dataset="../stacking_env/warmup_dataset_ego_obj%.01f.pkl" % obj_task_ratio,
    save_interval=20,
)
