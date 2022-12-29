from policies.mvp_hybrid_policy import HybridMlpStateGaussianPolicy
obj_task_ratio = 0.7
config = dict(
    env_id="BulletDrawerState-v1",
    num_workers=64,
    algo="ppo",
    name="debug_h20_sparsecont_obj%.01f_newsuccess_nojoint" % obj_task_ratio,
    # log_dir="logs/BulletDrawer-v1/test",
    total_timesteps=int(5e7),
    create_env_kwargs=dict(
        kwargs=dict(reward_type="sparse", use_gpu_render=False, obj_task_ratio=obj_task_ratio),
    ),
    policy_class=HybridMlpStateGaussianPolicy,
    policy=dict(
        state_obs_dim=15,
        n_primitive=3,
        act_dim=4,
        # num_bin=64,
        hidden_dim=256,
    ),
    train=dict(
        use_wandb=False,
        n_steps=1024,
        gamma=0.95,
    ),
    warmup_dataset="../stacking_env/warmup_dataset_state_obj%.01f.pkl" % obj_task_ratio,
    save_interval=20,
)
