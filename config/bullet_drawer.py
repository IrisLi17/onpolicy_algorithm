from policies.mvp_hybrid_policy import HybridMlpPolicy
obj_task_ratio = 0.7
view_mode = "third"
noisy_img = False
config = dict(
    env_id="BulletDrawer-v1",
    num_workers=64,
    algo="ppo",
    name="debug_discrete_%s_ile15b128%s_svalue" % (view_mode, "_noisy" if noisy_img else ""),
    # log_dir="logs/BulletDrawer-v1/test",
    total_timesteps=int(5e7),
    create_env_kwargs=dict(
        kwargs=dict(reward_type="sparse", view_mode=view_mode, obj_task_ratio=obj_task_ratio,
                    shift_params=(-5, 5) if noisy_img else (0, 0)),
    ),
    policy_class=HybridMlpPolicy,
    policy=dict(
        mvp_feat_dim=768, 
        state_obs_dim=7, 
        n_primitive=3, 
        act_dim=4, 
        num_bin=64,
        hidden_dim=128, 
        proj_img_dim=128, 
        proj_state_dim=64,
        use_privilege=True,
        state_only_value=True
    ),
    train=dict(
        use_wandb=False,
        n_steps=1024,
        gamma=0.95,
    ),
    warmup_dataset="../stacking_env/warmup_dataset_%s_obj%.01f%s.pkl" % (
        view_mode, obj_task_ratio, "_noisy" if noisy_img else ""),
    save_interval=20,
)
