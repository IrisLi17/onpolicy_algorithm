config = dict(
    env_id="PandaPushPixel-v0",
    num_workers=64,
    algo="ppo",
    name="test",
    total_timesteps=int(1e8),
    create_env_kwargs=dict(
        obs_keys=["image", "proprioception", "desired_goal"],
        done_when_success=True,
        reward_scale=1,
        normalize=False,
        kwargs=dict(reward_type="sparse"),
    ),
    policy_type=("policies.mvp.mvp_policy", "PixelActorCritic"),
    policy=dict(
        image_shape=(3, 224, 224),
        states_shape=(11,),
        actions_shape=(4,),
        initial_std=0.3,
        encoder_cfg=dict(
            model_type="maevit-s16",
            pretrain_dir="policies/mvp/pretrained",
            pretrain_type="hoi",
            freeze=True,
            emb_dim=128),
        policy_cfg=dict(pi_hid_sizes=[256, 128, 64], vf_hid_sizes=[256, 128, 64])
    ),
    train=dict(
        feature_only=True,
        use_wandb=False,
    ),
)