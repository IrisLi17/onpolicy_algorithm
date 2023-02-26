from policies.slot_attention_policy import SlotAttentionPoicy
import numpy as np

config = dict(
    env_id="BulletPixelStack-v1",
    num_workers=64,
    algo="ppo",
    # name="distill_expand5_alltask_aux_sdim64",
    name="slot_attn_rl_newdata_oldenc_round123_xy41",
    total_timesteps=int(5e7),
    create_env_kwargs=dict(
        use_raw_img=True,
        kwargs=dict(
            n_object=6, reward_type="sparse", action_dim=7, generate_data=True, primitive=True,
            n_to_stack=np.array([[1, 2, 3]]), name="allow_rotation", use_gpu_render=False,
            feature_dim=3 * 128 * 128,
        ),
    ),
    policy_class=SlotAttentionPoicy,
    policy=dict(
        encoder_path="logs/learn_oc_7slot/model_265000.pt",
        num_slots=7,
        act_dim=6, 
        num_bin=[41, 41, 21, 21, 21, 21],
        use_privilege=True,
    ),
    train=dict(
        store_raw_img=2*3*128*128,
        use_wandb=False,
        n_steps=256,
        # nminibatches=16,
        gamma=0.99,
        noptepochs=6,
        ent_coef=0.001,
        cliprange=0.3,
    ),
    # warmup_dataset="../stacking_env/warmup_dataset_stacking.pkl",
    # warmup_dataset="distill_dataset_stacking.pkl",
    save_interval=10,
)
