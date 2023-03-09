from policies.vit_policy.policies.policy_vit import ViTPoicy
import numpy as np


# expand_rounds = [1, 2, 3]
expand_rounds = [1]
config = dict(
    env_id="BulletPixelStack-v1",
    num_workers=64,
    algo="ppo",
    # name="distill_expand5_alltask_aux_sdim64",
    name="segvit_enc8_cls_attn_rl_round1",
    total_timesteps=int(1e6),
    create_env_kwargs=dict(
        use_raw_img=True,
        kwargs=dict(
            n_object=6, reward_type="sparse", action_dim=7, generate_data=True, primitive=True,
            n_to_stack=np.array([[1, 2, 3]]), name="allow_rotation", use_gpu_render=False,
            feature_dim=3 * 128 * 128,
        ),
    ),
    policy_class=ViTPoicy,
    policy=dict(
        resolution=(128, 128), 
        encoder_path="logs/segmenter/seg_tiny_mask_checkpoint.pth", 
        act_dim=6, 
        num_bin=[41, 41, 21, 21, 21, 21],
        type="mean",
        layer_id=8,
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
        task_files=["distill_tasks_new_raw_expand%d.pkl" % i for i in expand_rounds],
        demo_files=["distill_dataset_new_stacking_raw_expand%d.pkl" % i for i in expand_rounds],
    ),
    # warmup_dataset="../stacking_env/warmup_dataset_stacking.pkl",
    # warmup_dataset="distill_dataset_stacking.pkl",
    save_interval=10,
)
