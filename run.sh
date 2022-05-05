# Collect BC demo
python train_isaac.py --config config.isaac.panda_push_cartesian_pixel_config --collect_demo --load_path pretrained_state_model.pt
# BC then RL
python train_isaac.py --config config.isaac.panda_push_cartesian_pixel_config --imitation_pretrain
# DAGGER
python train_isaac.py --config config.isaac.panda_push_cartesian_pixel_config --expert_policy_path pretrained_state_model.pt

