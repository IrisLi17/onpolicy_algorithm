import argparse
import torch
import importlib


# Usage: to replay trajectories in the dataset
# python play_robot_trajectory.py --config config.bullet_pixel_stack_slot --traj_path distill_dataset_new_stacking_raw_expand3.pkl
# to predict object-level actions with a policy
# python play_robot_trajectory.py --config config.bullet_pixel_stack_slot --use_rl \
#     --load_path logs/ppo_BulletPixelStack-v1/slot_attn_rl_newdata_newenc_round123_xy41/model_0.pt \
#     --task_path test_tasks_raw.pkl

def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", type=str)
    parser.add_argument("--use_rl", action="store_true", default=False)
    parser.add_argument("--traj_path", type=str, default=None)
    parser.add_argument("--load_path", type=str, default=None)
    parser.add_argument("--task_path", type=str, default=None)
    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()
    # use a configuration file to pass in arguments
    cfg_module = importlib.import_module(args.config)
    config = cfg_module.config
    config["log_dir"] = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config["num_workers"] = 1
    import sys
    sys.path.append("../stacking_env")
    from bullet_envs.utils.make_vec_env import make_vec_env
    env = make_vec_env(
        config["env_id"], config["num_workers"], device, reset_when_done=False,
        log_dir=config["log_dir"], **config["create_env_kwargs"])
    print(env.observation_space, env.action_space)
    if config["policy"].get("use_privilege", False):
        config["policy"]["privilege_dim"] = env.get_attr("privilege_dim")[0]
        del config["policy"]["use_privilege"]
    sys.path.remove("../stacking_env")
    
    if not args.use_rl:
        assert args.traj_path is not None
        from utils.evaluation import trajectory_replay
        trajectory_replay(env, args.traj_path)

    else:
        assert args.load_path is not None
        assert args.task_path is not None
        policy = config["policy_class"](**config["policy"])
        policy.to(device)
        if config["algo"] == "ppo":
            from onpolicy import PPO
            model = PPO(env, policy, device, **config.get("train", {}))
        else:
            raise NotImplementedError
        model.load(args.load_path, eval=True)
        from utils.evaluation import evaluate_tasks
        evaluate_tasks(env, policy, args.task_path, 100, deterministic=False)


if __name__ == "__main__":
    main()
