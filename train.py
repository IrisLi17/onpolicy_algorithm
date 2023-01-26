import importlib

import torch, os
from policies.mvp_hybrid_policy import HybridMlpPolicy
import argparse
from utils import logger


def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", type=str)
    parser.add_argument("--play", action="store_true", default=False)
    parser.add_argument("--load_path", type=str, default=None)
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    # use a configuration file to pass in arguments
    cfg_module = importlib.import_module(args.config)
    config = cfg_module.config
    if args.play:
        config["log_dir"] = None
    else:
        config["log_dir"] = "logs/%s_%s/%s" % (config["algo"], config["env_id"], config["name"])
    logger.configure(config["log_dir"])
    if config["train"].get("use_wandb", False) and not args.play:
        import wandb
        wandb.init(config=config, project=config["algo"] + "_" + config["env_id"], name=config["name"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.play:
        config["num_workers"] = 1
        if config["env_id"] == "BulletDrawerState-v1":
            config["create_env_kwargs"]["kwargs"]["render_goal"] = True
    import sys
    sys.path.append("../stacking_env")
    from bullet_envs.utils.make_vec_env import make_vec_env
    env = make_vec_env(
        config["env_id"], config["num_workers"], device, reset_when_done=not args.play,
        log_dir=config["log_dir"], **config["create_env_kwargs"])
    print(env.observation_space, env.action_space)
    if config["policy"].get("use_privilege", False):
        config["policy"]["privilege_dim"] = env.get_attr("privilege_dim")[0]
        del config["policy"]["use_privilege"]
    sys.path.remove("../stacking_env")
    if config.get("policy_class") is not None:
        policy = config["policy_class"](**config["policy"])
    else:
        policy = HybridMlpPolicy(**config["policy"])
    policy.to(device)
    if (not args.play) and config.get("warmup_dataset") is not None:
        import pickle
        import numpy as np
        with open(config["warmup_dataset"], "rb") as f:
            dataset = pickle.load(f)
        warmup_dataset = dict()
        for k in dataset[0].keys():
            warmup_dataset[k] = np.concatenate([dataset[i][k] for i in range(2)], axis=0)
        config["train"]["warmup_dataset"] = warmup_dataset
    
    if config["algo"] == "ppo":
        from onpolicy import PPO
        model = PPO(env, policy, device, **config.get("train", {}))
    elif config["algo"] == "pair":
        raise NotImplementedError
        from onpolicy import PAIR
        eval_env = make_vec_env(config["env_id"], 20, device, **config["create_env_kwargs"])
        model = PAIR(env, policy, device=device, eval_env=eval_env, **config.get("train", {}))
    else:
        raise NotImplementedError

    def callback(locals, globals):
        if locals["j"] % config["save_interval"] == 0:
            model.save(os.path.join(config["log_dir"], "model_%d.pt" % locals["j"]))

    logger.log(config)
    if not args.play:
        if args.load_path is not None:
            model.load(args.load_path, eval=False)
            print("loaded", args.load_path)
        model.learn(config["total_timesteps"], callback)
    else:
        model.load(args.load_path, eval=True)
        from utils.evaluation import evaluate, evaluate_tasks
        evaluate_tasks(env, policy, "distill_tasks.pkl")
        # "../stacking_env/warmup_tasks.pkl"
        # "logs/ppo_BulletPixelStack-v1/base1/generated_tasks_61.pkl"
        # evaluate(env, policy, 10, task_file="distill_tasks_full.pkl")


if __name__ == "__main__":
    main()
