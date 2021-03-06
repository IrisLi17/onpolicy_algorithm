import importlib

import torch, os
from utils.make_env import make_vec_env
from policies.attention_discrete import AttentionDiscretePolicy
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
    env = make_vec_env(config["env_id"], config["num_workers"], device, log_dir=config["log_dir"],
                       training=(not args.play), **config["create_env_kwargs"])
    if config["policy_type"] == "attention_discrete":
        # implement obs_parser
        obs_parser = config["obs_parser"]
        policy = AttentionDiscretePolicy(obs_parser, env.action_space.shape[0], **config["policy"])
    elif config["policy_type"] == "attention_gaussian":
        obs_parser = config["obs_parser"]
        from policies.attention_discrete import AttentionGaussianPolicy
        policy = AttentionGaussianPolicy(obs_parser, env.action_space.shape[0], **config["policy"])
    elif config["policy_type"] == "mlp":
        from policies.mlp import MlpGaussianPolicy
        policy = MlpGaussianPolicy(env.observation_space.shape[0], env.action_space.shape[0], **config["policy"])
    elif isinstance(config["policy_type"], tuple):
        policy_class = importlib.import_module(config["policy_type"][0])
        policy = policy_class.__getattribute__(config["policy_type"][1])(**config["policy"])
    else:
        raise NotImplementedError
    policy.to(device)
    if config["algo"] == "ppo":
        from onpolicy import PPO
        model = PPO(env, policy, device, **config.get("train", {}))
    elif config["algo"] == "pair":
        from onpolicy import PAIR
        eval_env = make_vec_env(config["env_id"], 20, device, **config["create_env_kwargs"])
        model = PAIR(env, policy, device=device, eval_env=eval_env, **config.get("train", {}))
    else:
        raise NotImplementedError

    def callback(locals, globals):
        if locals["j"] % 50 == 0:
            model.save(os.path.join(config["log_dir"], "model_%d.pt" % locals["j"]))

    logger.log(config)
    if not args.play:
        if args.load_path is not None:
            model.load(args.load_path, eval=False)
            print("loaded", args.load_path)
        model.learn(config["total_timesteps"], [callback] + config.get("callback", []))
    else:
        model.load(args.load_path, eval=True)
        from utils.evaluation import evaluate
        evaluate(env, policy, 10)


if __name__ == "__main__":
    main()
