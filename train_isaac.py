import importlib

import os
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
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    headless = True
    if args.play:
        config["env_config"].env.num_envs = 1
        # headless = False
    env = config["entry_point"](config["env_config"], headless=headless)
    if config["policy_type"] == "mlp":
        from policies.mlp import MlpGaussianPolicy
        policy = MlpGaussianPolicy(env.num_obs, env.num_actions, **config["policy"])
    elif config["policy_type"] == "mlp_discrete":
        from policies.mlp import MlpPolicy
        policy = MlpPolicy(env.num_obs, env.num_actions, **config["policy"])
    elif isinstance(config["policy_type"], tuple):
        policy_class = importlib.import_module(config["policy_type"][0])
        policy = policy_class.__getattribute__(config["policy_type"][1])(**config["policy"])
    else:
        raise NotImplementedError
    policy.to(device)
    for name, param in policy.named_parameters():
        print(name, param.shape)
    if config["algo"] == "ppo":
        from onpolicy.ppo.ppo_isaac import PPO
        model = PPO(env, policy, device, **config.get("train", {}))
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
        evaluate(env, policy, 10)


def evaluate(env, policy, n_episode):
    import torch
    from PIL import Image
    import numpy as np
    import shutil, os
    episode_count = 0
    step_count = 0
    shutil.rmtree("tmp")
    os.makedirs("tmp", exist_ok=True)
    obs = env.reset()
    while episode_count < n_episode:
        env.render()
        image = env.get_camera_image()
        image = Image.fromarray(image.astype(np.uint8))
        filename = "tmp/tmp%d.png" % step_count
        image.save(filename)
        with torch.no_grad():
            _, actions, _, _ = policy.act(obs, deterministic=False)
        step_count += 1
        obs, reward, done, info = env.step(actions)
        if done[0]:
            print(obs[0])
            episode_count += 1
        
if __name__ == "__main__":
    main()
