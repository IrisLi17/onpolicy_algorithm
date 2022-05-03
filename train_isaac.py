import importlib

import os
import argparse
from utils import logger


def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", type=str)
    parser.add_argument("--play", action="store_true", default=False)
    parser.add_argument("--load_path", type=str, default=None)
    parser.add_argument("--collect_demo", action="store_true", default=False)
    parser.add_argument("--imitation_pretrain", action="store_true", default=False)
    parser.add_argument("--expert_policy_path", type=str, default=None)
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    # use a configuration file to pass in arguments
    cfg_module = importlib.import_module(args.config)
    config = cfg_module.config
    if args.play or args.collect_demo:
        config["log_dir"] = None
    else:
        config["log_dir"] = "logs/%s_%s/%s" % (config["algo"], config["env_id"], config["name"])
    logger.configure(config["log_dir"])
    if config["train"].get("use_wandb", False) and (not args.play and not args.collect_demo):
        import wandb
        wandb.init(config=config, project=config["algo"] + "_" + config["env_id"], name=config["name"])
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    headless = True
    if args.play:
        # pass
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
        if config.get("train", {}).get("dagger", False):
            from policies.mlp import MlpGaussianPolicy
            state_policy = MlpGaussianPolicy(env.num_state_obs, env.num_actions, hidden_size=64)
            state_policy.to(env.device)
            checkpoint = torch.load(args.expert_policy_path, map_location=env.device)
            state_policy.load_state_dict(checkpoint['policy'], strict=False)
            model.set_expert_policy(state_policy)
    else:
        raise NotImplementedError

    def callback(locals, globals):
        if locals["j"] % 50 == 0:
            model.save(os.path.join(config["log_dir"], "model_%d.pt" % locals["j"]))

    logger.log(config)
    if args.collect_demo:
        collect_imitation_demo(env, args.load_path, policy, 1000)
    else:
        if not args.play:
            if args.load_path is not None:
                model.load(args.load_path, eval=False)
                print("loaded", args.load_path)
            if args.imitation_pretrain:
                import pickle
                import numpy as np
                obs_buffer, actions_buffer = [], []
                with open("imitation_data.pkl", "rb") as f:
                    try:
                        while True:
                            traj = pickle.load(f)
                            obs_buffer.append(traj["image_obs"])
                            actions_buffer.append(traj["action"])
                    except EOFError:
                        pass
                obs_buffer = np.concatenate(obs_buffer, axis=0)
                actions_buffer = np.concatenate(actions_buffer, axis=0)
                model.pretrain(obs_buffer, actions_buffer, is_feature_input=True)
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
    episode_length = 0
    if os.path.exists("tmp"):
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
            if env.cfg.obs.type == "pixel":
                obs_image = obs[0, :3 * 224 * 224].reshape((3, 224, 224))
                obs_image = (obs_image * env.im_std + env.im_mean).permute(1, 2, 0) * 255
                obs_image = Image.fromarray(obs_image.cpu().numpy().astype(np.uint8))
                filename = "tmp/tmpobs%d.png" % step_count
                obs_image.save(filename)
                obs = policy.encode_obs(obs)
            _, actions, _, _ = policy.act(obs, deterministic=False)
        step_count += 1
        episode_length += 1
        obs, reward, done, info = env.step(actions)
        if done[0]:
            print(obs[0])
            print("episode length", episode_length, info)
            episode_count += 1
            episode_length = 0

def collect_imitation_demo(env, load_path, image_policy, n_episode):
    import torch
    import numpy as np
    import pickle
    from policies.mlp import MlpGaussianPolicy
    state_policy = MlpGaussianPolicy(18, env.num_actions, hidden_size=64)
    state_policy.to(env.device)
    checkpoint = torch.load(load_path, map_location=env.device)
    state_policy.load_state_dict(checkpoint['policy'], strict=False)
    episode_count = 0
    if os.path.exists("imitation_data.pkl"):
        os.remove("imitation_data.pkl")
    obs = env.reset()
    demo = [dict(image_obs=[], state_obs=[], action=[], reward=[]) for _ in range(env.num_envs)]
    while episode_count < n_episode:
        state_obs = env.get_state_obs()
        with torch.no_grad():
            image_features = image_policy.encode_obs(obs)
            _, actions, _, _ = state_policy.act(state_obs, deterministic=False)
        for i in range(env.num_envs):
            demo[i]["image_obs"].append(image_features[i].detach().cpu().numpy())
            demo[i]["state_obs"].append(state_obs[i].detach().cpu().numpy())
            demo[i]["action"].append(actions[i].cpu().numpy())
        obs, reward, done, info = env.step(actions)
        for i in range(env.num_envs):
            demo[i]["reward"].append(reward[i].cpu().numpy())
            if done[i]:
                if demo[i]["reward"][-1] >= 1:
                    demo[i]["image_obs"] = np.stack(demo[i]["image_obs"], axis=0)
                    demo[i]["state_obs"] = np.stack(demo[i]["state_obs"], axis=0)
                    demo[i]["action"] = np.stack(demo[i]["action"], axis=0)
                    demo[i]["reward"] = np.stack(demo[i]["reward"], axis=0)
                    episode_count += 1
                    # episode_reward.append(np.sum(demo[-1]["reward"]))
                    print("episode count", episode_count, "episode reward", np.sum(demo[i]["reward"]))
                    with open("imitation_data.pkl", "ab") as f:
                        pickle.dump(demo[i], f)
                demo[i] = dict(image_obs=[], state_obs=[], action=[], reward=[])
                if episode_count >= n_episode:
                    break

if __name__ == "__main__":
    main()
