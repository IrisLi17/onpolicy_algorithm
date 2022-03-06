from onpolicy import PPO
import torch, os
from utils.make_env import make_vec_env, ObsParser
from policies.attention_discrete import AttentionDiscretePolicy
import argparse
from utils import logger


def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--env", type=str, choices=["Handover", "fetchreach"])
    parser.add_argument("--play", action="store_true", default=False)
    parser.add_argument("--load_path", type=str)
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    # use a configuration file to pass in arguments
    if args.env == "Handover":
        from config.handover import config
    elif args.env == "fetchreach":
        from config.reach import config
    else:
        raise NotImplementedError
    if args.play:
        config["log_dir"] = None
    logger.configure(config["log_dir"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.play:
        config["num_workers"] = 1
    env = make_vec_env(config["env_id"], config["num_workers"], device, log_dir=config["log_dir"],
                       **config["create_env_kwargs"])
    if config["policy_type"] == "attention_discrete":
        # implement obs_parser
        obs_parser = ObsParser(**config["obs_parser"])
        policy = AttentionDiscretePolicy(obs_parser, env.action_space.shape[0], **config["policy"])
    elif config["policy_type"] == "attention_gaussian":
        obs_parser = ObsParser(**config["obs_parser"])
        from policies.attention_discrete import AttentionGaussianPolicy
        policy = AttentionGaussianPolicy(obs_parser, env.action_space.shape[0], **config["policy"])
    elif config["policy_type"] == "mlp":
        from policies.mlp import MlpGaussianPolicy
        policy = MlpGaussianPolicy(env.observation_space.shape[0], env.action_space.shape[0], **config["policy"])
    else:
        raise NotImplementedError
    policy.to(device)
    model = PPO(env, policy, device)

    def callback(locals, globals):
        if locals["j"] % 50 == 0:
            model.save(os.path.join(config["log_dir"], "model_%d.pt" % locals["j"]))

    logger.log(config)
    if not args.play:
        model.learn(config["total_timesteps"], callback)
    else:
        model.load(args.load_path, eval=True)
        from utils.evaluation import evaluate
        evaluate(env, policy, 10)


if __name__ == "__main__":
    main()
