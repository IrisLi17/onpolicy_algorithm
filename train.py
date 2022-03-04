from onpolicy import PPO
import torch, os
from utils.make_env import make_vec_env, ObsParser
from policies.attention_discrete import AttentionDiscretePolicy
import argparse
from utils import logger


def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--env", type=str, choices=["Handover"])
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    # use a configuration file to pass in arguments
    if args.env == "Handover":
        from config.handover import config
    else:
        raise NotImplementedError
    logger.configure(config["log_dir"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = make_vec_env(config["env_id"], config["num_workers"], device, log_dir=config["log_dir"],
                       **config["create_env_kwargs"])
    # implement obs_parser
    obs_parser = ObsParser(**config["obs_parser"])
    policy = AttentionDiscretePolicy(obs_parser, env.action_space.shape[0], **config["policy"])
    policy.to(device)
    model = PPO(env, policy, device)

    def callback(locals, globals):
        if locals["j"] % 50 == 0:
            model.save(os.path.join(config["log_dir"], "model_%d.pt" % locals["j"]))

    logger.log(config)
    model.learn(config["total_timesteps"], callback)


if __name__ == "__main__":
    main()
