from train_isaac import parse_arguments
import importlib


args = parse_arguments()
# use a configuration file to pass in arguments
cfg_module = importlib.import_module(args.config)
config = cfg_module.config
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
observation_shape = 20
action_shape = 4
if config["policy_type"] == "mlp":
    from policies.mlp import MlpGaussianPolicy
    policy = MlpGaussianPolicy(observation_shape, action_shape, **config["policy"])
elif isinstance(config["policy_type"], tuple):
    policy_class = importlib.import_module(config["policy_type"][0])
    policy = policy_class.__getattribute__(config["policy_type"][1])(**config["policy"])
checkpoint = torch.load(args.load_path, map_location=device)
policy.load_state_dict(checkpoint["policy"], strict=False)
scripted_policy = torch.jit.script(policy)
scripted_policy.save("scripted.pt")
