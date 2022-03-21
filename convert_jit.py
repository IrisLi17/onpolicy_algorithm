from policies.mlp import MlpGaussianPolicy
from train import parse_arguments
import importlib
import torch


args = parse_arguments()
# use a configuration file to pass in arguments
cfg_module = importlib.import_module(args.config)
config = cfg_module.config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
observation_shape = 20
action_shape = 4
policy = MlpGaussianPolicy(observation_shape, action_shape, **config["policy"])
checkpoint = torch.load(args.load_path, map_location=device)
policy.load_state_dict(checkpoint["policy"])
scripted_policy = torch.jit.script(policy)
scripted_policy.save("push_scripted.pt")
