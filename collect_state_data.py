from utils.make_env import make_vec_env
import torch
from policies.mvp_stacking_policy import AttentionDiscretePolicy
import argparse
import matplotlib.pyplot as plt


class StackingObsParser(object):
    def __init__(self, robot_dim, obj_dim, goal_dim, n_max_goal, primitive=False):
        self.robot_dim = robot_dim + 6  # 11+6*n
        self.arm_dim = robot_dim  # 11
        self.obj_dim = obj_dim  # 15+1
        self.goal_dim = goal_dim  # 3+6=9
        self.n_max_goal = n_max_goal
        self.primitive = primitive

    def forward(self, obs: torch.Tensor):
        if self.primitive:
            assert isinstance(obs, torch.Tensor)
            assert len(obs.shape) == 2
            robot_obs = torch.narrow(obs, dim=1, start=0, length=self.arm_dim)
            bsz = obs.shape[0]
            achieved_obs = torch.zeros((bsz, self.n_max_goal * 6))
            goal_obs = torch.zeros((bsz, self.n_max_goal * 6))

            for goal_idx in range(self.n_max_goal):
                achieved_obs[:, goal_idx * 6: (goal_idx + 1) * 6] = torch.narrow(
                    obs, dim=1, start=obs.shape[1] - 2 * self.goal_dim * self.n_max_goal + goal_idx * self.goal_dim,
                    length=6)
                goal_obs[:, goal_idx * 6: (goal_idx + 1) * 6] = torch.narrow(
                    obs, dim=1, start=obs.shape[1] - self.goal_dim * self.n_max_goal + goal_idx * self.goal_dim,
                    length=6)

            robot_obs = torch.cat([robot_obs.cuda(), achieved_obs.cuda(), goal_obs.cuda()], dim=-1)
            objects_obs = torch.narrow(obs, dim=1, start=self.arm_dim,
                                       length=obs.shape[1] - self.arm_dim - 2 * self.goal_dim * self.n_max_goal)
            objects_obs = torch.reshape(objects_obs, (objects_obs.shape[0], -1, self.obj_dim))
            type_embed = torch.zeros((objects_obs.shape[0], objects_obs.shape[1], 3)).cuda()
            type_embed[:, :, 0] = 1
            objects_obs = torch.cat([type_embed, objects_obs], dim=-1)
            masks = torch.norm(objects_obs[:, :, 3:] + 1, dim=-1) < 1e-3
            return robot_obs, objects_obs, masks
        assert isinstance(obs, torch.Tensor)
        assert len(obs.shape) == 2
        robot_obs = torch.narrow(obs, dim=1, start=0, length=self.arm_dim)
        bsz = obs.shape[0]
        achieved_obs = torch.zeros((bsz, self.n_max_goal * 3))
        goal_obs = torch.zeros((bsz, self.n_max_goal * 3))

        for goal_idx in range(self.n_max_goal):
            achieved_obs[:, goal_idx * 3: (goal_idx + 1) * 3] = torch.narrow(
                obs, dim=1, start=obs.shape[1] - 2 * self.goal_dim * self.n_max_goal + goal_idx * self.goal_dim,
                length=3)
            goal_obs[:, goal_idx * 3: (goal_idx + 1) * 3] = torch.narrow(
                obs, dim=1, start=obs.shape[1] - self.goal_dim * self.n_max_goal + goal_idx * self.goal_dim, length=3)

        robot_obs = torch.cat([robot_obs.cuda(), achieved_obs.cuda(), goal_obs.cuda()], dim=-1)
        objects_obs = torch.narrow(obs, dim=1, start=self.arm_dim,
                                   length=obs.shape[1] - self.arm_dim - 2 * self.goal_dim * self.n_max_goal)
        objects_obs = torch.reshape(objects_obs, (objects_obs.shape[0], -1, self.obj_dim))
        masks = torch.norm(objects_obs + 1, dim=-1) < 1e-3
        return robot_obs, objects_obs, masks


def main(args):
    config = dict(
        env_id="BulletStack-v2",
        num_workers=1,
        algo="ppo",
        name="debug",  # pyramid_base_9obj_3goal_roll
        total_timesteps=int(2e8),
        create_env_kwargs=dict(
            obs_keys=["observation", "achieved_goal", "desired_goal"],
            flexible_time_limit=True,
            allow_switch_goal=True,
            done_when_success=True,
            reward_offset=0.0,
            reward_scale=1,
            kwargs=dict(
                n_object=6,
                n_to_stack=[[1], [1], [1]],
                action_dim=7,
                reward_type="sparse",
                name="allow_rotation",
                primitive=True,
                generate_data=True,
                # action_dim=5,
            ),
        ),
        policy_type="attention_discrete",
        obs_parser=StackingObsParser(robot_dim=11, obj_dim=16, goal_dim=6+6, n_max_goal=3, primitive=True),
        policy=dict(
            hidden_size=64,
            num_bin=21,
            feature_extractor="self_attention",
            shared=False,
            n_critic_layers=1,
            n_actor_layers=1,
            n_object=6,
            tokenize=True,
            kwargs=dict(
                n_attention_blocks=2,
                n_heads=1,
            ),
        ),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = make_vec_env(config["env_id"], config["num_workers"], device, log_dir=None,
                       training=False, **config["create_env_kwargs"])
    obs_parser = config["obs_parser"]
    policy = AttentionDiscretePolicy(obs_parser, env.action_space.shape[0], **config["policy"])
    policy.to(device)
    checkpoint = torch.load(args.load_path, map_location=device)
    policy.load_state_dict(checkpoint['policy'], strict=False)
    env.set_attr("use_expand_goal_prob", 1.0)

    episode_count = 0
    obs = env.reset()
    while episode_count < 10:
        with torch.no_grad():
            _, action, _, _ = policy.act(obs)
        img = env.render(mode="rgb_array")
        
        obs, reward, done, info = env.step(action)
        if done[0]:
            episode_count += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--load_path", type=str, default=None)
    args = parser.parse_args()
    main(args)