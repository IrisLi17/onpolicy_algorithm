from utils.make_env import make_vec_env
import torch
from policies.mvp_stacking_policy import AttentionDiscretePolicy
import argparse
import matplotlib.pyplot as plt
import os, shutil
import numpy as np
import pybullet as p
import pickle
import sys
sys.path.append("../stacking_env")
from bullet_envs.env.pixel_stacking import get_n_to_move
sys.path.remove("../stacking_env")


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

    def bridge(self, obs: np.ndarray):
        robot_obs = obs[..., :self.arm_dim]
        scaled_finger = robot_obs[..., -2:-1] * 2 / 0.08
        eef_pos = robot_obs[..., :3]
        eef_euler = robot_obs[..., 3: 6]
        new_robot_obs = np.concatenate([scaled_finger, eef_pos, eef_euler], axis=-1)
        objects_obs = obs[..., self.arm_dim: obs.shape[1] - 2 * self.goal_dim * self.n_max_goal]
        objects_obs = np.reshape(objects_obs, (objects_obs.shape[0], -1, self.obj_dim))
        objects_pos = objects_obs[..., :3]
        objects_euler = objects_obs[..., 6: 9]
        flatten_objects_euler = objects_euler.reshape((-1, 3))
        flatten_objects_quat = np.array([
            self.euler2quat(flatten_objects_euler[i]) for i in range(flatten_objects_euler.shape[0])
        ])
        objects_quat = flatten_objects_quat.reshape((objects_euler.shape[0], -1, 4))
        new_objects_obs = np.concatenate([objects_pos, objects_quat], axis=-1).reshape((objects_obs.shape[0], -1))
        return new_robot_obs, new_objects_obs
    
    def euler2quat(self, euler):
        q = np.array(p.getQuaternionFromEuler(euler))
        if q[-1] < 0:
            q = -q
        return q


class ImageProcessor(object):
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        import mvp
        import torchvision.transforms
        self.mvp_model = mvp.load("vitb-mae-egosoup")
        self.mvp_model.to(self.device)
        self.mvp_model.freeze()
        # Norm should still be trainable, so I will not forward norm in this wrapper, and add a ln to policy
        self.image_transform = torchvision.transforms.Resize(224)
        self.im_mean = torch.Tensor([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1)).to(self.device)
        self.im_std = torch.Tensor([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1)).to(self.device)
    
    def _normalize_obs(self, obs):
        img = self.image_transform(torch.from_numpy(obs).float().to(self.device))
        normed_img = (img / 255.0 - self.im_mean) / self.im_std
        return normed_img
    
    def process_obs(self, obs):
        normed_img = self._normalize_obs(obs)
        with torch.no_grad():
            scene_feat = self.mvp_model.extract_feat(normed_img.float())
        return scene_feat


def main(args):
    config = dict(
        env_id="BulletStack-v2",
        num_workers=64,
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
    policy.load_state_dict(checkpoint['policy'])

    image_processor = ImageProcessor()
    # if os.path.exists("tmp"):
    #     shutil.rmtree("tmp")
    # os.makedirs("tmp")
    # create_generalize_task(env, image_processor, 1000)
    collect_data(env, policy, image_processor, 30_000)
    # create test tasks

    return
    if os.path.exists("tmp"):
        shutil.rmtree("tmp")
    os.makedirs("tmp")
    episode_count = 0
    step_count = 0
    obs = env.reset()
    while episode_count < 10:
        with torch.no_grad():
            _, action, _, _ = policy.act(obs)
        img = env.get_images()[0]
        plt.imsave("tmp/tmp%d.png" % step_count, img.transpose((1, 2, 0)))
        step_count += 1
        obs, reward, done, info = env.step(action)
        if done[0]:
            print(info[0]["is_success"])
            episode_count += 1


def collect_data(env, policy, image_processor, desired_num):
    device = image_processor.device
    n_object = 6
    im_dataset = [dict(obs=[], action=[], boundary=[]) for _ in range(n_object)]
    task_arrays = [[] for _ in range(n_object)]
    traj_buffer = [dict(img=[], state=[], action=[]) for _ in range(env.num_envs)]
    obs = env.reset()
    imgs = env.get_images()
    for i in range(env.num_envs):
        traj_buffer[i]["state"].append(obs[i].detach().cpu().numpy())
        traj_buffer[i]["img"].append(imgs[i].astype(np.uint8))
    n_data = [0 for _ in range(n_object)]
    while sum(n_data) < desired_num:
        with torch.no_grad():
            _, action, _, _ = policy.act(obs)
        obs, reward, done, info = env.step(action)
        for i in range(env.num_envs):
            traj_buffer[i]["action"].append(action[i].detach().cpu().numpy())
            traj_buffer[i]["state"].append(obs[i].detach().cpu().numpy())
            traj_buffer[i]["img"].append(info[i]["img"])
            if done[i]:
                if info[i]["is_success"] and len(traj_buffer[i]["action"]) < 10:
                    traj_buffer[i]["state"][-1] = info[i]["terminal_observation"]
                    # Bridge to pixel env observation, relabel
                    # TODO
                    # for j in range(len(traj_buffer[i]["img"])):
                    #     plt.imsave("tmp/tmp%d.png" % j, traj_buffer[i]["img"][j].transpose((1, 2, 0)))
                    # exit()
                    traj_states = np.stack(traj_buffer[i]["state"], axis=0)
                    traj_images = np.stack(traj_buffer[i]["img"], axis=0)
                    traj_features = image_processor.process_obs(traj_images).detach().cpu().numpy()
                    robot_obs, objects_obs = policy.obs_parser.bridge(traj_states)
                    for goal_idx in range(len(traj_buffer[i]["action"]), len(traj_buffer[i]["action"]) + 1):
                        goal_feature = traj_features[goal_idx:goal_idx + 1]
                        goal_state = objects_obs[goal_idx:goal_idx + 1]
                        new_obs = np.concatenate([
                            traj_features[:goal_idx], robot_obs[:goal_idx], 
                            np.tile(goal_feature, (goal_idx, 1)), 
                            objects_obs[:goal_idx], 
                            np.tile(goal_state, (goal_idx, 1))
                        ], axis=-1).astype(np.float32)
                        n_to_move = get_n_to_move(new_obs[0:1], n_object, 0.05, 0.75)[0]
                        im_dataset[n_to_move - 1]["obs"].append(new_obs)
                        im_dataset[n_to_move - 1]["action"].append(np.stack(traj_buffer[i]["action"][:goal_idx], axis=0).astype(np.float32))
                        im_dataset[n_to_move - 1]["boundary"].append(n_data[n_to_move - 1])
                        assert im_dataset[n_to_move - 1]["obs"][-1].shape[0] == im_dataset[n_to_move - 1]["action"][-1].shape[0]
                        n_data[n_to_move - 1] += new_obs.shape[0]
                        # print(n_data)
                        # get task array
                        task_array = np.concatenate([
                            robot_obs[0], objects_obs[0], objects_obs[goal_idx], traj_features[goal_idx]
                        ])
                        task_arrays[n_to_move - 1].append(task_array)
                    print(sum(n_data))
                reset_img = env.env_method("render", indices=i)[0]
                traj_buffer[i] = dict(
                    img=[reset_img.astype(np.uint8)], state=[obs[i].detach().cpu().numpy()], action=[]
                )
    for i in range(n_object):
        im_dataset[i]["obs"] = np.concatenate(im_dataset[i]["obs"], axis=0)
        im_dataset[i]["action"] = np.concatenate(im_dataset[i]["action"], axis=0)
        im_dataset[i]["boundary"] = np.array(im_dataset[i]["boundary"])
        task_arrays[i] = np.stack(task_arrays[i], axis=0)
    for i in range(n_object):
        print("N to move", i + 1, "count", im_dataset[i]["boundary"].shape[0])
    with open("distill_dataset_stacking.pkl", "wb") as f:
        pickle.dump(im_dataset, f)
    with open("distill_tasks.pkl", "wb") as f:
        pickle.dump(task_arrays, f)


def create_generalize_task(env, image_processor, desired_num):
    all_task_arrays = []
    count = 0
    while count < desired_num:
        tasks = env.env_method("create_generalize_task")
        robot_obs, init_states, goal_states, goal_images = map(lambda x: np.array(x), zip(*tasks))
        goal_feat = image_processor.process_obs(goal_images).cpu().numpy()
        task_arrays = np.concatenate([robot_obs, init_states, goal_states, goal_feat], axis=-1)
        all_task_arrays.append(task_arrays)
        count += task_arrays.shape[0]
    all_task_arrays = np.concatenate(all_task_arrays, axis=0)
    with open("test_tasks.pkl", "wb") as f:
        pickle.dump(all_task_arrays, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--load_path", type=str, default=None)
    args = parser.parse_args()
    main(args)