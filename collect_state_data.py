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
from bullet_envs.utils.image_processor import ImageProcessor
sys.path.remove("../stacking_env")


use_pretrain = False
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
                use_expand_goal_prob=1,
                expand_task_path=args.expand_task,
                use_gpu_render=True if use_pretrain else False,
                # permute_object=True,
                # action_dim=5,
            ),
        ),
        policy_type="attention_discrete",
        obs_parser=StackingObsParser(robot_dim=11, obj_dim=16, goal_dim=6+6, n_max_goal=6, primitive=True),
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
                       training=False, reset_when_done=True, **config["create_env_kwargs"]) # TODO: note reset when done
    obs_parser = config["obs_parser"]
    policy = AttentionDiscretePolicy(obs_parser, env.action_space.shape[0], **config["policy"])
    policy.to(device)
    if args.load_path is not None:
        checkpoint = torch.load(args.load_path, map_location=device)
        policy.load_state_dict(checkpoint['policy'])

    image_processor = ImageProcessor(device)
    # if os.path.exists("tmp"):
    #     shutil.rmtree("tmp")
    # os.makedirs("tmp")
    # create_generalize_task(env, image_processor, 1000, use_patch_feat=False, use_raw_img=False if use_pretrain else True, shape="Y")
    # create_canonical_view(env, image_processor, use_patch_feat=False, use_raw_img=True)
    n_traj = min(len(env.get_attr("offline_datasets")[0]), 10_000)
    collect_data(env, policy, image_processor, n_traj, balance=False, round_idx=args.round_idx, use_patch_feat=False, use_raw_img=False if use_pretrain else True)
    # follow_expand_data(env, obs_parser, image_processor, n_traj, args.round_idx)
    # create test tasks
    # prepare_visual_base_data(env, policy, image_processor, "collect_data_last_step.pkl", use_patch_feat=False, use_raw_img=False if use_pretrain else True)

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


def collect_data(env, policy, image_processor, desired_num, balance, round_idx, use_patch_feat, use_raw_img):
    dataset_fname = "distill_dataset_new_stacking_%sexpand%d.pkl" % ("raw_" if use_raw_img else "", round_idx)
    task_fname = "distill_tasks_new_%sexpand%d%s.pkl" % ("raw_" if use_raw_img else "", round_idx, "_balance" if balance else "")
    if os.path.exists(dataset_fname):
        ans = input(f'{dataset_fname} exists, remove? [Y|n]')
        if ans == "Y":
            os.remove(dataset_fname)
        else:
            print("Not removed")
            return
    assert env.get_attr("use_expand_goal_prob")[0] == 1
    device = image_processor.device
    n_object = 6
    task_arrays = [[] for _ in range(n_object)]
    traj_buffer = [dict(img=[], state=[], action=[]) for _ in range(env.num_envs)]
    obs = env.reset()
    imgs = env.get_images()
    for i in range(env.num_envs):
        traj_buffer[i]["state"].append(obs[i].detach().cpu().numpy())
        traj_buffer[i]["img"].append(imgs[i].astype(np.uint8))
    n_data = [0 for _ in range(n_object)]
    while sum([len(task_arrays[i]) for i in range(n_object)]) < desired_num and sum(n_data) < 6 * desired_num:
        with torch.no_grad():
            _, action, _, _ = policy.act(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        for i in range(env.num_envs):
            traj_buffer[i]["action"].append(action[i].detach().cpu().numpy())
            traj_buffer[i]["state"].append(obs[i].detach().cpu().numpy())
            traj_buffer[i]["img"].append(info[i]["img"])
            if done[i]:
                if info[i]["is_success"] and reward[i] == 1 and len(traj_buffer[i]["action"]) < 10:
                    traj_buffer[i]["state"][-1] = info[i]["terminal_observation"]
                    # Bridge to pixel env observation, relabel
                    # TODO
                    # for j in range(len(traj_buffer[i]["img"])):
                    #     plt.imsave("tmp/tmp%d.png" % j, traj_buffer[i]["img"][j].transpose((1, 2, 0)))
                    # exit()
                    traj_states = np.stack(traj_buffer[i]["state"], axis=0)
                    traj_images = np.stack(traj_buffer[i]["img"], axis=0)
                    traj_features = image_processor.mvp_process_image(
                        traj_images, use_patch_feat=use_patch_feat, use_raw_img=use_raw_img).detach().cpu().numpy()
                    robot_obs, objects_obs = policy.obs_parser.bridge(traj_states)
                    for goal_idx in range(len(traj_buffer[i]["action"]), len(traj_buffer[i]["action"]) + 1):
                        goal_feature = traj_features[goal_idx:goal_idx + 1]
                        goal_state = objects_obs[goal_idx:goal_idx + 1]
                        if not use_raw_img:
                            new_obs = np.concatenate([
                                traj_features[:goal_idx],
                                robot_obs[:goal_idx],
                                np.tile(goal_feature, (goal_idx, 1)),  
                                objects_obs[:goal_idx], 
                                np.tile(goal_state, (goal_idx, 1))
                            ], axis=-1).astype(np.float32)
                        else:
                            new_obs = np.concatenate([
                                traj_features[:goal_idx],
                                np.tile(goal_feature, (goal_idx, 1)),
                                robot_obs[:goal_idx],
                                objects_obs[:goal_idx],
                                np.tile(goal_state, (goal_idx, 1))
                            ], axis=-1).astype(np.float32)
                        # n_to_move = get_n_to_move(new_obs[0:1], n_object, 0.05, 0.75)[0]
                        n_to_move = len(set([int(traj_buffer[i]["action"][j][0]) for j in range(goal_idx)]))
                        if (not balance) or (balance and len(task_arrays[n_to_move - 1]) < desired_num / n_object):
                            # im_dataset[n_to_move - 1]["obs"].append(new_obs)
                            # im_dataset[n_to_move - 1]["action"].append(np.stack(traj_buffer[i]["action"][:goal_idx], axis=0).astype(np.float32))
                            # im_dataset[n_to_move - 1]["boundary"].append(n_data[n_to_move - 1])
                            # assert im_dataset[n_to_move - 1]["obs"][-1].shape[0] == im_dataset[n_to_move - 1]["action"][-1].shape[0]
                            n_data[n_to_move - 1] += new_obs.shape[0]
                            # print(n_data)
                            # get task array
                            task_array = np.concatenate([
                                robot_obs[0], objects_obs[0], objects_obs[goal_idx], traj_features[goal_idx]
                            ])
                            task_arrays[n_to_move - 1].append(task_array)
                            with open(dataset_fname, "ab") as f:
                                pickle.dump({"obs": new_obs, "action": np.stack(traj_buffer[i]["action"][:goal_idx], axis=0).astype(np.float32)}, f)
                    print(sum([len(task_arrays[i]) for i in range(n_object)]), sum(n_data))
                reset_img = env.env_method("render", indices=i)[0]
                traj_buffer[i] = dict(
                    img=[reset_img.astype(np.uint8)], state=[obs[i].detach().cpu().numpy()], action=[]
                )
    for i in range(n_object):
        print("N to move", i + 1, "count", len(task_arrays[i]))
    all_tasks = []
    for i in range(n_object):
        # task_arrays[i] = np.stack(task_arrays[i], axis=0)
        all_tasks.extend(task_arrays[i])
        # obs_shape = im_dataset[i]["obs"].shape[1]
        # action_shape = im_dataset[i]["action"].shape[1]
        # task_shape = task_arrays[i].shape[1]
    all_tasks = np.stack(all_tasks, axis=0)
    # with open("distill_dataset_stacking_raw_expand%d%s.pkl" % (round_idx, "_balance" if balance else ""), "wb") as f:
    #     pickle.dump(im_dataset, f)
    with open(task_fname, "wb") as f:
        pickle.dump(all_tasks, f)


# only prefix is recorded, do not use this function
def follow_expand_data(env, policy, image_processor, desired_num, round_idx):
    dataset_fname = f"distill_dataset_new_stacking_raw_expand{round_idx}.pkl" 
    task_fname = f"distill_tasks_new_raw_expand{round_idx}.pkl"
    if os.path.exists(dataset_fname):
        ans = input(f'{dataset_fname} exists, remove? [Y|n]')
        if ans == "Y":
            os.remove(dataset_fname)
        else:
            print("Not removed")
            return
    assert env.get_attr("use_expand_goal_prob")[0] == 1
    n_object = 6
    task_arrays = [[] for _ in range(n_object)]
    traj_buffer = [dict(img=[], state=[], action=[]) for _ in range(env.num_envs)]
    actions_to_follow = [None for _ in range(env.num_envs)]
    obs = env.reset()
    imgs = env.get_images()
    for i in range(env.num_envs):
        traj_buffer[i]["state"].append(obs[i].detach().cpu().numpy())
        traj_buffer[i]["img"].append(imgs[i].astype(np.uint8))
        actions_to_follow[i] = env.get_attr("expand_action_seq", indices=i)[0]
    n_data = [0 for _ in range(n_object)]
    while sum([len(task_arrays[i]) for i in range(n_object)]) < desired_num and sum(n_data) < 6 * desired_num:
        with torch.no_grad():
            _, action, _, _ = policy.act(obs)
        for i in range(env.num_envs):
            if len(actions_to_follow[i]):
                action[i] = torch.from_numpy(actions_to_follow[i][0]).to(obs.device)
                actions_to_follow[i] = actions_to_follow[i][1:]
        obs, reward, done, info = env.step(action)
        for i in range(env.num_envs):
            traj_buffer[i]["action"].append(action[i].detach().cpu().numpy())
            traj_buffer[i]["state"].append(obs[i].detach().cpu().numpy())
            traj_buffer[i]["img"].append(info[i]["img"])
            if done[i]:
                # TODO: if success
                # TODO: modify to auto reset to prevent infinite rollout

                # print("should done", done[i])
                # print("obs", obs[i], "goal", env.get_attr("goal", indices=i)[0])
                traj_states = np.stack(traj_buffer[i]["state"], axis=0)
                traj_images = np.stack(traj_buffer[i]["img"], axis=0)
                traj_features = image_processor.mvp_process_image(
                    traj_images, use_patch_feat=False, use_raw_img=True).detach().cpu().numpy()
                robot_obs, objects_obs = policy.obs_parser.bridge(traj_states)
                # print("last objects obs", objects_obs[-1])
                goal_idx = len(traj_buffer[i]["action"])
                goal_feature = traj_features[goal_idx:goal_idx + 1]
                goal_state = objects_obs[goal_idx:goal_idx + 1]     
                new_obs = np.concatenate([
                    traj_features[:goal_idx],
                    np.tile(goal_feature, (goal_idx, 1)),
                    robot_obs[:goal_idx],
                    objects_obs[:goal_idx],
                    np.tile(goal_state, (goal_idx, 1))
                ], axis=-1).astype(np.float32)
                n_to_move = len(set([int(traj_buffer[i]["action"][j][0]) for j in range(goal_idx)]))
                n_data[n_to_move - 1] += new_obs.shape[0]
                task_array = np.concatenate([
                    robot_obs[0], objects_obs[0], objects_obs[goal_idx], traj_features[goal_idx]
                ])
                task_arrays[n_to_move - 1].append(task_array)
                with open(dataset_fname, "ab") as f:
                    pickle.dump({"obs": new_obs, "action": np.stack(traj_buffer[i]["action"][:goal_idx], axis=0).astype(np.float32)}, f)
                print(sum([len(task_arrays[i]) for i in range(n_object)]), sum(n_data))
                obs[i] = torch.from_numpy(env.env_method("reset", indices=i)[0]).float().to(obs.device)
                actions_to_follow[i] = env.get_attr("expand_action_seq", indices=i)[0]
                reset_img = env.env_method("render", indices=i)[0]
                traj_buffer[i] = dict(
                    img=[reset_img.astype(np.uint8)], state=[obs[i].detach().cpu().numpy()], action=[]
                )
    for i in range(n_object):
        print("N to move", i + 1, "count", len(task_arrays[i]))
    all_tasks = []
    for i in range(n_object):
        all_tasks.extend(task_arrays[i])
    all_tasks = np.stack(all_tasks, axis=0)
    # with open("distill_dataset_stacking_raw_expand%d%s.pkl" % (round_idx, "_balance" if balance else ""), "wb") as f:
    #     pickle.dump(im_dataset, f)
    with open(task_fname, "wb") as f:
        pickle.dump(all_tasks, f)


def create_generalize_task(env, image_processor, desired_num, use_patch_feat, use_raw_img, shape="3T"):
    task_fname = "test_tasks_%s_%s.pkl" % ("raw" if use_raw_img else "", shape)
    all_task_arrays = []
    all_state_task_trajs = []
    count = 0
    while count < desired_num:
        tasks = env.env_method("create_generalize_task", shape=shape)
        robot_obs, init_states, goal_states, goal_images = map(lambda x: np.array(x), zip(*tasks))
        goal_feat = image_processor.mvp_process_image(
            goal_images, use_patch_feat=use_patch_feat, use_raw_img=use_raw_img
        ).cpu().numpy()
        task_arrays = np.concatenate([robot_obs, init_states, goal_states, goal_feat], axis=-1)
        all_task_arrays.append(task_arrays)
        count += task_arrays.shape[0]
        # also create state version
        for e_idx in range(robot_obs.shape[0]):
            state_task_array = [robot_obs[e_idx, 1:7], np.zeros(5)]
            for i in range(init_states.shape[1] // 7):
                state_task_array.append(
                    np.concatenate([
                        init_states[e_idx, 7 * i: 7 * i + 3], 
                        np.zeros(3),
                        np.array(p.getEulerFromQuaternion(init_states[e_idx, 7 * i + 3: 7 * (i + 1)])),
                        np.zeros(7),
                    ])
                )
            n_max_goal = env.get_attr("n_max_goal")[0]
            goal_idxs = np.argsort(goal_states[e_idx].reshape((-1, 7))[:, 2])[-n_max_goal:]
            for i in range(env.get_attr("n_max_goal")[0]):
                onehot = np.zeros(6)
                onehot[goal_idxs[i]] = 1
                state_task_array.append(
                    np.concatenate([
                        np.array([0., 1., 0.]), 
                        goal_states[e_idx, goal_idxs[i] * 7: goal_idxs[i] * 7 + 3],
                        onehot
                    ])
                )
            all_state_task_trajs.append({"obs": [np.concatenate(state_task_array)]})
    all_task_arrays = np.concatenate(all_task_arrays, axis=0)
    with open(task_fname, "wb") as f:
        pickle.dump(all_task_arrays, f)
    with open(f"test_state_tasks_{shape}.pkl", "wb") as f:
        pickle.dump({"expansion": all_state_task_trajs}, f)

def create_canonical_view(env, image_processor, use_patch_feat, use_raw_img):
    images = np.array(env.env_method("create_canonical_view")[0])
    image_feat = image_processor.mvp_process_image(
        images, use_patch_feat=use_patch_feat, use_raw_img=use_raw_img).cpu().numpy()
    print(images.shape, image_feat.shape)
    with open("canonical_view.pkl", "wb") as f:
        pickle.dump(image_feat, f)

def prepare_visual_base_data(env, policy, image_processor, base_data_path, use_patch_feat, use_raw_img):
    dataset_fname = "base_dataset_stacking.pkl"
    if os.path.exists(dataset_fname):
        os.remove(dataset_fname)
    with open(base_data_path, "rb") as f:
        data = pickle.load(f)
    import tqdm
    for i in tqdm.tqdm(range(len(data) // env.num_envs)):
        for e_idx in range(env.num_envs):
            env.env_method("set_obs_debug", data[env.num_envs * i + e_idx]["obs"][0], indices=e_idx)
        obs = env.reset()
        reset_img = env.env_method("render")
        actions_np = np.stack([data[j]["actions"][0] for j in range(env.num_envs * i, env.num_envs * (i + 1))], axis=0)
        actions = torch.from_numpy(actions_np).to(obs.device)
        _, reward, done, infos = env.step(actions)
        goal_img = [info["img"] for info in infos]
        for e_idx in range(env.num_envs):
            if not done[e_idx]:
                print("Error: taking action does not lead to success")
                continue
            goal_obs = infos[e_idx]["terminal_observation"]
            traj_images = np.stack([reset_img[e_idx], goal_img[e_idx]], axis=0)
            traj_features = image_processor.mvp_process_image(
                traj_images, use_patch_feat=use_patch_feat, use_raw_img=use_raw_img).detach().cpu().numpy()
            traj_states = np.stack([obs[e_idx].detach().cpu().numpy(), goal_obs], axis=0)
            robot_obs, objects_obs = policy.obs_parser.bridge(traj_states)
            goal_feature = traj_features[1: 2]
            goal_state = objects_obs[1: 2]
            new_obs = np.concatenate([
                traj_features[:1],
                goal_feature,
                robot_obs[:1],
                objects_obs[:1],
                goal_state,
            ], axis=-1).astype(np.float32)
            with open(dataset_fname, "ab") as f:
                pickle.dump({"obs": new_obs, "action": np.expand_dims(actions_np[e_idx], axis=0).astype(np.float32)}, f)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--load_path", type=str, default=None)
    parser.add_argument("--expand_task", type=str, default="primitive_cuboid_expand5.pkl")
    parser.add_argument("--round_idx", type=int)
    args = parser.parse_args()
    main(args)