import torch, os, shutil
import matplotlib.pyplot as plt
import numpy as np
import pickle


def evaluate(env, policy, n_episode, task_file="generated_tasks_0.pkl"):
    with open(task_file, "rb") as f:
        new_tasks = pickle.load(f)
    if isinstance(new_tasks, list):
        new_tasks = np.concatenate(new_tasks[0:6], axis=0)
    task_idx = np.arange(new_tasks.shape[0])
    np.random.shuffle(task_idx)
    task_per_env = new_tasks.shape[0] // env.num_envs if (
        new_tasks.shape[0] % env.num_envs) == 0 else new_tasks.shape[0] // env.num_envs + 1
    task_per_env = min(100, task_per_env)
    for i in range(env.num_envs):
        env.env_method("add_tasks", new_tasks[task_idx[task_per_env * i: task_per_env * (i + 1)]])
                
    if os.path.exists("tmp"):
        shutil.rmtree("tmp")
    os.makedirs("tmp", exist_ok=True)
    episode_count = 0
    frame_count = 0
    obs = env.reset()
    feat_dim = env.get_attr("feature_dim")[0]
    print("goal state", env.get_attr("goal_dict")[0]["full_state"])
    goal_img = env.env_method("get_goal_image")[0]
    plt.imsave("tmp/goal%d.png" % episode_count, goal_img)
    episode_reward = 0
    episode_length = 0
    reset_step = 0
    # env.env_method("start_rec", "output_0", indices=0)
    fig, ax = plt.subplots(1, 1)
    if hasattr(env, "obs_rms"):
        init_obs_mean = env.obs_rms.mean.copy()
        init_obs_std = env.obs_rms.var.copy()
    while episode_count < n_episode:
        img = env.render(mode="rgb_array")
        ax.cla()
        ax.imshow(img)
        plt.imsave("tmp/tmp%d_%d.png" % (episode_count, frame_count), img)
        plt.pause(0.01)
        with torch.no_grad():
            value_pred, actions, _, _ = policy.act(obs, deterministic=False)
        # if frame_count == reset_step:
        #     print("action at reset step", actions)
        print("value", value_pred[0])
        obs, reward, done, info = env.step(actions)
        episode_reward += reward[0]
        episode_length += 1
        frame_count += 1
        # print("action", actions[0], "info", info[0]["handle_joint"])
        if hasattr(env, "obs_rms"):
            assert np.linalg.norm(env.obs_rms.mean - init_obs_mean) < 1e-5
            assert np.linalg.norm(env.obs_rms.var - init_obs_std) < 1e-5
        if done[0]:
            # env.env_method("end_rec", indices=0)
            img = env.render(mode="rgb_array")
            plt.imsave("tmp/tmp%d_%d.png" % (episode_count, frame_count), img)
            frame_count += 1
            print("terminal obs", info[0]["terminal_observation"][feat_dim * 2 + 7:].reshape(2, -1))
            print(episode_count, "episode reward", episode_reward, "episode length", episode_length)
            obs = env.reset()
            print("goal", env.get_attr("goal_dict")[0]["full_state"])
            episode_count += 1
            episode_reward = 0
            episode_length = 0
            reset_step = frame_count
            # env.env_method("start_rec", "output_%d" % episode_count, indices=0)
            goal_img = env.env_method("get_goal_image")[0]
            plt.imsave("tmp/goal%d.png" % episode_count, goal_img)


def evaluate_fixed_states(env, policy, device, initial_states, goals, n_episode=100, deterministic=True, debug=False):
    if initial_states is not None:
        n_episode = len(initial_states)
    env_id = env.get_attr("spec")[0].id
    obs = env.reset()
    evaluated_episode = 0
    n_used_state = 0
    if env_id == "BulletStack-v1":
        total_episode = [0] * env.get_attr("n_object")[0]
        success_episode = [0] * env.get_attr("n_object")[0]
    elif env_id == "BulletDrawer-v1" or env_id == "BulletDrawerState-v1":
        total_episode = [0, 0]
        success_episode = [0, 0]
    else:
        total_episode, success_episode = 0, 0
    valid_mask = [True] * env.num_envs
    if initial_states is not None:
        for i in range(env.num_envs):
            if n_used_state < len(initial_states):
                env.env_method("set_state", initial_states[n_used_state], indices=i)
                env.env_method("set_goals", [goals[n_used_state]], indices=i)
                if env_id == "BulletStack-v1":
                    env.env_method("sync_attr", indices=i)
                n_used_state += 1
            else:
                valid_mask[i] = False
        obs = env.get_obs()
    recurrent_mask = torch.ones(env.num_envs, 1, device=device)
    while evaluated_episode < n_episode:
        with torch.no_grad():
            values, actions, log_probs, recurrent_hidden_state = \
                policy.act(obs, deterministic=deterministic)
        obs, rewards, dones, infos = env.step(actions)
        for e_idx, done in enumerate(dones):
            if done:
                if valid_mask[e_idx]:
                    evaluated_episode += 1
                    if env_id == "BulletStack-v1":
                        total_episode[infos[e_idx]["n_to_stack"] - 1] += 1
                        success_episode[infos[e_idx]["n_to_stack"] - 1] += infos[e_idx]["is_success"]
                    elif env_id == "BulletDrawer-v1" or env_id == "BulletDrawerState-v1":
                        if infos[e_idx]["is_goal_move_drawer"]:
                            total_episode[0] += 1
                            success_episode[0] += infos[e_idx]["is_success"]
                        else:
                            total_episode[1] += 1
                            success_episode[1] += infos[e_idx]["is_success"]
                    else:
                        total_episode += 1
                        success_episode += infos[e_idx]["is_success"]
                if initial_states is not None:
                    if n_used_state < len(initial_states):
                        env.env_method(
                            "set_state", initial_states[n_used_state],
                            indices=e_idx
                        )
                        env.env_method(
                            "set_goals", [goals[n_used_state]],
                            indices=e_idx
                        )
                        if env_id == "BulletStack-v1":
                            env.env_method("sync_attr", indices=e_idx)
                        n_used_state += 1
                        e_obs = env.get_obs(indices=e_idx)[0]
                        obs[e_idx] = e_obs
                    else:
                        valid_mask[e_idx] = False
                recurrent_mask[e_idx] = 0.
    # print(n_to_stack_stats, n_to_stack_success)
    return success_episode, total_episode


def evaluate_tasks(env, policy, task_file="test_tasks.pkl", evaluate_episode=200, deterministic=False):
    with open(task_file, "rb") as f:
        new_tasks = pickle.load(f)
    if isinstance(new_tasks, list):
        new_tasks = np.concatenate(new_tasks[0:6], axis=0)
    # new_tasks = new_tasks[77:78]
    if task_file == "test_tasks_raw.pkl":
        new_tasks = new_tasks[:1]
        new_tasks[:, 7 + 4 * 7: 7 + 5 * 7] = np.array([0.38, -0.23, 0.025, 0., 0., np.sin(np.pi / 4), np.cos(np.pi / 4)])
        new_tasks[:, 7 + 5 * 7: 7 + 6 * 7] = np.array([0.3, 0.105, 0.025, 0., 0., np.sin(np.pi / 5), np.cos(np.pi / 5)])
    elif task_file == "test_tasks_raw_I.pkl":
        new_tasks = new_tasks[14:15]
        new_tasks[:, 7: 7 + 1 * 7] = np.array([0.5185, -0.1691, 0.025, 0., 0., np.sin(-np.pi / 7), np.cos(-np.pi / 7)])
        new_tasks[:, 7 + 2 * 7: 7 + 3 * 7] = np.array([0.4054, -0.1705, 0.025, 0., 0., np.sin(np.pi * 3 / 11), np.cos(np.pi * 3 / 11)])
        new_tasks[:, 7 + 3 * 7: 7 + 4 * 7] = np.array([0.3350, 0.1617, 0.025, 0., 0., np.sin(np.pi * 3 / 14), np.cos(np.pi * 3 / 14)])
        print(new_tasks[0, 7 :7 + 42].reshape(6, 7), new_tasks[0, 7 + 42: 7 + 42 * 2].reshape(6, 7))
    task_idx = np.arange(new_tasks.shape[0])
    goal_img = (new_tasks[0, -3 * 128 * 128:].reshape(3, 128, 128).transpose(1, 2, 0)).astype(np.uint8)
    plt.imsave("tmp/tmp_goal.png", goal_img)
    # np.random.shuffle(task_idx)
    assert env.num_envs == 1
    task_per_env = new_tasks.shape[0] // env.num_envs if (
        new_tasks.shape[0] % env.num_envs) == 0 else new_tasks.shape[0] // env.num_envs + 1
    task_per_env = min(task_per_env, 500)
    print("all tasks", new_tasks.shape[0])
    print("task per env", task_per_env)
    env.env_method("clear_tasks")
    for i in range(env.num_envs):
        env.env_method("add_tasks", new_tasks[task_idx[task_per_env * i: task_per_env * (i + 1)]])
    # env.env_method("set_dist_threshold", 0.08)
    n_episode = 0
    n_success = 0
    frame_count = 0
    detail_stats = [[0, 0] for _ in range(7)]
    obs = env.reset()
    n_to_move = env.env_method("oracle_feasible", obs.detach().cpu().numpy())[0][0]
    if task_file == "test_tasks_raw.pkl":
        action_seq = [
            torch.tensor([ 3.0000, -0.6000 + 0.6 * 0,  0.9000 - 0.5 * 0, -0.8000 + 0.1,  1.0000,  1.0000,  0.4000]),
            torch.tensor([ 2.0000,  0.6000,  0.4000, -0.8000 + 0.1,  0.0000,  1.0000, -0.5000]),
            torch.tensor([ 1.0000,  0.7000,  0.4000, -0.3000 + 0.1,  0.0000,  0.0000, -1.0000]),
            torch.tensor([ 5.0000, -0.4000,  0.0000, -0.9000 + 0.2, -0.1000,  1.0000, -0.1000]),
            torch.tensor([ 4.0000, -0.6000,  0.9000, -0.2000 + 0.1,  0.0000,  0.0000,  0.4000]),
            torch.tensor([ 0.0000, -0.4000, -0.1000 + 0.1, -0.3000 + 0.1,  0.0000,  0.0000, -0.3000])
        ]
    elif task_file == "test_tasks_raw_I.pkl":
        action_seq = [
            torch.tensor([ 3.0000, -0.8000, -0.3000 + 0.4 * 0, -0.8000 + 0.1,  0.2000,  1.0000,  0.2000]),
            torch.tensor([ 0.0000, -0.8000, -0.3000 + 0.4 * 0,  0.4000 - 0.6,  0.0000,  0.0000, -0.9000]),
            torch.tensor([ 1.0000, -0.7000, -0.4000 + 0.4 * 0,  0.2000 + 0.1,  0.1000,  1.0000,  0.0000]),
        ]
    while n_episode < evaluate_episode:
        with torch.no_grad():
            _, actions, _, _ = policy.act(obs, deterministic=deterministic)
        img = env.render(mode="rgb_array")
        plt.imsave("tmp/tmp%d_%d.png" % (n_episode, frame_count), img)
        actions = action_seq[frame_count].unsqueeze(dim=0)
        print(actions[0])
        # with torch.no_grad():
        #     aux_pos_loss, aux_rot_loss = policy.get_aux_loss(obs)
        # print("aux pos loss", aux_pos_loss, "aux rot loss", aux_rot_loss)
        obs, reward, done, info = env.step(actions)
        frame_count += 1
        if done[0]:
            img = env.render(mode="rgb_array")
            plt.imsave("tmp/tmp%d_%d.png" % (n_episode, frame_count), img)
            if info[0]["is_success"]:
                return
            print("reset")
            n_success += info[0]["is_success"]
            n_episode += 1
            detail_stats[n_to_move][0] += info[0]["is_success"]
            detail_stats[n_to_move][1] += 1
            obs = env.reset()
            n_to_move = env.env_method("oracle_feasible", obs.detach().cpu().numpy())[0][0]
    print("success rate", n_success / n_episode, "detail stats", detail_stats)


def _parse_task_from_obs(obs: np.ndarray):
    assert len(obs.shape) == 1 and obs.shape[0] == 6 * 128 * 128 + 7 + 84
    task = np.concatenate([obs[-91:], obs[3 * 128 * 128: 6 * 128 * 128]])
    return task


def trajectory_replay(env, dataset_file="distill_dataset_new_stacking_raw_expand3.pkl"):
    assert env.num_envs == 1
    with open(dataset_file, "rb") as f:
        try:
            while True:
                traj = pickle.load(f)
                obs_seq = traj["obs"]
                action_seq = traj["action"]
                task = _parse_task_from_obs(obs_seq[0])
                env.env_method("clear_tasks")
                env.env_method("add_tasks", np.expand_dims(task, 0))
                env.reset()
                for step_idx in range(action_seq.shape[0]):
                    obs, reward, done, info = env.step(torch.from_numpy(action_seq[step_idx]).unsqueeze(dim=0))
                print(info)
        except EOFError:
            pass
