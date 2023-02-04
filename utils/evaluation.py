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
                
    episode_count = 0
    frame_count = 0
    obs = env.reset()
    feat_dim = env.get_attr("feature_dim")[0]
    print("goal state", env.get_attr("goal_dict")[0]["full_state"])
    goal_img = env.env_method("get_goal_image")[0]
    episode_reward = 0
    episode_length = 0
    reset_step = 0
    # env.env_method("start_rec", "output_0", indices=0)
    fig, ax = plt.subplots(1, 1)
    if os.path.exists("tmp"):
        shutil.rmtree("tmp")
    os.makedirs("tmp", exist_ok=True)
    if hasattr(env, "obs_rms"):
        init_obs_mean = env.obs_rms.mean.copy()
        init_obs_std = env.obs_rms.var.copy()
    while episode_count < n_episode:
        img = env.render(mode="rgb_array")
        ax.cla()
        ax.imshow(img)
        plt.imsave("tmp/tmp%d.png" % frame_count, np.concatenate([img, goal_img], axis=1))
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
            plt.imsave("tmp/tmp%d.png" % frame_count, np.concatenate([img, goal_img], axis=1))
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


def evaluate_tasks(env, policy, task_file="test_tasks.pkl"):
    with open(task_file, "rb") as f:
        new_tasks = pickle.load(f)
    if isinstance(new_tasks, list):
        new_tasks = np.concatenate(new_tasks[0:6], axis=0)
    task_idx = np.arange(new_tasks.shape[0])
    np.random.shuffle(task_idx)
    assert env.num_envs == 1
    task_per_env = new_tasks.shape[0] // env.num_envs if (
        new_tasks.shape[0] % env.num_envs) == 0 else new_tasks.shape[0] // env.num_envs + 1
    print("all tasks", new_tasks.shape[0])
    print("task per env", task_per_env)
    for i in range(env.num_envs):
        env.env_method("add_tasks", new_tasks[task_idx[task_per_env * i: task_per_env * (i + 1)]])
    # env.env_method("set_dist_threshold", 0.08)
    n_episode = 0
    n_success = 0
    detail_stats = [[0, 0] for _ in range(7)]
    obs = env.reset()
    n_to_move = env.env_method("oracle_feasible", obs.detach().cpu().numpy())[0][0]
    while n_episode < 500:
        with torch.no_grad():
            _, actions, _, _ = policy.act(obs, deterministic=False)
        with torch.no_grad():
            aux_pos_loss, aux_rot_loss = policy.get_aux_loss(obs)
        print("aux pos loss", aux_pos_loss, "aux rot loss", aux_rot_loss)
        obs, reward, done, info = env.step(actions)
        if done[0]:
            n_success += info[0]["is_success"]
            n_episode += 1
            detail_stats[n_to_move][0] += info[0]["is_success"]
            detail_stats[n_to_move][1] += 1
            obs = env.reset()
            n_to_move = env.env_method("oracle_feasible", obs.detach().cpu().numpy())[0][0]
    print("success rate", n_success / n_episode, "detail stats", detail_stats)
