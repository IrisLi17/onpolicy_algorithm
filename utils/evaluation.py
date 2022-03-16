import torch, os, shutil
import matplotlib.pyplot as plt
import numpy as np


def evaluate(env, policy, n_episode):
    episode_count = 0
    frame_count = 0
    obs = env.reset()
    episode_reward = 0
    fig, ax = plt.subplots(1, 1)
    if os.path.exists("tmp"):
        shutil.rmtree("tmp")
    os.makedirs("tmp", exist_ok=True)
    while episode_count < n_episode:
        img = env.render(mode="rgb_array")
        ax.cla()
        ax.imshow(img)
        plt.imsave("tmp/tmp%d.png" % frame_count, img)
        plt.pause(0.01)
        with torch.no_grad():
            _, actions, _, _ = policy.act(obs, deterministic=False)
        obs, reward, done, info = env.step(actions)
        episode_reward += reward[0]
        frame_count += 1
        if done[0]:
            print("episode reward", episode_reward)
            episode_count += 1
            episode_reward = 0


def evaluate_fixed_states(env, policy, device, initial_states, goals, n_episode=100, deterministic=True, debug=False):
    if initial_states is not None:
        n_episode = len(initial_states)
    env_id = env.get_attr("spec")[0].id
    env.reset()
    evaluated_episode = 0
    n_used_state = 0
    if env_id == "BulletStack-v1":
        total_episode = [0] * env.get_attr("n_object")[0]
        success_episode = [0] * env.get_attr("n_object")[0]
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

