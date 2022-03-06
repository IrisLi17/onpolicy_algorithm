import torch, os, shutil
import matplotlib.pyplot as plt


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
        plt.pause(0.1)
        with torch.no_grad():
            _, actions, _, _ = policy.act(obs, deterministic=False)
        obs, reward, done, info = env.step(actions)
        episode_reward += reward[0]
        frame_count += 1
        if done[0]:
            print("episode reward", episode_reward)
            episode_count += 1
            episode_reward = 0
