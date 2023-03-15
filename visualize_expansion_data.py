import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import gym


def visualize_expansion():
    count = 0
    with open("distill_dataset_new_stacking_raw_expand2.pkl", "rb") as f:
        try:
            while True:
                dataset = pickle.load(f)
                traj_obs = dataset['obs']
                count += 1
                if max(traj_obs[0][-42:].reshape(6, 7)[:, 2]) < 0.15:
                    continue
                # init_obs = traj_obs[0][:3 * 128 * 128].reshape((3, 128, 128)).transpose((1, 2, 0)).astype(np.uint8)
                goal_obs = traj_obs[0][3 * 128 * 128: 6 * 128 * 128].reshape((3, 128, 128)).transpose((1, 2, 0)).astype(np.uint8)
                for i in range(traj_obs.shape[0]):
                    cur_obs = traj_obs[i][:3 * 128 * 128].reshape((3, 128, 128)).transpose((1, 2, 0)).astype(np.uint8)
                    plt.imsave("tmp%d.png" % i, cur_obs)
                plt.imsave("tmp%d.png" % traj_obs.shape[0], goal_obs)
                res = input("Continue? [Y|n]")
                if res == "Y":
                    for i in range(traj_obs.shape[0] + 1):
                        os.remove("tmp%d.png" % i)                
                    continue
                break
        except:
            pass

def check_expansion_data_size():
    count = 0
    with open("distill_dataset_new_stacking_raw_expand1.pkl", "rb") as f:
        try:
            while True:
                dataset = pickle.load(f)
                traj_obs = dataset['obs']
                count += traj_obs.shape[0]
        except EOFError:
            pass
    print(count)

def visualize_base():
    import sys
    sys.path.append("../stacking_env")
    import bullet_envs
    env = gym.make("BulletStack-v2", n_object=6,
                n_to_stack=[[1], [1], [1]],
                action_dim=7,
                reward_type="sparse",
                name="allow_rotation",
                primitive=True,
                generate_data=True,
                use_expand_goal_prob=1,)
    sys.path.remove("../stacking_env")
    with open("collect_data_last_step.pkl", "rb") as f:
        data = pickle.load(f)
    print(len(data))
    traj_idx = np.random.choice(np.arange(len(data)), 20)
    for idx in traj_idx:
        env.set_obs_debug(data[idx]["obs"][0])
        env.reset()
        img = env.render(mode="rgb_array", width=128, height=128)
        plt.imsave(f"tmp{idx}_0.png", img)
        for i in range(data[idx]["actions"].shape[0]):
            env.step(data[idx]["actions"][i])
            img = env.render(mode="rgb_array", width=128, height=128)
            plt.imsave(f"tmp{idx}_{i + 1}.png", img)

if __name__ == "__main__":
    # visualize_base()
    visualize_expansion()
    # check_expansion_data_size()
