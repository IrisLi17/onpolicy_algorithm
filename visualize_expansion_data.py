import pickle
import numpy as np
import matplotlib.pyplot as plt
import os


with open("distill_dataset_new_stacking_raw_expand3.pkl", "rb") as f:
    try:
        while True:
            dataset = pickle.load(f)
            traj_obs = dataset['obs']
            if max(traj_obs[0][-42:].reshape(6, 7)[:, 2]) < 0.32:
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