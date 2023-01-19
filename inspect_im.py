import pickle
import numpy as np
import importlib
import torch
import sys
sys.path.append("../stacking_env")
from bullet_envs.env.pixel_stacking import quat_apply_batch
import matplotlib.pyplot as plt


def inspect_task():
    cfg_module = importlib.import_module("config.bullet_pixel_stack")
    config = cfg_module.config
    config["num_workers"] = 1
    config["create_env_kwargs"]["kwargs"]["use_gpu_render"] = False
    config["log_dir"] = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from bullet_envs.utils.make_vec_env import make_vec_env
    env = make_vec_env(config["env_id"], config["num_workers"], device, log_dir=config["log_dir"], **config["create_env_kwargs"])
        
    with open("logs/ppo_BulletPixelStack-v1/base1/generated_tasks_0.pkl", "rb") as f:
        task_arrays = pickle.load(f)
    print(task_arrays.shape)
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    for i in range(min(task_arrays.shape[0], 10)):
        task_array = task_arrays[i]
        env.env_method("set_task", task_array)
        init_img = env.render(mode="rgb_array")
        goal_img = env.env_method("get_goal_image")[0]
        ax[0].imshow(init_img)
        ax[1].imshow(goal_img)
        plt.savefig("tmp/tmp%d.png" % i)
    priv_info = task_arrays[:, 7: -768].reshape((-1, 2, 6, 7))
    cur_state = priv_info[:, 0]
    goal_state = priv_info[:, 1]
    pos_cond = np.linalg.norm(cur_state[:, :, :3] - goal_state[:, :, :3], axis=-1) < 0.05
    achieved_x_vec = quat_apply_batch(cur_state[:, :, 3:], np.array([1., 0., 0.]).reshape((1, 1, 3)))
    goal_x_vec = quat_apply_batch(goal_state[:, :, 3:], np.array([1., 0., 0.]).reshape((1, 1, 3)))
    rot_cond = np.abs(np.sum(achieved_x_vec * goal_x_vec, axis=-1)) > 0.75
    match_count = np.sum(np.logical_and(pos_cond, rot_cond), axis=-1)
    n_object_to_move = 6 - match_count
    for i in range(7):
        print("%d object to move %d" % (i, np.sum(n_object_to_move == i)))

if __name__ == "__main__":
    inspect_task()
# with open("im_dataset_0.pkl", "rb") as f:
#     dataset = pickle.load(f)
# obs = dataset["obs"]
# terminate_obs = dataset["terminate_obs"]
# original_obs = dataset["original_obs"]
# boundary = dataset["boundary"]
# for i in range(len(terminate_obs)):
#     print("=" * 10)
#     print("obs", obs[boundary[i]: boundary[i + 1], 768 * 2 + 7:], terminate_obs[i][768 * 2 + 7:], 
#           "robot", obs[boundary[i]: boundary[i + 1], 768: 768 + 7])
#     assert (np.linalg.norm(obs[boundary[i]: boundary[i + 1], 768 + 7: 768 * 2 + 7] - obs[boundary[i]: boundary[i] + 1, 768 + 7: 768 * 2 + 7]) < 1e-3)
#     # print("original obs", original_obs[boundary[i], 768 * 2 + 7:], original_obs[boundary[i + 1] - 1, 768 * 2 + 7:])
#     print("initial value", dataset["initial_value"][i])
#     print("interm value", dataset["interm_value"][i])
