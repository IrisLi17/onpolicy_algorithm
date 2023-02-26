from policies.slot_attention_policy import SlotAttentionAutoEncoder, detect_background, assign_object
import pickle
import numpy as np
import torch
from collections import deque
from utils import logger
import os
import argparse


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# im_mean = torch.Tensor([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1)).to(device)
# im_std = torch.Tensor([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1)).to(device)
im_mean = torch.Tensor([0.5, 0.5, 0.5]).reshape((1, 3, 1, 1)).to(device)
im_std = torch.Tensor([0.5, 0.5, 0.5]).reshape((1, 3, 1, 1)).to(device)

def train(args):
    data_rounds = [3,]
    total_dataset = []
    for data_round in data_rounds:
        with open("distill_dataset_new_stacking_%sexpand%d.pkl" % ("raw_", data_round), "rb") as f:
            try:
                while True:
                    dataset = pickle.load(f)
                    if isinstance(dataset, list):
                        total_dataset.extend(dataset)
                    else:
                        total_dataset.append(dataset)
            except EOFError:
                pass
    il_dataset = dict()
    for k in total_dataset[0].keys():
        il_dataset[k] = np.concatenate([total_dataset[i][k] for i in range(len(total_dataset))], axis=0) 
    obs_dataset = il_dataset["obs"][:, :3 * 128 * 128]
    # load eval dataset
    with open("test_tasks_raw.pkl", "rb") as f:
        data = pickle.load(f)
    eval_obs_dataset = data[:, -3 * 128 * 128:]

    model = SlotAttentionAutoEncoder(resolution=(128, 128), num_slots=args.num_slots, num_iterations=3)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    batch_size = 64
    metric = torch.nn.MSELoss()

    logger.configure(args.log_dir)
    losses = deque(maxlen=100)
    for n_steps in range(args.num_train_steps):
        mb_indices = np.random.randint(0, obs_dataset.shape[0], size=batch_size)
        if n_steps < args.warmup_steps:
            lr = n_steps / args.warmup_steps * args.learning_rate
        else:
            lr = args.learning_rate
        lr = lr * (args.decay_rate ** (n_steps / args.decay_steps))
        for p in optimizer.param_groups:
            p['lr'] = lr
        mb_obs = torch.from_numpy(obs_dataset[mb_indices]).float().to(device)
        mb_obs = mb_obs.reshape((mb_obs.shape[0], 3, 128, 128))
        mb_obs = (mb_obs / 255.0 - im_mean) / im_std
        mb_obs = mb_obs.permute((0, 2, 3, 1))
        out = model.forward(mb_obs)
        recon_combined = out[0]
        loss = metric.forward(recon_combined, mb_obs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        # eval
        if n_steps % 1000 == 0:
            eval_losses = []
            for mb_idx in range(eval_obs_dataset.shape[0] // batch_size):
                mb_obs = torch.from_numpy(eval_obs_dataset[mb_idx * batch_size: (mb_idx + 1) * batch_size]).float().to(device)
                mb_obs = mb_obs.reshape((mb_obs.shape[0], 3, 128, 128))
                mb_obs = (mb_obs / 255.0 - im_mean) / im_std
                mb_obs = mb_obs.permute((0, 2, 3, 1))
                with torch.no_grad():
                    out = model.forward(mb_obs)
                    recon_combined = out[0]
                    eval_loss = metric.forward(recon_combined, mb_obs)
                    eval_losses.append(eval_loss.item())
                logger.logkv("eval_loss", np.mean(eval_losses))
        if n_steps % 100 == 0:
            logger.logkv("loss", np.mean(losses))
            logger.logkv("n_steps", n_steps)
            logger.logkv("lr", lr)
            logger.dump_tabular()
        
        if n_steps % 5000 == 0:
            save_dict = {
                "param": model.state_dict(), "im_mean": im_mean, "im_std": im_std
            }
            torch.save(save_dict, os.path.join(logger.get_dir(), "model_%d.pt" % n_steps))


def eval(args):
    # '''
    with open("test_tasks_raw.pkl", "rb") as f:
        data = pickle.load(f)
    obs = data[:, -3 * 128 * 128:]
    '''
    total_dataset = []
    with open("distill_dataset_new_stacking_raw_expand3.pkl", "rb") as f:
        for i in range(50):
            dataset = pickle.load(f)
            if isinstance(dataset, list):
                total_dataset.extend(dataset)
            else:
                total_dataset.append(dataset)
        data = np.concatenate([total_dataset[i]["obs"] for i in range(len(total_dataset))], axis=0) 
    obs = data[:, :3 * 128 * 128]
    '''
    model = SlotAttentionAutoEncoder(resolution=(128, 128), num_slots=args.num_slots, num_iterations=3)
    model.to(device)
    model.load_state_dict(torch.load(args.load_path)["param"])
    # obs_batch = obs[np.random.randint(0, obs.shape[0], size=5)]
    obs_batch = obs[:5]
    visualize(model, obs_batch)

    
def visualize(model, obs_batch: np.ndarray):
    mb_obs = torch.from_numpy(obs_batch).float().to(device)
    mb_obs = mb_obs.reshape((mb_obs.shape[0], 3, 128, 128))
    mb_obs = (mb_obs / 255.0 - im_mean) / im_std
    mb_obs = mb_obs.permute((0, 2, 3, 1))
    with torch.no_grad():
        out = model.forward(mb_obs)
        recon_combined, recon, mask, slot_feature = out
        per_slot_recon = recon * mask
        loss = torch.nn.MSELoss().forward(recon_combined, mb_obs)
    print(torch.cuda.memory_allocated(device))
    print("loss", loss)
    colors = np.array([[1.0, 0, 0], [1, 1, 0], [0.2, 0.8, 0.8], [0.8, 0.2, 0.8], [0.2, 0.8, 0.2], [0.0, 0.0, 1.0]])
    # normalize colors
    # colors = (colors - im_mean.squeeze(dim=-1).squeeze(dim=-1)) / im_std.squeeze(dim=-1).squeeze(dim=-1)
    # with open("debug.pkl", "wb") as f:
    #     pickle.dump({"mb_obs": mb_obs, "recon_combined": recon_combined, "recon": recon, "mask": mask}, f)
    object_feature, assignment = assign_object(mb_obs * 0.5 + 0.5, mask, colors, slot_feature)
    print("object assignment", assignment)
    recon_denorm = (recon_combined.permute((0, 3, 1, 2)) * im_std + im_mean).cpu().numpy() * 255
    per_slot_recon_denorm = (per_slot_recon.permute((0, 1, 4, 2, 3)) * im_std.unsqueeze(dim=1) + im_mean.unsqueeze(dim=1)).cpu().numpy() * 255
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, min(5, recon_denorm.shape[0]))
    for i in range(len(ax[0])):
        ax[0][i].imshow(obs_batch[i].reshape(3, 128, 128).transpose(1, 2, 0).astype(np.uint8))
        ax[1][i].imshow(recon_denorm[i].transpose(1, 2, 0).astype(np.uint8))
    plt.savefig("tmp/tmp0.png")
    fig, ax = plt.subplots(5, per_slot_recon.shape[1])
    for i in range(per_slot_recon.shape[1]):
        for j in range(5):
            ax[j][i].imshow(per_slot_recon_denorm[j][i].transpose(1, 2, 0).astype(np.uint8))
    plt.savefig("tmp/tmp1.png")

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    arg_parser.add_argument("--log_dir", type=str)
    arg_parser.add_argument("--num_slots", type=int, default=7)
    arg_parser.add_argument("--learning_rate", type=float, default=0.0004)
    arg_parser.add_argument("--num_train_steps", type=int, default=500000)
    arg_parser.add_argument("--warmup_steps", type=int, default=10000)
    arg_parser.add_argument("--decay_rate", type=float, default=0.5)
    arg_parser.add_argument("--decay_steps", type=int, default=100000)
    arg_parser.add_argument("--eval", action="store_true", default=False)
    arg_parser.add_argument("--load_path", type=str)
    args = arg_parser.parse_args()
    if not args.eval:
        train(args)
    else:
        eval(args)
