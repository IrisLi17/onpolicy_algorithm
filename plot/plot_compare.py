import sys, os
import numpy as np
import pandas
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


if __name__ == '__main__':
    option = sys.argv[1]
    log_paths = sys.argv[2:]
    # assert option in ['success_rate', 'eval', 'entropy', 'aug_ratio', 'self_aug_ratio']
    window = 1
    L = 10000
    def get_item(log_file, label):
        data = pandas.read_csv(log_file, index_col=None, comment='#', error_bad_lines=True)
        return data[label].values
    def smooth(array, window):
        out = np.zeros(array.shape[0] - window)
        for i in range(out.shape[0]):
            out[i] = np.mean(array[i:i + window])
        return out
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    for idx, log_path in enumerate(log_paths):
        progress_file = os.path.join(log_path, 'progress.csv')
        eval_file = os.path.join(log_path, 'eval.csv')
        # if 'ds' in log_path:
        #     success_rate = get_item(progress_file, 'ep_success_rate')
        # else:
        # success_rate = get_item(progress_file, 'is_success')
        total_timesteps = get_item(progress_file, 'total_timesteps')
        try:
            eval_reward = get_item(eval_file, 'mean_eval_reward')
            n_update = get_item(eval_file, 'n_updates')
        except:
            pass
        # success_rate = smooth(success_rate, window)
        # total_timesteps = smooth(total_timesteps, window)
        if option == 'success_rate':
            print("label", log_path[-50:])
            success_rate = get_item(progress_file, "is_success")
            if idx == 0:
                success_rate = success_rate[0:]
            ax.plot(smooth(total_timesteps, window), smooth(success_rate, window), label=log_path[-50:])
        elif option == 'eval':
            # ax[0].plot(n_updates*65536, eval_reward, label=log_path)
            
            ax[0].plot(smooth(total_timesteps[n_updates-1], window), smooth(eval_reward, window), label=log_path)
        # elif option == 'entropy':
        #     entropy = get_item(progress_file, 'policy_entropy')
        #     ax[0].plot(smooth(total_timesteps, window), smooth(entropy, window), label=log_path)
        elif option == 'aug_ratio':
            original_success = get_item(progress_file, 'original_success')
            total_success = get_item(progress_file, 'total_success')
            aug_ratio = (total_success - original_success) / (total_success + 1e-8)
            print(total_timesteps.shape, aug_ratio.shape)
            ax[0].plot(smooth(total_timesteps, 2), smooth(aug_ratio, 2), label=log_path)
        elif option == 'self_aug_ratio':
            self_aug_ratio = get_item(progress_file, 'self_aug_ratio')
            ax[0].plot(smooth(total_timesteps, window), smooth(self_aug_ratio, window), label=log_path)
        else:
            values = get_item(progress_file, option)
            # ax.plot(n_update, values, label=log_path[-50:])
            ax.plot(total_timesteps[:L], values[:L], label=log_path[-50:])
        try:
            original_steps = get_item(progress_file, 'original_timesteps')[0]
            augment_steps = get_item(progress_file, 'augment_steps') / original_steps
            # augment_steps = smooth(augment_steps, window)
        except:
            augment_steps = np.zeros(total_timesteps.shape)
        # ax[1].plot(smooth(total_timesteps, window), smooth(augment_steps, window), label=log_path)
    if option == 'success_rate':
        ax.set_title('success rate')
    elif option == 'eval':
        ax[0].set_title('eval success rate')
    # elif option == 'entropy':
    #     ax[0].set_title('entropy')
    elif option == 'aug_ratio':
        ax[0].set_title('aug success episode / total success episode')
    elif option == 'self_aug_ratio':
        ax[0].set_title('self_aug_ratio')
    # ax[1].set_title('augment steps / original rollout steps')
    ax.grid()
    # ax[1].grid()
    plt.legend(loc="best")
    plt.savefig("figure.png")
    # plt.show()
    
