import sys, os
import numpy as np
import pandas
import matplotlib.pyplot as plt


if __name__ == '__main__':
    option = sys.argv[1]
    log_paths = sys.argv[2:]
    window = 20
    def get_item(log_file, label):
        data = pandas.read_csv(log_file, index_col=None, comment='#', error_bad_lines=True)
        return data[label].values
    def smooth(array, window):
        out = np.zeros(array.shape[0] - window)
        for i in range(out.shape[0]):
            out[i] = np.mean(array[i:i + window])
        return out
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    for log_path in log_paths:
        progress_file = os.path.join(log_path, 'progress.csv')
        data = get_item(progress_file, option)
        total_timesteps = get_item(progress_file, 'total_timesteps')
        # success_rate = smooth(success_rate, window)
        # total_timesteps = smooth(total_timesteps, window)
        ax.plot(smooth(total_timesteps, window), smooth(data, window), label=log_path)
    ax.set_title(option)
    ax.grid()
    plt.legend()
    plt.savefig("figure.png")
    plt.show()
    
