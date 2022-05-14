from cProfile import run
from email.mime import base
import os
import math
import copy
import argparse
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

index_label = "base_env"
column_label = "target_env"

def plot_matrix(matrix_df, val_label, title, save_path, vmin=0, vmax=100):
    matrix_pivot = matrix_df.pivot(index=index_label,
                                    columns=column_label,
                                    values=val_label)
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax_divider = make_axes_locatable(ax)
    cbar_ax = ax_divider.append_axes("right", size="7%", pad="10%")
    cmap = copy.copy(mpl.cm.get_cmap("RdYlGn"))
    cmap.set_bad("black")
    sns.heatmap(
        matrix_pivot,
        annot=True,
        # cmap="RdYlGn_r",
        cmap=cmap,
        fmt=".2f",
        square=True,
        ax=ax,
        vmin=vmin,
        vmax=vmax,
        cbar_ax=cbar_ax,
        linewidths=0.003,
        rasterized=False,
    )

    ax.set(
        title=title,
        xlabel="Target Environment",
        ylabel="Base Environment",)

    # set_spines_visible(ax)
    ax.figure.tight_layout()
    fig.tight_layout()
    fig.savefig(save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e',
                        '--exp_dir',
                        required=True,
                        help='Experiment directory of the matrix experiment.')

    # Read in experiment log file
    opt = parser.parse_args()
    exp_dir = opt.exp_dir

    matrix = {
        "base_env": [],
        "target_env": [],
        "first_5M_avg_reward": [],
        "second_5M_avg_reward": [],
    }

    upper_bound = int(1e7)

    for run_csv in os.listdir(exp_dir):
        print(run_csv)

        if not run_csv.endswith(".csv"):
            continue

        base_env = run_csv.split("_")[1]
        target_env = run_csv.split("_")[3]

        run_full_path = os.path.join(exp_dir, run_csv)
        data = pd.read_csv(run_full_path)

        episode_rewards = data["Value"]
        env_steps = data["Step"]

        # Find split point
        for split, env_step in enumerate(env_steps):
            if int(env_step) > int(5e6):
                break


        first_5M_avg_reward = np.mean(episode_rewards.iloc[:split+1])
        second_5M_avg_reward = np.mean(episode_rewards.iloc[split+1:])

        matrix["base_env"].append(base_env)
        matrix["target_env"].append(target_env)
        matrix["first_5M_avg_reward"].append(first_5M_avg_reward)
        matrix["second_5M_avg_reward"].append(second_5M_avg_reward)

    matrix_df = pd.DataFrame(matrix)

    # make the plot
    # Write the map for the cell fitness

    plot_matrix(
        matrix_df,
        "first_5M_avg_reward",
        title="Atari Planes Env Avg Reward (First 5M Env Steps)",
        save_path=os.path.join(exp_dir, "first_5M.png"),
    )

    plot_matrix(
        matrix_df,
        "second_5M_avg_reward",
        title="Atari Planes Env Avg Reward (Second 5M Env Steps)",
        save_path=os.path.join(exp_dir, "second_5M.png"),
    )