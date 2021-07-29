"""
Module for plotting learning curves
"""
import os
os.environ["KMP_WARNINGS"] = "off"
import logging
import matplotlib

matplotlib.use("pdf")
import matplotlib.pyplot as plt
import numpy as np
import math
import argparse
from commonroad_rl.utils_run.plot_util import smooth
from commonroad_rl.utils_run.plot_util import plot_results as plot_results_baselines
from commonroad_rl.utils_run.plot_util import load_results as load_results_baselines

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
LOGGER.addHandler(handler)

LATEX = False
LABELPAD = 15
FIGSIZE = (14, 12)

if LATEX:
    # use Latex font
    FONTSIZE = 28
    plt.rcParams['text.latex.preamble']=[r"\usepackage{lmodern}"]
    pgf_with_latex = {                      # setup matplotlib to use latex for output
        "pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
        "text.usetex": True,                # use LaTeX to write all text
        "font.family": 'lmodern',
        # blank entries should cause plots
        "font.sans-serif": [], #['Avant Garde'],              # to inherit fonts from the document
        # 'text.latex.unicode': True,
        "font.monospace": [],
        "axes.labelsize": FONTSIZE,               # LaTeX default is 10pt font.
        "font.size": FONTSIZE-10,
        "legend.fontsize": FONTSIZE,               # Make the legend/label fonts
        "xtick.labelsize": FONTSIZE,               # a little smaller
        "ytick.labelsize": FONTSIZE,
        "pgf.preamble": [
            r"\usepackage[utf8x]{inputenc}",    # use utf8 fonts
            r"\usepackage[T1]{fontenc}",         # plots will be generated
            r"\usepackage[detect-all,locale=DE]{siunitx}",
        ]                                   # using this preamble
    }
    matplotlib.rcParams.update(pgf_with_latex)

def argsparser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-f", "--log-folder", help="Log folder", type=str, default="log")
    parser.add_argument("--model_path", "-model", type=str, nargs="+", default=(),
                        help="(tuple) Relative path of the to be plotted model from the log folder")
    parser.add_argument("--legend_name", "-legend", type=str, nargs="+", default=(),
                        help="(tuple) Legend informations in the same order as the model_name")
    parser.add_argument("--no_render", "-nr", action="store_true", help="Whether to render images")
    parser.add_argument("-t", "--title", help="Figure title", type=str, default="result")
    # TODO: integrate sliding window size
    parser.add_argument("--smooth", action="store_true", help="Smooth learning curves (average around a sliding window)")

    return parser.parse_args()


def ts2reward(results):
    """
    Decompose a timesteps variable to x ans ys

    :param timesteps: (Pandas DataFrame) the input data
    :param xaxis: (str) the axis for the x and y output
        (can be X_TIMESTEPS='timesteps', X_EPISODES='episodes' or X_WALLTIME='walltime_hrs')
    :return: (np.ndarray, np.ndarray) the x and y output
    """

    x_var = np.cumsum(results.monitor.l * 1e-3)
    y_var = smooth(results.monitor.r.values, radius=1)
    # pandas.set_option("display.max_rows", None, "display.max_columns", None)
    # print(results.monitor)
    return x_var, y_var


def ts2goal(results):
    """
    Decompose a timesteps variable to x ans ys

    :param timesteps: (Pandas DataFrame) the input data
    :param xaxis: (str) the axis for the x and y output
        (can be X_TIMESTEPS='timesteps', X_EPISODES='episodes' or X_WALLTIME='walltime_hrs')
    :return: (np.ndarray, np.ndarray) the x and y output
    """

    x_var = np.cumsum(results.monitor.l.values * 1e-3)
    y_var = smooth(results.monitor.is_goal_reached.values, radius=1)

    return x_var, y_var


def ts2collision(results):
    """
    Decompose a timesteps variable to x ans ys

    :param timesteps: (Pandas DataFrame) the input data
    :param xaxis: (str) the axis for the x and y output
        (can be X_TIMESTEPS='timesteps', X_EPISODES='episodes' or X_WALLTIME='walltime_hrs')
    :return: (np.ndarray, np.ndarray) the x and y output
    """

    x_var = np.cumsum(results.monitor.l * 1e-3)
    if hasattr(results.monitor, "valid_collision"):
        y_var = smooth(results.monitor.valid_collision, radius=1)
    else:
        y_var = smooth(results.monitor.is_collision, radius=1)

    return x_var, y_var


def ts2rule_violation(results):
    """
    Decompose a timesteps variable to x ans ys

    :param timesteps: (Pandas DataFrame) the input data
    :param xaxis: (str) the axis for the x and y output
        (can be X_TIMESTEPS='timesteps', X_EPISODES='episodes' or X_WALLTIME='walltime_hrs')
    :return: (np.ndarray, np.ndarray) the x and y output
    """

    x_var = np.cumsum(results.monitor.l.values)
    y_var = results.monitor.num_traffic_rule_violation.values/results.monitor.l.values
    # y_var = []
    # for is_collision, num_traffic_rule_violation, ep_len in zip(results.monitor.is_collision.values,
    #     results.monitor.num_traffic_rule_violation.values, results.monitor.l.values
    # ):
    #     if is_collision:
    #         y_var.append(is_collision)
    #     else:
    #         y_var.append(num_traffic_rule_violation/ep_len)
    # y_var = np.array(y_var)

    return x_var, y_var

def ts2valid_rule_violation(results):
    """
    Decompose a timesteps variable to x ans ys

    :param timesteps: (Pandas DataFrame) the input data
    :param xaxis: (str) the axis for the x and y output
        (can be X_TIMESTEPS='timesteps', X_EPISODES='episodes' or X_WALLTIME='walltime_hrs')
    :return: (np.ndarray, np.ndarray) the x and y output
    """

    x_var = np.cumsum(results.monitor.l.values)
    # y_var = results.monitor.num_traffic_rule_violation.values/results.monitor.l.values
    y_var = []
    for is_collision, is_off_road, num_traffic_rule_violation, ep_len in zip(results.monitor.is_collision.values,
        results.monitor.is_off_road.values,
        results.monitor.num_traffic_rule_violation.values, results.monitor.l.values
    ):
        if is_off_road or is_collision:
            y_var.append(1)
        else:
            y_var.append(num_traffic_rule_violation/ep_len)
    y_var = np.array(y_var)

    return x_var, y_var


def ts2friction_violation(results):
    """
    Decompose a timesteps variable to x ans ys

    :param timesteps: (Pandas DataFrame) the input data
    :param xaxis: (str) the axis for the x and y output
        (can be X_TIMESTEPS='timesteps', X_EPISODES='episodes' or X_WALLTIME='walltime_hrs')
    :return: (np.ndarray, np.ndarray) the x and y output
    """

    # TODO: Implement xaxis
    x_var = np.cumsum(results.monitor.l.values * 1e-3)
    y_var = smooth(results.monitor.is_friction_violation.values, radius=50)

    return x_var, y_var


def ts2max_time(results):
    """
    Decompose a timesteps variable to x ans ys

    :param timesteps: (Pandas DataFrame) the input data
    :param xaxis: (str) the axis for the x and y output
        (can be X_TIMESTEPS='timesteps', X_EPISODES='episodes' or X_WALLTIME='walltime_hrs')
    :return: (np.ndarray, np.ndarray) the x and y output
    """

    # TODO: Implement xaxis
    x_var = np.cumsum(results.monitor.l.values * 1e-3)
    y_var = smooth(results.monitor.is_time_out.values, radius=1)

    return x_var, y_var


def ts2off_road(results):
    """
    Decompose a timesteps variable to x ans ys

    :param timesteps: (Pandas DataFrame) the input data
    :param xaxis: (str) the axis for the x and y output
        (can be X_TIMESTEPS='timesteps', X_EPISODES='episodes' or X_WALLTIME='walltime_hrs')
    :return: (np.ndarray, np.ndarray) the x and y output
    """

    x_var = np.cumsum(results.monitor.l * 1e-3)
    if hasattr(results.monitor, "valid_off_road"):
        y_var = smooth(results.monitor.valid_off_road.values, radius=1)
    else:
        y_var = smooth(results.monitor.is_off_road.values, radius=1)

    return x_var, y_var


def ts2monitor_reward(results):
    """
    Decompose a timesteps variable to x ans ys

    :param timesteps: (Pandas DataFrame) the input data
    :param xaxis: (str) the axis for the x and y output
        (can be X_TIMESTEPS='timesteps', X_EPISODES='episodes' or X_WALLTIME='walltime_hrs')
    :return: (np.ndarray, np.ndarray) the x and y output
    """

    x_var = np.cumsum(results.monitor.l)
    y_var = smooth(results.monitor.total_monitor_reward.values, radius=50)

    return x_var, y_var


def ts2gym_reward(results):
    """
    Decompose a timesteps variable to x ans ys

    :param timesteps: (Pandas DataFrame) the input data
    :param xaxis: (str) the axis for the x and y output
        (can be X_TIMESTEPS='timesteps', X_EPISODES='episodes' or X_WALLTIME='walltime_hrs')
    :return: (np.ndarray, np.ndarray) the x and y output
    """

    x_var = np.cumsum(results.monitor.l)
    y_var = smooth(results.monitor.total_gym_reward.values, radius=50)

    return x_var, y_var


def ts2gym_reward_step(results):
    """
    Decompose a timesteps variable to x ans ys

    :param timesteps: (Pandas DataFrame) the input data
    :param xaxis: (str) the axis for the x and y output
        (can be X_TIMESTEPS='timesteps', X_EPISODES='episodes' or X_WALLTIME='walltime_hrs')
    :return: (np.ndarray, np.ndarray) the x and y output
    """

    x_var = np.cumsum(results.monitor.l)
    y_var = results.monitor.total_gym_reward.values/results.monitor.l.values

    return x_var, y_var


def ts2monitor_reward_step(results):
    """
    Decompose a timesteps variable to x ans ys

    :param timesteps: (Pandas DataFrame) the input data
    :param xaxis: (str) the axis for the x and y output
        (can be X_TIMESTEPS='timesteps', X_EPISODES='episodes' or X_WALLTIME='walltime_hrs')
    :return: (np.ndarray, np.ndarray) the x and y output
    """
    x_var = np.cumsum(results.monitor.l)
    y_var = results.monitor.total_monitor_reward.values / results.monitor.l.values
    print(f"min robustness = {np.min(results.monitor.min_robustness.values)}")
    print(f"max robustness = {np.max(results.monitor.max_robustness.values)}")

    return x_var, y_var


def ts2u1(results):
    """
    Decompose a timesteps variable to x ans ys

    :param timesteps: (Pandas DataFrame) the input data
    :param xaxis: (str) the axis for the x and y output
        (can be X_TIMESTEPS='timesteps', X_EPISODES='episodes' or X_WALLTIME='walltime_hrs')
    :return: (np.ndarray, np.ndarray) the x and y output
    """

    x_var = np.cumsum(results.monitor.l.values) * 1e-3
    y_var = np.abs(results.monitor.u_cbf_1.values)
    return x_var, y_var


def ts2u2(results):
    """
    Decompose a timesteps variable to x ans ys

    :param timesteps: (Pandas DataFrame) the input data
    :param xaxis: (str) the axis for the x and y output
        (can be X_TIMESTEPS='timesteps', X_EPISODES='episodes' or X_WALLTIME='walltime_hrs')
    :return: (np.ndarray, np.ndarray) the x and y output
    """

    x_var = np.cumsum(results.monitor.l.values) * 1e-3
    y_var = np.abs(results.monitor.u_cbf_2.values)
    return x_var, y_var


def ts2active_step_robustness(results):
    """
    Decompose a timesteps variable to x ans ys

    :param timesteps: (Pandas DataFrame) the input data
    :param xaxis: (str) the axis for the x and y output
        (can be X_TIMESTEPS='timesteps', X_EPISODES='episodes' or X_WALLTIME='walltime_hrs')
    :return: (np.ndarray, np.ndarray) the x and y output
    """

    x_var = np.cumsum(results.monitor.l.values)
    y_var = []
    for is_off_road, is_collision, ep_len, total_robustness in zip(
            results.monitor.is_off_road.values, results.monitor.is_collision.values, results.monitor.l.values,
            results.monitor.total_monitor_reward.values
    ):
        if is_off_road or is_collision:
            y_var.append(0.)
        else:
            y_var.append(total_robustness/ep_len)
    y_var = np.array(y_var)

    return x_var, np.array(y_var)


def ts2active_total_robustness(results):
    """
    Decompose a timesteps variable to x ans ys

    :param timesteps: (Pandas DataFrame) the input data
    :param xaxis: (str) the axis for the x and y output
        (can be X_TIMESTEPS='timesteps', X_EPISODES='episodes' or X_WALLTIME='walltime_hrs')
    :return: (np.ndarray, np.ndarray) the x and y output
    """

    x_var = np.cumsum(results.monitor.l.values)
    y_var = []
    for is_off_road, is_collision, ep_len, total_robustness in zip(
            results.monitor.is_off_road.values, results.monitor.is_collision.values, results.monitor.l.values,
            results.monitor.total_monitor_reward.values
    ):
        if is_off_road or is_collision:
            y_var.append(0.)
        else:
            y_var.append(total_robustness)
    y_var = np.array(y_var)

    return x_var, y_var


def ts2min_robustness(results):
    """
    Decompose a timesteps variable to x ans ys

    :param timesteps: (Pandas DataFrame) the input data
    :param xaxis: (str) the axis for the x and y output
        (can be X_TIMESTEPS='timesteps', X_EPISODES='episodes' or X_WALLTIME='walltime_hrs')
    :return: (np.ndarray, np.ndarray) the x and y output
    """

    x_var = np.cumsum(results.monitor.l.values) * 1e-3
    y_var = results.monitor.min_robustness.values
    print(f"min robustness = {np.min(results.monitor.min_robustness.values)}")
    return x_var, y_var


def ts2max_robustness(results):
    """
    Decompose a timesteps variable to x ans ys

    :param timesteps: (Pandas DataFrame) the input data
    :param xaxis: (str) the axis for the x and y output
        (can be X_TIMESTEPS='timesteps', X_EPISODES='episodes' or X_WALLTIME='walltime_hrs')
    :return: (np.ndarray, np.ndarray) the x and y output
    """

    x_var = np.cumsum(results.monitor.l.values) * 1e-3
    y_var = results.monitor.max_robustness.values
    # print(f"max robustness = {np.max(results.monitor[results.monitor.max_robustness < 0.9].max_robustness.values)}")
    print(f"max robustness = {np.max(results.monitor.max_robustness.values)}")
    return x_var, y_var


def violation2step_robustness(results):
    plt.figure()
    # x_var = results.monitor.num_traffic_rule_violation.values / results.monitor.l.values
    x_var = []
    for is_off_road, is_collision, ep_len, rule_violation in zip(
            results.monitor.is_off_road.values, results.monitor.is_collision.values, results.monitor.l.values,
            results.monitor.num_traffic_rule_violation.values
    ):
        if is_off_road or is_collision:
            x_var.append(0.)
        else:
            x_var.append(rule_violation/ep_len)
    x_var = np.array(x_var)
    y_var = []
    for is_off_road, is_collision, ep_len, total_robustness in zip(
            results.monitor.is_off_road.values, results.monitor.is_collision.values, results.monitor.l.values,
            results.monitor.total_monitor_reward.values
    ):
        if is_off_road or is_collision:
            y_var.append(0.)
        else:
            y_var.append(total_robustness/ep_len)
    y_var = np.array(y_var)
    plt.plot(x_var, y_var, "x")
    plt.savefig("bla.png")
    return x_var, y_var


def violation2step_robustness_tmp(results):
    plt.figure()
    # x_var = results.monitor.num_traffic_rule_violation.values / results.monitor.l.values
    x_var = []
    for ep_len, rule_violation in zip(
            results.monitor.l.values,
            results.monitor.num_traffic_rule_violation.values
    ):
        x_var.append(rule_violation/ep_len)
    x_var = np.array(x_var)
    y_var = []
    for ep_len, total_robustness in zip(
            results.monitor.l.values,
            results.monitor.total_monitor_reward.values
    ):
        y_var.append(total_robustness/ep_len)
    y_var = np.array(y_var)
    plt.plot(x_var, y_var, "x")
    plt.xlabel("rule violation rate")
    plt.ylabel("avg. step robustness")
    plt.savefig("bla.png")
    return x_var, y_var

PLOT_DICT = {
    # "Mean Reward": ts2ep_rew_mean,
    "Total Reward": ts2reward,
    "Goal-Reaching Rate": ts2goal,
    "Collision Rate": ts2collision,
    "Off-Road Rate": ts2off_road,
    "Time-Out Rate": ts2max_time,
    # "Total Robustness reward": ts2monitor_reward,
    # "Total Sparse reward": ts2gym_reward,
    # "Min Robustness": ts2min_robustness,
    # "Max Robustness": ts2max_robustness,
    # "Avg. Step Robustness reward": ts2monitor_reward_step,
    # "Avg. Step Sparse reward": ts2gym_reward_step,
    # "True traffic rule violation": ts2rule_violation,
    # "Valid traffic rule violation": ts2valid_rule_violation,
    # "Active step robustness reward": ts2active_step_robustness,
    # "Active total robustness reward": ts2active_total_robustness,
    # "Step robustness reward vs num violation": violation2step_robustness_tmp,
    # "$|u_1 - u_\mathrm{RL1}|$   [rad/$\mathrm{s}^2$]": ts2u1,
    # "$|u_2 - u_\mathrm{RL2}|$   [m/$\mathrm{s}^2$]": ts2u2
    # "Friction violation": ts2friction_violation,
}


def group_fn(results):
    return os.path.basename(results.dirname)


def main():
    args = argsparser()

    log_dir = args.log_folder
    model_paths = tuple(args.model_path)
    legend_names = tuple(args.legend_name)

    num_of_columns = 2
    num_of_rows = math.ceil(len(PLOT_DICT) / num_of_columns)

    for idx, model in enumerate(model_paths):
        fig, axarr = plt.subplots(num_of_rows, num_of_columns, sharex=False, squeeze=True, figsize=FIGSIZE, dpi=100)
        results = load_results_baselines(os.path.join(log_dir, model))

        for i, (k, xy_fn) in enumerate(PLOT_DICT.items()):
            legend = i == 0
            plot_line = i >1
            try:
                fig, axarr = plot_results_baselines(results, fig, axarr,
                                                    nrows=num_of_rows, ncols=num_of_columns, xy_fn=xy_fn,
                                                    idx_row=i//num_of_columns, idx_col=i%num_of_columns,
                                                    average_group=False, resample=args.smooth,
                                                    group_fn=group_fn,
                                                    xlabel="Training Steps * 1000", ylabel=k,
                                                    # xlabel="\\textbf{Training Steps * 1000}",
                                                    # ylabel="\\textbf{" + k + "}",
                                                    labelpad=LABELPAD,
                                                    legend_outside=True,
                                                    legend=legend, plot_line=plot_line)
            except AttributeError:
                continue
        format = "pdf"
        plt.tight_layout()
        # plt.show()
        fig.savefig(os.path.join(log_dir, model, f"{model}.{format}"), format=format, bbox_inches='tight')
        LOGGER.info(f"Saved {model}.{format} to {log_dir}/{model}")#, figure, os.path.join(log_dir, model))


if __name__ == "__main__":
    main()
