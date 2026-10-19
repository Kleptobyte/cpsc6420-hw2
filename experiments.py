"""Q-learning runs, results, plots, and video."""

import csv
from dataclasses import replace
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np

from q_learning import Config, episode, evaluate, make_env, train


OUT = Path(__file__).resolve().parent / "results"
SEED = 11
EVAL_EPISODES = 30
BASELINE = Config()
CONFIGS = {
    "baseline": BASELINE,
    "alpha_0p1": replace(BASELINE, alpha=0.1),
    "alpha_0p9": replace(BASELINE, alpha=0.9),
    "gamma_0p5": replace(BASELINE, gamma=0.5),
    "gamma_0p9": replace(BASELINE, gamma=0.9),
    "decay_0p98": replace(BASELINE, epsilon_decay=0.98),
    "decay_0p999": replace(BASELINE, epsilon_decay=0.999),
}


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def evaluate_policy(condition, q, noise_p, config):
    rows = evaluate(q, noise_p, config.max_steps, EVAL_EPISODES)
    return {"condition": condition, "noise_p": noise_p, "episodes": len(rows),
            "mean_reward": float(np.mean([r["reward"] for r in rows])),
            "success_rate": float(np.mean([r["success"] for r in rows]))}


def rolling(rewards, window=100):
    cumulative = np.r_[0.0, np.cumsum(rewards)]
    ends = np.arange(1, len(rewards) + 1)
    starts = np.maximum(0, ends - window)
    return (cumulative[ends] - cumulative[starts]) / (ends - starts)


def save_plot(fig, name):
    fig.savefig(OUT / "figures" / f"{name}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_training(series, title, filename):
    fig, ax = plt.subplots(figsize=(6.8, 3.8), layout="constrained")
    lower = 0.0
    for label, rewards in series:
        curve = rolling(rewards)
        lower = min(lower, float(curve.min()))
        x = np.arange(1, len(curve) + 1)
        keep = (x <= 10) | (x % 10 == 0)
        ax.plot(x[keep], curve[keep], label=label)
    ax.set(xlabel="Training episode", ylabel="Return (100-episode trailing mean)",
           title=title, xlim=(1, BASELINE.episodes),
           ylim=(500 * np.floor(lower / 500), 0))
    ax.grid(alpha=0.2)
    ax.legend()
    save_plot(fig, filename)


def create_figures(histories, selected, evaluation):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False})
    plot_training([("Baseline", histories["baseline"])],
                  "Baseline Q-learning", "baseline_learning")
    comparisons = [
        ("alpha_comparison", "Learning rate", "alpha",
         ["alpha_0p1", "baseline", "alpha_0p9"]),
        ("gamma_comparison", "Discount factor", "gamma",
         ["gamma_0p5", "gamma_0p9", "baseline"]),
        ("epsilon_comparison", "Exploration decay", "epsilon_decay",
         ["decay_0p98", "baseline", "decay_0p999"]),
    ]
    for filename, title, parameter, names in comparisons:
        plot_training([(f"{parameter} = {getattr(CONFIGS[n], parameter):g}",
                        histories[n]) for n in names], title, filename)
    plot_training([("p = 0", histories[selected]),
                   ("p = 0.10", histories["noise_0p1"]),
                   ("p = 0.25", histories["noise_0p25"])],
                  "Training with action noise", "stochastic_learning")

    zero_condition = "baseline" if selected == "baseline" else "selected"
    zero = next(r["mean_reward"] for r in evaluation
                if r["condition"] == zero_condition)
    fig, ax = plt.subplots(figsize=(6.8, 3.8), layout="constrained")
    lower = zero
    for condition, label in [("original", "Original"), ("retrained", "Retrained")]:
        rewards = [zero] + [r["mean_reward"] for r in evaluation
                            if r["condition"] == condition]
        lower = min(lower, min(rewards))
        ax.plot([0, 0.1, 0.25], rewards, marker="o", label=label)
    ax.set(xlabel="Evaluation noise p", ylabel="Mean reward",
           title="Greedy evaluation under action noise", xticks=[0, 0.1, 0.25],
           xlim=(0, 0.25), ylim=(50 * np.floor(lower / 50), 0))
    ax.grid(alpha=0.2)
    ax.legend()
    save_plot(fig, "robustness")


def record_video(q, config):
    env = make_env(0.0, config.max_steps, render_mode="rgb_array")
    frames = []
    try:
        episode(env, q, 700000, 800000, frames=frames)
    finally:
        env.close()
    with imageio.get_writer(OUT / "selected_policy.mp4", fps=8, codec="libx264",
                            macro_block_size=None, pixelformat="yuv420p") as writer:
        for frame in [frames[0]] * 8 + frames + [frames[-1]] * 16:
            writer.append_data(frame)


def main():
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    qs, histories = {}, {}
    for name, config in CONFIGS.items():
        qs[name], histories[name] = train(config, SEED)
    # First configuration wins an exact tie.
    selected = max(CONFIGS, key=lambda n: np.mean(histories[n][-100:]))
    config, q = CONFIGS[selected], qs[selected]
    evaluation = [evaluate_policy("baseline", qs["baseline"], 0.0, BASELINE)]
    if selected != "baseline":
        evaluation.append(evaluate_policy("selected", q, 0.0, config))
    for p in [0.1, 0.25]:
        name = f"noise_{str(p).replace('.', 'p')}"
        noisy_q, histories[name] = train(config, SEED, p)
        evaluation.append(evaluate_policy("original", q, p, config))
        evaluation.append(evaluate_policy("retrained", noisy_q, p, config))
    write_csv(OUT / "training.csv",
              [{"config": name, "episode": i + 1, "reward": reward}
               for name, rewards in histories.items() for i, reward in enumerate(rewards)])
    write_csv(OUT / "evaluation.csv", evaluation)
    create_figures(histories, selected, evaluation)
    record_video(q, config)


if __name__ == "__main__":
    main()
