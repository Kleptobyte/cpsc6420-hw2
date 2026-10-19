# CPSC 6420: Homework 2

Q-learning in CliffWalking. Learning rate, discount factor, exploration decay,
and action noise.

Requires Python 3.12. CPU only, no display needed.

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python experiments.py
```

Parameters in `Config`
(`q_learning.py`) and `CONFIGS` (`experiments.py`).

CSVs, plots, and video saved in `results/`.
[Video](results/selected_policy.mp4)
