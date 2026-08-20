# HW2 Setup — Actor-Critic / PPO with MetaWorld

## Venv

HW2 uses a **separate uv venv** inside this directory due to dependency conflicts with the
root project (legacy gym API, dm-control, MetaWorld). The root venv uses numpy 2.x and
gymnasium 1.x; HW2 needs gym (old API) and dm-control.

## Dependency notes

The original `conda_env.yml` specifies several packages that need modernizing for Apple Silicon
and uv compatibility:

| Original | Replacement | Reason |
|---|---|---|
| `mujoco_py==2.1.2.14` | `mujoco>=3.0` | mujoco_py has no Apple Silicon support |
| `gym==0.26.2` | `gym` + `gymnasium` | Keep gym for MetaWorld compat, add gymnasium |
| `cudatoolkit=11.5` | omit | CUDA-only, irrelevant on Mac |
| `numpy=1.24` | `numpy>=2.0` | Old pin unnecessary |
| `hydra-core==1.1.0` | `hydra-core>=1.3` | Updated API |

## mw.py patch required

`mw.py` has `import mujoco_py` at line 6 but **never calls any mujoco_py functions** — it is
an unused import left over from an earlier version. Remove it:

```python
# Delete this line from mw.py:
import mujoco_py
```

MetaWorld handles MuJoCo internally; mw.py itself does not need to call mujoco_py directly.

Also verify the MetaWorld import path after installation:
```python
from metaworld.envs.mujoco.env_dict import ALL_V2_ENVIRONMENTS  # mw.py line 17
```
If the chosen MetaWorld commit has moved this (newer versions use `metaworld.envs`), update
the import path in `MetaWorldEnv.__init__`.

## Python 3.14 compatibility note

Most dependencies (mujoco, dm-control, hydra-core, torch) are actively maintained and support
3.14. The main risk is `gym>=0.26.2` (the old unmaintained Gym package, last released 2022) —
it may not have 3.14-compatible wheels and could fail to install. If it does, fall back to
`uv venv --python 3.13` instead; do not go all the way back to 3.11.

## Steps to create the venv

```bash
cd exercises/stanford/cs224_spring_2026/hw2

# 1. Create pyproject.toml (see below)
# 2. Create and populate the venv
uv venv --python 3.14
uv sync
```

## pyproject.toml to create

```toml
[project]
name = "cs224r-hw2"
version = "0.1.0"
requires-python = ">=3.14"
dependencies = [
    "torch>=2.9.1",
    "numpy>=2.0",
    "hydra-core>=1.3.0",
    "omegaconf>=2.3.0",
    "mujoco>=3.0.0",
    "dm-control>=1.0.0",
    "gymnasium>=1.0.0",
    "gym>=0.26.2",
    "metaworld @ git+https://github.com/Farama-Foundation/Metaworld.git@a98086ababc81560772e27e7f63fe5d120c4cc50",
    "wandb",
    "imageio>=2.33.0",
    "imageio-ffmpeg",
    "opencv-python",
    "pytest",
    "ipykernel",
]

[build-system]
requires = ["setuptools>=69"]
build-backend = "setuptools.build_meta"
```

## IDE / PyLance

Create `.vscode/settings.json` in this directory:

```json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/exercises/stanford/cs224_spring_2026/hw2/.venv/bin/python",
  "python.analysis.extraPaths": [
    "${workspaceFolder}/exercises/stanford/cs224_spring_2026/hw2"
  ]
}
```

Also add a second `executionEnvironments` entry to the root `pyrightconfig.json`:

```json
{
  "root": "exercises/stanford/cs224_spring_2026/hw2",
  "pythonVersion": "3.11",
  "extraPaths": [
    "exercises/stanford/cs224_spring_2026/hw2",
    "exercises/stanford/cs224_spring_2026/hw2/.venv/lib/python3.14/site-packages"
  ]
}
```

## Running scripts

```bash
cd exercises/stanford/cs224_spring_2026/hw2

# Gridworld Q-learning (no MuJoCo needed — good for initial testing)
uv run python gridworld_q_learning.py

# On-policy (PPO) — requires MetaWorld + MuJoCo
uv run python train_on_policy.py

# Off-policy (Actor-Critic) — requires MetaWorld + MuJoCo
uv run python train_off_policy.py
```

## Notebooks

```python
import sys
sys.path.insert(0, "../../stanford/cs224_spring_2026/hw2")

import utils
from on_policy import ...
```

## Import style

All imports are flat (local modules imported directly by name):

```python
import mw
import utils
from logger import Logger
from replay_buffer import ReplayBufferStorage, make_replay_loader
from video import TrainVideoRecorder, VideoRecorder
```
