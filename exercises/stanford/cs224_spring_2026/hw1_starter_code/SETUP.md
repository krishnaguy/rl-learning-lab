# HW1 Setup — Flappy Bird Imitation Learning

## Venv

HW1 uses the **root project venv** at `/path/to/rl-learning-lab/.venv`. No separate venv needed.

## Dependencies

All required packages (torch, gymnasium, imageio, numpy) are already in the root venv.
`pygame` was added to the root `pyproject.toml` and installed via `uv sync`.

```bash
# From repo root
uv sync
```

## Running scripts

```bash
# From repo root
uv run exercises/stanford/cs224_spring_2026/hw1_starter_code/main.py --method bc_reg

# Or from the hw1 directory (flat imports resolve because you're in the directory)
cd exercises/stanford/cs224_spring_2026/hw1_starter_code
uv run main.py --method bc_reg
```

## IDE / PyLance

The hw1 directory is added to `extraPaths` in both `pyrightconfig.json` and `.vscode/settings.json`
at the repo root. This lets PyLance resolve flat imports like `from networks import BCPolicy`
without needing the files installed as a package.

## Notebooks

To import hw1 modules from a notebook (e.g. in `exercises/notebooks/`):

```python
import sys
sys.path.insert(0, "../../stanford/cs224_spring_2026/hw1_starter_code")

from networks import BCPolicy
from losses import mse_loss
```

## Import style

All imports in hw1 are flat (no `cs224r.` namespace), e.g.:

```python
from networks import BCPolicy, FlowMatchingPolicy
from losses import mse_loss, flow_matching_loss
from expert import collect_expert_data
```

The `setup.py` in this directory declares `packages=['cs224r']` but no such directory exists —
it is a leftover from the starter code template and can be ignored.
