# flightdelay-sjsu

## Setup

```bash
# Install uv, on macOS and Linux.
curl -LsSf https://astral.sh/uv/install.sh | sh

# set up venv
uv run main.py
source .venv/bin/activate

# To use ipynb notebooks, run this
uv run ipython kernel install --user --name=uv

# To add new packages
uv add <package_name>
# try this if uv doesn't install it
uv sync --active

```

