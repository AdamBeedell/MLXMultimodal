#!/usr/bin/env bash
# run like `source setup.sh` on any remote to ensure active shell is set up with venv

# ensure we have all the utils we need
apt-get update
apt-get install -y vim rsync git nvtop htop tmux curl ca-certificates git-lfs lsof nano
apt-get upgrade -y

# get env vars sent via send script and load into shell
cp ~/.env .env
set -a
source .env
set +a

# install uv (and setup custom cache dirs if we have runpod storage)
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
if [[ -n "$SSH_CONNECTION" && -d /workspace/ ]]; then
  echo "🐧 Running on remote runpod with storage attached - setting custom uv/hf cache dir"
  mkdir -p /workspace/.cache/uv
  mkdir -p /workspace/.cache/datasets_cache
  set -a
  export UV_CACHE_DIR="/workspace/.cache/uv"
  export HF_DATASETS_CACHE="/workspace/.cache/datasets_cache"
  set +a
fi

uv venv --system-site-packages

# install python packages (using nightly index for latest torch if beast mode enabled)
if [[ "$BEAST_MODE" == "1" ]]; then
  echo "🔥 BEAST_MODE enabled - using nightly config for torch prereleases"
  uv sync --prerelease=allow --config-file uv.nightly.toml
else
  uv sync
fi

source .venv/bin/activate
