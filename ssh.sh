#!/usr/bin/env bash
# run like `source ssh.sh` on tmp runpod, after sending your private key and this script

if [[ -z "${1-}" ]]; then
    PRIVATE_KEY="id_ed25519"
else
    PRIVATE_KEY="$1"
fi

# start ssh agent, add key, go to /workspace
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/$PRIVATE_KEY

if [[ -n "$SSH_CONNECTION" && -d /workspace/ ]]; then
  echo "🐧 Running on remote runpod with storage attached - moving to /workspace"
  cd /workspace
fi

# ensure we have git, clone repo, cd in etc.
apt-get update && apt-get install -y git
git clone git@github.com:AdamBeedell/MLXMultimodal.git || true
cd MLXMultimodal
git pull
git status

# chain into setup.sh
echo "Chaining into setup.sh..."
source setup.sh
