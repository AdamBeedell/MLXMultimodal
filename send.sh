#!/usr/bin/env bash
# run like `./send.sh` on local, to prepare remote to continue setup

if [[ -z "${1-}" ]]; then
    REMOTE="mlx"
else
    REMOTE="$1"
fi

# move private key, ssh.sh script and .env file to remote
scp ~/.ssh/id_ed25519 "$REMOTE:~/.ssh/id_ed25519"
scp ssh.sh "$REMOTE:ssh.sh"
scp .env "$REMOTE:.env"
