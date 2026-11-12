#!/bin/bash
set -euo pipefail

APT_GET_INSTALL="apt-get install -yq --no-install-recommends"

# update, qq: quiet
apt-get update -qq

# ensure certificates are up to date
${APT_GET_INSTALL} --reinstall ca-certificates

# base utilities
${APT_GET_INSTALL} git curl jq build-essential sudo

# python (distro default), plus `python` -> python3 alias
${APT_GET_INSTALL} python3 python3-dev python3-venv python3-pip python-is-python3

# clean up
apt-get clean && rm -rf /var/lib/apt/lists/*
rm -rf /root/.cache

# add user runner
useradd --create-home --shell /bin/bash --uid 1001 runner \
    && groupadd --gid 123 docker \
    && usermod -aG sudo,docker runner \
    && echo "%sudo ALL=(ALL:ALL) NOPASSWD:ALL" > /etc/sudoers.d/runner \
    && echo 'Defaults env_keep += "DEBIAN_FRONTEND"' >> /etc/sudoers.d/runner \
    && chmod 0440 /etc/sudoers.d/runner
