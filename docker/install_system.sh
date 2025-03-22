# Commands
APT_GET_INSTALL="apt-get install -yq --no-install-recommends"

# update, qq: quiet
apt-get update -qq

# ensure certificates are up to date
${APT_GET_INSTALL} --reinstall ca-certificates

# base utilities
${APT_GET_INSTALL} git software-properties-common gpg-agent curl jq

# add repo for python installation
add-apt-repository ppa:deadsnakes/ppa
apt update -qq

${APT_GET_INSTALL} $PYTHON-full

# pip
if [[ "$PYTHON" == "python3.10" ]]; then
    # System python on ubuntu does not support ensurepip
    ${APT_GET_INSTALL} python3-pip
else
    $PYTHON -m ensurepip --upgrade
fi
$PYTHON -m pip install --upgrade pip --no-cache-dir

# create alias for installed python version
ln -s /usr/bin/$PYTHON /usr/local/bin/python
ln -s /usr/bin/$PYTHON /usr/local/bin/python3
ln -s /usr/local/bin/pip3 /usr/local/bin/pip

# clean up
apt-get clean && rm -rf /var/lib/apt/lists/*
rm -rf /root/.cache

# add user runner
adduser --disabled-password --gecos "" --uid 1001 runner \
    && groupadd docker --gid 123 \
    && usermod -aG sudo runner \
    && usermod -aG docker runner \
    && echo "%sudo   ALL=(ALL:ALL) NOPASSWD:ALL" > /etc/sudoers \
    && echo "Defaults env_keep += \"DEBIAN_FRONTEND\"" >> /etc/sudoers
