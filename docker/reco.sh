# define apt-get installation command
APT_GET_INSTALL="apt-get install -yq --no-install-recommends"

# update, qq: quiet
apt-get update -qq

# ensure certificates are up to date
${APT_GET_INSTALL} --reinstall ca-certificates

# base utilities
${APT_GET_INSTALL} git software-properties-common gpg-agent

# add repo for python installation
add-apt-repository ppa:deadsnakes/ppa
apt update -qq
${APT_GET_INSTALL} $PYTHON-full

# create alias for installed python version
ln -s /usr/bin/$PYTHON /usr/local/bin/python
ln -s /usr/bin/$PYTHON /usr/local/bin/python3

pip install matplotlib

# clone repo to get requirements
git clone https://github.com/PTB-MR/mrpro --depth 1 /opt/mrpro
cd /opt/mrpro
python -m ensurepip --upgrade
python -m pip install --upgrade pip

# create alias to ensure pip works in the same way as pip3
ln -s /usr/local/bin/pip3 /usr/local/bin/pip

# pre-install cpu-version of torch to avoid installation of cuda-version via dependencies
python -m pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# install mrpro and dependencies
python -m pip install --upgrade --upgrade-strategy -e .[notebook,test,docs]

# clean up
rm -r /opt/mrpro
apt-get clean && rm -rf /var/lib/apt/lists/*

# add user runner
adduser --disabled-password --gecos "" --uid 1001 runner \
    && groupadd docker --gid 123 \
    && usermod -aG sudo runner \
    && usermod -aG docker runner \
    && echo "%sudo   ALL=(ALL:ALL) NOPASSWD:ALL" > /etc/sudoers \
    && echo "Defaults env_keep += \"DEBIAN_FRONTEND\"" >> /etc/sudoers
