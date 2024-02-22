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
${APT_GET_INSTALL} python3.11-full

# create alias for python3.11
echo 'alias python="/usr/bin/python3.11"' >> /root/.bashrc
source /root/.bashrc
ln -s /usr/bin/python3.11 /usr/local/bin/python

# clone repo to get requirements.txt
git clone https://github.com/PTB-MR/mrpro --depth 1 /opt/mrpro
python -m ensurepip --upgrade
python -m pip install --upgrade pip
# pre-install cpu-version of torch to avoid installation of cuda-version via requirements.txt
python -m pip install --no-cache-dir torch==2.2.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
# exclude any nvidia packages and triton to save space
grep -v '^ *#\|^nvidia*\|^triton*' /opt/mrpro/binder/requirements.txt | grep . > /opt/mrpro/binder/requirements_slim.txt
python -m pip install --no-cache-dir -r /opt/mrpro/binder/requirements_slim.txt
rm -r /opt/mrpro

apt-get clean && rm -rf /var/lib/apt/lists/*
