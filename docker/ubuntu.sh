# define apt-get installation command
APT_GET_INSTALL="apt-get install -yq --no-install-recommends"

# update, qq: quiet
apt-get update -qq
${APT_GET_INSTALL} apt-utils locales

# ensure certificates are up to date
${APT_GET_INSTALL} --reinstall ca-certificates

# base utilities
${APT_GET_INSTALL} build-essential python3-dev wget git tmux zsh vim htop unzip

apt-get clean
