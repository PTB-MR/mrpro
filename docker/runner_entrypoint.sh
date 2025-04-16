#!/bin/sh
# the 1st argument must be the runner registration token
# (see  https://docs.github.com/en/rest/actions/self-hosted-runners?apiVersion=2022-11-28#create-a-registration-token-for-an-organization 
#  example in .github/slurm_dispatcher.py)
if [ -z ${1+x} ]; then
    echo "The runner registration token is missing" 1>&2
    exit 1
fi

# if the 2nd argument for the script is given: run in ephemeral mode (one runner takes one job)
# constructing the unique name for the runner (see .github/slurm_dispatcher.py)
./config.sh \
    --name $(hostname)-${2:-default} \
    --token $1 \
    --labels my-runner \
    --url https://github.com/${GITHUB_OWNER} \
    --work "/work" \
    --unattended \
    --replace \
    ${2:+--ephemeral}

remove() {
    ./config.sh remove --token "$1"
}

trap 'remove; exit 130' INT
trap 'remove; exit 143' TERM

# ./run.sh actually does not need any arguments if those were specified in ./config.sh
./run.sh &

wait $!
