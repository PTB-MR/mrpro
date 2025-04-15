#!/bin/sh
# the 1st argument must be the token for the runner registration
# (see .github/slurm_dispatcher.py)

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
