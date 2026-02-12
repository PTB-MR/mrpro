#!/bin/sh
# This script registers a GitHub Actions self-hosted runner.
#
# Arguments:
#   $1 - Runner registration token.
#        See GitHub REST API documentation:
#        https://docs.github.com/en/rest/actions/self-hosted-runners?apiVersion=2022-11-28#create-a-registration-token-for-an-organization
#        Also referenced in .github/slurm_cronjob.py.
#
#   $2 - GitHub organization name.
#        Example: fzimmermann89
#
#   $3 - (Optional) If provided, enables ephemeral mode: the runner will handle only one job.
#        The runner name is constructed uniquely, following the logic in .github/slurm_cronjob.py.
#
#   $4 - (Optional) If provided, the runner is added to the specified group.

if [ -z ${1+x} ]; then
    echo "The runner registration token is missing" 1>&2
    exit 1
fi

if [ -z ${2+x} ]; then
    echo "Organization name is missing" 1>&2
    exit 1
fi

./config.sh \
    --token $1 \
    --url https://github.com/$2 \
    --name $(hostname)-${3:-default} \
    --labels my-runner \
    --work "/work" \
    --unattended \
    --replace \
    ${3:+--ephemeral} \
    ${4:+--runnergroup $4}

remove() {
    ./config.sh remove --token "$1"
}

trap 'remove; exit 130' INT
trap 'remove; exit 143' TERM

./run.sh &

wait $!
