#!/bin/sh
registration_url="https://api.github.com/orgs/${GITHUB_OWNER}/actions/runners/registration-token"
echo "Requesting registration URL at '${registration_url}'"

payload=$(curl -sX POST -H "Authorization: token ${GITHUB_PERSONAL_TOKEN}" ${registration_url})
export RUNNER_TOKEN=$(echo $payload | jq .token --raw-output)

# if the 1st argument for the script is given: run in ephemeral mode (one runner takes one job)
# constructing the unique name for the runner (see .github/slurm_dispatcher.py)
./config.sh \
    --name $(hostname)-${1:-default} \
    --token ${RUNNER_TOKEN} \
    --labels my-runner \
    --url https://github.com/${GITHUB_OWNER} \
    --work "/work" \
    --unattended \
    --replace \
    ${1:+--ephemeral}

remove() {
    ./config.sh remove --unattended --token "${RUNNER_TOKEN}"
}

trap 'remove; exit 130' INT
trap 'remove; exit 143' TERM

# ./run.sh actually does not need any arguments if those were specified in ./config.sh
./run.sh &

wait $!
