"""Cron-scheduled slurm test dispatcher."""

import logging
import os
import subprocess

import requests as r

"""
This script is intended to run periodically by the cron.

Add the following line into the list of cron jobs:
*/5 * * * * . /path/to/mr2.rc; /path/to/python /path/to/slurm_cronjob.py >> /path/to/err.log 2>&1

Example of the mr2.rc file content with the required environmental variables:

# name of the organization
export GITHUB_OWNER="fzimmermann89"
# name of the repository
export GITHUB_REPOSITORY="mr2"
# name of the system user
export USER_NAME="XXX"
# id of the GitHub workflow to be periodically checked
# (see the number in the URL field at https://docs.github.com/en/rest/actions/workflows?apiVersion=2022-11-28#list-repository-workflows)
export WORKFLOW_ID="XXX"
# GitHub personal token (see pytest_selfhosted.yml)
export GITHUB_PERSONAL_TOKEN="XXX"
# Runner group
export RUNNER_GROUP="XXX"

# specify any other required environmental variables as http(s)_proxy
export https_proxy="XXX"
export http_proxy="XXX"
"""

logger = logging.getLogger('SlurmDispatcher')


REQUIRED_ENV_VARS = (
    'GITHUB_OWNER',
    'GITHUB_REPOSITORY',
    'GITHUB_PERSONAL_TOKEN',
    'WORKFLOW_ID',
    'USER_NAME',
    'RUNNER_GROUP',
)
ENV_VARS = {var_name: os.environ.get(var_name) for var_name in REQUIRED_ENV_VARS}

UNSET_VARS = [key for key, value in ENV_VARS.items() if value is None]

if UNSET_VARS:
    error_message = f'{", ".join(str(name) for name in UNSET_VARS)} - not set'
    logger.critical(error_message)
    raise ValueError(error_message)

SBATCH_SUBMIT_COMMAND = """#!/bin/bash
#SBATCH --job-name=mr2-runner-{RUN_ID} # name of the job
#SBATCH --ntasks=6  # number of "tasks" (default: allocates 1 core per task)
#SBATCH --mem=64G
#SBATCH -t 0-00:60:00   # time in d-hh:mm:ss
#SBATCH -o /home/%u/slurm_output/slurm.%j.out # file to save job's STDOUT (%j = JobId)
#SBATCH -e /home/%u/slurm_output/slurm.%j.err # file to save job's STDERR (%j = JobId)
#SBATCH --export=NONE   # Purge the job-submitting shell environment
#SBATCH --gres=gpu:A100mig:1 # reserved gpu
#SBATCH --qos=urgent # priority
#SBATCH -p equipment_typeG # Request GPU

# display the config file
singularity exec --nv --pwd /actions-runner --writable-tmpfs --contain docker://hpcharbor.berlin.ptb.de/abt81/mr2_runner:latest\
    /actions-runner/entrypoint.sh {RUNNER_TOKEN} {GITHUB_OWNER} {RUN_ID} {RUNNER_GROUP}
sleep 15"""


def get_running_job_names() -> list[str]:
    """
    Get the names of currently running jobs from the SLURM scheduler.

    This function executes a command to list all running jobs and extracts their names
    from the output. It returns a list of job names that are currently active.

    Returns
    -------
        list[str]: A list of names of currently running jobs.
    """
    # ruff: noqa: S603
    process = subprocess.run(
        ['/usr/bin/squeue', '--user', ENV_VARS['USER_NAME'], '--noheader', '--format=%j'],
        capture_output=True,
        text=True,
    )
    if process.returncode != 0:
        logger.error(f'Error retrieving running jobs: {process.stderr.strip()}')
        return []
    return process.stdout.strip().split('\n')


def dispatch_run(runner_token: str, run_id: int) -> None:
    """
    Dispatches and monitors a subprocess for a given run in an asynchronous manner.

    This function launches a subprocess using the SLURM submission command,
    uniquely identified by the provided `run_id` and authenticated via `runner_token`.
    It updates the `scheduled_runs` dictionary to reflect the current state of the run,
    logs process output (stdout and stderr), and ensures cleanup after execution.

    Please note that the runner should be configured to run in "ephemeral" mode.

    Args:
        runner_token (str): A token used to authenticate the runner process.
        run_id (int): A unique identifier for the run, used to track and name the process.
    """
    # ruff: noqa: S603
    process = subprocess.run(
        # pass the run_id as a unique identifier for the runner name
        ['/usr/bin/sbatch'],
        input=SBATCH_SUBMIT_COMMAND.format(
            RUNNER_TOKEN=runner_token,
            RUN_ID=run_id,
            GITHUB_OWNER=ENV_VARS['GITHUB_OWNER'],
            USER_NAME=ENV_VARS['USER_NAME'],
            RUNNER_GROUP=ENV_VARS['RUNNER_GROUP'],
        ),
        text=True,
        capture_output=True,
    )
    logger.info(f'[run_id={run_id}]: dispatched, job name: mr2-runner-{run_id}, stdout: {process.stdout.strip()}')


API_HEADERS = {
    'Authorization': 'Bearer {GITHUB_PERSONAL_TOKEN}'.format(**ENV_VARS),
    'Accept': 'application/vnd.github+json',
    'X-GitHub-Api-Version': '2022-11-28',
}
""" headers required by GitHub REST API """

# check if the is any workflow for WORKFLOW_ID is queued
RUNS_CHECK_QUERY = (
    'https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPOSITORY}/actions/workflows/{WORKFLOW_ID}/runs'.format(
        **ENV_VARS
    )
)
""" GitHub REST API query to obtain a list of queued jobs """

RUNS_CHECK_PARAMS = {'status': 'queued'}
"""  GitHub REST API list of queued jobs filters """

RUNNER_TOKEN_QUERY = 'https://api.github.com/orgs/{GITHUB_OWNER}/actions/runners/registration-token'.format(**ENV_VARS)
""" GitHub REST API query to obatin a runner registration token """


def main() -> None:
    """
    Periodically polls the GitHub API for new workflow runs and dispatches runners as needed.

    This coroutine continuously sends GET requests to check for pending GitHub Actions
    workflow runs. If a new unhandled run is found, it requests a runner token and
    schedules it for execution via the `dispatch_run` function.

    Runs indefinitely until cancelled or the event loop is stopped.
    """
    logger.debug(f'Request GitHub API, query: {RUNS_CHECK_QUERY}, params: {RUNS_CHECK_PARAMS}')
    resp = r.get(RUNS_CHECK_QUERY, params=RUNS_CHECK_PARAMS, headers=API_HEADERS, timeout=10)
    logger.debug(f'Response code: {resp.status_code}')
    if resp.status_code == 200:
        payload = resp.json()
        if payload.get('total_count') > 0:
            running_job_names = get_running_job_names()
            for workflow_run in payload.get('workflow_runs'):
                run_id = workflow_run.get('id')
                # check if there is a job for such run_id in slurm
                if f'mr2-runner-{run_id}' in running_job_names:
                    logger.info(f'[run_id={run_id}]: already queued')
                    continue
                # this means there is no runner dispatched for the run
                resp_token = r.post(RUNNER_TOKEN_QUERY, headers=API_HEADERS, timeout=10)
                if resp_token.status_code == 201:
                    runner_token = resp_token.json().get('token')
                    if runner_token is not None:
                        logger.info(f'[run_id={run_id}]: registered')
                        dispatch_run(runner_token, run_id)


if __name__ == '__main__':
    logging.basicConfig(
        filename='slurm_cronjob.log',
        level=logging.INFO,
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    logger.info('Cronjob started...')
    main()
    logger.info('Cronjob ended...')
