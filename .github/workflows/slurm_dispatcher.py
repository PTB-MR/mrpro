import asyncio
import logging
import os
from enum import StrEnum

import requests as r


class JobState(StrEnum):
    """
    Enumeration of possible job states in the runner dispatch workflow.

    Attributes
    ----------
        REGISTERED: The job has been detected and registered but not yet dispatched.
        DISPATCHED: The job has been assigned to a runner and is currently in progress.
        FINISHED: The job has completed execution and is no longer tracked.
    """

    REGISTERED = 'REGISTERED'
    DISPATCHED = 'DISPATCHED'
    FINISHED = 'FINISHED'


scheduled_runs = {}
logger = logging.getLogger('SlurmDispatcher')


REQUIRED_ENV_VARS = ('GITHUB_OWNER', 'GITHUB_REPOSITORY', 'GITHUB_PERSONAL_TOKEN', 'WORKFLOW_ID')
ENV_VARS = {var_name: os.environ.get(var_name) for var_name in REQUIRED_ENV_VARS}

unset_vars = list(filter(lambda kv: kv[1] is None, ENV_VARS.items()))

if unset_vars:
    error_message = f'{", ".join(unset_vars)} are not set'
    logger.critical(error_message)
    raise ValueError(error_message)

SLURM_SUBMIT_COMMAND = [
    '/usr/bin/srun',
    '--cpus-per-task=6',
    '-p',
    'equipment_typeG',
    '--gres=gpu:1',
    '-t',
    '0-00:15:00',
    'singularity',
    'exec',
    '--nv',
    '--env-file=mrpro.env',
    '--pwd=/actions-runner',
    '--writable-tmpfs',
    '--contain',
    'mrpro-runner.sif',
    '/actions-runner/entrypoint.sh',
]

GITHUB_API_DELAY = 0.5 * 60
""" time in seconds to wait after the runner completed """

REQUEST_DELAY_TIME = 1.5 * 60
""" period in seconds to ask the GitHub REST API for new jobs  """

EXTENDED_DELAY_TIME = 10 * 60
""" time in seconds to wait befory response if GitHub REST API returns an error"""


async def dispatch_run(runner_token: str, run_id: int) -> None:
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
    process = await asyncio.create_subprocess_exec(
        # pass the run_id as a unique identifier for the runner name
        *[*SLURM_SUBMIT_COMMAND, runner_token, str(run_id)],
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    scheduled_runs[run_id]['pid'] = process.pid
    scheduled_runs[run_id]['state'] = JobState.DISPATCHED
    logger.info(f'[run_id={run_id}]: dispatched, pid: {process.pid}')
    stdout, stderr = await process.communicate()
    if stdout:
        logger.info(f'[run_id={run_id}]: stdout: {stdout.decode().strip()}')
    if stderr:
        logger.info(f'[run_id={run_id}]: stderr: {stderr.decode().strip()}')
    await process.wait()
    scheduled_runs[run_id]['state'] = JobState.FINISHED
    logger.info(f'[run_id={run_id}]: finished, pid: {process.pid}')
    logger.debug(f'[run_id={run_id}]: wait {GITHUB_API_DELAY}s')
    await asyncio.sleep(GITHUB_API_DELAY)
    scheduled_runs.pop(run_id)


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
""" GitHub REST API query to obain a list of queued jobs """

RUNS_CHECK_PARAMS = {'status': 'queued'}
"""  GitHub REST API list of queued jobs filters """

RUNNER_TOKEN_QUERY = 'https://api.github.com/orgs/{GITHUB_OWNER}/actions/runners/registration-token'.format(**ENV_VARS)
""" GitHub REST API query to obatin a runner registration token """


async def main() -> None:
    """
    Periodically polls the GitHub API for new workflow runs and dispatches runners as needed.

    This coroutine continuously sends GET requests to check for pending GitHub Actions
    workflow runs. If a new unhandled run is found, it requests a runner token and
    schedules it for execution via the `dispatch_run` function.

    Runs indefinitely until cancelled or the event loop is stopped.
    """
    while True:
        logger.debug(f'Request GitHub API, query: {RUNS_CHECK_QUERY}, params: {RUNS_CHECK_PARAMS}')
        resp = r.get(RUNS_CHECK_QUERY, params=RUNS_CHECK_PARAMS, headers=API_HEADERS, timeout=10)
        logger.debug(f'Response code: {resp.status_code}')
        if resp.status_code == 200:
            payload = resp.json()
            if payload.get('total_count') > 0:
                for workflow_run in payload.get('workflow_runs'):
                    run_id = workflow_run.get('id')
                    run_details = scheduled_runs.get(run_id)
                    # this means there is no runner dispatched for the run
                    if run_details is None:
                        resp_token = r.post(RUNNER_TOKEN_QUERY, headers=API_HEADERS, timeout=10)
                        if resp_token.status_code == 201:
                            runner_token = resp_token.json().get('token')
                            if runner_token is not None:
                                scheduled_runs[run_id] = {'pid': None, 'state': JobState.REGISTERED}
                                logger.info(f'[run_id={run_id}]: registered')
                                asyncio.create_task(dispatch_run(runner_token, run_id))
                    else:
                        logger.info(f'[run_id={run_id}]: {run_details}')
            logger.debug(f'Sleep {REQUEST_DELAY_TIME}s before next request')
            await asyncio.sleep(REQUEST_DELAY_TIME)
        else:
            await asyncio.sleep(EXTENDED_DELAY_TIME)


if __name__ == '__main__':
    logging.basicConfig(
        filename='slurm_dispatcher.log',
        level=logging.INFO,
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    logger.info('Started scheduler...')
    try:
        asyncio.run(main())
    finally:
        logger.info('Shutdown scheduler...')
        for _, run_details in scheduled_runs.items():
            logger.info(f'Unhandled processes {run_details}')
