import asyncio
import logging
import os
from enum import StrEnum

import requests as r


class JobState(StrEnum):
    REGISTERED = 'REGISTERED'
    DISPATCHED = 'DISPATCHED'
    FINISHED = 'FINISHED'


GITHUB_OWNER = os.environ.get('GITHUB_OWNER')
GITHUB_REPOSITORY = os.environ.get('GITHUB_REPOSITORY')
GITHUB_PERSONAL_TOKEN = os.environ.get('GITHUB_PERSONAL_TOKEN')
# this workflow_id corresponds to the self-hosted pytest workflow
WORKFLOW_ID = '132920098'
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

# all times in seconds

# how long to wait for the job to be fetched as complete
# since the github runner possibly ensures that the job is finished before turning off
# this can be reduced to 0
GITHUB_API_DELAY = 0.5 * 60
# period to ask the GitHub API for new jobs
REQUEST_DELAY_TIME = 1.5 * 60
# if the GitHub API returns an error, wait for this time before retrying
EXTENDED_DELAY_TIME = 10 * 60


# we need to propagate the run_id to runners to have the unique names of runners for each run
# moreover, the runner should start with `--ephemeral` to take only one job (see the docker/runner_entrypoint.sh)
async def dispatch_run(runner_token: str, run_id: int):
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


if None in (GITHUB_OWNER, GITHUB_REPOSITORY, GITHUB_PERSONAL_TOKEN):
    raise ValueError('Specify the environment variables')

api_headers = {
    'Authorization': 'Bearer ' + GITHUB_PERSONAL_TOKEN,
    'Accept': 'application/vnd.github+json',
    'X-GitHub-Api-Version': '2022-11-28',
}

# check if the is any workflow for WORKFLOW_ID is queued
runs_query = f'https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPOSITORY}/actions/workflows/{WORKFLOW_ID}/runs'
params = {'status': 'queued'}

# obtain a token to register a runner
# the idea to do it here to isolate the GITHUB_PERSONAL_TOKEN from runner container
runner_token_query = f'https://api.github.com/orgs/{GITHUB_OWNER}/actions/runners/registration-token'

scheduled_runs = {}
logger = logging.getLogger('SlurmDispatcher')


async def main():
    while True:
        logger.debug(f'Request GitHub API, query: {runs_query}, params: {params}')
        resp = r.get(runs_query, params=params, headers=api_headers, timeout=10)
        logger.debug(f'Response code: {resp.status_code}')
        if resp.status_code == 200:
            payload = resp.json()
            if payload.get('total_count') > 0:
                for workflow_run in payload.get('workflow_runs'):
                    run_id = workflow_run.get('id')
                    run_details = scheduled_runs.get(run_id)
                    # this means there is no runner dispatched for the run
                    if run_details is None:
                        resp_token = r.post(runner_token_query, headers=api_headers, timeout=10)
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
