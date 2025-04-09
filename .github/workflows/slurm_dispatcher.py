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
GITHUB_REPO = os.environ.get('GITHUB_REPO')
GITHUB_TOKEN = os.environ.get('GITHUB_TOKEN')
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
async def dispatch_run(run_id: int):
    process = await asyncio.create_subprocess_exec(
        # pass the run_id as a unique identifier for the runner name
        *[*SLURM_SUBMIT_COMMAND, str(run_id)],
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    scheduled_runs[run_id]['pid'] = process.pid
    scheduled_runs[run_id]['state'] = JobState.DISPATCHED
    logger.info(f'Dispatched run_id: {run_id}, pid: {process.pid}')
    await process.wait()
    scheduled_runs[run_id]['state'] = JobState.FINISHED
    logger.info(f'Runner finished for run_id: {run_id}, pid: {process.pid}')
    logger.debug(f'Wait for {GITHUB_API_DELAY} till job is fetched as complete')
    await asyncio.sleep(GITHUB_API_DELAY)
    scheduled_runs.pop(run_id)


if None in (GITHUB_OWNER, GITHUB_REPO, GITHUB_TOKEN):
    raise ValueError('Specify the environment variables')

api_headers = {
    'Authorization': 'Bearer ' + GITHUB_TOKEN,
    'Accept': 'application/vnd.github+json',
    'X-GitHub-Api-Version': '2022-11-28',
}

query = f'https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/actions/workflows/{WORKFLOW_ID}/runs'
params = {'status': 'queued'}

scheduled_runs = {}
logger = logging.getLogger('SlurmDispatcher')


async def main():
    while True:
        logger.debug(f'Request GitHub API, query: {query}, params: {params}')
        resp = r.get(query, params=params, headers=api_headers, timeout=10)
        logger.debug(f'Response code: {resp.status_code}')
        if resp.status_code == 200:
            payload = resp.json()
            if payload.get('total_count') > 0:
                for workflow_run in payload.get('workflow_runs'):
                    run_id = workflow_run.get('id')
                    run_details = scheduled_runs.get(run_id)
                    if run_details is None:
                        scheduled_runs[run_id] = {'pid': None, 'state': JobState.REGISTERED}
                        logger.info(f'Registered {run_id}')
                        asyncio.create_task(dispatch_run(run_id))
                    else:
                        logger.info(f'Run {run_id}: {run_details}')
            logger.debug(f'Sleep {REQUEST_DELAY_TIME} before next request')
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
