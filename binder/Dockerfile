# not able to set "latest" tag, need to be changed from the docker workflow anyways
FROM ghcr.io/ptb-mr/mrpro_py312@sha256:236b7217a39cbc764cc76ee19e038705fb0f8bc93b3cbfa3b39d981e52d268a3

RUN python -m pip install --no-cache-dir notebook jupyterlab

ARG NB_USER runner
ARG NB_UID 1000
ENV USER runner
ENV NB_UID ${NB_UID}
ENV HOME /home/${NB_USER}

USER ${NB_USER}
