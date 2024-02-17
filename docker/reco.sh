#!/usr/bin/env bash
set -ev

conda update -n base -c defaults conda
conda env create --file py311.yml
source /opt/conda/bin/activate mrpro_py311
conda install pytorch torchvision torchaudio cpuonly -c pytorch
pip install torchkbnufft==1.4
