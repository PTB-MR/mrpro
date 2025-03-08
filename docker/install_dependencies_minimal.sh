# install the minimal dependencies for the project from the pyproject.toml file
python -m pip install --no-cache-dir torch==2.3.1 \
                                     torchvision==0.18.1 \
                                     --index-url https://download.pytorch.org/whl/cpu

python -m pip install --no-cache-dir numpy==1.23 \
                                     ismrmrd==1.14.1 \
                                     einops \
                                     pydicom==3.0.1 \
                                     pypulseq==1.4.2 \
                                     pytorch-finufft==0.1.0 \
                                     cufinufft==2.3.1 \
                                     scipy==1.12 \
                                     ptwt==0.1.8 \
                                     typing-extensions==4.12

#clean up
rm -rf /root/.cache
