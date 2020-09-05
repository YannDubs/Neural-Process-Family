# Neural Process Family

## Install

### Pip

`pip install -r requirements.txt`

### Docker

1. Install [nvidia-docker](https://github.com/NVIDIA/nvidia-docker)

2. Build your image using `Dockerfile` or pull `docker pull yanndubs/npf:gpu`

3. Create and run a container, e.g.:
`docker run --gpus all --init -d --ipc=host --name npf -v .:/Neural-Process-Family -p 8888:8888 -p 6006:6006 yanndubs/npf:gpu jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root`