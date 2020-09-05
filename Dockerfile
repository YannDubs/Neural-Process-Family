# ==================================================================
# module list
# ------------------------------------------------------------------
# python                3.8    (apt)
# pytorch               latest (pip)
# jupyterlab            latest (pip)
# requirements          latest (pip)
# jupyter/requirements  latest (pip)
# ==================================================================
# Credits : modified from https://github.com/ufoym/deepo
# ------------------------------------------------------------------

FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04
ENV LANG C.UTF-8
ARG PYTHON_VERSION=3.8

RUN APT_INSTALL="apt-get install -y --no-install-recommends" && \
    PIP_INSTALL="python -m pip --no-cache-dir install --upgrade" && \
    GIT_CLONE="git clone --depth 10" && \

    rm -rf /var/lib/apt/lists/* \
           /etc/apt/sources.list.d/cuda.list \
           /etc/apt/sources.list.d/nvidia-ml.list && \

    apt-get update && \

# ==================================================================
# tools
# ------------------------------------------------------------------

    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        build-essential \
        apt-utils \
        ca-certificates \
        wget \
        git \
        vim \
        libssl-dev \
        curl \
        unzip \
        unrar \
        && \

    $GIT_CLONE https://github.com/Kitware/CMake ~/cmake && \
    cd ~/cmake && \
    ./bootstrap && \
    make -j"$(nproc)" install && \

# ==================================================================
# python
# ------------------------------------------------------------------

    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        software-properties-common \
        && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        python"$PYTHON_VERSION" \
        python"$PYTHON_VERSION"-dev \
        python3-distutils-extra \
        && \
    wget -O ~/get-pip.py \
        https://bootstrap.pypa.io/get-pip.py && \
    python"$PYTHON_VERSION" ~/get-pip.py && \
        ln -s /usr/bin/python"$PYTHON_VERSION" /usr/local/bin/python3 && \
        ln -s /usr/bin/python"$PYTHON_VERSION" /usr/local/bin/python && \
        $PIP_INSTALL \
            setuptools \
            && \

# ==================================================================
# jupyter lab
# ------------------------------------------------------------------

    $PIP_INSTALL \
        jupyterlab \
        && \


# ==================================================================
# pytorch
# ------------------------------------------------------------------

    $PIP_INSTALL \
        future \
        numpy \
        protobuf \
        enum34 \
        pyyaml \
        typing \
        && \
    $PIP_INSTALL \
        torch torchvision 

# ==================================================================
# requirements
# ------------------------------------------------------------------

COPY requirements.txt /tmp/
RUN pip install -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt 


# ==================================================================
# jupyter/requirements
# ------------------------------------------------------------------

COPY jupyter/requirements.txt /tmp/
RUN pip install -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt 

# ==================================================================
# config & cleanup
# ------------------------------------------------------------------

RUN ldconfig && \
        apt-get clean && \
        apt-get autoremove && \
        rm -rf /var/lib/apt/lists/* /tmp/* ~/*

EXPOSE 8888 8889 6006



