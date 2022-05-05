FROM phusion/baseimage:focal-1.2.0
LABEL maintainer="Joshua Griffin Dunn"
LABEL version="0.3.0"

ARG DEBIAN_FRONTEND=noninteractive
ENV HOME=/root
ENV MPLBACKEND=agg
ENV HOSTNAME=minihmm

RUN apt-get update \
    && apt-get install \
        --assume-yes \
        --verbose-versions \
        --allow-change-held-packages \
        -o Dpkg::Options::="--force-confdef" \
        build-essential \
        curl \
        git \
        gfortran \
        libatlas3-base \
        libatlas-base-dev \
        libfreetype6 \
        libfreetype6-dev \
        liblapack3 \
        liblapack-dev \
        libpng16-16 \
        libpng-dev \
        libssl-dev \
        pkg-config \
        software-properties-common \
        sudo \
        vim \
        zlib1g-dev \
    && apt-get build-dep \
        --assume-yes \
        python-numpy

# Install Python 3.6 and 3.9
RUN add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install \
        --assume-yes \
        --verbose-versions \
        --allow-change-held-packages \
        -o Dpkg::Options::="--force-confdef" \
        python3.6-full \
        python3.6-dev \
        python3.6-distutils \
        python3.9-full \
        python3.9-dev \
        python3.9-venv \
        python3.9-distutils \
        python3.9-lib2to3 \
        python3.9-tk

# Copy source code into project
ENV PROJECT_HOME=/usr/src/minihmm
WORKDIR $PROJECT_HOME
COPY . .

# Install pip and tox, which will handle installation of all dependencies
# inside virtual environments running various version of Python
RUN curl -o get-pip.py -sSL https://bootstrap.pypa.io/get-pip.py \
    && python3 get-pip.py "pip==22.0.4" \
    && pip install -r test-requirements.txt


RUN cat requirements.txt | sed -e "s/=.*//" > requirements-latest.txt \
    && tox -r --notest

# Boot into bash terminal rather than run tests, because tests are slow
# and sometimes we only want to run a subset of them
CMD ["/bin/bash"]
