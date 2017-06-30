FROM debian:stretch

MAINTAINER Toni Heittola <toni.heittola@gmail.com>

# Debian
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update -qq \
 && apt-get install --no-install-recommends -y \
    # Essentials
    build-essential \
    g++ \
    git \
    locales

# Setup locales
RUN dpkg-reconfigure locales && \
    locale-gen C.UTF-8 && \
    /usr/sbin/update-locale LANG=C.UTF-8

ENV LC_ALL C.UTF-8

RUN apt-get install -y curl
RUN apt-get install -qq libsndfile1 sndfile-tools
#RUN apt-get install -qq libglib2.0-0
#RUN apt-get -y install libgl1-mesa-glx
RUN apt-get clean

# Install miniconda to /miniconda
RUN curl -LO http://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh
RUN bash Miniconda-latest-Linux-x86_64.sh -p /miniconda -b
RUN rm Miniconda-latest-Linux-x86_64.sh
ENV PATH=/miniconda/bin:${PATH}
RUN conda update -y conda

RUN apt-get install -qq python-tk
##RUN apt-get install -qq python-qt4

# Python packages from conda
ARG python_version=2.7.13
RUN conda install -y python=${python_version}
RUN conda install -y mkl
RUN conda install -y numpy
RUN conda install -y scipy
RUN conda install -y scikit-learn==0.18.1
RUN conda install -y IPython
RUN conda install -y six

# Lock version for key libraries only
RUN pip install keras==2.0.2
RUN pip install theano==0.9.0
RUN pip install librosa==0.5.0
RUN pip install sed_eval==0.1.8
RUN pip install pandas==0.19.2

RUN pip install coloredlogs msgpack-python pydot-ng pyyaml soundfile matplotlib tqdm h5py

RUN dpkg-query -l > /dpkg-query-l.txt
RUN pip freeze > /pip2-freeze.txt

RUN conda clean -yt

ENV NB_USER dcase
ENV NB_UID 1000

RUN useradd -m -s /bin/bash -N -u $NB_UID $NB_USER

USER dcase

# Theano
ADD theanorc /home/dcase/.theanorc

# Single thread mode
ENV OMP_NUM_THREADS=1
ENV OMP_DYNAMIC=FALSE
##ENV GOTO_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV MKL_DYNAMIC=FALSE
##ENV KMP_DETERMINISTIC_REDUCTION=1
ENV MKL_CBWR=COMPATIBLE

WORKDIR /DCASE2017-baseline-system