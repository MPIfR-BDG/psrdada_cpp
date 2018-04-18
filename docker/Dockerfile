# Copyright (C) 2016 by Ewan Barr
# Licensed under the Academic Free License version 3.0
# This program comes with ABSOLUTELY NO WARRANTY.
# You are free to modify and redistribute this code as long
# as you do not remove the above attribution and reasonably
# inform receipients that you have modified the original work.

FROM nvidia/cuda:8.0-devel-ubuntu16.04

MAINTAINER Ewan Barr "ebarr@mpifr-bonn.mpg.de"

# Suppress debconf warnings
ENV DEBIAN_FRONTEND noninteractive

# Switch account to root and adding user accounts and password
USER root

# Create space for ssh daemon and update the system
RUN echo 'deb http://us.archive.ubuntu.com/ubuntu trusty main multiverse' >> /etc/apt/sources.list && \
    mkdir /var/run/sshd && \
    apt-get -y check && \
    apt-get -y update && \
    apt-get install -y apt-utils apt-transport-https software-properties-common python-software-properties && \
    apt-get -y update --fix-missing && \
    apt-get -y upgrade 

# Install dependencies
RUN apt-get --no-install-recommends -y install \
    build-essential \
    autoconf \
    autotools-dev \
    automake \
    autogen \
    libtool \
    csh \
    gcc \
    gfortran \
    wget \
    git \
    cvs \
    expect \
    libcfitsio-dev \
    libltdl-dev \
    gsl-bin \
    libgsl-dev \
    libgsl2 \
    hwloc \
    libhwloc-dev \
    libboost1.58-all-dev \
    pkg-config \
    cmake

ENV PSRHOME /software/
ENV OSTYPE linux
RUN mkdir -p $PSRHOME
WORKDIR $PSRHOME

# Install PSRDADA
COPY psrdada_cvs_login $PSRHOME
RUN  chmod +x psrdada_cvs_login &&\
    ./psrdada_cvs_login && \
    cvs -z3 -d:pserver:anonymous@psrdada.cvs.sourceforge.net:/cvsroot/psrdada co -P psrdada
ENV PSRDADA_HOME $PSRHOME/psrdada
WORKDIR $PSRDADA_HOME
RUN mkdir build/ && \
    ./bootstrap && \
    ./configure --prefix=$PSRDADA_HOME/build && \
    make && \
    make install && \
    make clean 
ENV PATH $PATH:$PSRDADA_HOME/build/bin
ENV PSRDADA_BUILD $PSRDADA_HOME/build/
ENV PACKAGES $PSRDADA_BUILD

WORKDIR $PSRHOME
RUN git clone https://github.com/ewanbarr/psrdada_cpp.git && \
    cd psrdada_cpp/ &&\
    git checkout meerkat &&\
    mkdir build/ &&\
    cd build/ &&\
    cmake -DENABLE_CUDA=true ../ &&\
    make -j 4 &&\
    make install

RUN env | awk '{print "export ",$0}' > $HOME/.profile && \
    echo "source $HOME/.profile" >> $HOME/.bashrc
