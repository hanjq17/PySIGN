# base mirror
FROM nvidia/cuda:11.0-cudnn8-devel-ubuntu16.04

ENV LD_LIBRARY_PATH "/usr/local/cuda-11.0/lib64:${LD_LIBRARY_PATH}"


# install essential dependencies
COPY ubuntu16_sources.list /tmp/ubuntu16_sources.list
RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN cp -rf /tmp/ubuntu16_sources.list /etc/apt/sources.list
RUN apt update -y
RUN apt install -y --no-install-recommends \
    cmake \
    cron \
    expect \
    gcc-4.9 \
    grsync \
    g++-4.9 \
    iproute2 \
    iputils-ping \
    lsof \
    net-tools \
    openssh-client \
    openssh-server \
    rsync \
    tcl \
    telnet \
    tk \
    unzip \
    vim \
    wget \
    zip

# install Miniconda
# RUN bash /softwares/extras/Miniconda3-py37_4.11.0-Linux-x86_64.sh -b -p /softwares/miniconda
RUN wget --no-check-certificate https://repo.continuum.io/miniconda/Miniconda3-py38_4.9.2-Linux-x86_64.sh \
    && /bin/bash Miniconda3-py38_4.9.2-Linux-x86_64.sh   -b -p /softwares/miniconda \
    && rm -v Miniconda3-py38_4.9.2-Linux-x86_64.sh
ENV PATH "/softwares/miniconda/bin:${PATH}"
RUN pip config set global.index-url https://mirrors.cloud.tencent.com/pypi/simple

# install Python dependencies
RUN pip install --upgrade pip
# Torch and PYG
RUN pip install torch==1.7.1 --extra-index-url https://download.pytorch.org/whl/cu110
RUN pip install torch_scatter==2.0.7 \
                torch_sparse==0.6.9 \
                torch_cluster==1.5.8 \
                torch-geometric==2.0.4
# Others
RUN pip install scikit-learn \
                easydict  \
                tqdm \
                matplotlib \
                sympy \
                pyyaml \
                lie-learn \
                atom3d

# install Horovod
ENV HOROVOD_GPU_OPERATIONS "NCCL"
RUN pip install horovod

# create a symbolic link for libcusolver.so
RUN ln -s /usr/local/cuda-11.0/lib64/libcusolver.so.10 /usr/local/cuda-11.0/lib64/libcusolver.so.11

# provide defaults for the executing container
CMD [ "/bin/bash" ]