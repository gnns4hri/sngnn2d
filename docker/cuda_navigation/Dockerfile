# FROM ubuntu:18.04
FROM nvidia/cuda:10.1-base-ubuntu18.04

RUN apt-get update && apt-get install -y apt-transport-https ca-certificates && \
    rm -rf /var/lib/apt/lists/*
    
RUN apt-get update && apt-get install -y --no-install-recommends apt-utils gnupg2 curl && \
    curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub | apt-key add - && \
    echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda.list && \
    echo "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list &&\
    apt-get purge --autoremove -y curl && \
rm -rf /var/lib/apt/lists/*

ENV CUDA_VERSION 10.1.243
ENV NCCL_VERSION 2.4.8
ENV CUDA_PKG_VERSION 10-1=$CUDA_VERSION-1
ENV CUDNN_VERSION 7.6.5.32

RUN apt-get update && apt-get install -y --no-install-recommends \
        cuda-cudart-$CUDA_PKG_VERSION \
        cuda-compat-10-1 && \
    ln -s cuda-10.1 /usr/local/cuda && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --allow-unauthenticated --no-install-recommends \
        cuda-libraries-$CUDA_PKG_VERSION \
        cuda-nvtx-$CUDA_PKG_VERSION \
        libcublas10=10.2.1.243-1 \
        libnccl2=$NCCL_VERSION-1+cuda10.1 && \
    apt-mark hold libnccl2 && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --allow-unauthenticated --no-install-recommends \
        cuda-libraries-dev-$CUDA_PKG_VERSION \
        cuda-nvml-dev-$CUDA_PKG_VERSION \
        cuda-minimal-build-$CUDA_PKG_VERSION \
        cuda-command-line-tools-$CUDA_PKG_VERSION \
        libnccl-dev=$NCCL_VERSION-1+cuda10.1 \
        libcublas-dev=10.2.1.243-1 \
        && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends \
    libcudnn7=$CUDNN_VERSION-1+cuda10.1 \
    libcudnn7-dev=$CUDNN_VERSION-1+cuda10.1 \
    && \
    apt-mark hold libcudnn7 && \
    rm -rf /var/lib/apt/lists/*


ENV LIBRARY_PATH /usr/local/cuda/lib64/stubs

# NVIDIA docker 1.0.
LABEL com.nvidia.volumes.needed="nvidia_driver"
LABEL com.nvidia.cuda.version="${CUDA_VERSION}"

RUN echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf && \
    echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf

ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64

# NVIDIA container runtime.
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV NVIDIA_REQUIRE_CUDA "cuda>=10.0 brand=tesla,driver>=384,driver<385 brand=tesla,driver>=410,driver<411"

# PyTorch (Geometric) installation
RUN rm /etc/apt/sources.list.d/cuda.list && \
    rm /etc/apt/sources.list.d/nvidia-ml.list 

RUN apt-get update &&  apt-get install -y curl ca-certificates vim sudo  git bzip2 libx11-6 
RUN rm -rf /var/lib/apt/lists/*

# Create a working directory.
RUN mkdir /app
WORKDIR /app

# Create a non-root user and switch to it.
RUN adduser --disabled-password --gecos '' --shell /bin/bash user \
 && chown -R user:user /app
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user

# All users can use /home/user as their home directory.
ENV HOME=/home/user
RUN chmod 777 /home/user

# Install Miniconda.
RUN curl -so ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
 && chmod +x ~/miniconda.sh \
 && ~/miniconda.sh -b -p ~/miniconda \
 && rm ~/miniconda.sh
ENV PATH=/home/user/miniconda/bin:$PATH
ENV CONDA_AUTO_UPDATE_CONDA=false

# Create a Python 3.6 environment.
RUN /home/user/miniconda/bin/conda install conda-build \
 && /home/user/miniconda/bin/conda create -y --name py36 python=3.6.5 \
 && /home/user/miniconda/bin/conda clean -ya
ENV CONDA_DEFAULT_ENV=py36
ENV CONDA_PREFIX=/home/user/miniconda/envs/$CONDA_DEFAULT_ENV
ENV PATH=$CONDA_PREFIX/bin:$PATH

# CUDA 10.0-specific steps.
RUN conda install -y -c pytorch \
    cudatoolkit=10.1 \
    "pytorch=1.7.0=py3.6_cuda10.1.243_cudnn7.6.3_0" \
 && conda clean -ya
    # torchvision=0.5.0=py36_cu101 \

# Install HDF5 Python bindings.
RUN conda install -y h5py=2.8.0 \
 && conda clean -ya
RUN pip install h5py-cache==1.0

# Install TorchNet, a high-level framework for PyTorch.
# RUN pip install torchnet==0.0.4

# Install Requests, a Python library for making HTTP requests.
RUN conda install -y requests=2.19.1 \
 && conda clean -ya

# Install Graphviz.
RUN conda install -y graphviz=2.40.1 python-graphviz=0.8.4 \
 && conda clean -ya

# Install OpenCV3 Python bindings.
RUN sudo apt-get update && sudo apt-get install -y --no-install-recommends \
    libgtk2.0-0 \
    libcanberra-gtk-module \
 && sudo rm -rf /var/lib/apt/lists/*
RUN conda install -y -c menpo opencv3=3.1.0 \
 && conda clean -ya

# Install PyTorch Geometric.
RUN CPATH=/usr/local/cuda/include:$CPATH \
 && LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH \
 && DYLD_LIBRARY_PATH=/usr/local/cuda/lib:$DYLD_LIBRARY_PATH

RUN python3 -m pip install dgl-cu102
RUN python3 -m pip install torch-scatter==2.0.5+cu102 -f https://pytorch-geometric.com/whl/torch-1.7.0.html
RUN python3 -m pip install torch-sparse==0.6.8+cu102 -f https://pytorch-geometric.com/whl/torch-1.7.0.html
RUN python3 -m pip install torch-cluster==1.5.8+cu102 -f https://pytorch-geometric.com/whl/torch-1.7.0.html
RUN python3 -m pip install torch-spline-conv==1.2.0+cu102 -f https://pytorch-geometric.com/whl/torch-1.7.0.html
USER root
# RUN DEBIAN_FRONTEND=noninteractive apt-get install --yes python3-numba
RUN python3 -m pip install torch-geometric

# RUN pip install torch-scatter==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.7.0.html \
#  && pip install torch-sparse==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.7.0.html \
#  && pip install torch-cluster==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.7.0.html \
#  && pip install torch-spline-conv==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.7.0.html \
#  && pip install torch-geometric





# docker run -ti --rm -e DISPLAY=$DISPLAY  -v /tmp/.X11-unix/:/tmp/.X11-unix -v $HOME:$HOME --security-opt apparmor=unconfined  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility -e NVIDIA_VISIBLE_DEVICES=all --gpus all ljmanso/gnns4hri:base terminator
# docker run -ti --rm -e DISPLAY=$DISPLAY  -v /tmp/.X11-unix/:/tmp/.X11-unix -v $HOME:$HOME --security-opt apparmor=unconfined  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility -e NVIDIA_VISIBLE_DEVICES=all --gpus all ljmanso/gnns4hri:base terminator

# install nvidia-container-runtime
# install nvidia-container-runtime

###https://collabnix.com/introducing-new-docker-cli-api-support-for-nvidia-gpus-under-docker-engine-19-03-0-beta-release/
###https://collabnix.com/introducing-new-docker-cli-api-support-for-nvidia-gpus-under-docker-engine-19-03-0-beta-release/

# docker push ljmanso/gnns4hri:base
# docker push ljmanso/gnns4hri:base

# docker build -t ljmanso/gnns4hri:base
# docker build -t ljmanso/gnns4hri:base


# Update Ubuntu Software repository
USER root
RUN DEBIAN_FRONTEND=noninteractive apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y firefox sudo terminator firefox xterm dbus dbus-x11 gnupg
RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections



# Install Robocomp
RUN DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends --yes git git-annex cmake make g++ libgsl-dev libopenscenegraph-dev cmake-qt-gui freeglut3-dev libboost-system-dev libboost-thread-dev qt5-default libqt5xmlpatterns5-dev libxt-dev libboost-test-dev libboost-filesystem-dev  libccd-dev  libqt5opengl5-dev libxml2-dev
RUN DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends --yes libzeroc-ice3.7 libzeroc-icestorm3.7 python3-dbus python3-zeroc-ice zeroc-glacier2 zeroc-ice-slice zeroc-ice-utils zeroc-icebox zeroc-icegrid zeroc-icepatch2 zeroc-icebridge libzeroc-ice-dev zeroc-ice-all-dev zeroc-ice-compilers
RUN DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends --yes python3-pip python3-setuptools python3-pyparsing python3-numpy python3-libxml2 python3-xmltodict



RUN pip3 install networkx pyside2 argcomplete termcolor cogapp prompt_toolkit
USER user
WORKDIR /home/user
ENV HOME /home/user
ENV ROBOCOMP=~/robocomp
ENV PATH=$PATH:/opt/robocomp/bin
RUN git clone https://github.com/robocomp/robocomp.git
RUN echo "export ROBOCOMP=/home/user/robocomp" >> ~/.bashrc
RUN echo "export PATH=$PATH:/opt/robocomp/bin" >> ~/.bashrc
RUN export ROBOCOMP=/home/user/robocomp
RUN export PATH=$PATH:/opt/robocomp/bin
RUN cd /home/user/robocomp && mkdir build && cd build && cmake .. && make -j$(nproc)
USER root
RUN cd /home/user/robocomp/build && make install
RUN sh -c "echo '/opt/robocomp/lib/' >> /etc/ld.so.conf"
RUN ldconfig
RUN sudo ln -s ~ /home/robocomp

# DBUS
RUN dbus-uuidgen > /var/lib/dbus/machine-id






USER user
WORKDIR /home/user
ENV HOME /home/user
RUN wget https://www.coppeliarobotics.com/files/CoppeliaSim_Edu_V4_1_0_Ubuntu18_04.tar.xz
RUN tar -xJf CoppeliaSim_Edu_V4_1_0_Ubuntu18_04.tar.xz
RUN echo 'export PATH=$PATH:/home/user/CoppeliaSim_Edu_V4_1_0_Ubuntu18_04/' >> /home/user/.bashrc
RUN git clone https://github.com/stepjam/PyRep.git
WORKDIR "/home/user/PyRep"
RUN python3 -m pip install -r requirements.txt
RUN echo 'export COPPELIASIM_ROOT=/home/user/CoppeliaSim_Edu_V4_1_0_Ubuntu18_04/' >> /home/user/.bashrc
ENV COPPELIASIM_ROOT /home/user/CoppeliaSim_Edu_V4_1_0_Ubuntu18_04/
RUN echo $COPPELIA_ROOT
USER root
RUN DEBIAN_FRONTEND=noninteractive apt-get install --yes python3-dev
RUN python3 setup.py install 


USER user


# SONATA Graph2Image
USER root
RUN DEBIAN_FRONTEND=noninteractive apt-get install --yes python3-opencv
RUN DEBIAN_FRONTEND=noninteractive apt-get install --yes python3-dev 
RUN python3 -m pip install shapely scikit-build dgl-cu101
USER user
WORKDIR /home/user
RUN git clone https://github.com/gnns4hri/sonata
RUN git clone https://github.com/gnns4hri/sngnn2d

WORKDIR /home/user/sngnn2d
RUN sudo find . -name \*.ice -exec cp {} /opt/robocomp/interfaces/ \;

# RUN conda update -n base -c defaults conda

RUN conda install -c bioconda zeroc-ice
USER root
RUN conda install -c conda-forge pyside2 

RUN apt-get install -y mesa-utils and libgl1-mesa-glx
RUN apt-get update && apt-get install -y libglib2.0-0 libgl1-mesa-glx xcb "^libxcb.*" libx11-xcb-dev
RUN apt-get update && apt-get install -y libglu1-mesa-dev libxrender-dev libxi6 libdbus-1-3 libfontconfig1 xvfb
RUN apt-get update && apt-get install -y libxcursor-dev libxcomposite-dev software-properties-common build-essential libssl-dev
RUN apt-get update && apt-get install -y libxcb1-dev libx11-dev libgl1-mesa-dev libudev-dev qt5-default qttools5-dev qtdeclarative5-dev
RUN apt-get update && apt-get install -y qtpositioning5-dev qtbase5-dev 

