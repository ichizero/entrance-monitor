FROM ichizero/cuda-opencv:latest

LABEL maintainer="ichizero"

RUN apt-get -y update
RUN apt-get install -y --fix-missing \
    build-essential \
    cmake \
    gfortran \
    git \
    wget \
    curl \
    graphicsmagick \
    libgraphicsmagick1-dev \
    libatlas-dev \
    libavcodec-dev \
    libavformat-dev \
    libgtk2.0-dev \
    libjpeg-dev \
    liblapack-dev \
    libswscale-dev \
    pkg-config \
    software-properties-common \
    zip \
    && apt-get clean && rm -rf /tmp/* /var/tmp/*

RUN rm /usr/bin/python && \
  ln -s /usr/bin/python3 /usr/bin/python

RUN pip3 install  --no-cache-dir \
  argparse \
  tqdm \
  flask

RUN cd ~ && \
    mkdir -p dlib && \
    git clone -b 'v19.9' --single-branch https://github.com/davisking/dlib.git dlib/ && \
    cd  dlib/ && \
    python3 setup.py install --yes USE_AVX_INSTRUCTIONS

# The rest of this file just runs an example script.

# If you wanted to use this Dockerfile to run your own app instead, maybe you would do this:
# COPY . /root/your_app_or_whatever
# RUN cd /root/your_app_or_whatever && \
#     pip3 install -r requirements.txt
# RUN whatever_command_you_run_to_start_your_app

COPY . /root/face_recognition
RUN cd /root/face_recognition && \
    pip3 install -r requirements.txt && \
    python3 setup.py install

# CMD cd /root/face_recognition/examples && \
#     python3 recognize_faces_in_pictures.py
