FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-devel

RUN apt-get update && apt-get install -y git pbzip2 curl unzip ffmpeg libsm6 libxext6 libgl1-mesa-glx

RUN git clone https://github.com/StanfordHCI/human-pose-estimation.pytorch.git
WORKDIR /workspace/human-pose-estimation.pytorch
RUN git pull
RUN pip3 install -r requirements.txt
RUN pip3 install watchdog pycocotools tensorboard

WORKDIR /workspace/human-pose-estimation.pytorch/lib
RUN make

WORKDIR /workspace/
USER root
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
RUN unzip awscliv2.zip
RUN ./aws/install

RUN git clone https://github.com/StanfordHCI/VideoPose3D.git videopose3d
WORKDIR /workspace/videopose3d
RUN git pull
RUN git checkout "model_v1.9_new"
RUN pip3 install -r requirements.txt

RUN apt-get install -y wget
WORKDIR /workspace/
RUN git clone --recursive https://github.com/StanfordHCI/simple-HRNet.git resnetpose
WORKDIR /workspace/resnetpose
RUN pip3 install -r requirements.txt
WORKDIR /workspace/resnetpose/models/detectors/yolo/weights
RUN bash ./download_weights.sh


COPY *.py /workspace/

USER root
WORKDIR /workspace/human-pose-estimation.pytorch