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

COPY train.py /workspace/

USER root
WORKDIR /workspace/human-pose-estimation.pytorch