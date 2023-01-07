FROM nvidia/cuda:11.4.0-base-ubuntu20.04

RUN apt update
RUN apt-get install -y python3 python3-pip

COPY requirements.txt  /tmp/

RUN python3 -m pip install -r /tmp/requirements.txt

ENV PROJECT_DIR $HOME/app
ENV CUDA_VISIBLE_DEVICES 0
RUN mkdir $PROJECT_DIR

COPY ./model ./model
COPY ./src/* ./

CMD ["python","process.py"]

