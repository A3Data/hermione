FROM ubuntu:latest
# Set a docker label to advertise multi-model support on the container
LABEL com.amazonaws.sagemaker.capabilities.multi-models=false
# Set a docker label to enable container to use SAGEMAKER_BIND_TO_PORT environment variable if present
LABEL com.amazonaws.sagemaker.capabilities.accept-bind-to-port=true

# No question/dialog is asked during apt-get install
ARG DEBIAN_FRONTEND=noninteractive

# Setting the Timezone Environment Variable
ENV TZ=America/Sao_Paulo

# install ubuntu libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        python3.7 \
        python3-dev \
        python3-pip \
        ca-certificates \
        git \
        curl \
        nginx \
        openjdk-8-jre-headless\
        wget &&\
    rm -rf /var/lib/apt/lists/*
   
# Create folders for code
RUN mkdir /opt/ml && \
    mkdir /opt/ml/output && \
    mkdir /opt/ml/code && \
    mkdir /opt/ml/code/train && \
    mkdir /opt/ml/code/src 

# Install requirements
COPY requirements.txt /opt/ml/code/src/requirements.txt
RUN pip3 install --no-cache -r /opt/ml/code/src/requirements.txt

# Install the SageMaker Training Toolkit 
RUN pip3 install --no-cache \
    boto3 \
    sagemaker \
    sagemaker-training

# copy folders for code
COPY src/config/ /opt/ml/code/src/config/
COPY src/ml/ /opt/ml/code/src/ml/
COPY src/util.py /opt/ml/code/src/util.py
COPY train/train.py /opt/ml/code/train.py

# Copy entrypoint script to the image and make it executable
WORKDIR /opt/ml/code

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=UTF-8 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8
    
# Setting PYTHONPATH to access the copied code
ENV PYTHONPATH="/opt/ml/code:${PATH}"

# Add a Python script and configure Docker to run it
RUN chmod +x train.py
ENV SAGEMAKER_PROGRAM train.py
