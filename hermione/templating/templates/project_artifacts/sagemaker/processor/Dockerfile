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
    mkdir /opt/ml/processing && \
    mkdir /opt/ml/processing/input && \
    mkdir /opt/ml/processing/input/raw_data && \
    mkdir /opt/ml/processing/input/preprocessing && \
    mkdir /opt/ml/processing/input/expectations && \
    mkdir /opt/ml/processing/output && \
    mkdir /opt/ml/processing/output/processed && \
    mkdir /opt/ml/processing/output/processed/train && \
    mkdir /opt/ml/processing/output/processed/val && \
    mkdir /opt/ml/processing/output/processed/inference && \
    mkdir /opt/ml/processing/output/expectations && \
    mkdir /opt/ml/processing/output/validations
        
# Install requirements
COPY requirements.txt /opt/ml/code/src/requirements.txt
RUN pip3 install --no-cache -r /opt/ml/code/src/requirements.txt

# Copy entrypoint script to the image and make it executable
COPY src/config/ /opt/ml/code/src/config/
COPY src/ml/ /opt/ml/processing/ml/
COPY src/util.py /opt/ml/processing/util.py
COPY processor/preprocessor.py /opt/ml/processing/preprocessor.py

# Change working directory
WORKDIR /opt/ml/processing

# Setting PYTHONPATH to access the copied code
ENV PYTHONPATH="/opt/ml/processing:${PATH}"

# Add a Python script and configure Docker to run it
ENTRYPOINT ["python3", "preprocessor.py"]
