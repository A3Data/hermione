FROM ubuntu:latest
# Set a docker label to advertise multi-model support on the container
LABEL com.amazonaws.sagemaker.capabilities.multi-models=false
# Set a docker label to enable container to use SAGEMAKER_BIND_TO_PORT environment variable if present
LABEL com.amazonaws.sagemaker.capabilities.accept-bind-to-port=true

# Install some handful libraries like curl, wget, git, build-essential, zlib
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        python3.7 \
        python3-dev \
        python3-pip \
        ca-certificates \
        git \
        curl \
	    openjdk-8-jre-headless\
        wget &&\
    rm -rf /var/lib/apt/lists/*

# install the SageMaker Inference Toolkit 
RUN pip3 install --no-cache \
    multi-model-server \
    sagemaker-inference \
    retrying
    
# Change working directory
WORKDIR /

# Install requirements
COPY requirements.txt /opt/ml/code/src/requirements.txt
RUN pip3 install --no-cache -r /opt/ml/code/src/requirements.txt

# set some environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=UTF-8 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8
      
# copy folders for code    
COPY src/config/ /opt/ml/code/config/
COPY src/ml/ /opt/ml/code/ml/
COPY src/util.py /opt/ml/code/util.py

# Copy entrypoint script to the image and make it executable
COPY inference/main.py /opt/ml/code/main.py
COPY inference/handler.py /opt/ml/code/serving/handler.py

# install sagemaker training
RUN pip3 install --no-cache --upgrade \
    boto3 \
    sagemaker
    
# Setting PYTHONPATH to access the copied code
ENV PYTHONPATH="/opt/ml/code:${PATH}"

# Add a Python script and configure Docker to run it
ENTRYPOINT ["python3", "/opt/ml/code/main.py"]
