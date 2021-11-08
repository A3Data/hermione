FROM ubuntu:latest

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
        openjdk-8-jre-headless\
        wget &&\
    rm -rf /var/lib/apt/lists/*

# Create folders for code
RUN mkdir /opt/ml && \
    mkdir /opt/ml/output && \
    mkdir /opt/ml/code && \
    mkdir /opt/ml/code/src

# Install requirements
COPY requirements.txt /opt/ml/code/src/requirements.txt
RUN pip3 install --no-cache -r /opt/ml/code/src/requirements.txt

# Copy project files
COPY output/titanic_model_rf.pkl /opt/ml/code/src/api/model/titanic_model_rf.pkl
COPY src/api/ /opt/ml/code/src/api/
COPY src/config/ /opt/ml/code/src/config/
COPY src/ml/ /opt/ml/code/src/ml/
COPY src/util.py /opt/ml/code/src/util.py

# Change working directory
WORKDIR /opt/ml/code/src/api


# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=UTF-8

ENV PYTHONPATH="/opt/ml/code/src:${PATH}"

EXPOSE 5000

# Execution command
CMD ["gunicorn", "-w", "3", "-b", ":5000", "-t", "360", "--reload", "wsgi:app"]