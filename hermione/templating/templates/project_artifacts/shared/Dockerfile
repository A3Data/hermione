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
RUN mkdir /opt/ml

# Install requirements
COPY requirements.txt /opt/ml/requirements.txt
RUN pip3 install --no-cache -r /opt/ml/requirements.txt

# Copy project files
COPY data /opt/ml/data
COPY src /opt/ml/src
COPY api /opt/ml/api
COPY config /opt/ml/config

# Change working directory
WORKDIR /opt/ml/api


# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=UTF-8

ENV PYTHONPATH="/opt/ml/:${PATH}"

EXPOSE 5000

# Execution command

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "5000"]