FROM jupyter/pyspark-notebook:latest

ARG HADOOP_VERSION=3.3.1
ARG AWS_SDK_VERSION=1.11.901

USER root

RUN wget -q "https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/${HADOOP_VERSION}/hadoop-aws-${HADOOP_VERSION}.jar" -P ${SPARK_HOME}/jars/ && \
    wget -q "https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk-bundle/${AWS_SDK_VERSION}/aws-java-sdk-bundle-${AWS_SDK_VERSION}.jar" -P ${SPARK_HOME}/jars/

RUN echo 'spark.serializer org.apache.spark.serializer.KryoSerializer' >> "${SPARK_HOME}/conf/spark-defaults.conf" && \
    echo 'spark.hadoop.fs.s3a.fast.upload True' >> "${SPARK_HOME}/conf/spark-defaults.conf" && \
    echo 'spark.hadoop.fs.s3a.impl org.apache.hadoop.fs.s3a.S3AFileSystem' >> "${SPARK_HOME}/conf/spark-defaults.conf"

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt && \
    pip install "jupyterlab>=3" "ipywidgets>=7.6"