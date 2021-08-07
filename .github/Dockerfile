FROM python:3.7

RUN \
    apt-get update && \
    apt-get install ffmpeg libsm6 libxext6  -y

RUN pip3 install --upgrade pip setuptools wheel --no-cache-dir

RUN pip3 install --no-cache-dir git+https://github.com/aniketmaurya/chitra@master
