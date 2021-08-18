FROM python:3.7

RUN \
    apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libsm6 libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && pip3 install --upgrade pip setuptools wheel --no-cache-dir \
    && pip3 install --no-cache-dir git+https://github.com/aniketmaurya/chitra@master
