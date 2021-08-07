FROM python:3.7
RUN \
    apt-get update && \
    apt-get install ffmpeg libsm6 libxext6  -y


WORKDIR /app

COPY ./requirements.txt requirements.txt

RUN pip3 install --upgrade pip setuptools wheel --no-cache-dir
RUN pip3 install --no-cache-dir git+https://github.com/aniketmaurya/chitra@master --no-deps

RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE PORT

CMD [ "uvicorn", "main:app", "--host=0.0.0.0", "--port=PORT"]
