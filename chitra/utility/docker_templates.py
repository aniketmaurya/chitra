API_DOCKERFILE = """

FROM python:3.7

WORKDIR /app

COPY ./requirements.txt requirements.txt

RUN pip3 install -r requirements.txt

COPY . .

EXPOSE PORT

CMD [ "uvicorn", "main:app", "--host=0.0.0.0", "--port=PORT"]
"""
