FROM aniketmaurya/chitra:latest


WORKDIR /app

COPY ./requirements.txt requirements.txt

RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE PORT

CMD [ "uvicorn", "main:app", "--host=0.0.0.0", "--port=PORT"]
