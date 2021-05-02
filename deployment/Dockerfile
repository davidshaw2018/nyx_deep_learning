FROM python:3.7

WORKDIR /app

COPY ./results/model/ /app/model
COPY ./app.py /app/
RUN pip install flask tensorflow tensorflow_hub tensorflow_text

ENV FLASK_APP=/app/app.py