FROM python:3.7

WORKDIR /app

COPY ./results/model.h5
COPY ./app.py
RUN pip install flask tensorflow tensorflow_hub

ENV FLASK_APP=/app/app.py

CMD ['flask run']