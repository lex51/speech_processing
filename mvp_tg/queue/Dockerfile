FROM python:3.6-alpine

ENV CELERY_BROKER_URL redis://:devpassword@redis:6379/0
ENV CELERY_RESULT_BACKEND redis://:devpassword@redis:6379/0
#ENV C_FORCE_ROOT true

COPY . /queue
WORKDIR /queue

RUN pip install -r requirements.txt

CMD ["celery", "-A", "tasks", "worker", "--loglevel=info"]
