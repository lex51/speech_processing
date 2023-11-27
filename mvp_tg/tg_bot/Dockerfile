FROM python:3.6-alpine
ENV CELERY_BROKER_URL redis://:devpassword@redis:6379/0
ENV CELERY_RESULT_BACKEND redis://:devpassword@redis:6379/0
#ENV C_FORCE_ROOT true
ENV DEBUG true
COPY . /tg_bot
WORKDIR /tg_bot
# install requirements
RUN pip install -r requirements.txt
# expose the app port
# run the app server
# CMD ["gunicorn", "--bind", "0.0.0.0:5001", "--workers", "3", "app:server"]
CMD ["python", "main.py"]