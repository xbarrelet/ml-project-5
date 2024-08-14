FROM python:3.12

ADD Dockerfile /
ADD Dockerrun.aws.json /
ADD inferring_application.py /
ADD inferring_application_requirements.txt /
ADD models/best_supervised_model.model /
ADD models/tags.json /

RUN pip3 install -r inferring_application_requirements.txt

EXPOSE 8000

CMD ["gunicorn", "--bind", "0.0.0.0:8000", "inferring_application:app"]