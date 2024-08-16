FROM python:3.12

ADD inferring_application_requirements.txt /
RUN pip3 install -r inferring_application_requirements.txt

ADD inferring_application.py /

ADD models/best_supervised_model.model /
ADD models/tags.json /

EXPOSE 8000

CMD ["gunicorn", "--bind", "0.0.0.0:8000", "inferring_application:app"]