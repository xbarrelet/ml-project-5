FROM python:3.12

ADD inferring_application_requirements.txt /
RUN pip3 install -r inferring_application_requirements.txt

RUN python -m nltk.downloader wordnet
RUN python -m nltk.downloader punkt

ADD inferring_application.py /

EXPOSE 8000

CMD ["python3", "inferring_application.py"]