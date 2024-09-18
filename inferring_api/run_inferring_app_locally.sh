docker build . -t inferring_app &&
docker run --rm -p 8000:8000 inferring_app