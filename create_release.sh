cd inferring_api &&
zip -r release.zip Dockerfile Dockerrun.aws.json inferring_application.py inferring_application_requirements.txt models/ &&
mv release.zip ../