import os

import pytest

from inferring_api.conftest import test_db_user, test_db_password, test_db_database, test_db_port
from inferring_api.inferring_application import create_app

os.environ["RDS_USERNAME"] = test_db_user
os.environ["RDS_PASSWORD"] = test_db_password
os.environ["RDS_HOSTNAME"] = "localhost"
os.environ["RDS_PORT"] = test_db_port
os.environ["RDS_DB_NAME"] = test_db_database
os.environ["MODEL_URL"] = "https://inferring-api-models.s3.eu-west-3.amazonaws.com/test/best_supervised_model.model"
os.environ["TAGS_URL"] = "https://inferring-api-models.s3.eu-west-3.amazonaws.com/test/tags.json"


@pytest.fixture
def test_client():
    test_client = create_app().test_client()
    yield test_client


def test_health_check(test_client):
    # WHEN
    health_check = test_client.get('/')

    # THEN
    assert health_check.status_code == 200
    assert health_check.data == b'Inferring service up!'
