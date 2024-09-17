import os

import pytest

from conftest import test_db_user, test_db_password, test_db_database, test_db_port
from inferring_application import create_app

os.environ["RDS_USERNAME"] = test_db_user
os.environ["RDS_PASSWORD"] = test_db_password
os.environ["RDS_HOSTNAME"] = "localhost"
os.environ["RDS_PORT"] = test_db_port
os.environ["RDS_DB_NAME"] = test_db_database

# This model has been trained on a low number of questions for a reduced size and thus the predictions won't be that accurate
os.environ["MODEL_URL"] = "https://inferring-api-models.s3.eu-west-3.amazonaws.com/test/best_supervised_model.model"
os.environ["ML_BINARIZER_URL"] = "https://inferring-api-models.s3.eu-west-3.amazonaws.com/test/best_ml_binarizer.model"
os.environ["EMBEDDER_URL"] = "https://inferring-api-models.s3.eu-west-3.amazonaws.com/test/embedder_model.model"


@pytest.fixture
def test_client():
    test_client = create_app().test_client()
    yield test_client


def test_health_check(test_client):
    # GIVEN + WHEN
    health_check = test_client.get('/')

    # THEN
    assert health_check.status_code == 200
    assert health_check.data == b'Inferring service up!'


def test_prediction_and_events_logging(test_client):
    # GIVEN
    test_payload = {
        "title": "Does Python have a string &#39;contains&#39; substring method?",
        "body": "<p>I'm looking for a <code>string.contains</code> or <code>string.indexof</code> method in Python.</p>\n\n<p>I want to do:</p>\n\n<pre><code>if not somestring.contains(\"blah\"):\n   continue\n</code></pre>\n"
    }
    expected_predicted_tags = ['git', 'javascript', 'python']

    # WHEN
    prediction_response = test_client.post('/predict', json=test_payload)
    events_response = test_client.get('/events')

    # THEN
    assert prediction_response.status_code == 200
    assert events_response.status_code == 200

    assert "predicted_tags" in prediction_response.json
    assert prediction_response.json['predicted_tags'] == expected_predicted_tags

    assert len(events_response.json) == 1
    assert events_response.json[0]['title'] == test_payload['title']
    assert events_response.json[0]['body'] == test_payload['body']
    assert events_response.json[0]['tags'] == f"({','.join(expected_predicted_tags)})"

