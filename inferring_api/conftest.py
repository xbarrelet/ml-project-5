import docker
import pytest
from docker.models.containers import Container

# TEST CONFIG
test_db_user = 'test'
test_db_password = 'test123'
test_db_database = 'test_db'
test_db_port = '5433'

test_container: Container
smtp_server_container: Container


@pytest.hookimpl()
def pytest_sessionstart(session):
    docker_client = docker.from_env()

    # Init db and run ddl script
    session.test_container = docker_client.containers.run('postgres:12',
                                                          auto_remove=True,
                                                          ports={'5432/tcp': int(test_db_port)},
                                                          environment=['POSTGRES_USER=' + test_db_user,
                                                                       'POSTGRES_PASSWORD=' + test_db_password,
                                                                       'POSTGRES_DB=' + test_db_database],
                                                          detach=True)


@pytest.hookimpl()
def pytest_sessionfinish(session):
    session.test_container.stop()
