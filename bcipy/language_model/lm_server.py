"""Manages the creation and shutdown of the language model server"""
import logging
import os
import time
from typing import Dict, List
import docker
import requests
from bcipy.language_model.errors import DockerDownError, ConnectionErr, \
    StatusCodeError
log = logging.getLogger(__name__)
# pylint: disable=too-few-public-methods,too-many-arguments


class LmServerConfig():
    """Config for starting the language model server (docker container).
    Parameters:
    -----------
        image - docker image name
        host - host on which to start the server
        port - port to access server
        docker_port - port within the container
        volumes - mapping between local file and docker container file.
    """

    def __init__(self, image: str, host: str = "127.0.0.1",
                 port: int = 5000, docker_port: int = 5000,
                 volumes: Dict = None):
        self.image = image
        self.host = host
        self.port = port
        self.docker_port = docker_port
        self.volumes = volumes or {}
        self.command = "python server.py"

    def asdict(self):
        """Returns a dict representation of the class."""
        return docker_run_params(self)


def docker_run_params(config: LmServerConfig) -> Dict:
    """Dict used to parameterize docker-py's container run method."""
    return dict(image=config.image,
                command=config.command,
                ports={str(config.docker_port) +
                       '/tcp': (config.host, str(config.port))},
                volumes=docker_volumes(config.volumes),
                detach=True,
                remove=True)


def running_containers(client, server_config: LmServerConfig) -> List:
    """List the running Containers given the server_config"""
    return client.containers.list(filters={"ancestor": server_config.image})


def stop(server_config: LmServerConfig):
    """Stop the given docker image if it is currently running.
    """
    try:
        client = docker.from_env()
    except BaseException:
        raise DockerDownError  # docker ps for instance
    for con in running_containers(client, server_config):
        try:
            log.debug(f"Stopping existing container: {con.name}")
            con.stop()
            # Remove may throw an exception
            con.remove()
        except BaseException:
            pass


def start(server_config: LmServerConfig, max_wait: int = 16,
          start_complete_msg: str = "Running"):
    """Starts the language model server. This is currently implemented as a
    docker container that needs to be started, given the supplied config.
    Assumes that docker is installed and the correct server images have been
    loaded.

    Parameters:
    -----------
        server_config - configuration to startup the docker image
        max_wait - max seconds to wait during startup before returning.
        start_complete_msg - message that should be displayed in the (docker)
            logs when startup has completed.
    """
    try:
        client = docker.from_env()
    except BaseException:
        raise DockerDownError  # docker ps for instance

    # remove existing containers
    stop(server_config)

    run_params = server_config.asdict()
    t_start = time.time()
    try:
        container = client.containers.run(**run_params)
    except Exception:
        raise Exception("Error starting container. Try restarting docker")

    # wait for initialization
    log.debug("INITIALIZING LM SERVER.")
    log.debug(run_params)

    # wait for message in the logs or max_wait before continuing.
    sleep_interval = 0.5
    for _ in range(int(max_wait / sleep_interval)):
        if start_complete_msg in str(container.logs()):
            break
        time.sleep(sleep_interval)
    t_end = time.time()
    if start_complete_msg in str(container.logs()):
        log.debug(f"Container started in {t_end - t_start} seconds")
    else:
        log.debug("Wait time exceeded. Container may not be fully initialized.")

    # assert a new container was generated
    running_container_ids = [
        c.short_id for c in running_containers(client, server_config)]
    if not container.short_id in running_container_ids:
        raise Exception("Container did not correctly start.")


def post_json_request(server_config: LmServerConfig,
                      path: str, data: Dict = None):
    """Posts a JSON request to the given url.
    Returns:
    --------
        json response if any or None.
    """
    data = data or {}
    host = server_config.host
    port = server_config.port
    url = f'http://{host}:{port}/{path}'
    try:
        response = requests.post(url, json=data)
    except requests.ConnectionError:
        raise ConnectionErr(host, port)
    if not response.status_code == requests.codes.ok:
        raise StatusCodeError(response.status_code)
    try:
        return response.json()
    except ValueError:
        # no JSON returned
        return None


def docker_volumes(volumes: Dict) -> Dict:
    """Converts {local_path: docker_path} dict to the format required by
    docker py. Checks validity of local path.
    """
    vols = {}
    for local_path, docker_path in volumes.items():
        assert os.path.exists(os.path.dirname(local_path)
                              ), "%r is not a valid path" % local_path
        vols[local_path] = {'bind': docker_path, 'mode': 'ro'}
    return vols


def main():
    """Starts a docker container for the given language model."""
    from bcipy.language_model.lm_modes import LmType
    import argparse

    lm_options = '; '.join(
        [f'{i+1} => {str(opt)}' for i, opt in enumerate(LmType)])
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--type', default=1, type=int,
                        help=f'Language model type. Options: ({lm_options})')
    args = parser.parse_args()

    config = LmType(args.type).model.DEFAULT_CONFIG
    start(config)


if __name__ == '__main__':
    main()
