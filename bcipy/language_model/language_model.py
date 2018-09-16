import os
import docker
import requests
import time
import re
import logging
import unittest
from errors import ConnectionErr, StatusCodeError, DockerDownError
from bcipy.helpers.bci_task_related import alphabet
from subprocess import Popen, PIPE
import platform
ALPHABET = alphabet()


class LangModel:

    def __init__(self, localpath2fst, host="127.0.0.1", port="5000", logfile="log"):
        """
        Initiate the langModel class. Primarily initializing
        is aimed at establishing the tcp/ip connection
        between the host (local machine) and its server
        (the docker machine)
        Establishing the connection and running the server
        are done in a single operation
        Input:
          localpath2fst (str) - the local path to the fst file
          host (str) - host machine ip address
          port (str) - the port used in docker
          logfile (str) - a valid filename to function as a logger
        """

        # Windows requires a special handling on to setup the access to Docker.
        os_version = platform.platform()
        if os_version.startswith('Windows'):
            # Setup the environment variables.
            docker_env_cmd = Popen('docker-machine env --shell cmd', stdout=PIPE)
            docker_instructions = docker_env_cmd.stdout.read().decode().split('\n')
            for instruction in docker_instructions:
                if instruction.startswith('SET'):
                    environ_pair_str = instruction[instruction.find(' ')+1:]
                    var_name, var_value = environ_pair_str.split('=')
                    os.environ[var_name] = var_value
            # Overrides the local ip as Windows 7 uses docker machine hence would
            # fail to bind.
            docker_machine_ip_cmd = Popen('docker-machine ip', stdout=PIPE)
            host = str(docker_machine_ip_cmd.stdout.read().strip())


        # assert input path validity
        assert os.path.exists(os.path.dirname(
            localpath2fst)), "%r is not a valid path" % localpath2fst

        # Correct the format of path in Windows
        if os_version.startswith('Windows'):
            path = re.compile(r'([a-zA-Z]):\\(.*)')
            match = path.search(localpath2fst)
            if match:
                drive = match.group(1).lower()
                remaining_path = match.group(2).replace("\\", "/")
                localpath2fst = "/{0}/{1}".format(drive, remaining_path)

        # assert strings
        assert type(host) == str, "%r is not a string type" % host
        assert type(port) == str, "%r is not a string type" % port
        # assert docker is on
        try:
            client = docker.from_env()
        except BaseException:
            raise DockerDownError  # docker ps for instance

        self.host = host
        self.port = port
        logging.basicConfig(filename=logfile, level=logging.INFO)
        dockerpath2fst = "/opt/lm/brown_closure.n5.kn.fst"
        volume = {localpath2fst: {'bind': dockerpath2fst, 'mode': 'ro'}}

        try:
            # remove existing containers
            self.__rm_cons__(client)
        except:
            pass

        # create a new container from image
        self.container = client.containers.run(
            image='lmimage',
            command='python server.py',
            detach=True,
            ports={
                self.port + '/tcp': (
                    self.host,
                    self.port)},
            volumes=volume,
            remove=True)
        # wait for initialization
        print("INITIALIZING SERVER..")
        time.sleep(1)
        # assert a new container was generated
        con_id = str(self.container.short_id)
        con_list = str(client.containers.list())
        con_id_fromlist = re.findall('Container: (.+?)>', con_list)[0]
        assert con_id == con_id_fromlist, \
            "internal container exsistance failed"

    def __rm_cons__(self, client):
        """
        Remove existing containers as they
        occupy the required ports
        """
        con_list = str(client.containers.list())
        con_ids = re.findall('Container: (.+?)>', con_list)
        if con_ids:
            for container in con_ids:
                open_con = client.containers.get(container)
                open_con.stop()
                try:
                    open_con.remove()
                except BaseException:
                    pass

    def init(self):
        """
        Initialize the language model (on the server side)
        """
        try:
            r = requests.post(
                'http://' +
                self.host +
                ':' +
                self.port +
                '/init')
        except requests.ConnectionError:
            raise ConnectionErr(self.host, self.port)
        if not r.status_code == requests.codes.ok:
            raise StatusCodeError(r.status_code)

    def reset(self):
        """
        Clean observations of the language model use reset
        """
        try:
            r = requests.post(
                'http://' +
                self.host +
                ':' +
                self.port +
                '/reset')
        except requests.ConnectionError:
            raise ConnectionErr(self.host, self.port)
        if not r.status_code == requests.codes.ok:
            raise StatusCodeError(r.status_code)
        logging.info("\ncleaning history\n")

    def state_update(self, decision):
        """
        Provide a prior distribution of the language model
        in return to the system's decision regarding the
        last observation
        Input:
            decision - a character (or a string)
        Output:
            priors - a json dictionary with the priors
        """
        # assert the input contains a valid symbol
        assert isinstance(decision, list), "%r is not list" % decision
        for symbol in decision:
            assert symbol in ALPHABET or ' ', \
                "%r contains invalid symbol" % decision

        for symbol in decision:
            try:
                r = requests.post(
                    'http://' +
                    self.host +
                    ':' +
                    self.port +
                    '/state_update',
                    json={
                        'decision': symbol.lower()})
            except requests.ConnectionError:
                raise ConnectionErr(self.host, self.port)
            if not r.status_code == requests.codes.ok:
                raise StatusCodeError(r.status_code)
            self.priors = r.json()
            self.decision = symbol.upper()
            # self._logger()

        self.priors['prior'] = [
            [letter.upper(), prob]
            if letter != '#'
            else ["_", prob]
            for (letter, prob) in self.priors['prior']]

        return self.priors

    def _logger(self):
        """
        Log the priors given the recent decision
        """
        # print a json dict of the priors
        logging.info('\nThe priors for {0} are:\n'.format(self.decision))
        for k in self.priors.keys():
            priors = self.priors[k]
            for (symbol, pr) in priors:
                logging.info('{0} {1:.4f}'.format(symbol, pr))

    def recent_priors(self):
        """
        Display the priors given the recent decision
        """
        try:
            self.priors
            self.decision
        except BaseException:
            print("There are no priors in the history")
        # print a json dict of the priors
        return self.priors

