import os
import sys
sys.path.insert(0, ".")
import docker
import requests
import json
import time
import re
import logging
import unittest
from bcipy.oclm_language_model.errors import ConnectionErr, StatusCodeError, DockerDownError, NBestHighValue, EvidenceDataStructError, NBestError
from bcipy.helpers.bci_task_related import alphabet
from subprocess import Popen, PIPE
import platform
ALPHABET = alphabet()


class LangModel:

    def __init__(self, host="127.0.0.1", port="5000", logfile="log"):
        """
        Initiate the langModel class. Primarly initializing
        is aimed at establishing the tcp/ip connection
        between the host (local machine) and its server
        (the docker machine)
        Establishing the connection and running the server
        are done in a single operartion
        Input:
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
                    environ_pair_str = instruction[instruction.find(' ') + 1:]
                    var_name, var_value = environ_pair_str.split('=')
                    os.environ[var_name] = var_value
            # Overides the local ip as Windows 7 uses docker machine hence would
            # fail to bind.
            docker_machine_ip_cmd = Popen('docker-machine ip', stdout=PIPE)
            host = docker_machine_ip_cmd.stdout.read().decode().strip()

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

        try:
            # remove existing containers
            self.__rm_cons__(client)
        except:
            pass

        # create a new contaienr from image
        self.container = client.containers.run(
            image='oclmimage',
            command='python server.py',
            detach=True,
            ports={
                self.port + '/tcp': (
                    self.host,
                    self.port)},
            remove=True)
        # wait for initialization
        print("INITIALIZING SERVER..\n")
        time.sleep(16)
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

    def init(self, nbest):
        """
        Initialize the language model (on the server side)
        Input:
            nbest - top N symbols from evidence
        """
        try:
            assert isinstance(nbest, int)
        except BaseException:
            raise NBestError(nbest)

        if nbest > 4:
            raise NBestHighValue(nbest)
        try:
            r = requests.post(
                'http://' +
                str(self.host) +
                ':' +
                self.port +
                '/init',
                json={'nbest': nbest})
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

    def state_update(self, evidence, return_mode='letter'):
        """
        Provide a prior distribution of the language model
        in return to the system's decision regarding the
        last observation
        Input:
            evidence - a list of (list of) tuples [[(sym1, prob), (sym2, prob2)]]
        Output:
            priors - a json dictionary with character priors
            word - a json dictionary w word probabilites
        """
        # assert the input contains a valid symbol
        assert isinstance(evidence, list), "%r is not list" % evidence
        try:
            clean_evidence = []
            for tmp_evidence in evidence:
                tmp = []
                for (symbol, pr) in tmp_evidence:
                    assert symbol in ALPHABET, \
                        "%r contains invalid symbol" % evidence
                    if symbol == "_":
                        tmp.append(("#", pr))
                    else:
                        tmp.append((symbol.lower(), pr))
                clean_evidence.append(tmp)
        except:
            raise EvidenceDataStructError

        try:
            r = requests.post(
                'http://' +
                self.host +
                ':' +
                self.port +
                '/state_update',
                json={
                    'evidence': clean_evidence, 'return_mode': return_mode})
        except requests.ConnectionError:
            raise ConnectionErr(self.host, self.port)
        if not r.status_code == requests.codes.ok:
            raise StatusCodeError(r.status_code)
        output = r.json()
        self.priors = {}

        self.priors['letter'] = [
            [letter.upper(), prob]
            if letter != '#'
            else ["_", prob]
            for (letter, prob) in output['letter']]

        if return_mode != 'letter':
            self.priors['word'] = output['word']

        return self.priors

        # self._logger()

    def _logger(self):
        """
        Log the priors given the recent decision
        """
        # print a json dict of the priors
        logging.info('\nThe priors are:\n')
        for k in self.priors.keys():
            priors = self.priors[k]
            logging.info('\nThe priors for {0} type are:\n'.format(k))
            for (symbol, pr) in priors:
                logging.info('{0} {1:.4f}'.format(symbol, pr))

    def recent_priors(self):
        """
        Display the priors given the recent decision
        """
        try:
            self.priors
        except BaseException:
            print("There are no priors in the history")
        # print a json dict of the priors
        return self.priors


if __name__ == '__main__':
    unittest.main()
