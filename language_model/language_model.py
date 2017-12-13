import os
import sys
import docker
import requests
import json
import time
import re
import logging
from errors import ConnectionErr, StatusCodeError, DockerDownError
from helpers.bci_task_related import alphabet

ALPHABET = [
    'a',
    'b',
    'c',
    'd',
    'e',
    'f',
    'g',
    'h',
    'i',
    'j',
    'k',
    'l',
    'm',
    'n',
    'o',
    'p',
    'q',
    'r',
    's',
    't',
    'u',
    'v',
    'w',
    'x',
    'y',
    'z',
    '<',
    '#']


class LangModel:

    def __init__(self, localpath2fst, host, port, logfile):
        """
        Initiate the langModel class. Primarly initializing
        is aimed at establishing the tcp/ip connection
        between the host (local machine) and its server
        (the docker machine)
        Establishing the connection and running the server
        are done in a single operartion
        Input:
          localpath2fst (str) - the local path to the fst file
          host (str) - host machine ip address
          port (str) - the port used in docker
          logfile (str) - a valid filename to function as a logger 
        """
        # assert input path validity
        assert os.path.exists(os.path.dirname(
            localpath2fst)), "%r is not a valid path" % localpath2fst
        # assert strings
        assert type(host)==str, "%r is not a string type" % host
        assert type(port)==str, "%r is not a string type" % port
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

        # create a new contaienr from image
        self.container = client.containers.run(
            image='lmimage',
            command='python server.py',
            detach=True,
            ports={
                self.port + '/tcp': (
                    self.host,
                    self.port)},
            volumes=volume,
            auto_remove=True)
        # wait for initialization
        print "INITIALIZING SERVER.."
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
            assert symbol in ALPHABET, \
                "%r contains invalid symbol" % decision

        err_msg = "Connection was not extablished\nstate update failed"
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
            self._logger()
        
        self.priors = [[letter.upper(), prob] for (letter, prob) in self.priors]
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
            print "There are no priors in the history"
        # print a json dict of the priors
        return self.priors
