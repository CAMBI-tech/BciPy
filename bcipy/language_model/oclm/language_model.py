import json
import logging
import platform
import re
import sys
sys.path.insert(0, ".")
import time
import unittest
from subprocess import PIPE, Popen

import docker
import requests

from bcipy.helpers.bci_task_related import alphabet
from bcipy.language_model.oclm.errors import (ConnectionErr, DockerDownError,
                                              EvidenceDataStructError,
                                              NBestError, NBestHighValue,
                                              StatusCodeError)

ALPHABET = alphabet()


class LangModel:

    def __init__(self, host="127.0.0.1", port="6000", logfile="log"):
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
        self.priors = {}
#        logging.basicConfig(filename=logfile, level=logging.INFO)
#
#        try:
#            # remove existing containers
#            self.__rm_cons__(client)
#        except:
#            pass
#
#        # create a new contaienr from image
#        self.container = client.containers.run(
#            image='oclmimage:version2.0',
#            command='python server.py',
#            detach=True,
#            ports={
#                '5000/tcp': (
#                #self.port + '/tcp': (
#                    self.host,
#                    self.port)},
#            remove=True)
#        # wait for initialization
#        print("INITIALIZING SERVER..\n")
#        time.sleep(16)
#        # assert a new container was generated
#        con_id = self.container.short_id
#        for con in client.containers.list(filters={"ancestor": "oclmimage:version2.0"}):
#            con_id_fromlist = con.short_id
#        assert con_id == con_id_fromlist, \
#            "internal container exsistance failed"

    def __rm_cons__(self, client):
        """
        Remove existing containers as they
        occupy the required ports
        """
        for con in client.containers.list(filters={"ancestor": "oclmimage:version2.0"}):
            con.stop()
            con.remove()
            time.sleep(16)

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

    def recent_priors(self, return_mode):
        """
        Display the priors given the recent decision
        """
        if not bool(self.priors):
            try:
                r = requests.post(
                    'http://' +
                    self.host +
                    ':' +
                    self.port +
                    '/recent_priors',
                    json={
                        'return_mode': return_mode})
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
