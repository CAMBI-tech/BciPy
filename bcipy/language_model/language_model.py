import os
import requests
import docker
import time
import json
import logging
import sys
sys.path.append('.')
from bcipy.helpers.bci_task_related import alphabet
from bcipy.language_model.errors import (ConnectionErr, DockerDownError,
                                              EvidenceDataStructError,
                                              NBestError, NBestHighValue,
                                              StatusCodeError)
ALPHABET = alphabet()


class LangModel:

    def __init__(self, lmtype, logfile="log"):
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
        assert type(
            lmtype.host) == str, "%r is not a string type" % host
        assert type(
            lmtype.port) == str, "%r is not a string type" % port
        # assert docker is on
        try:
            client = docker.from_env()
        except BaseException:
            raise DockerDownError  # docker ps for instance

        self.host = lmtype.host
        self.port = lmtype.port
        self.dport = lmtype.dport
        self.image = lmtype.image
        self.lmtype = lmtype.type
        self.priors = {}
        logging.basicConfig(filename=logfile, level=logging.INFO)
        try:
            # remove existing containers
            self.__rm_cons__(client)
        except:
            pass
        # create a new contaienr from image
        if self.lmtype == 'oclm':

            self.container = client.containers.run(
                image=self.image,
                command='python server.py',
                detach=True,
                ports={
                    self.dport+'/tcp': (
                        self.host,
                        self.port)},
                remove=True)

        elif self.lmtype == 'prelm':
            # assert input path validity
            assert os.path.exists(os.path.dirname(
                lmtype.localfst)), "%r is not a valid path" % lmtype.localfst

            dockerpath2fst = "/opt/lm/brown_closure.n5.kn.fst"
            volume = {lmtype.localfst: {
                'bind': dockerpath2fst, 'mode': 'ro'}}

            # create a new container from image
            self.container = client.containers.run(
                image=self.image,
                command='python server.py',
                detach=True,
                ports={
                    self.dport + '/tcp': (
                        self.host,
                        self.port)},
                volumes=volume,
                remove=True)

        # wait for initialization
        print("INITIALIZING SERVER..\n")
        time.sleep(16)
        # assert a new container was generated
        con_id = self.container.short_id
        for con in client.containers.list(filters={"ancestor": self.image}):
            con_id_fromlist = con.short_id
        assert con_id == con_id_fromlist, \
            "internal container exsistance failed"

    def __rm_cons__(self, client):
        """
        Remove existing containers as they
        occupy the required ports
        """
        for con in client.containers.list(filters={"ancestor": self.image}):
            con.stop()
            con.remove()

    def init(self, nbest=1):
        """
        Initialize the language model (on the server side)
        Input:
            nbest - top N symbols from evidence
        """
        if self.lmtype == 'oclm':
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
                json={'nbest': nbest})  # prelm accept nbest but ignore
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
        Both lm types allow providing more the one timestep
        input. Pay attention to the data struct expected.
        OCLM
        Input:
            evidence - a list of (list of) tuples [[(sym1, prob), (sym2, prob2)]]
            the numbers are assumed to be in the log probabilty domain
        Output:
            priors - a json dictionary with character priors
            word - a json dictionary w word probabilites
            both in the Negative Log probabilty domain
        PRELM
        Input:
            decision - a symbol or a string of symbols in encapsulated in a
            list
        Output:
            priors - a json dictionary with the priors
                     in the Negative Log probabilty domain
        """
        # assert the input contains a valid symbol
        if self.lmtype == 'oclm':

            assert isinstance(
                evidence, list), "%r is not list" % evidence
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

        elif self.lmtype == 'prelm':

            decision = evidence  # in prelm the we treat it as a decision
            assert isinstance(
                decision, list), "%r is not list" % decision
            for symbol in decision:
                assert symbol in ALPHABET or ' ', \
                    "%r contains invalid symbol" % decision
            clean_evidence = []
            for symbol in decision:
                if symbol == '_':
                    symbol = '#'
                clean_evidence.append(symbol.lower())
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
            assert self.lmtype == 'oclm', "%r is not allowing for non-letters output" % self.lmtype
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

    def recent_priors(self, return_mode='letter'):
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
                assert self.lmtype == 'oclm', "%r is not allowing for non-letters output" % self.lmtype
                self.priors['word'] = output['word']
        return self.priors
