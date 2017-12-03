import os
import sys
import docker
import requests
import json
import time
import re
import logging
from errors import ConnectionErr, StatusCodeError, DockerDownError


class LangModel:

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

    def __init__(self, localpath2fst, host, port, logfile):
        """
        Initiiate the langModel class. Primarly initializing
        is aimed at establishing the tcp/ip connection
        between the host (local machine) and its server
        (the docker machine)
        Establishing the connection and running the server
        are done in a single operartion
        """
        # assert input path validity
        assert os.path.exists(os.path.dirname(
            localpath2fst)), "%r is not a valid path" % localpath2fst
        # assert docker is on
        try:
            docker.from_env()
        except BaseException:
            raise DockerDownError  # docker ps for instance

        self.host = host
        self.port = port
        logging.basicConfig(filename=logfile, level=logging.INFO)
        dockerpath2fst = "/opt/lm/brown_closure.n5.kn.fst"
        volume = {localpath2fst: {'bind': dockerpath2fst, 'mode': 'ro'}}
        # get docker client
        client = docker.from_env()
        # remove existing containers
        self.__rm_cons__(client)
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
            assert symbol in LangModel.ALPHABET, \
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
                        'decision': symbol})
            except requests.ConnectionError:
                raise ConnectionErr(self.host, self.port)
            if not r.status_code == requests.codes.ok:
                raise StatusCodeError(r.status_code)
            self.priors = r.json()
            self.decision = symbol
            self._logger()
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

if __name__ == '__main__':

    localfst = "/Users/dudy/CSLU/bci/5th_year/letters/pywrapper/lm/brown_closure.n5.kn.fst"
    lmodel = LangModel(
        localfst,
        host='127.0.0.1',
        port='5000',
        logfile="lmwrap.log")
    lmodel.init()
    priors = lmodel.state_update('t')
    lmodel.recent_priors()
    priors = lmodel.state_update('h')
    lmodel.recent_priors()
    priors = lmodel.state_update('e')
    lmodel.reset()
    lmodel.recent_priors()
    priors = lmodel.state_update('t')
    lmodel.recent_priors()
