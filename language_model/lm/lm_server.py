from __future__ import division
import sys
import subprocess
import random
from bitweight import *
import math
from collections import defaultdict
import csv
import pywrapfst as fst

class server:
    def __init__(self, filename):
        # read LM
        self.lm = fst.Fst.read(filename)
        self.lm.arcsort(st="ilabel")
        # read symbols
        self.lm_syms = self.lm.input_symbols()
        # Sigma machine
        self.sigma = None
        # History array of this typing process.
        self.history = []
        # Do the actual initialization.
        self.init()
        # Set up the valid characters and corresponding indices.
        self.legit_ch_dict = {\
                'a':0,\
                'b':1,\
                'c':2,\
                'd':3,\
                'e':4,\
                'f':5,\
                'g':6,\
                'h':7,\
                'i':8,\
                'j':9,\
                'k':10,\
                'l':11,\
                'm':12,\
                'n':13,\
                'o':14,\
                'p':15,\
                'q':16,\
                'r':17,\
                's':18,\
                't':19,\
                'u':20,\
                'v':21,\
                'w':22,\
                'x':23,\
                'y':24,\
                'z':25,\
                '<':26,\
                '#':27} # Space is represented as '#'

    def init(self):
        # build sigma
        self.sigma = fst.Fst()
        self.sigma.set_input_symbols(self.lm_syms)
        self.sigma.set_output_symbols(self.lm_syms)
        self.sigma.add_state()
        self.sigma.set_start(0)
        for code, ch in self.lm_syms:
            if code == 0:  # Don't include epsilon in sigma machine.
                continue
            self.sigma.add_arc(0, fst.Arc(code, code, None, 1))
        self.sigma.add_state()
        self.sigma.set_final(1)

        # since there's no history do not concat sigma with anything.
        # The history for now will be an array of characters,
        # because for creating arcs in pyfst, it needs a string instead of the
        # integer id.
        self.history = []


    def reset(self):
        # History array of this typing process.
        self.history = []
        # Do the actual initialization.
        self.init()

    def undo(self):
        '''
        Undo previous typing. This effectively deletes the last character
        in typing history.
        '''
        self.history = self.history[:-1]

    def get_prior_array(self):
        '''
        Get negative-log space probability distribution over the next character.

        OUTPUTS:
            an array of minus log probabilities, sorted according to the value
            in self.legit_ch_dict.
        '''
        sorted_prior = self.get_prior()
        sorted_by_key = sorted(sorted_prior, \
                key=lambda prior:self.legit_ch_dict[prior[0]])
        probs = [prob for _, prob in sorted_by_key]
        # Manually adds the probability of backspace to the array, which is
        # always 0 for now.
        probs.insert(self.legit_ch_dict['<'], 100.00)
        return probs

    def get_real_prior_array(self):
        '''
        Get real-space probability distribution over the next character.

        OUTPUTS:
            an array of real probabilities, sorted according to the value in
            self.legit_ch_dict.
        '''
        probs = self.get_prior_array()
        return [math.exp(-p) for p in probs]


    def get_prior(self):
        '''
        set an array with priors
        in future priors are given from rsvp EEG vector

        OUTPUTS:
            an array of tuples, which consists of the character and the
            corresponding probabilities.
        '''
        sigma_h = self.create_machine_history()

        # intersect
        sigma_h.arcsort(st="olabel")
        output_dist = fst.intersect(sigma_h, self.lm)

        # process result
        output_dist = fst.rmepsilon(output_dist)
        output_dist = fst.determinize(output_dist)
        output_dist.minimize()
        output_dist = fst.push(output_dist, push_weights=True, to_final=True)

        # worth converting this history to np.array if vector computations
        # will be involeved
        #output_dist.arcsort(st="olabel")

        # traverses the shortest path until we get to the second to
        # last state. And the arcs from that state to the final state contain
        # the distribution that we want.
        prev_stateid = curr_stateid = None
        for state in output_dist.states():
            if not curr_stateid is None:
                prev_stateid = curr_stateid
            curr_stateid = state
        priors = []
        for arc in output_dist.arcs(prev_stateid):
            ch = self.lm_syms.find(arc.ilabel) #ilabel and olabel are the same.
            w = float(arc.weight)

            # TODO: for this demo we only need distribution over the characters
            # from 'a' to 'z'
            if len(ch) == 1 and ch in self.legit_ch_dict:
                priors.append((ch, w))

        # assuming the EEG input is an array like [("a", 0.3),("b", 0.2),...]
        # sort the array depending on the probability
        priors = sorted(priors, key=lambda prior: prior[1])
        normalized_dist = self._normalize([prob for _, prob in priors])
        return zip([ch for ch, _ in priors], normalized_dist)

    def update_symbol(self, decision):
        '''
        update history given new input from rsvp
        add arc to last state
        '''
        if decision == "<":
          undo()
        else:
          self.history.append(decision)

    def get_history(self):
        '''
        Return the symbols in the history.
        '''
        return self.history

    def create_machine_history(self):
        '''
        Creates a fst which contains the history and ready to compute
        the distribution over next character.
        '''
        # initiate sigma_h with sigma
        sigma_h = self.sigma
        if self.history:
            # build fst with previous "chosen" characters
            sigma_h = fst.Fst()
            sigma_h.set_input_symbols(self.lm_syms)
            sigma_h.set_output_symbols(self.lm_syms)
            sigma_h.add_state()
            sigma_h.set_start(0)
            for i, ch in enumerate(self.history):
                key = self.lm_syms.find(ch)
                sigma_h.add_arc(i, fst.Arc(key, key, None, i+1))
                sigma_h.add_state()
            sigma_h.set_final(i+1)

            # add sigma at the end
            sigma_h.concat(self.sigma)
            sigma_h = fst.rmepsilon(sigma_h)
        return sigma_h

    def symbol(self):
        symbols = self.legit_ch_dict.keys()
        for i in xrange(len(symbols)):
            if symbols[i] == '#':
                symbols[i] = ' '
        return symbols


    def _normalize(self, distribution):
        '''
        Normalize the distribution which is in negative log (e based) space.

        INPUTS:
            distribution, an array of floating values which are in negative log
            (e based space).

        OUTPUTS:
            an array of normalized distribution in negative log (e based) space.
        '''
        transformed_dist = [BitWeight(math.e**(-prob)) for prob in distribution]
        total = sum(transformed_dist)
        normalized_dist = [(prob / total).to_real for prob in transformed_dist]
        return [-math.log(prob) for prob in normalized_dist]

def sum_up(pr_vector1, pr_vector2):
    '''
    INPUT:
       two vectors of symbol distribution. Each cell
       is of the form (symbol, float)
    OUTPUT:
       one vector with the coresponding sum over shared
       symbols.
    '''
    dist1 = sorted(pr_vector1, key=lambda symbol: symbol[0])
    dist2 = sorted(pr_vector2, key=lambda symbol: symbol[0])
    dist1_2 = [(s1, p1+p2) for ((s1, p1),(s2, p2)) in zip(dist1,dist2)]
    return sorted(dist1_2, key=lambda symbol: symbol[1])

def min_symbol(dist):
    return dist[0][0]

def error_rate(ref, pred):
    '''
    INPUT:
       two strings
    DISPLAY:
       error rate
    '''
    errs = [s1 for s1, s2 in zip(ref,pred) if s1 != s2]
    print "The word ",ref, " had ", len(errs) / len(ref) * 100, "% error rate"
    return len(errs)
