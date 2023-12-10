# -*- coding: utf-8 -*-


import numpy as np
import re, json, unicodedata
from itertools import chain
from functools import partial
from tqdm import tqdm, trange
from base64 import b64encode, b64decode
from multiprocessing import Pool, Queue
import ahocorasick
from . import faster


def normalize(text, maxlen=0, isolate_digits=False):
    text = unicodedata.normalize('NFC', text)
    if maxlen > 0:
        if isolate_digits:
            regex = '\d|[^\n\d]{,%d}\n{1,100}|[^\n\d]{1,%d}' % (maxlen, maxlen)
        else:
            regex = '.{,%d}\n{1,100}|.{1,%d}' % (maxlen, maxlen)
    else:
        if isolate_digits:
            regex = '\d|[^\n\d]*\n+|[^\n\d]+'
        else:
            regex = '.*\n+|.+'
    return [t.encode() for t in re.findall(regex, text)]

class Tokenizer:
    """Unigram tokenizer with Aho-Corasick automaton
    """
    def __init__(self, dict_path="deepctrl_dict.txt", seed=None):
        self._pieces = {}
        for line in open(dict_path):
            line = line.strip()
            word, freq, pos = line.split()
            self._pieces[word] = [int(freq), pos]
       
        # Aho-Corasick automaton
        log_total = np.log(sum([_[0] for _ in self._pieces.values()]))
        self._automaton = ahocorasick.Automaton()
        for k, v in self._pieces.items():
            self._automaton.add_word(k, (len(k), np.log(v[0]) - log_total, v[1]))
        self._automaton.make_automaton()
        self.set_seed(seed)

    def set_seed(self, seed):
        if seed is not None:
            faster.set_seed(seed)

    def _tokenize(self, text):
        return faster._tokenize(self, text)

    def tokenize(self, text, iter=False):
        pieces = chain(*(self._tokenize(t) for t in normalize(text)))
        if iter:
            return pieces
        return list(pieces)


if __name__ == "__main__":
    tokenizer = Tokenizer()
    text = "今天真高兴啊"
    res = tokenizer.tokenize(text)
    print(res)


    

    

    