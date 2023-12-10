# cython: language_level=3
from libc.time cimport time
from libc.stdlib cimport RAND_MAX, rand, srand
from libc.math cimport INFINITY, exp, log

srand(time(NULL))


cpdef set_seed(unsigned int seed):
    srand(seed)


cdef inline double logsumexp(double x, double y):
    if x < y:
        x, y = y, x
    return x + log(1 + exp(y - x))



def _tokenize(self, str text):
    cdef int e, k, s
    cdef double score
    cdef list v
    cdef list scores = [0] + [-INFINITY] * len(text)
    cdef list routes = list(range(len(text) + 1))
    cdef list tokens = []
    for e, (k, v) in self._automaton.iter(text):
        s, e = e - k + 1, e + 1
        score = scores[s] + v[0]
        if score > scores[e]:
            scores[e], routes[e] = score, s
       
    while text:
        s = routes[e]
        tokens.append(text[s:e])
        text, e = text[:s], s
    return tokens[::-1]
