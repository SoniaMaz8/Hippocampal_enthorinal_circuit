import numpy as np
import pylab as plt
import scipy.signal
from scipy.stats import vonmises
import random
import copy


def roots(z, n):
    """"
    n roots of z.
    """
    nthRootOfr = np.abs(z)**(1.0/n)
    t = np.angle(z)
    return map(lambda k: nthRootOfr*np.exp((t+2*k*np.pi)*1j/n), range(n))


def cvecl(N, loopsize=None):
    """"
    N random unit roots of order loopsize (e^2kpi*i/loopsize).
    This choice is made for the translation to loop: translation to the right make the
       image appear on the left.
    If loopsize=None, then loopsize=N.
    """
    if loopsize is None:
        loopsize = N

    unity_roots = np.array(list(roots(1.0 + 0.0j, loopsize)))
    root_idxs = np.random.randint(loopsize, size=N)
    X1 = unity_roots[root_idxs]

    return X1



def crvec(N, D=1):
    """"
    Random complex vector of size N in the format a+jb
    """
    rphase = 2*np.pi * np.random.rand(D, N)
    return np.cos(rphase) + 1.0j * np.sin(rphase)




def norm(X):
    # normalize between -1 and 1
    return 2*(X-np.min(X, axis=0))/(np.max(X)-np.min(X))-1


def update_resonator_digit_async(codebooks, resonator, scene):
    resonator_update = copy.copy(resonator)
    for i in range(len(codebooks)):
        new_code = scene
        for j in range(len(codebooks)):
            if i != j:
                new_code = new_code*(resonator_update[j, :]**-1)
        new_code = np.dot(codebooks[i].T, np.dot(
            np.conj(codebooks[i]), new_code.T))
#         new_code = np.dot(outer_products[i],new_code.T)
        new_code = new_code / np.abs(new_code)
        resonator_update[i, :] = new_code
    return resonator_update



def update_resonator_digit(codebooks, resonator, scene):
    resonator_update = np.ones((len(codebooks)), dtype=complex)
    for i in range(len(codebooks)):
        new_code = scene
        for j in range(len(codebooks)):
            if i != j:
                new_code = new_code*(resonator[j, :]**-1)
        new_code = np.dot(codebooks[i].T, np.dot(
            np.conj(codebooks[i]), new_code.T))
        new_code = new_code / np.abs(new_code)
        resonator_update[i, :] = new_code
    return resonator_update


def update_resonator_digit_async(codebooks, resonator, scene):
    resonator_update = copy.copy(resonator)
    for i in range(len(codebooks)):
        new_code = scene
        for j in range(len(codebooks)):
            if i != j:
                new_code = new_code*(resonator_update[j, :]**-1)
        new_code = np.dot(codebooks[i].T, np.dot(
            np.conj(codebooks[i]), new_code.T))
        new_code = new_code / np.abs(new_code)
        resonator_update[i, :] = new_code
    return resonator_update


def g(x):
    return x / np.abs(x)


def gen_res_digit(resonator, codebooks, max_iters, tree):
    res_hist = []
    res_curr = resonator
    for i in range(max_iters):
        res_hist.append(copy.copy(res_curr))
        res_curr = update_resonator_digit_async(codebooks, res_curr, tree)
        if np.mean(np.cos(np.angle(np.ndarray.flatten(res_curr))-np.angle(np.ndarray.flatten(res_hist[-1])))) > 0.99:
            break
    res_hist.append(copy.copy(res_curr))
    return i+1, res_hist


def dot_complex(vec1, vec2):
    num = np.dot(np.conj(vec1), vec2)
    denom = np.linalg.norm(vec1)*np.linalg.norm(vec2)
    return np.abs(num)/denom