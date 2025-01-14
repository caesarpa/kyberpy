import numpy as np
import hashlib as hl
import os
from util import *
from hash import *
from params import *
from ntt import *


def PKE_CPA_KeyGen(d: bytes):
    # d is a seed of length 32 bytes
    seed = d + intToBytes(k)
    rho, sigma = G(seed)
    counter = 0
    A_hat = [[None for _ in range(k)] for _ in range(k)]

    for i in range(0, k):
       
       for j in range(0, k):
           A_hat[i][j] = sampleNTT(rho + intToBytes(j) + intToBytes(i)) # generate matrix of ntt polynomials with 34 byte input
    
    s = [None] * k
    e = [None] * k
    for i in range(0, k):

        secretSeed = PRF(sigma, intToBytes(counter), η1)
        errorSeed = PRF(sigma, intToBytes(counter + 2), η1)

        s[i] = CBD(η1, secretSeed)
        e[i] = CBD(η1, errorSeed)

        counter += 1
    
    s_hat = [None] * k
    e_hat = [None] * k
    for i in range(len(s)):
        s_hat[i] = NTT(s[i])
        e_hat[i] = NTT(e[i])
    
    t_hat = vector_addition(matrix_vector_ntt_product(A_hat, s_hat), e_hat)

    ek = [t_hat, rho] #TODO: perhaps use encode? but not necessary
    dk = s_hat

    return ek, dk

def PKE_Encrypt(ek, m: bytes, r: bytes):

    counter = 0
    t_hat = ek[0]
    rho = ek[1]

    A_hat = [[None for _ in range(k)] for _ in range(k)] #Recompute matrix A_hat so it does not need to be sent

    for i in range(0, k):
       for j in range(0, k):
           A_hat[i][j] = sampleNTT(rho + intToBytes(j) + intToBytes(i)) # generate matrix of ntt polynomials with 34 byte input
    
    y = [None] * k
    for i in range(k):
        y[i] = CBD(η1, PRF(r, intToBytes(counter), η1))
        counter += 1
    
    e1 = [None] * k
    for i in range(k):
        e1[i] = CBD(η2, PRF(r, intToBytes(counter), η2))
        counter += 1

    e2 = CBD(η2, PRF(r, intToBytes(counter), η2))
    y_hat = [None] * k
    for i in range(k):
        y_hat[i] = NTT(y[i])
    
    interim = transpose_matrix_vector_ntt_product(A_hat, y_hat)
    u = [None] * k
    for i in range(k):
        u[i] = INTT(interim[i])
    u = vector_addition(u, e1)

    interim = byteDecode(m, 1) #converts m to an bit array len 256
    mu = []
    for i in range(n):
        mu.append(decompress(interim[i], 1))

    interim = vector_ntt_dot_product(t_hat, y_hat)
    interim = INTT(interim)
    interim = poly_addition(interim, e2)
    v = poly_addition(interim, mu)

    compressed_u = [None] * k
    for i in range(k):
        compressed_u[i] = [None] * n
        for j in range(n):
            compressed_u[i][j] = compress(u[i][j], du)
    
    c1 = [None] * k
    for i in range(k):
        c1[i] = byteEncode(compressed_u[i], du)
    
    compressed_v = [None] * n
    for i in range(n):
        compressed_v[i] = compress(v[i], dv)
    
    c2 = byteEncode(compressed_v, dv)

    return [c1, c2]

def PKE_Decrypt(dk, c):
    c1 = c[0]
    c2 = c[1]

    for i in range(k):
        c1[i] = byteDecode(c1[i], du)
    
    u = [None] * k
    for i in range(k):
        u[i] = [None] * n
        for j in range(n):
            u[i][j] = decompress(c1[i][j], du)

    c2 = byteDecode(c2, dv)
    v = [None] * n
    for i in range(n):
        v[i] = decompress(c2[i], dv)

    s_hat = dk

    u_hat = []
    for i in range(k):
        u_hat.append(NTT(u[i]))
    
    interim = vector_ntt_dot_product(s_hat, u_hat)
    interim = INTT(interim)
    w = poly_subtraction(v, interim)

    for i in range(n):
        w[i] = compress(w[i], 1)
    
    m = byteEncode(w, 1)
    return m # TODO this is bytes, make sure its correct to do so
