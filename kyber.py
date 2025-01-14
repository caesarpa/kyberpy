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
    c = [c[0][:], c[1][:]] #copy c to avoid modifying the original

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
    return m


def ML_KEM_KEYGEN_INTERNAL(d, z):
    ek, dk = PKE_CPA_KeyGen(d)

    ek_bytes = b''
    for i in range(k):
        ek_bytes += byteEncode(ek[0][i], 12)
    ek_bytes += ek[1]

    dk = [dk, ek, H(ek_bytes), z]

    return ek, dk

def ML_KEM_ENCAPS(ek, m):

    ek_bytes = b''
    for i in range(k):
        ek_bytes += byteEncode(ek[0][i], 12)
    ek_bytes += ek[1]

    K, r = G(m + H(ek_bytes))

    c = PKE_Encrypt(ek, m, r)

    return K, c

def ML_KEM_DECAPS(dk_ML, c):

    dk = dk_ML[0]
    ek = dk_ML[1]
    h = dk_ML[2]
    z = dk_ML[3]
    m = PKE_Decrypt(dk, c)

    K_prime, r_prime = G(m + h)
    c_bytes = b'' #Transform c into byte form to pass into J
    for i in range(k):
        c_bytes += c[0][i]
    c_bytes += c[1]

    K_line = J(z + c_bytes)
    c_prime = PKE_Encrypt(ek, m, r_prime)
    if c == c_prime:
        print("Success")
        return K_prime
    else:
        print("Failure")
        return K_line

#Test decaps
d = os.urandom(32)
z = os.urandom(32)
ek, dk = ML_KEM_KEYGEN_INTERNAL(d, z)
m = os.urandom(32)
K, c = ML_KEM_ENCAPS(ek, m)
K_prime = ML_KEM_DECAPS(dk, c)
print(K_prime)
print(len(K_prime))
