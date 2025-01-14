import hashlib as hl
import numpy as np
from util import *
from hash import *

def sampleNTT(input_bytes: bytes):
    # 1. Initialize XOF context
    ctx = XOF()

    # 2. Absorb the 34-byte input into the XOF
    ctx.absorb(input_bytes)

    # We'll produce 256 coefficients in Z_q
    a_hat = [None] * 256
    j = 0

    # 3. Keep generating coefficients until we've filled a_hat
    while j < 256:
        # 4–5. Squeeze 3 bytes from the XOF
        C = ctx.squeeze(3)
        
        # Interpret these 3 bytes as two 12-bit integers d1 and d2
        #   d1 uses the lower 4 bits of C[1],
        #   d2 uses the upper 4 bits of C[1].
        d1 = C[0] + 256 * (C[1] & 0x0F)   # 12 bits from [C[0], lower 4 bits of C[1]]
        d2 = (C[1] >> 4) + 16 * C[2]      # 12 bits from [upper 4 bits of C[1], C[2]]

        # 6–9. Accept d1 and/or d2 if they are < q
        if d1 < q:
            a_hat[j] = d1
            j += 1
        if d2 < q and j < 256:
            a_hat[j] = d2
            j += 1

    # 10. Return the array of 256 coefficients in Z_q
    return a_hat

def NTT(poly):
    nttPoly = poly.copy()
    counter = 1
    l = 128
    while l >= 2:

        start = 0
        while start < 256:
            z = (zeta ** BitRev7(counter)) % q
            counter += 1

            for j in range(start, start + l):
                t = z * nttPoly[j + l] % q
                nttPoly[j + l] = (nttPoly[j] - t) % q
                nttPoly[j] = (nttPoly[j] + t) % q

            start = start + 2*l

        l = l//2
    
    return nttPoly

def INTT(nttPoly):
    poly = nttPoly.copy()
    counter = 127
    l = 2
    while l <= 128:

        start = 0
        while start < 256:
            z = (zeta ** BitRev7(counter)) % q
            counter -= 1

            for j in range(start, start + l):
                t = poly[j]
                poly[j] = (t + poly[j + l]) % q
                poly[j + l] = (z * (poly[j + l] - t)) % q

            start = start + 2*l

        l = l*2
    
    for i in range(256):
        poly[i] = (poly[i] * 3303) % q
    
    return poly

def baseCaseMultiply(a0, a1, b0, b1, gamma):

    c0 = (a0 * b0 + a1 * b1 * gamma) % q  # Step 1
    c1 = (a0 * b1 + a1 * b0) % q          # Step 2
    return c0, c1

def multiplyNTTs(f_hat, g_hat):

    h_hat = [None] * 256  # Initialize the result array

    for i in range(128):
        # Extract f[2i], f[2i+1], g[2i], g[2i+1]
        f0, f1 = f_hat[2 * i], f_hat[2 * i + 1]
        g0, g1 = g_hat[2 * i], g_hat[2 * i + 1]

        # Compute baseCaseMultiply
        gamma = (zeta ** (2 * BitRev7(i) + 1)) % q
        c0, c1 = baseCaseMultiply(f0, f1, g0, g1, gamma)

        # Store the results in h_hat
        h_hat[2 * i], h_hat[2 * i + 1] = c0, c1

    return h_hat

def matrix_vector_ntt_product(A_hat, u_hat):

    w_hat = [None] * k  # Initialize result vector
    
    for i in range(k):
        # Compute the sum of polynomial products for each entry of w_hat
        result = [0] * n  # Initialize to zero polynomial
        for j in range(k):
            # Multiply A_hat[i][j] and u_hat[j] using multiplyNTTs
            product = multiplyNTTs(A_hat[i][j], u_hat[j])
            # Add the product to the result polynomial (component-wise)
            result = [(result[h] + product[h]) % q for h in range(256)]
        w_hat[i] = result  # Store the resulting polynomial
    
    return w_hat

def transpose_matrix_vector_ntt_product(A_hat, u_hat):

    y_hat = [None] * k  # Initialize result vector

    for i in range(k):
        # Compute the sum of polynomial products for each entry of y_hat
        result = [0] * n  # Initialize to zero polynomial
        for j in range(k):
            # Multiply A_hat[j][i] (transposed element) and u_hat[j]
            product = multiplyNTTs(A_hat[j][i], u_hat[j])
            # Add the product to the result polynomial (component-wise)
            result = [(result[h] + product[h]) % q for h in range(n)]
        y_hat[i] = result  # Store the resulting polynomial

    return y_hat

def vector_ntt_dot_product(u_hat, v_hat):

    result = [0] * n  # Initialize to zero polynomial

    for j in range(k):
        # Multiply u_hat[j] and v_hat[j] using multiplyNTTs
        product = multiplyNTTs(u_hat[j], v_hat[j])
        # Add the product to the result polynomial (component-wise)
        result = [(result[h] + product[h]) % q for h in range(n)]

    return result

def poly_addition(u, v):
    result = [None] * n
    for i in range(n):
        result[i] = (u[i] + v[i]) % q
    return result

def poly_subtraction(u, v):
    result = [None] * n
    for i in range(n):
        result[i] = (u[i] - v[i]) % q
    return result

def vector_addition(u, v):
    result = []
    for i in range(k):
        result.append(poly_addition(u[i], v[i]))
    return result