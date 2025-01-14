import numpy as np
import hashlib as hl
import os
from hash import *
from params import *

def intToBytes(x):
    return x.to_bytes((x.bit_length() + 7) // 8, 'big')

def bytesToInt(x):
    return int.from_bytes(x, 'big')

def generateBits(length):
    return np.random.randint(0, 2, length)

def bytesToBits(bytes):
    bits = []
    for byte in bytes:
        # Convert each byte into its 8-bit binary representation
        for i in range(8):
            bits.append((byte >> i) & 1)  # Extract the i-th bit
    return bits

def bitsToBytes(bits):

    bytes = []
    for i in range(0, len(bits), 8):
        byte = 0
        for j in range(8):
            if i + j < len(bits):  # Check to avoid index out of range
                byte |= (bits[i + j] << j)  # Set the j-th bit of the byte
        bytes.append(byte)
    return bytes

def BitRev7(r):
    """
    Perform bit reversal of a 7-bit integer.

    Args:
        r (int): An integer representing a 7-bit number (0 to 127).

    Returns:
        int: The bit-reversed integer.
    """
    if r < 0 or r > 127:
        raise ValueError("Input must be a 7-bit integer (0 to 127).")

    # Extract bits from the integer
    bits = [(r >> i) & 1 for i in range(7)]  # Extract the 7 bits (r0 to r6)

    # Reverse the bits and calculate the new value
    reversed_value = sum(bits[i] * (1 << (6 - i)) for i in range(7))

    return reversed_value
    
def binomial_sampling(η):
    # Generate two random binary strings of length η
    a = np.random.randint(0, 2, η)
    b = np.random.randint(0, 2, η)

    # Sum the element-wise difference of the two bit strings, resulting in a binomially distributed sample [-η, η].
    sample = np.sum(a - b)

    return sample

def CBD(η, bytes):
    coefficients = []
    # 64*η bytes input are needed
    # Sample n coefficients from the binomial distribution
    bits = bytesToBits(bytes)

    for i in range(n):
        # Select η bits from the bit string for a and b and calculate the difference of the sum to get a binomially distributed coefficient.
        x = sum(bits[2 * η * i + j] for j in range(η)) 
        y = sum(bits[2 * η * i + η + j] for j in range(η))
        f_i = (x - y) % q
        coefficients.append(f_i)
    return coefficients

def compress (x: int, d):
    compressed_x = round((2**d / q) * x) % 2**d
    return compressed_x

def decompress (compressed_x: int,d):
    x = round((q / 2**d) * compressed_x)
    return x

def byteEncode(f, d):

    m = 2**d if d < 12 else q  # Calculate m based on d
    b = [0] * (256 * d)  # Initialize bit array

    for i in range(n):
        a = f[i] % m # Take the i-th integer
        for j in range(d):
            b[i * d + j] = a % 2  # Extract the least significant bit
            a = (a - b[i * d + j]) // 2  # Update a (divide by 2)
    
    B = bitsToBytes(b)  # Convert bits to bytes
    return B

def byteDecode(B, d):
    m = 2**d if d < 12 else q  # Calculate m based on d
    b = bytesToBits(B)  # Convert bytes to bits
    F = [0] * n  # Initialize the integer array

    for i in range(n):
        F[i] = sum(b[i * d + j] * (2**j) for j in range(d)) % m  # Compute integer mod m

    return F