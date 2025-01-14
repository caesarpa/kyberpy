import hashlib as hl
import numpy as np
import os
from params import *

def G(seed):
    ctx = hl.sha3_512()
    ctx.update(seed)
    output = ctx.digest()
    a = output[:32]
    b = output[32:]
    return a, b

class XOF:
    def __init__(self):
        self.shake = hl.shake_128()
        self.output_buffer = b""  # Buffer to hold the output bytes
        self.position = 0         # Pointer to track current position in the buffer

    def absorb(self, data: bytes):
        self.shake.update(data)

    def squeeze(self, length: int) -> bytes:
        # Ensure the buffer has enough bytes for the request
        if self.position + length > len(self.output_buffer):
            self.output_buffer += self.shake.digest(1024)  # Extend the buffer with more output

        # Extract the requested slice and update the position
        result = self.output_buffer[self.position:self.position + length]
        self.position += length
        return result
    
def PRF(seed: bytes, b: bytes, eta: int):

    #concatenate the seed with the 32-bit integer b
    input_data = seed + b

    shake = hl.shake_256()
    shake.update(input_data)

    return shake.digest(64*eta)

def H(bytes):
    ctx = hl.sha3_256()
    ctx.update(bytes)
    return ctx.digest()

def J (bytes):
    ctx = hl.shake_256()
    ctx.update(bytes)
    return ctx.digest(32)