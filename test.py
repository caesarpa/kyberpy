import os
import hashlib as hl
from kyber import PKE_CPA_KeyGen, PKE_Encrypt, PKE_Decrypt
from util import *
from params import *

def test_PKE_CPA_KeyGen():
    seed = os.urandom(32)
    ek, dk = PKE_CPA_KeyGen(seed)
    assert len(ek) == 2
    assert len(ek[0]) == k
    assert len(dk) == k

def test_PKE_Encrypt():
    seed = os.urandom(32)
    ek, dk = PKE_CPA_KeyGen(seed)
    message = os.urandom(32)
    r = os.urandom(32)
    ciphertext = PKE_Encrypt(ek, message, r)
    assert len(ciphertext) == 2

def test_PKE_Decrypt():
    seed = os.urandom(32)
    ek, dk = PKE_CPA_KeyGen(seed)
    message = os.urandom(32)
    r = os.urandom(32)
    ciphertext = PKE_Encrypt(ek, message, r)
    decrypted_message = PKE_Decrypt(dk, ciphertext)
    print(decrypted_message)
    x = byteDecode(message,1)
    x = bitsToBytes(x)
    print(x)
    assert decrypted_message == x

if __name__ == "__main__":
    test_PKE_CPA_KeyGen()
    test_PKE_Encrypt()
    test_PKE_Decrypt()
    print("All tests passed.")