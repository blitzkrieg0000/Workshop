# RANDOM UID - TIME-HASH
import time
import hashlib as hl

def getUID():
    return int.from_bytes(hl.sha3_512(str(time.time()).encode()).digest(),"big")

x = getUID()
print(x)