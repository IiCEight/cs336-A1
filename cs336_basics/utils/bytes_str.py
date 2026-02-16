
from loguru import logger


def str2tuple_of_bytes(x:str)->tuple[bytes]:
    # return tuple(map(lambda ch: bytes([ch]), x.encode("utf-8")))

    return tuple(bytes([ch]) for ch in x.encode("utf-8"))

def str2bytes(x:str)->bytes:
    return bytes(x.encode("utf-8"))

def bytes2str(x:bytes)->str:
    return x.decode("utf-8")

def tuple_of_bytes2str(x :tuple[bytes])->str:
    res = b""
    for one_bytes in x:
        res += one_bytes
    
    return res.decode("utf-8")


if __name__ == "__main__":
    logger.debug(str2tuple_of_bytes("saber"))
    logger.debug(tuple_of_bytes2str(str2tuple_of_bytes("saber")))
    logger.debug(tuple([str2bytes("love")]))
    logger.debug(bytes2str(str2bytes("love")))


