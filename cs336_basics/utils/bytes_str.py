
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

def gpt2_bytes_to_unicode() -> dict[int, str]:
    """
    Returns a mapping between every possible byte (an integer from 0 to 255) to a
    printable unicode string character representation. This function is taken
    from the GPT-2 code.

    For example, `chr(0)` is `\x00`, which is an unprintable character:

    >>> chr(0)
    '\x00'
    >>> print(chr(0))

    As a result, this function returns a dictionary `d` where `d[0]` returns `Ā`.
    The bytes that are visually printable keep their original string representation [1].
    For example, `chr(33)` returns `!`, and so accordingly `d[33]` returns `!`.
    Note in particular that the space character `chr(32)` becomes `d[32]`, which
    returns 'Ġ'.

    For unprintable characters, the function shifts takes the integer representing
    the Unicode code point of that character (returned by the Python `ord`) function
    and shifts it by 256. For example, `ord(" ")` returns `32`, so the the space character
    ' ' is shifted to `256 + 32`. Since `chr(256 + 32)` returns `Ġ`, we use that as the
    string representation of the space.

    This function can simplify the BPE implementation and makes it slightly easier to
    manually inspect the generated merges after they're serialized to a file.
    """
    # These 188 integers can used as-is, since they are not whitespace or control characters.
    # See https://www.ssec.wisc.edu/~tomw/java/unicode.html.
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    # now get the representations of the other 68 integers that do need shifting
    # each will get mapped chr(256 + n), where n will grow from 0...67 in the loop
    # Get printable representations of the remaining integers 68 integers.
    n = 0
    for b in range(2**8):
        if b not in bs:
            # If this integer isn't in our list of visually-representable
            # charcters, then map it to the next nice character (offset by 256)
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    characters = [chr(n) for n in cs]
    d = dict(zip(bs, characters))
    return d


def gpt2_unicode_to_bytes() -> dict[str, int]:
    """
    Returns the exact inverse mapping of gpt2_bytes_to_unicode().
    Maps the printable unicode string characters back to their original byte values (0-255).
    """
    # Get the forward mapping
    bytes_to_unicode = gpt2_bytes_to_unicode()
    
    # Invert the dictionary: {unicode_char: byte_int}
    unicode_to_bytes_map = {v: k for k, v in bytes_to_unicode.items()}
    
    return unicode_to_bytes_map


if __name__ == "__main__":
    logger.debug(str2tuple_of_bytes("saber"))
    logger.debug(tuple_of_bytes2str(str2tuple_of_bytes("saber")))
    logger.debug(tuple([str2bytes("love")]))
    logger.debug(bytes2str(str2bytes("love")))


