from functools import partial
from multiprocessing import Pool
import os
from typing import BinaryIO
from heapq import merge
from hmac import new
from re import I
from click import command
from loguru import logger

from cs336_basics.constant.constant import ONE_BYTES_SIZE
from ..config import loggerConfig
import typer
import regex as re


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def load_data_and_pretokenize(input_path:str) -> str:
    with open(input_path, "rb") as f:
        num_processes = 4
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.

        chunks = []

        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            # Since we open file using byte mode, we need to decode it into
            # readable characters using utf-8.
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            logger.debug(f"Type of chunk {type(chunk)}")
            chunks.append(chunk)

        # Run pre-tokenization on your chunk and store the counts for each pre-token
        with Pool(num_processes) as pool:
            pretokenized_maps = pool.map(partial(pretokenizer, "<|endoftext|>"), chunks)

        # Combine these maps into one.
        pretokenized_map = {}
        for chunk_map in pretokenized_maps:
            for word, count in chunk_map.items():
                pretokenized_map[word] = pretokenized_map.get(word, 0) + count
        
        # This can be elegantly rewrite as :
        # counters = map(Counter, pretokenized_maps)
        # reduce(lambda a b: a + b, counters)

    return pretokenized_map

# use Regular expression to tokenize first (coarse-grained)
def pretokenizer(special_token: bytes, data: str)->dict[tuple[bytes], int]:
    logger.info("Pre-Tokenization....")

    # First spilt by special_token.
    # re.escape is a helper function in Python that neutralizes special characters 
    # in a string so they are treated as plain text by the Regular Expression engine.
    patten = re.escape(special_token)
    parts = re.split(patten, data)

    regularExp = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    
    data_split = []
    for part in parts:
        data_split.extend(re.finditer(regularExp, part))

    pretokenized_map = {}

    # My implements
    for match in data_split:
        # Turn str into tuple[bytes
        key = tuple(match.group())
        logger.debug(f"Current match word: {key}")
        # Return the value for key if key is in the dictionary, 
        # else default(0).
        pretokenized_map[key] = pretokenized_map.get(key, 0) + 1

    # Python already did this using C implmentation
    # pretokenized_map = collections.Counter(data_split)


    logger.debug(f"pretokenized_map :{pretokenized_map}")

    return pretokenized_map


def train_bpe(input_path :str, vocab_size:int, special_tokens: list[bytes]):
    
    pretokenized_map = load_data_and_pretokenize(input_path)

    bytes_plus_special_tokens_len = ONE_BYTES_SIZE + len(special_tokens)
    merge_times = vocab_size - bytes_plus_special_tokens_len

    vocabulary = {i:bytes([i]) for i in range(ONE_BYTES_SIZE)}
    for i, x in enumerate(special_tokens):
        vocabulary[i + ONE_BYTES_SIZE] = x
    # See page 9 for definitation.
    merges = []

    logger.info(f"Start merging pairs.... merge times {merge_times}")

    # while True:
    for merge_index in range(merge_times):

        byte_pair_map = {}
        
        for word, count in pretokenized_map.items():
            for pair in zip(word, word[1:]):
                # Return the value for key if key is in the dictionary, 
                # else default(0).
                byte_pair_map[pair] =  byte_pair_map.get(pair, 0) + count
        
        logger.debug(f"Current byte_pair_map {byte_pair_map}")
        
        # Find pair that to be merged. O(N)
        max_pair, max_count = None, -1
        for pair, count in byte_pair_map.items():
            if count > max_count:
                max_count = count
                max_pair = pair
            elif count == max_count:
                max_pair = max(max_pair, pair)
            
        logger.debug(f"max_pair {max_pair}, max_count {max_count}")
        # Add new token into vocabulary
        vocabulary[merge_index + bytes_plus_special_tokens_len] = max_pair[0] + max_pair[1]
        merges.append(max_pair)

        merged_pretokenized_map = {}

        # merge max pair O(N)
        for word, count in pretokenized_map.items():
            new_word = []
            len_word, i, is_last_merged = len(word), 0, False
            # skip last one since we iterate the pair
            while i < len_word - 1:
                pair = (word[i], word[i + 1])
                if pair != max_pair:
                    new_word.append(word[i])
                else:
                    is_last_merged = i == len_word - 2
                    new_word.append(word[i] + word[i + 1])
                    i += 1 # skip word[i + 1]
                # move to next word.
                i += 1
            # Not forget last bytes if there is no merge in last pair.
            if is_last_merged == False:
                new_word.append(word[-1])
            
            # Turn list[bytes] into tuple[bytes] and update byte_pair_map
            merged_pretokenized_map[tuple(new_word)] = count

        pretokenized_map = merged_pretokenized_map
        logger.debug(f"Merged pretokenized_map {pretokenized_map}")

    logger.info(f"Finial vocabulary {vocabulary}, merges {merges}")

    return vocabulary, merges
