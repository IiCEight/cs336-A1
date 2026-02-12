from collections import Counter, defaultdict
from enum import unique
from functools import partial
from hmac import new
from multiprocessing import Pool
import os
from typing import BinaryIO

# from loguru import logger
from sortedcontainers import SortedDict, SortedSet
from sympy import denom

from cs336_basics.constant.constant import ONE_BYTES_SIZE
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


def load_data_and_pretokenize(input_path: str, special_tokens:list[bytes]) -> Counter:
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
            # # logger.debug(f"Type of chunk {type(chunk)}")
            chunks.append(chunk)

        # Run pre-tokenization on your chunk and store the counts for each pre-token
        with Pool(num_processes) as pool:
            pretokenized_counters= pool.map(partial(pretokenizer, special_tokens), chunks)

        def word2bytes(word:str)->tuple[bytes,...]:
            return tuple(map(lambda char: bytes([char]), word.encode("utf-8")))

        # Combine these maps into one.
        pretokenized_counter = Counter()
        for chunk_map in pretokenized_counters:
            for word, count in chunk_map.items():
                pretokenized_counter[word2bytes(word)] += count

        # This can be elegantly rewrite as :
        # counters = map(Counter, pretokenized_maps)
        # reduce(lambda a b: a + b, counters)

    return pretokenized_counter


# use Regular expression to tokenize first (coarse-grained)
def pretokenizer(special_tokens: list[str], data: str) -> Counter:
    # logger.info("Pre-Tokenization....")

    # First spilt by special_tokens.
    # re.escape is a helper function in Python that neutralizes special characters
    # in a string so they are treated as plain text by the Regular Expression engine.
    regularExp = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    if not special_tokens:
        return re.findall(regularExp, data)

    toks = sorted(special_tokens, key=len, reverse=True)
    union = "|".join(re.escape(t) for t in toks)
    parts = re.split(f"{union}", data)

    data_split = []
    for part in parts:
        data_split.extend(re.finditer(regularExp, part))

    pretokenized_counter = Counter()

    # My implements
    for match in data_split:
        # type of map_key is str
        map_key = match.group()
        # # logger.debug(f"Current match word: {key}")
        # Return the value for key if key is in the dictionary,
        # else default(0).
        pretokenized_counter[map_key] += 1

    # Python already did this using C implmentation
    # pretokenized_map = collections.Counter(data_split)

    # logger.debug(f"pretokenized_map :{pretokenized_counter}")

    return pretokenized_counter


# Define my own data structure that supports
# find max value and update any items.
# Each item is a (count, pair(tuple[bytes]))
class orderedSet:
    def __init__(self, func):
        self.sortedset = SortedSet()
        self.find_count_of_pair = func

    #  Use to prin
    def __str__(self):
        """Called when you run print(obj) or str(obj)"""
        return self.sortedset.__str__()

    def self_check(self):
        pair_list = list(map(lambda x:x[1], self.sortedset))
        unique_pair_list = set(pair_list)
        if len(unique_pair_list) != len(pair_list):
            logger.error("ordered set failed. Duplicate pair!")
            return

    def add(self, pair : tuple[bytes], count:int):
        self.sortedset.add((count, pair))
        # self.self_check()

    def increment(self, pair, val):
        """
        Find pair and increase pair.
        If pair doesn't exist, only add new one.
        """
        old_count = self.find_count_of_pair(pair)
        self.sortedset.discard((old_count, pair))
        self.add(pair, old_count + val)
    
    def decrement(self, pair, val):
        """
        Find pair and decrease pair.
        If pair doesn't exist, Error!.
        If decrease to zero, remove this pair.
        """
        old_count = self.find_count_of_pair(pair)
        self.sortedset.remove((old_count, pair))
        if old_count - val > 0:
            self.add(pair, old_count - val)

    def discard(self, pair):
        count = self.find_count_of_pair(pair)
        self.sortedset.discard((count, pair))

    def get_max(self):
        """
        If there is no element, return -1
        """
        if len(self.sortedset) == 0:
            return -1
        return self.sortedset[-1]
    
    
def train_bpe(input_path: str, vocab_size: int, special_tokens: list[bytes]):

    pretokenized_counter = load_data_and_pretokenize(input_path, special_tokens)
    # pretokenized_map = linear_pretokenizer(input_path, special_tokens)
    # Turn all tuple into list, since we need to modify it.
    word_list:list = list(map(lambda pair_count : [list(pair_count[0]), pair_count[1]], 
                         pretokenized_counter.items()))

    bytes_plus_special_tokens_len = ONE_BYTES_SIZE + len(special_tokens)
    merge_times = vocab_size - bytes_plus_special_tokens_len

    vocabulary= {}
    token_size = len(special_tokens)

    for i, x in enumerate(special_tokens):
        vocabulary[i] = bytes(x.encode("utf-8"))
    for i in range(ONE_BYTES_SIZE):
        vocabulary[i + token_size] = bytes([i])
    # See page 9 for definitation.
    merges = []

    # logger.debug(f"Start merging pairs.... merge times {merge_times}")


    # Used to store the which word pair occurs.
    # NOTE: Use defaultdict will simplify a lot!!!!
    pair_index = defaultdict(set)
    # Used to store the count of occurrence of a pair.
    pair_count = defaultdict(int)
    # Use an ordered set to maintain the max occurence pair and modify the
    # occurence of any pairs. 
    # Item of it is (count, pair(i.e., tuple[bytes]))

    for index, (word, count) in enumerate(pretokenized_counter.items()):
        for pair in zip(word, word[1:]):
            pair_index[pair].add(index)
            pair_count[pair] += count

    # for pair, count in pair_count.items():
    #     counter.add(pair, count)

    for merge_index in range(merge_times):

        # if (merge_index + 1) % 100 == 0:
        #     # logger.debug(f"{merge_index + 1}/{merge_times} merge start!")

        # max_count_pair = counter.get_max()
        # Iterate all pairs to find max O(pairs)
        if len(pair_count) == 0:
            # logger.info("No pair to merge!!!!!")
            break
            
        # Since we don't find max pair frequently, we can directly brute force
        # Use a O(log pairs) structure will slow it down a lot! Since check and modify will 
        # become O(log pairs) and a dict does them O(1) for Average Case
        max_pair, max_count = max(pair_count.items(), key = lambda x : (x[1], x[0]))

        # logger.debug(f"Max pair {max_pair}, max count {max_count}")
        
        merged_bytes = max_pair[0] + max_pair[1]

        vocabulary[merge_index + bytes_plus_special_tokens_len] = merged_bytes
        merges.append(max_pair)

        # # logger.debug(f"Current pair_index {pair_index}")
        # For each occurrence of max_pair we need to update our data structure


        # # logger.debug(f"len of occurences of pair in different words {len(occurrences)}")
        
        # never modify pair_index[max_pair] so it's safe.
        for i in pair_index[max_pair]:
            word, count = word_list[i]
            # 1. find the all positions of max_pair in word
            # match_positions = [i for i in len(word) - 1 if word[i:i + 2] == merged_bytes]

            pos = 0
            while pos < len(word) -1:
                pair = (word[pos], word[pos + 1])
                if pair != max_pair:
                    pos += 1
                    continue
                
                # logger.debug(f"Starting merge pair! Word {word}, pos {pos}, max_pair {max_pair}")

                def update_piar_count_and_counter(old_pair, new_pair):
                    # logger.debug(f"update_piar_count_and_counter: old_pair {old_pair},  new_pair {new_pair}")
                    # # logger.debug(f"Before counter {counter}")
                    # update counter first since it depends on pair_count
                    # counter.decrement(old_pair, count)
                    # counter.increment(new_pair, count)
                    # # logger.debug(f"After counter {counter}")

                    pair_count[old_pair] -= count
                    pair_count[new_pair] += count 
                    if pair_count[old_pair] <= 0:
                        pair_count.pop(old_pair)
                    # We don't need to remove index of old_pair. 
                    # Since the case of overlap(oooo),
                    # otherwise, we keep it, and handle it when iterate word.
                    pair_index[new_pair].add(i)

                # 2. upate left neighbor
                if pos > 0:
                    old_pair = (word[pos - 1], word[pos])
                    new_pair = (word[pos - 1], merged_bytes)

                    update_piar_count_and_counter(old_pair, new_pair)

                # 3. update right neighbor
                if pos + 1 < len(word) - 1:
                    old_pair = (word[pos + 1], word[pos + 2])
                    new_pair = (merged_bytes, word[pos + 2])

                    update_piar_count_and_counter(old_pair, new_pair)

                # 4. merge max_pair and update word_list
                #   This will change word we are iterating.
                #   And it can handle the overlap case. Since word[pos] is changed, 
                #   when iterate word to find next pos, it will find correct pair.
                word[pos] = merged_bytes
                del word[pos + 1]

                # 6. move to next
                pos += 1

            #   Since we merged all max_pair in this word, so pair will never occur in word
            # assert max_pair not in word, "Merged max_pair failed."
        
        
        # logger.debug(f"pair_index {pair_index}")

        # 7. Now all max_pair are merged, we can directly delete max_pair
        # counter.discard(max_pair)
        pair_index.pop(max_pair, 0)
        pair_count.pop(max_pair, 0)
        # logger.debug(f"Finish merging one max_pair {max_pair}, pair_index {pair_index}")
        
    # # logger.debug(f"Finial merges {merges}")
    # # logger.debug(f"Finial vocabulary {vocabulary}")


    return vocabulary, merges
