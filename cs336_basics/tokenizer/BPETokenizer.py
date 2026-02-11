from functools import partial
from multiprocessing import Pool
import os
from typing import BinaryIO

from loguru import logger
from sortedcontainers import SortedSet

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


def load_data_and_pretokenize(input_path: str) -> dict[tuple[bytes] : int]:
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
            # logger.debug(f"Type of chunk {type(chunk)}")
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
def pretokenizer(special_token: bytes, data: str) -> dict[tuple[bytes], int]:
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
        # logger.debug(f"Current match word: {key}")
        # Return the value for key if key is in the dictionary,
        # else default(0).
        pretokenized_map[key] = pretokenized_map.get(key, 0) + 1

    # Python already did this using C implmentation
    # pretokenized_map = collections.Counter(data_split)

    logger.debug(f"pretokenized_map :{pretokenized_map}")

    return pretokenized_map


# Use a doubly linked list to store a split word
class Node:
    # __slots__ reduces memory usage by ~60% vs normal classes
    __slots__ = ["count", "val", "prev", "next"]

    def __init__(self, val, count):
        # the ocurrence of val in original word.
        # The number of count in the same linked list is the same
        self.count = count
        self.val = val
        self.prev = None
        self.next = None


def build_linked_list(word: tuple[bytes], count: int) -> list[Node]:
    """Converts a tuple of bytes (word) into a chain of Nodes."""
    if not word:
        return []

    # 1. Create all nodes first
    nodes = [Node(t, count) for t in word]

    # 2. Link them up
    for i in range(len(nodes) - 1):
        nodes[i].next = nodes[i + 1]
        nodes[i + 1].prev = nodes[i]

    return nodes


def train_bpe(input_path: str, vocab_size: int, special_tokens: list[bytes]):

    pretokenized_map = load_data_and_pretokenize(input_path)

    bytes_plus_special_tokens_len = ONE_BYTES_SIZE + len(special_tokens)
    merge_times = vocab_size - bytes_plus_special_tokens_len

    vocabulary = {i: bytes([i]) for i in range(ONE_BYTES_SIZE)}
    for i, x in enumerate(special_tokens):
        vocabulary[i + ONE_BYTES_SIZE] = x
    # See page 9 for definitation.
    merges = []

    logger.info(f"Start merging pairs.... merge times {merge_times}")

    # Use an ordered set to maintain the max occurence pair and modify the
    # occurence of any pairs.
    # Item of it is (count, pair(i.e., tuple[bytes]))
    sorted_count = SortedSet()
    # Used to store the node of occurrence of a pair.
    byte_pair_index: dict[tuple[bytes] : list[tuple[Node]]] = {}
    # Used to store the count of occurrence of a pair.
    byte_pair_count: dict[tuple[bytes] : int] = {}

    for word, count in pretokenized_map.items():
        linked_list_word = build_linked_list(word, count)
        for node_l, node_r in zip(linked_list_word, linked_list_word[1:]):
            pair = (node_l.val, node_r.val)
            # Return the value for key if key is in the dictionary,
            # else default(0).
            if pair not in byte_pair_index:
                byte_pair_index[pair] = []
            byte_pair_index[pair].append((node_l, node_r))
            byte_pair_count[pair] = byte_pair_count.get(pair, 0) + count

    for pair, count in byte_pair_count.items():
        sorted_count.add((count, pair))

    def update_sorted_count(old_val, new_val):
        sorted_count.discard(old_val)
        sorted_count.add(new_val)

    def update_pair(neighbor_node, origin_node, merged_node, old_pair, new_pair, is_left_neighbor):
        # delete old_piar pair TODO: what about count becomes 0.

        sorted_count.remove((byte_pair_count[old_pair], old_pair))
        logger.debug(f"remove from sorted_count {(byte_pair_count[old_pair], old_pair)}")
        byte_pair_count[old_pair] -= node_l.count
        if is_left_neighbor:
            byte_pair_index[old_pair].remove((neighbor_node, origin_node))
        else:
            byte_pair_index[old_pair].remove((origin_node, neighbor_node))

        sorted_count.add((byte_pair_count[old_pair], old_pair))
        logger.debug(f"Add into sorted_count {(byte_pair_count[old_pair], old_pair)}")
        # add new_pair
        if new_pair not in byte_pair_index:
            byte_pair_index[new_pair] = []

        if is_left_neighbor:
            byte_pair_index[new_pair].append((neighbor_node, merged_node))
        else:
            byte_pair_index[new_pair].append((merged_node, neighbor_node))

        new_pair_count = byte_pair_count.get(new_pair, 0)
        # NOTE: Since we are updating sorted_count, we need first to remove
        # and then add.
        sorted_count.discard((new_pair_count, new_pair))
        logger.debug(f"remove from sorted_count {(new_pair_count, new_pair)}")

        byte_pair_count[new_pair] = new_pair_count + node_l.count
        sorted_count.add((byte_pair_count[new_pair], new_pair))
        logger.debug(f"Add into sorted_count {(byte_pair_count[new_pair], new_pair)}")

    for merge_index in range(merge_times):
        # logger.debug(f"Current byte_pair_map {byte_pair_count}")

        max_count_pair = sorted_count[-1]

        max_pair = max_count_pair[1]

        logger.debug(f"Max pair {max_pair}, max count {max_count_pair[0]}")
        
        # No pair to merge
        if max_count_pair[0] <= 0:
            logger.info("No pair to merge!!!!!")
            break

        vocabulary[merge_index + bytes_plus_special_tokens_len] = merged_bytes = (
            max_pair[0] + max_pair[1])
        merges.append(merged_bytes)

        # For all max_pair, find this occurrence (node_l, node_r),
        # and merge these two nodes and update pair(node_l.prev, node_l)
        # and (node_r, node_r.next)

        occurrences = byte_pair_index[max_pair]
        # NOTE: create a new one to avoid modify it when iterating!!!
        occurrences = list(byte_pair_index[max_pair])
        logger.debug(f"Len of occurrences {len(occurrences)}")

        for id, (node_l, node_r) in enumerate(occurrences):
            # NOTE: The case of overlapping like aaaa and max_pair = (a, a).

            logger.debug(f"{id}:max_pair {max_pair}, (node_l.val, node_r.val) {(node_l.val, node_r.val)}")

            if node_l.val == merged_bytes or node_r.val == merged_bytes:
                logger.debug("There is an overlapping!")
                # we do not merge but update the count.
                current_count_max_pair = byte_pair_count[max_pair]
                update_sorted_count(
                    (current_count_max_pair, max_pair), 
                    (current_count_max_pair - node_l.count, max_pair)
                    )
                byte_pair_count[max_pair] = node_l.count
                continue

            # Change one of node_l and node_r into merged_node
            merged_node = Node(merged_bytes, node_l.count)
            merged_node.prev = node_l.prev
            merged_node.next = node_r.next
            if node_l.prev:
                node_l.prev.next = merged_node
            if node_r.next:
                node_r.next.prev = merged_node

            # Left Neighbor
            if node_l.prev is not None:
                prev_node = node_l.prev
                old_pair = (prev_node.val, node_l.val)
                new_pair = (prev_node.val, merged_node.val)

                update_pair(prev_node, node_l, merged_node, old_pair, new_pair, True)

            if node_r.next is not None:
                next_node = merged_node.next
                old_pair = (node_r.val, next_node.val)
                new_pair = (merged_node.val, next_node.val)
                
                update_pair(next_node, node_r, merged_node, old_pair, new_pair, False)

            # Finially, we need mark node_l and node_r are merged for iteration later
            node_l.val = node_r.val = merged_bytes
        
        # Delete this pair
        sorted_count.discard((byte_pair_count[max_pair], max_pair))
        byte_pair_count.pop(max_pair)
        byte_pair_index.pop(max_pair)


    logger.info(f"Finial vocabulary {vocabulary}, merges {merges}")

    return vocabulary, merges
