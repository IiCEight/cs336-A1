import itertools
import json
from loguru import logger
import regex as re
from collections.abc import Iterable, Iterator

from cs336_basics.utils.bytes_str import  str2bytes, str2tuple_of_bytes


class Tokenizer:
    def __init__(self, vocabulary: dict[int, bytes], 
                 merges: list[tuple[bytes, bytes]], 
                 special_tokens: list[str]| None = None):
        
        logger.info("Size of vocabulary {}, size of merges {}", len(vocabulary), len(merges))

        self.special_tokens = set([])
        # Add new.
        if special_tokens is not None:
            # we need to avoid duplicate vocabulary.
            vocab_values = set(vocabulary.values())
            next_id = len(vocabulary)

            for s_t in (special_tokens):
                self.special_tokens.add(s_t)
                
                s_t_bytes = str2bytes(s_t)
                # If this is a new vocab add it to vocabulary
                if s_t_bytes not in vocab_values:
                    vocabulary[next_id] = s_t_bytes
                    vocab_values.add(s_t_bytes)
                    next_id += 1

        self.vocabulary = vocabulary
        # Make sure value is unique
        assert len(set(vocabulary.values())) == len(vocabulary.values())
        self.reversed_vocabulary = {value:key for key, value in vocabulary.items()}
        self.merges = merges
        # Used to memory the map from pretoken to true token id for
        # speeding up.
        self.pretoken2token_id = {}
        
        # Use to speed up merging.
        self.merge_rank = {pair: len(merges) - i for i, pair in enumerate(self.merges)}
        # logger.debug("self.reversed_vocabulary {}", self.reversed_vocabulary)
        logger.debug("self.special_tokens {}", self.special_tokens)

    def pre_tokenizer(self, text:str)->list[tuple[bytes]]:
        logger.info("Start pre-tokening...")
        regularExp = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        logger.debug("s.t. {}, text {}", self.special_tokens, text)
        
        if not self.special_tokens:
            logger.info("No special tokens!")
            words = re.findall(regularExp, text)
            return [str2tuple_of_bytes(word) for word in words]

        toks = sorted(self.special_tokens, key=len, reverse=True)
        union = "|".join(re.escape(t) for t in toks)
        # use () to wrap the regular expression to avoid discard special token 
        # when spliting.
        parts = re.split(f"({union})", text)
        
        pre_tokens = []
        for part in parts:
            if part in self.special_tokens:
                pre_tokens.append(tuple([str2bytes(part)]))
            else:
                words = re.findall(regularExp, part)
                pre_tokens.extend([str2tuple_of_bytes(word) for word in words])

        return pre_tokens

    # Class method (like static method in java)
    @classmethod
    def from_files(cls, vocab_filepath:str, merges_filepath:str, 
                   special_tokens:list[str]|None=None):
        # 1. Load Vocabulary (JSON: {"token_str": id})
        logger.info(f"Loading vocabulary from {vocab_filepath}")
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            # The JSON keys are Strings (e.g. "\u0120"), we need to convert to Bytes
            vocab_raw = json.load(f)
            
        vocabulary = {}
        for token_str, token_id in vocab_raw.items():
            # JSON forces keys to be strings. We must encode them back to bytes.
            # Note: GPT-2 uses a specific 'bytes_to_unicode' map, but for standard BPE:
            # logger.debug("token_str {}, type of token_str {}, token_id {}",token_str, type(token_str), token_id)
            vocabulary[int(token_id)] = str2bytes(token_str)

        # 2. Load Merges (Text file)
        # Format: "t h" (space separated)
        logger.info(f"Loading merges from {merges_filepath}")
        merges = []
        with open(merges_filepath, "r", encoding="utf-8") as f:
            # Skip the first line if it's a version/comment (common in HuggingFace files)
            lines = f.readlines()
            start_idx = 1 if lines[0].startswith("#") or "version" in lines[0] else 0
            
            for line in lines[start_idx:]:
                line = line.strip()
                if not line: continue
                
                # Split "t h" -> ("t", "h")
                parts = line.split()
                if len(parts) == 2:
                    # logger.debug("parts[0] {}, type of it {}, parts[1] {}", parts[0], type(parts[0]), parts[1])
                    # Convert to bytes
                    merges.append((str2bytes(parts[0]), str2bytes(parts[1])))

        return cls(vocabulary, merges, special_tokens)

    def merge_one_token(self, pretoken:tuple[bytes], index, pre_tokens_len)->list[int]:
        if (index + 1) % 10000 == 0:
            logger.info("Working on {}/{} pretoken", index + 1, pre_tokens_len)

        if self.pretoken2token_id.get(pretoken) is not None:
            return self.pretoken2token_id[pretoken]
        
        word = list(pretoken)
        logger.debug("Begin to merge pretoken {}", pretoken)
        while len(word) > 1:
            # find all possible pairs
            pairs = list(zip(word, word[1:]))

            # find the highest rank pair in merges to merge
            
            merge = max(pairs, key=lambda pair: self.merge_rank.get(pair, -1))

            if merge not in self.merge_rank:
                break

            # combine all pairs that is equal to merge
            first, second = merge
            new_word = []
            pos = 0
            while pos < len(word):
                # Try to find the pair at index i
                if pos < len(word) - 1 and word[pos] == first and word[pos+1] == second:
                    new_word.append(first + second)
                    pos += 2 # Skip the next character (we just merged it)
                else:
                    new_word.append(word[pos])
                    pos += 1
            word = new_word
        logger.debug("Merged fininshed. Final token {}", word)
        # FP way
        # token = list(map(lambda x: self.reversed_vocabulary[x], token))

        token = [self.reversed_vocabulary[merged_bytes] for merged_bytes in word]
        self.pretoken2token_id[pretoken] = token

        return token

    def encode(self, text:str)-> list[int]:
        pre_tokens = self.pre_tokenizer(text)
        logger.info("Apply merges....")

        pre_tokens_len = len(pre_tokens)

        # 1. Create a generator of lists (Lazy)
        nested_lists = (self.merge_one_token(pretoken, i, pre_tokens_len) 
                        for i, pretoken in enumerate(pre_tokens))

        # 2. Chain them together instantly
        tokens = list(itertools.chain.from_iterable(nested_lists))

        logger.info("Complete encoding! Final length of all tokens {}", len(tokens))
        return tokens

    def encode_iterable(self, iterable:Iterable[str])->Iterator[int]:
        """
        Given an iterable of strings (e.g., a Python file handle), return a 
        generator that lazily yields token IDs. This is required for 
        memory-efficient tokenization  of large files that we cannot directly 
        load into memory.
        """
        for chunck in iterable:
            ids = self.encode(chunck)
            for token_id in ids:
                yield token_id


    def decode(self, ids:list[int])-> str:
        logger.info("Start decoding....")
        text = b""

        # Don't do this, bytes a imutable, so concatenate they will 
        # use a large time!!!
        # len_ids = len(ids)
        # for i, token_id in enumerate(ids):
        #     if (i + 1) % 1000 == 0:
        #         logger.info("Decoding on {}/{} token", i + 1, len_ids)
        #     text += self.vocabulary[token_id]


        text_list = [self.vocabulary[token_id] for token_id in ids]
        
        # Join them all at once (Optimized C implementation)
        # This will be much faster!
        text_bytes = b"".join(text_list)

        logger.info("Begin to transform bytes into Unicode!")
        text = text_bytes.decode("utf-8", errors="replace")

        logger.debug("Finished decoding. Text {}", text)
            
        return text