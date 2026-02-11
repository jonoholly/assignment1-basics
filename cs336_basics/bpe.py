import regex as re
from collections import Counter, defaultdict
import multiprocessing
import os
from functools import partial

def gpt2_bytes_to_unicode() -> dict[int, str]:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    characters = [chr(n) for n in cs]
    return dict(zip(bs, characters))

# GPT-2 pre-tokenization regex as specified in the assignment
PAT = r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"

def pretokenize_part(part):
    counts = Counter()
    for match in re.finditer(PAT, part):
        counts[match.group()] += 1
    return counts

def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Pre-tokenization with multiprocessing
    word_counts = Counter()
    if special_tokens:
        special_pattern = "|".join(re.escape(st) for st in special_tokens)
        parts = re.split(f"({special_pattern})", text)
        special_tokens_set = set(special_tokens)
        
        # Filter out special tokens and empty parts
        text_parts = [p for p in parts if p and p not in special_tokens_set]
        
        with multiprocessing.Pool() as pool:
            results = pool.map(pretokenize_part, text_parts)
            for res in results:
                word_counts.update(res)
    else:
        # If no special tokens, we can still chunk the text for parallelism
        # For simplicity, if no special tokens, we just do it in a single process 
        # or we could split by whitespace. But the assignment says special tokens delimits docs.
        for match in re.finditer(PAT, text):
            word_counts[match.group()] += 1

    # Initial vocabulary
    vocab: dict[int, bytes] = {}
    next_id = 0
    for st in special_tokens:
        vocab[next_id] = st.encode("utf-8")
        next_id += 1

    byte_encoder = gpt2_bytes_to_unicode()
    byte_order = list(byte_encoder.keys())
    for b in byte_order:
        vocab[next_id] = bytes([b])
        next_id += 1

    # Map byte -> its token ID
    byte_to_id = {b: i for i, b in enumerate(byte_order, start=len(special_tokens))}

    # Prepare words for BPE: list of [token_ids, frequency]
    word_list = []
    for word, freq in word_counts.items():
        word_list.append([[byte_to_id[b] for b in word.encode("utf-8")], freq])

    # Initial pair counts and pair-to-word mapping
    pair_counts = Counter()
    pair_to_word_indices = defaultdict(set)
    for i, (tokens, freq) in enumerate(word_list):
        for j in range(len(tokens) - 1):
            pair = (tokens[j], tokens[j+1])
            pair_counts[pair] += freq
            pair_to_word_indices[pair].add(i)

    merges: list[tuple[bytes, bytes]] = []
    num_merges = vocab_size - len(vocab)

    for _ in range(num_merges):
        if not pair_counts:
            break
        
        # Find the max frequency
        max_freq = max(pair_counts.values())
        if max_freq <= 0:
            break
        
        # Find all pairs with max frequency
        candidates = [pair for pair, freq in pair_counts.items() if freq == max_freq]
        
        # Tie-breaking: preferring the lexicographically greater pair (on bytes)
        best_pair_ids = max(candidates, key=lambda p: (vocab[p[0]], vocab[p[1]]))
        
        token1, token2 = best_pair_ids
        merges.append((vocab[token1], vocab[token2]))
        new_token_id = next_id
        vocab[new_token_id] = vocab[token1] + vocab[token2]
        next_id += 1

        # Update words containing the best_pair
        affected_indices = list(pair_to_word_indices[best_pair_ids])
        for i in affected_indices:
            tokens, freq = word_list[i]
            
            # Remove all old pairs for this word from the counts and index
            for j in range(len(tokens) - 1):
                p = (tokens[j], tokens[j+1])
                pair_counts[p] -= freq
                pair_to_word_indices[p].discard(i)
                if pair_counts[p] <= 0:
                    del pair_counts[p]
            
            # Perform merge
            new_tokens = []
            j = 0
            while j < len(tokens):
                if j < len(tokens) - 1 and tokens[j] == token1 and tokens[j+1] == token2:
                    new_tokens.append(new_token_id)
                    j += 2
                else:
                    new_tokens.append(tokens[j])
                    j += 1
            word_list[i][0] = new_tokens
            tokens = new_tokens
            
            # Add all new pairs for this word
            for j in range(len(tokens) - 1):
                p = (tokens[j], tokens[j+1])
                pair_counts[p] += freq
                pair_to_word_indices[p].add(i)

    return vocab, merges
