from tests.common import FIXTURES_PATH, gpt2_bytes_to_unicode

def print_ref():
    reference_merges_path = FIXTURES_PATH / "train-bpe-reference-merges.txt"
    gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
    with open(reference_merges_path, encoding="utf-8") as f:
        gpt2_reference_merges = [tuple(line.rstrip().split(" ")) for line in f]
        reference_merges = [
            (
                bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
                bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
            )
            for merge_token_1, merge_token_2 in gpt2_reference_merges
        ]
    
    for i, m in enumerate(reference_merges[:35]):
        print(f"{i}: {m}")

print_ref()
