import json
import time
import psutil
import os
from cs336_basics.bpe import train_bpe

def main():
    input_path = "data/TinyStoriesV2-GPT4-train.txt"
    vocab_size = 10000
    special_tokens = ["<|endoftext|>"]
    
    print(f"Starting BPE training on {input_path}...")
    start_time = time.time()
    process = psutil.Process(os.getpid())
    
    vocab, merges = train_bpe(input_path, vocab_size, special_tokens)
    
    end_time = time.time()
    mem_info = process.memory_info()
    
    duration = end_time - start_time
    print(f"Training took {duration:.2f} seconds ({duration/3600:.24f} hours)")
    print(f"Peak memory usage: {mem_info.rss / (1024**3):.2f} GB")
    
    # Save to disk
    serializable_vocab = {k: v.decode('utf-8', errors='replace') for k, v in vocab.items()}
    
    from tests.common import gpt2_bytes_to_unicode
    byte_encoder = gpt2_bytes_to_unicode()
    def b_to_s(b):
        return "".join(byte_encoder[bt] for bt in b)
    
    serializable_merges = [(b_to_s(m1), b_to_s(m2)) for m1, m2 in merges]
    
    output_vocab_path = "vocab.json"
    output_merges_path = "merges.txt"
    
    with open(output_vocab_path, "w", encoding="utf-8") as f:
        json.dump(serializable_vocab, f, indent=4)
        
    with open(output_merges_path, "w", encoding="utf-8") as f:
        for m1, m2 in serializable_merges:
            f.write(f"{m1} {m2}\n")
            
    print(f"Saved vocab to {output_vocab_path} and merges to {output_merges_path}")
    
    # Longest token
    longest_token = max(vocab.values(), key=len)
    print(f"Longest token: {longest_token}")
    print(f"Longest token length: {len(longest_token)} bytes")
    try:
        decoded = longest_token.decode('utf-8')
    except UnicodeDecodeError:
        decoded = longest_token.decode('utf-8', errors='replace')
    print(f"Longest token (decoded): {decoded}")

if __name__ == "__main__":
    main()