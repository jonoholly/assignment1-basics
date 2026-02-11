from tests.common import gpt2_bytes_to_unicode

d = gpt2_bytes_to_unicode()
keys = list(d.keys())
print(f"First 5 keys: {keys[:5]}")
print(f"Keys 185-195: {keys[185:195]}")
print(f"Index of byte 33 (!): {keys.index(33)}")
print(f"Index of byte 0: {keys.index(0)}")
