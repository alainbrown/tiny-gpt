from collections import Counter
import json
from itertools import islice

class BPETokenizer:
    def __init__(self):
        # Base vocabulary: every possible byte.
        self.vocab = {i: bytes([i]) for i in range(256)}

        # Special tokens live outside byte range.
        self.eos_id = 256
        self.vocab[self.eos_id] = b""
        
        self.byte_encoder = self.bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}

        # Dictionary mapping pair to new_id:
        # {(left_id, right_id): new_id}
        self.merges = {}

    def text_to_ids(self, text):
        return list(text.encode("utf-8"))

    def get_pair_counts(self, sequences):
        counts = Counter()

        for ids in sequences:
            for pair in zip(ids, islice(ids, 1, None)):
                counts[pair] += 1

        return counts

    def apply_merge(self, ids, pair, new_id):
        out = []
        i = 0

        while i < len(ids):
            if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
                out.append(new_id)
                i += 2
            else:
                out.append(ids[i])
                i +=1
        return out

    def train(self, texts, vocab_size):
        assert vocab_size > 257

        # Tokenizer training corpus.
        # This can be a subset of the full dataset.
        sequences = [self.text_to_ids(text) for text in texts]

        next_id = 257

        while next_id < vocab_size:
            counts = self.get_pair_counts(sequences)

            if len(counts) == 0:
                break

            pair, count = counts.most_common(1)[0]

            if count < 2:
                break

            new_id = next_id
            next_id += 1
            self.merges[pair] = new_id

            left, right = pair
            self.vocab[new_id] = self.vocab[left] + self.vocab[right]

            sequences = [
                self.apply_merge(ids, pair, new_id)
                for ids in sequences
            ]

    def encode(self, text, add_eos=False):
        ids = self.text_to_ids(text)
        while len(ids) >= 2:
            pairs_zip = zip(ids, islice(ids, 1, None))
            pair = min(pairs_zip, key=lambda p: self.merges.get(p, float("inf")))
            
            if pair not in self.merges:
                break

            new_id = self.merges[pair]
            ids = self.apply_merge(ids, pair, new_id)

        if add_eos:
            ids.append(self.eos_id)

        return ids

    def decode(self, ids, skip_special=True):
        chunks = []

        for idx in ids:
            if idx == self.eos_id and skip_special:
                continue

            chunks.append(self.vocab[idx])

        return b"".join(chunks).decode("utf-8", errors="replace")

    def save(self, path):
        vocab_export = {}
        for idx, token_bytes in self.vocab.items():
            if idx == self.eos_id:
                continue
            token_str = "".join([self.byte_encoder[b] for b in token_bytes])
            vocab_export[token_str] = idx

        merges_export = []
        sorted_merges = sorted(self.merges.items(), key=lambda x: x[1])
        for (left, right), new_id in sorted_merges:
            left_str = "".join([self.byte_encoder[b] for b in self.vocab[left]])
            right_str = "".join([self.byte_encoder[b] for b in self.vocab[right]])
            merges_export.append(f"{left_str} {right_str}")

        data = {
            "version": "1.0",
            "added_tokens": [
                {"id": self.eos_id, "content": "<EOS>", "special": True}
            ],
            "model": {
                "type": "BPE",
                "vocab": vocab_export,
                "merges": merges_export
            }
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path):
        tokenizer = cls()

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        tokenizer.eos_id = data["added_tokens"][0]["id"]
        
        vocab_import = data["model"]["vocab"]
        tokenizer.vocab = {}
        for token_str, idx in vocab_import.items():
            token_bytes = bytes([tokenizer.byte_decoder[c] for c in token_str])
            tokenizer.vocab[idx] = token_bytes
            
        tokenizer.vocab[tokenizer.eos_id] = b""
        
        tokenizer.merges = {}
        for merge_str in data["model"]["merges"]:
            left_str, right_str = merge_str.split(" ")
            merged_str = left_str + right_str
            
            left_id = vocab_import[left_str]
            right_id = vocab_import[right_str]
            new_id = vocab_import[merged_str]
            
            tokenizer.merges[(left_id, right_id)] = new_id

        return tokenizer

    @staticmethod
    def bytes_to_unicode():
        bs = list(range(ord("!"), ord("~")+1)) + list(range(ord("¡"), ord("¬")+1)) + list(range(ord("®"), ord("ÿ")+1))
        cs = bs[:]
        n = 0
        for b in range(2**8):
            if b not in bs:
                bs.append(b)
                cs.append(2**8 + n)
                n += 1
        cs = [chr(n) for n in cs]
        return dict(zip(bs, cs))
