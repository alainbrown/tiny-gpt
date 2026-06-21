from collections import Counter
import json

class BPETokenizer:
    def __init__(self):
        # Base vocabulary: every possible byte.
        self.vocab = {i: bytes([i]) for i in range(256)}

        # Special tokens live outside byte range.
        self.eos_id = 256
        self.vocab[self.eos_id] = b""

        # Ordered list of merges:
        # [((left_id, right_id), new_id), ...]
        self.merges = []

    def text_to_ids(self, text):
        return list(text.encode("utf-8"))

    def get_pair_counts(self, sequences):
        counts = Counter()

        for ids in sequences:
            for pair in zip(ids, ids[1:]):
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

            self.merges.append((pair, new_id))

            left, right = pair
            self.vocab[new_id] = self.vocab[left] + self.vocab[right]

            sequences = [
                self.apply_merge(ids, pair, new_id)
                for ids in sequences
            ]

    def encode(self, text, add_eos=False):
        ids = self.text_to_ids(text)

        for pair, new_id in self.merges:
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
        data = {
            "eos_id": self.eos_id,
            "merges": [
                [left, right, new_id]
                for (left, right), new_id in self.merges
            ],
        }

        with open(path, "w") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path):
        tokenizer = cls()

        with open(path, "r") as f:
            data = json.load(f)

        tokenizer.eos_id = data["eos_id"]
        tokenizer.merges = []

        for left, right, new_id in data["merges"]:
            pair = (left, right)
            tokenizer.merges.append((pair, new_id))
            tokenizer.vocab[new_id] = (
                tokenizer.vocab[left] + tokenizer.vocab[right]
            )

        return tokenizer
