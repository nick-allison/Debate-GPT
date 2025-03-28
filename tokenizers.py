import json
<<<<<<< HEAD
import re


class BPETokenizer:
    """
    A placeholder BPE tokenizer that loads vocab from a JSON file, then can encode and decode.
    See the other(messy) folder for notebooks with the code that I used for training the tokenizer.
    """
    def __init__(self, bpe_file= 'vocab.json'):
=======

class CharacterTokenizer:
    """
    Uses a simple character-level approach: each unique char is a token.
    """
    def __init__(self, text):
        chars = sorted(list(set(text)))
        self.vocab = chars
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}

    def encode(self, text):
        return [self.stoi[ch] for ch in text]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)

    def __len__(self):
        return len(self.vocab)

class BPETokenizer:
    """
    A placeholder BPE tokenizer that loads from a JSON file.
    See the other(messy) folder for notebooks with the code that I used for
    tokenization.
    """
    def __init__(self, bpe_file):
>>>>>>> 7d5b632a9bd1c8ce37d6e30c350fbe1b062b9b2d
        with open(bpe_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            self.id_to_token = data
            self.token_to_id = {tok: i for i, tok in enumerate(data)}
        elif isinstance(data, dict):
            self.token_to_id = data
            self.id_to_token = [None] * (max(data.values())+1)
            for t, i in data.items():
                self.id_to_token[i] = t
        else:
            raise ValueError("BPE file format not recognized.")

    def encode(self, text):
<<<<<<< HEAD
        # Create regex pattern to match all keys in mapping
        pattern = "|".join(re.escape(k) for k in list(self.token_to_id.keys())[::-1])

        # Split string but keep the delimiters
        parts = re.split(f'({pattern})', text)

        # Replace found substrings with their corresponding numbers
        result = [self.token_to_id[p] if p in self.token_to_id else p for p in parts if p]

        return result

=======
        # simplistic example (split on space)
        tokens = text.split(" ")
        ids = []
        for t in tokens:
            if t in self.token_to_id:
                ids.append(self.token_to_id[t])
            else:
                # unknown token fallback
                ids.append(self.token_to_id.get("<UNK>", 0))
        return ids
>>>>>>> 7d5b632a9bd1c8ce37d6e30c350fbe1b062b9b2d

    def decode(self, ids):
        tokens = []
        for i in ids:
            if i < len(self.id_to_token) and self.id_to_token[i] is not None:
                tokens.append(self.id_to_token[i])
<<<<<<< HEAD
        return "".join(tokens)
=======
        return " ".join(tokens)
>>>>>>> 7d5b632a9bd1c8ce37d6e30c350fbe1b062b9b2d

    def __len__(self):
        return len(self.id_to_token)
