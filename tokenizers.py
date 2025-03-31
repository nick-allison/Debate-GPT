import json
import re


class BPETokenizer:
    """
    A placeholder BPE tokenizer that loads vocab from a JSON file, then can encode and decode.
    See the other(messy) folder for notebooks with the code that I used for training the tokenizer.
    """
    def __init__(self, bpe_file= 'vocab.json'):
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
        # Create regex pattern to match all keys in mapping
        pattern = "|".join(re.escape(k) for k in list(self.token_to_id.keys())[::-1])

        # Split string but keep the delimiters
        parts = re.split(f'({pattern})', text)

        # Replace found substrings with their corresponding numbers
        result = [self.token_to_id[p] if p in self.token_to_id else p for p in parts if p]

        return result


    def decode(self, ids):
        tokens = []
        for i in ids:
            if i < len(self.id_to_token) and self.id_to_token[i] is not None:
                tokens.append(self.id_to_token[i])
        return "".join(tokens)

    def __len__(self):
        return len(self.id_to_token)
