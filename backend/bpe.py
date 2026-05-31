"""
Byte Pair Encoding (BPE) tokenizer implemented from scratch.
Extends the lab implementation into a trainable, serializable class.
"""
import json
import re
from collections import Counter, defaultdict


def clean_text(text: str) -> str:
    """Lowercase and remove special characters, keep letters/digits/space."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


class BPETokenizer:
    END = "</w>"

    def __init__(self):
        self.merges = []          # ordered list of merged pairs [(a, b), ...]
        self.merge_ranks = {}     # (a, b) -> rank (lower = earlier merge)
        self.vocab = set()        # set of subword tokens
        self.num_merges = 0

    # ---------- training ----------
    @staticmethod
    def _word_to_symbols(word):
        return list(word) + [BPETokenizer.END]

    @staticmethod
    def _get_word_freqs(corpus_words):
        freqs = Counter(corpus_words)
        # represent each word as a tuple of symbols
        return {tuple(BPETokenizer._word_to_symbols(w)): c for w, c in freqs.items()}

    @staticmethod
    def _get_stats(word_freqs):
        pairs = defaultdict(int)
        for symbols, freq in word_freqs.items():
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq
        return pairs

    @staticmethod
    def _merge(pair, word_freqs):
        a, b = pair
        merged = a + b
        new_freqs = {}
        for symbols, freq in word_freqs.items():
            new_symbols = []
            i = 0
            while i < len(symbols):
                if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                    new_symbols.append(merged)
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            new_freqs[tuple(new_symbols)] = freq
        return new_freqs

    def train(self, words, num_merges, log_first=10):
        """Train BPE on a list of words. Returns the first `log_first` merge rules."""
        self.num_merges = num_merges
        word_freqs = self._get_word_freqs(words)
        self.merges = []
        first_rules = []
        for i in range(num_merges):
            stats = self._get_stats(word_freqs)
            if not stats:
                break
            best = max(stats, key=stats.get)
            word_freqs = self._merge(best, word_freqs)
            self.merges.append(best)
            if i < log_first:
                first_rules.append({"step": i + 1, "pair": list(best), "freq": stats[best]})
        self.merge_ranks = {tuple(p): r for r, p in enumerate(self.merges)}
        # final vocabulary = all surviving symbols
        self.vocab = set()
        for symbols in word_freqs:
            self.vocab.update(symbols)
        return first_rules

    # ---------- inference ----------
    def tokenize_word(self, word):
        symbols = self._word_to_symbols(word)
        while True:
            # find adjacent pair with the lowest merge rank
            best_rank = None
            best_idx = None
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                rank = self.merge_ranks.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_idx = i
            if best_idx is None:
                break
            symbols = (
                symbols[:best_idx]
                + [symbols[best_idx] + symbols[best_idx + 1]]
                + symbols[best_idx + 2:]
            )
        return symbols

    def tokenize(self, text):
        text = clean_text(text)
        out = []
        for w in text.split():
            out.extend(self.tokenize_word(w))
        return out

    # ---------- stats ----------
    def top_subwords(self, words, n=20):
        """Most frequent learned multi-character subwords across a word list."""
        counter = Counter()
        for w in words:
            for tok in self.tokenize_word(w):
                clean = tok.replace(self.END, "")
                if len(clean) > 1:
                    counter[tok] += 1
        return counter.most_common(n)

    # ---------- persistence ----------
    def to_dict(self):
        return {
            "num_merges": self.num_merges,
            "merges": [list(p) for p in self.merges],
            "vocab": sorted(self.vocab),
        }

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            d = json.load(f)
        obj = cls()
        obj.num_merges = d["num_merges"]
        obj.merges = [tuple(p) for p in d["merges"]]
        obj.merge_ranks = {tuple(p): r for r, p in enumerate(obj.merges)}
        obj.vocab = set(d["vocab"])
        return obj
