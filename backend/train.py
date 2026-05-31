"""
Post-lab pipeline:
1. Load AG News, use text only, clean.
2. 80/20 train-test split.
3. Train BPE at 100 / 500 / 1000 merges on TRAIN only.
4. Apply to test sentences, collect metrics, save artifacts.
"""
import csv
import json
import random
import statistics
from pathlib import Path

from bpe import BPETokenizer, clean_text

HERE = Path(__file__).parent
DATA = HERE / "data" / "ag_train.csv"
MODELS = HERE / "models"
MODELS.mkdir(exist_ok=True)

MERGE_SIZES = [100, 500, 1000]
N_SENTENCES = 4000
SEED = 42


def load_sentences(n):
    rows = []
    with open(DATA, encoding="utf-8") as f:
        for label, title, desc in csv.reader(f):
            text = clean_text(f"{title} {desc}")
            if len(text.split()) >= 4:
                rows.append(text)
            if len(rows) >= n:
                break
    return rows


def main():
    random.seed(SEED)
    sentences = load_sentences(N_SENTENCES)
    random.shuffle(sentences)
    split = int(0.8 * len(sentences))
    train_sents, test_sents = sentences[:split], sentences[split:]

    train_words = [w for s in train_sents for w in s.split()]
    print(f"Sentences: {len(sentences)} (train {len(train_sents)}, test {len(test_sents)})")
    print(f"Train word tokens: {len(train_words)}, unique words: {len(set(train_words))}")

    report = {
        "dataset": "AG News (text only)",
        "n_sentences": len(sentences),
        "n_train_sentences": len(train_sents),
        "n_test_sentences": len(test_sents),
        "n_train_words": len(train_words),
        "n_unique_train_words": len(set(train_words)),
        "experiments": {},
        "sample_tokenizations": {},
    }

    # fixed sample test sentences (first 5 of test set)
    sample_sents = test_sents[:5]
    test_words = [w for s in test_sents for w in s.split()]
    train_vocab_words = set(train_words)

    for m in MERGE_SIZES:
        tok = BPETokenizer()
        first_rules = tok.train(train_words, num_merges=m, log_first=10)
        tok.save(MODELS / f"bpe_{m}.json")

        # metrics on test set
        token_counts = []
        single_char_tokens = 0
        total_tokens = 0
        for s in test_sents:
            toks = tok.tokenize(s)
            token_counts.append(len(toks))
            for t in toks:
                total_tokens += 1
                if len(t.replace(BPETokenizer.END, "")) == 1:
                    single_char_tokens += 1

        unseen_words = sum(1 for w in set(test_words) if w not in train_vocab_words)
        avg_tokens_per_sent = statistics.mean(token_counts)
        # avg subwords per word on test
        tw = len(test_words)
        avg_tokens_per_word = total_tokens / tw

        report["experiments"][str(m)] = {
            "merges": m,
            "vocab_size": len(tok.vocab),
            "avg_tokens_per_sentence": round(avg_tokens_per_sent, 2),
            "avg_tokens_per_word": round(avg_tokens_per_word, 3),
            "total_test_tokens": total_tokens,
            "single_char_token_pct": round(100 * single_char_tokens / total_tokens, 2),
            "unseen_test_words": unseen_words,
            "unique_test_words": len(set(test_words)),
            "first_10_merges": first_rules,
            "top_subwords": [[t, c] for t, c in tok.top_subwords(train_words, 20)],
        }

        # sample tokenizations
        report["sample_tokenizations"][str(m)] = [
            {"sentence": s, "tokens": tok.tokenize(s)} for s in sample_sents
        ]
        print(f"merges={m}: vocab={len(tok.vocab)}, "
              f"avg tok/word={avg_tokens_per_word:.3f}, "
              f"single-char%={100*single_char_tokens/total_tokens:.1f}")

    with open(MODELS / "report_data.json", "w") as f:
        json.dump(report, f, indent=2)
    print("Saved models + report_data.json")


if __name__ == "__main__":
    main()
