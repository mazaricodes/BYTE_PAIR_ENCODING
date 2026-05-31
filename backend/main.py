"""
FastAPI backend for the BPE Tokenizer project.
Loads pre-trained BPE models (100/500/1000 merges) and exposes endpoints
for tokenization, merge rules, vocabulary, and experiment metrics.
"""
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from bpe import BPETokenizer

HERE = Path(__file__).parent
MODELS = HERE / "models"
MERGE_SIZES = [100, 500, 1000]

app = FastAPI(title="BPE Tokenizer API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# load all models once at startup
TOKENIZERS = {m: BPETokenizer.load(MODELS / f"bpe_{m}.json") for m in MERGE_SIZES}
REPORT = json.load(open(MODELS / "report_data.json"))


class TokenizeRequest(BaseModel):
    text: str
    merges: int = 1000


@app.get("/")
def root():
    return {"status": "ok", "available_merges": MERGE_SIZES}


@app.post("/tokenize")
def tokenize(req: TokenizeRequest):
    if req.merges not in TOKENIZERS:
        raise HTTPException(400, f"merges must be one of {MERGE_SIZES}")
    if not req.text.strip():
        raise HTTPException(400, "text is empty")
    tok = TOKENIZERS[req.merges]
    tokens = tok.tokenize(req.text)
    return {
        "merges": req.merges,
        "vocab_size": len(tok.vocab),
        "tokens": tokens,
        "num_tokens": len(tokens),
        "num_words": len([w for w in req.text.split() if w.strip()]),
    }


@app.post("/compare")
def compare(req: TokenizeRequest):
    """Tokenize the same text across all three merge sizes."""
    if not req.text.strip():
        raise HTTPException(400, "text is empty")
    out = []
    for m in MERGE_SIZES:
        tok = TOKENIZERS[m]
        tokens = tok.tokenize(req.text)
        out.append({
            "merges": m,
            "vocab_size": len(tok.vocab),
            "tokens": tokens,
            "num_tokens": len(tokens),
        })
    return {"text": req.text, "results": out}


@app.get("/merges/{m}")
def merge_rules(m: int):
    if str(m) not in REPORT["experiments"]:
        raise HTTPException(404, "no such model")
    return REPORT["experiments"][str(m)]["first_10_merges"]


@app.get("/vocab/{m}")
def vocab(m: int):
    if m not in TOKENIZERS:
        raise HTTPException(404, "no such model")
    return {"merges": m, "vocab": sorted(TOKENIZERS[m].vocab)}


@app.get("/subwords/{m}")
def subwords(m: int):
    if str(m) not in REPORT["experiments"]:
        raise HTTPException(404, "no such model")
    return REPORT["experiments"][str(m)]["top_subwords"]


@app.get("/report")
def report():
    return REPORT
