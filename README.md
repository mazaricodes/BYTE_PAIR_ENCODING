# BPE Tokenizer — NLP Lab 11 (Batch AI-23)

Byte Pair Encoding tokenizer implemented **from scratch**, trained on the AG News corpus,
served through a FastAPI backend and an interactive React UI. Covers the full post-lab task:
dataset prep, 80/20 split, training at 100/500/1000 merges, applying to unseen test data,
and vocabulary-size analysis.

```
bpe-tokenizer/
├── backend/
│   ├── bpe.py            # BPE implemented from scratch (trainable + serializable)
│   ├── train.py          # load AG News → clean → split → train → save models + metrics
│   ├── main.py           # FastAPI: /tokenize /compare /merges /subwords /vocab /report
│   ├── requirements.txt
│   ├── data/ag_train.csv # AG News
│   └── models/           # trained models (bpe_100/500/1000.json) + report_data.json
├── frontend/             # React + Vite UI
│   └── src/App.jsx
├── make_report.js        # generates the Word report from real metrics
├── BPE_Project_Report.docx
└── README.md
```

## Run the backend
```bash
cd backend
pip install -r requirements.txt
python train.py            # optional: rebuilds models/ (already included)
uvicorn main:app --reload  # http://localhost:8000
```

## Run the frontend
```bash
cd frontend
npm install
npm run dev                # http://localhost:5173
```
The UI reads `VITE_API_URL` (default `http://localhost:8000`). See `.env.example`.

## API
| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/tokenize` | POST `{text, merges}` | tokenize text with one model |
| `/compare` | POST `{text}` | tokenize across 100/500/1000 merges |
| `/merges/{m}` | GET | first 10 merge rules |
| `/subwords/{m}` | GET | top learned subwords |
| `/vocab/{m}` | GET | full vocabulary |
| `/report` | GET | all experiment metrics |

## Results (test set, 800 sentences)
| Merges | Vocab | Tokens/word | Single-char % |
|--------|-------|-------------|---------------|
| 100  | 137  | 3.36 | 56.1% |
| 500  | 534  | 2.35 | 32.4% |
| 1000 | 1026 | 1.97 | 24.1% |

More merges → larger vocabulary → shorter token sequences → less single-character fallback.
