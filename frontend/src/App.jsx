import { useState, useEffect, useCallback } from 'react'

const API = import.meta.env.VITE_API_URL || 'http://localhost:8000'
const MERGES = [100, 500, 1000]

function Token({ t }) {
  const eow = t.endsWith('</w>')
  const core = eow ? t.slice(0, -4) : t
  const single = core.length === 1
  return (
    <span className={`token ${single ? 'single' : 'multi'}`}>
      {core}{eow && <span className="eow">·</span>}
    </span>
  )
}

export default function App() {
  const [online, setOnline] = useState(null)
  const [text, setText] = useState('The government announced lower taxes for unhappy taxpayers worldwide')
  const [merges, setMerges] = useState(1000)
  const [result, setResult] = useState(null)
  const [compare, setCompare] = useState(null)
  const [report, setReport] = useState(null)
  const [rules, setRules] = useState([])
  const [subwords, setSubwords] = useState([])
  const [err, setErr] = useState('')

  useEffect(() => {
    fetch(`${API}/`).then(r => r.json()).then(() => setOnline(true)).catch(() => setOnline(false))
    fetch(`${API}/report`).then(r => r.json()).then(setReport).catch(() => {})
  }, [])

  const loadMeta = useCallback((m) => {
    fetch(`${API}/merges/${m}`).then(r => r.json()).then(setRules).catch(() => {})
    fetch(`${API}/subwords/${m}`).then(r => r.json()).then(setSubwords).catch(() => {})
  }, [])
  useEffect(() => { loadMeta(merges) }, [merges, loadMeta])

  const run = async () => {
    setErr('')
    try {
      const r = await fetch(`${API}/tokenize`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, merges }),
      })
      if (!r.ok) throw new Error((await r.json()).detail)
      setResult(await r.json())
      const c = await fetch(`${API}/compare`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      })
      setCompare(await c.json())
    } catch (e) { setErr(String(e.message || e)) }
  }

  useEffect(() => { if (online) run() }, [online]) // eslint-disable-line

  const exp = report?.experiments

  return (
    <div className="wrap">
      <header className="masthead">
        <div className="kicker">NLP Lab 11 · Subword Tokenization</div>
        <h1 className="title">Byte Pair<br />Encoding.</h1>
        <p className="subtitle">A from-scratch BPE tokenizer trained on the AG News corpus — built, merged, and analyzed.</p>
        <div className="meta-row">
          <span>Batch <b>AI-23</b></span>
          <span>Dataset <b>AG News</b></span>
          <span>Merges <b>100 / 500 / 1000</b></span>
          <span className="status">
            <span className={`dot ${online === null ? '' : online ? 'on' : 'off'}`} />
            <b>{online === null ? 'connecting' : online ? 'API online' : 'API offline'}</b>
          </span>
        </div>
      </header>

      {online === false && (
        <div className="error">
          Backend not reachable at {API}. Start it with: <code>uvicorn main:app --reload</code> inside <code>/backend</code>.
        </div>
      )}

      {/* INTERACTIVE TOKENIZER */}
      <section>
        <div className="sec-head"><span className="sec-num">01</span><span className="sec-title">Tokenize</span></div>
        <p className="sec-desc">Enter text and tokenize it with the trained model. Tokens are split into subwords; <b>·</b> marks an end-of-word boundary. Teal = learned subword, orange = fallback single character.</p>
        <div className="card">
          <textarea value={text} onChange={e => setText(e.target.value)} placeholder="Type a sentence..." />
          <div className="controls">
            <div className="seg">
              {MERGES.map(m => (
                <button key={m} className={merges === m ? 'active' : ''} onClick={() => setMerges(m)}>{m} merges</button>
              ))}
            </div>
            <button className="btn" onClick={run}>Tokenize →</button>
          </div>
          {err && <div className="error">{err}</div>}
          {result && (
            <>
              <div className="tokens" style={{ marginTop: 18 }}>
                {result.tokens.map((t, i) => <Token key={i} t={t} />)}
              </div>
              <div className="count-strip">
                <div className="stat"><b>{result.num_words}</b><span>words</span></div>
                <div className="stat"><b>{result.num_tokens}</b><span>tokens</span></div>
                <div className="stat"><b>{(result.num_tokens / Math.max(result.num_words,1)).toFixed(2)}</b><span>tokens / word</span></div>
                <div className="stat"><b>{result.vocab_size}</b><span>vocab size</span></div>
              </div>
            </>
          )}
        </div>
      </section>

      {/* COMPARE */}
      {compare && (
        <section>
          <div className="sec-head"><span className="sec-num">02</span><span className="sec-title">Vocabulary Size Experiment</span></div>
          <p className="sec-desc">Same input, three vocabularies. Larger vocabularies merge more frequent patterns, producing fewer, longer tokens.</p>
          <div className="cmp-grid">
            {compare.results.map(c => (
              <div className="cmp-col" key={c.merges}>
                <h4>{c.merges} merges</h4>
                <div className="vc">vocab {c.vocab_size} · {c.num_tokens} tokens</div>
                <div className="tokens">{c.tokens.map((t, i) => <Token key={i} t={t} />)}</div>
              </div>
            ))}
          </div>
        </section>
      )}

      {/* METRICS TABLE */}
      {exp && (
        <section>
          <div className="sec-head"><span className="sec-num">03</span><span className="sec-title">Test-Set Metrics</span></div>
          <p className="sec-desc">Measured on the held-out 20% test split ({report.n_test_sentences} sentences, {report.n_train_words.toLocaleString()} train word tokens).</p>
          <div className="card" style={{ padding: 0, overflowX: 'auto' }}>
            <table>
              <thead>
                <tr><th>Merges</th><th>Vocab size</th><th>Avg tokens / word</th><th>Single-char %</th><th>Unseen test words</th></tr>
              </thead>
              <tbody>
                {MERGES.map(m => (
                  <tr key={m}>
                    <td><b>{m}</b></td>
                    <td className="num">{exp[m].vocab_size}</td>
                    <td className="num trend-down">{exp[m].avg_tokens_per_word}</td>
                    <td className="num">{exp[m].single_char_token_pct}%</td>
                    <td className="num">{exp[m].unseen_test_words} / {exp[m].unique_test_words}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="hint">As merges increase, average tokens-per-word falls ({exp[100].avg_tokens_per_word} → {exp[1000].avg_tokens_per_word}) and the share of un-merged single characters drops ({exp[100].single_char_token_pct}% → {exp[1000].single_char_token_pct}%). Unseen test words are still tokenizable — BPE falls back to subwords/characters, so there are no true unknowns.</p>
        </section>
      )}

      {/* MERGE RULES + SUBWORDS */}
      <section>
        <div className="sec-head"><span className="sec-num">04</span><span className="sec-title">Merge Rules & Subwords</span></div>
        <p className="sec-desc">First 10 learned merge rules and the most frequent multi-character subwords for the <b>{merges}-merge</b> model.</p>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(300px,1fr))', gap: 20 }}>
          <div>
            <p className="kicker" style={{ marginBottom: 10 }}>First 10 merges</p>
            <div className="chips">
              {rules.map(r => (
                <span className="chip" key={r.step}><span className="n">{r.step}</span>{r.pair[0]}+{r.pair[1]}<span className="f">{r.freq.toLocaleString()}×</span></span>
              ))}
            </div>
          </div>
          <div>
            <p className="kicker" style={{ marginBottom: 10 }}>Top subwords</p>
            <div className="chips">
              {subwords.slice(0, 18).map(([t, c], i) => (
                <span className="chip" key={i}>{t.replace('</w>', '·')}<span className="f">{c.toLocaleString()}</span></span>
              ))}
            </div>
          </div>
        </div>
      </section>

      <footer>
        BPE Tokenizer · NLP Lab 11 · Batch AI-23 · Built from scratch in Python (FastAPI) + React · Trained on AG News
      </footer>
    </div>
  )
}
