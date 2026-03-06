# 🎙️ Multi-Agent Voice AI — Google ADK

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Google ADK](https://img.shields.io/badge/Google_ADK-1.0-4285F4?style=for-the-badge&logo=google&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-LLaMA_3.3_70B-F55036?style=for-the-badge)
![HuggingFace](https://img.shields.io/badge/🤗_Spaces-Live_Demo-FFD21E?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-10b981?style=for-the-badge)

**A production-grade multi-agent voice AI system built with Google ADK, featuring domain-specialized agents for Healthcare, Finance, and Retail — with parallel tool execution, BM25 RAG, reasoning traces, and a live metrics dashboard.**

[🚀 Live Demo](https://huggingface.co/spaces/SaiTejaSrivilli/voice-agent-adk) · [📖 Architecture](#architecture) · [⚡ Quick Start](#quick-start)

</div>

---

## 🌐 Live Demo

> **Try it now — no setup required:**
> ## 👉 [huggingface.co/spaces/SaiTejaSrivilli/voice-agent-adk](https://huggingface.co/spaces/SaiTejaSrivilli/voice-agent-adk)

<div align="center">

| Feature | Try This |
|---------|----------|
| 🌐 General | *"What's the weather in Tokyo?"* |
| 💹 Finance | *"Risk score for AAPL, TSLA, GOOGL"* |
| 🏥 Healthcare | *"Check interaction between aspirin and ibuprofen"* |
| 🛒 Retail | *"Product sentiment for Nike shoes"* |
| 🧠 Memory | Ask *"Who invented the telephone?"* then *"When did he die?"* |
| 📚 RAG | Upload a PDF → ask *"Summarize the document"* |

</div>

---

## 🎯 What This Project Demonstrates

This project was built to align with the **Google Cloud AI Research** engineering role requirements:

| JD Requirement | Implementation |
|---|---|
| Multi-agent systems | Google ADK Planner → Executor architecture |
| Information retrieval | BM25 RAG with TF-IDF normalization |
| Large-scale system design | Multi-user session isolation, metrics tracking |
| Distributed computing | Parallel async tool execution via `ThreadPoolExecutor` |
| NLP | Intent classification, reasoning traces, entity extraction |
| AI for healthcare/finance/retail | Domain-specialized agents with custom toolsets |
| Agent safety | Input guardrails, toxicity filtering |
| Rate limiting | Token bucket algorithm (10 req/min per user) |
| Speech/Audio | Groq Whisper transcription + pyttsx3/gTTS synthesis |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     USER INPUT                               │
│              (Voice / Text / Uploaded Doc)                   │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                  SAFETY GUARDRAILS                           │
│         (regex patterns · length limits · PII)              │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              GOOGLE ADK — PLANNER AGENT                      │
│                  (LLaMA 3.3 70B / Groq)                     │
│  Identifies: INTENT · TOOLS_NEEDED · QUERIES · TASK         │
└──────────────────────┬──────────────────────────────────────┘
                       │
          ┌────────────┴────────────┐
          ▼                         ▼
  ┌───────────────┐       ┌──────────────────┐
  │ PARALLEL TOOL │       │  DOMAIN CONTEXT  │
  │  EXECUTION    │       │  (Healthcare /   │
  │               │       │  Finance/Retail) │
  │ web_search    │       └──────────────────┘
  │ get_weather   │
  │ calculate     │
  │ stock_price   │
  │ BM25 RAG      │
  │ drug_check    │
  │ risk_score    │
  │ sentiment     │
  └───────┬───────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────┐
│            GOOGLE ADK — EXECUTOR AGENT                       │
│                  (LLaMA 3.3 70B / Groq)                     │
│     Synthesizes tool results + conversation context          │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                REASONING TRACE GENERATOR                     │
│          Step 1 → Step 2 → Step 3 → Conclusion              │
└──────────────────────┬──────────────────────────────────────┘
                       │
          ┌────────────┴────────────┐
          ▼                         ▼
   ┌─────────────┐          ┌──────────────┐
   │  TEXT OUTPUT │          │ VOICE OUTPUT │
   │  (Gradio UI) │          │  (pyttsx3 /  │
   └─────────────┘          │    gTTS)     │
                             └──────────────┘
```

---

## ✨ Features

### 🤖 Google ADK Multi-Agent System
- **Planner Agent** — analyzes intent, selects tools, creates execution plan
- **Executor Agent** — runs tools, synthesizes results, generates spoken answer
- Full **session state management** via ADK `InMemoryRunner`
- **Multi-user isolation** — each user gets their own session UUID

### 🏥💹🛒 Domain-Specialized Agents
Switch between four specialized modes, each with its own system prompt and toolset:

| Domain | Tools | Use Case |
|--------|-------|----------|
| 🌐 General | web_search, weather, calculator, stocks | General assistant |
| 🏥 Healthcare | drug_interaction_check, web_search, RAG | Clinical decision support |
| 💹 Finance | financial_risk_score, stock_price, calculator | Investment analysis |
| 🛒 Retail | product_sentiment, web_search, calculator | Market analysis |

### ⚡ Parallel Async Tool Execution
Multiple tools run **concurrently** via `ThreadPoolExecutor` — a query needing both stock price and web search runs both simultaneously, cutting latency in half.

```python
futures = {
    executor_pool.submit(run_one, tool, query): tool
    for tool, query in zip(tools_needed, queries)
}
```

### 📚 BM25 RAG (Document Retrieval)
Upload `.pdf`, `.txt`, or `.md` files — ask questions in natural language. Uses proper **BM25 scoring** with TF normalization and document length penalty (not just keyword overlap).

```
Score = Σ IDF(t) × [TF(t,d) × (k1+1)] / [TF(t,d) + k1 × (1 - b + b × |d|/avgdl)]
```

### 🧠 Reasoning Traces
Every query generates an explicit step-by-step reasoning chain before the final answer:
```
Step 1: User is asking about AAPL stock volatility
Step 2: Retrieved 30-day price history, computed daily returns
Step 3: Annualized std dev = 22.3% → Medium risk tier
```

### 🛡️ Safety Guardrails
Input validation layer blocks:
- Injection attacks (`sql injection`, `hack`, `exploit`)
- Harmful content patterns
- PII exposure risks (SSN, credit card numbers)
- Oversized inputs (> 2000 chars)

### 🚦 Rate Limiting (Token Bucket Algorithm)
Per-user rate limiting using the **token bucket algorithm** — the same approach used in production APIs:

```
Capacity : 10 tokens (max burst)
Refill   : 1 token per 6 seconds → ~10 requests/minute
Cost     : 1 token per query
Response : HTTP-style message with wait time if exceeded
```

Each user gets their own isolated bucket. Burst traffic is absorbed, sustained overuse is throttled. Rate limit blocks are tracked in the metrics dashboard.

### 📊 Live Metrics Dashboard
Real-time tracking of:
- Average & P95 query latency
- Token usage per query
- Tool call distribution
- Domain usage breakdown
- Guardrail block count & errors

### 🎙️ Voice I/O
- **Input**: Groq `whisper-large-v3-turbo` — fast, accurate, free
- **Output**: `pyttsx3` (offline, no rate limits) with `gTTS` fallback

---

## 💬 Example Queries

**🌐 General**
```
"What's the weather in Tokyo?"
"What is the stock price of NVDA?"
"Calculate compound interest on $10,000 at 7% for 5 years"
```

**🏥 Healthcare**
```
"Check interaction between aspirin and ibuprofen"
"What are the symptoms of type 2 diabetes?"
```

**💹 Finance**
```
"Risk score for AAPL, TSLA, GOOGL portfolio"
"What are the latest Fed interest rate decisions?"
```

**🛒 Retail**
```
"Product sentiment for Nike shoes"
"What are trending consumer products in 2025?"
```

**🧠 Multi-turn memory**
```
Turn 1: "Who invented the telephone?"
Turn 2: "When did he die?"        ← remembers context
Turn 3: "What was his net worth?" ← still remembers
```

---

## ⚡ Quick Start

### Run on Hugging Face Spaces
👉 **[Live Demo](https://huggingface.co/spaces/SaiTejaSrivilli/voice-agent-adk)** — no setup required

### Run Locally

```bash
# Clone
git clone https://github.com/SaiTejaSrivilli/voice-agent-adk
cd voice-agent-adk

# Install dependencies
pip install -r requirements.txt

# Set your Groq API key (free at https://console.groq.com)
export GROQ_API_KEY="your_key_here"

# Run
python app.py
```

Then open `http://localhost:7860` in your browser.

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Agent Framework | Google ADK 1.0 |
| LLM | LLaMA 3.3 70B via Groq |
| Speech-to-Text | Groq Whisper large-v3-turbo |
| Text-to-Speech | pyttsx3 (offline) + gTTS fallback |
| RAG | BM25 scoring (custom implementation) |
| Parallel Execution | Python `ThreadPoolExecutor` |
| UI | Gradio 6.x |
| Deployment | Hugging Face Spaces (Python 3.11) |
| Weather API | Open-Meteo (free, no key) |
| Finance API | Yahoo Finance (free, no key) |
| Search | DuckDuckGo HTML scraping |

---

## 📁 Project Structure

```
voice-agent-adk/
├── app.py              # Main application (all-in-one)
├── requirements.txt    # Python dependencies
├── packages.txt        # System dependencies (espeak for TTS)
├── .python-version     # Python 3.11 pin for HF Spaces
└── README.md
```

---

## 🔑 Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GROQ_API_KEY` | ✅ | Free at [console.groq.com](https://console.groq.com) |

---

## 📈 Roadmap

- [ ] Add vector embeddings for semantic RAG (replacing BM25)
- [ ] Add Agent-to-Agent (A2A) protocol support
- [ ] Persistent session storage (Redis/PostgreSQL)
- [ ] Evaluation benchmarks (latency, answer quality)
- [ ] Add more domain agents (Legal, Education)

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

Built with ❤️ using [Google ADK](https://google.github.io/adk-docs/) · [Groq](https://console.groq.com) · [Hugging Face Spaces](https://huggingface.co/spaces)

⭐ **Star this repo if you found it useful!**

</div>
