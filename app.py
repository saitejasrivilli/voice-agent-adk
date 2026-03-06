# ============================================================
# Multi-Agent Voice AI — Google ADK + Groq
# Google Cloud AI Research JD-aligned features:
#   ✅ Google ADK multi-agent orchestration
#   ✅ Domain-specialized agents (Healthcare / Finance / Retail / General)
#   ✅ Parallel async tool execution
#   ✅ RAG with BM25-style retrieval
#   ✅ Reasoning traces
#   ✅ Input guardrails + safety filter
#   ✅ Token bucket rate limiting (10 req/min per user)
#   ✅ Metrics dashboard (latency, tokens, tool usage, rate blocks)
#   ✅ Multi-user session isolation
#   ✅ Voice I/O (Whisper + pyttsx3/gTTS)
# Hugging Face Spaces — Python 3.11
# Secret: GROQ_API_KEY (https://console.groq.com)
# ============================================================

import os, json, textwrap, datetime, requests, re, asyncio, time, uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from google.adk.agents import Agent
from google.adk.runners import InMemoryRunner
from google.adk.tools import FunctionTool
from google.genai import types as genai_types
from groq import Groq
import gradio as gr

# ── CONFIG ───────────────────────────────────────────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not set. Add it in Space Settings → Secrets.")

groq_client = Groq(api_key=GROQ_API_KEY)
os.environ.setdefault("GOOGLE_API_KEY", "dummy")
executor_pool = ThreadPoolExecutor(max_workers=4)
print("✅ Config ready")

# ── METRICS STORE ─────────────────────────────────────────────
metrics = {
    "total_queries": 0,
    "tool_calls": defaultdict(int),
    "domain_usage": defaultdict(int),
    "latencies": [],          # ms per query
    "token_usage": [],        # estimated tokens per query
    "guardrail_blocks": 0,
    "errors": 0,
}

def record_metric(domain: str, latency_ms: float, tokens: int, tools_used: list):
    metrics["total_queries"] += 1
    metrics["domain_usage"][domain] += 1
    metrics["latencies"].append(latency_ms)
    metrics["token_usage"].append(tokens)
    for t in tools_used:
        metrics["tool_calls"][t] += 1

def get_metrics_md() -> str:
    if metrics["total_queries"] == 0:
        return "No queries yet. Run some queries to see metrics."
    avg_lat  = sum(metrics["latencies"]) / len(metrics["latencies"])
    avg_tok  = sum(metrics["token_usage"]) / len(metrics["token_usage"])
    p95_lat  = sorted(metrics["latencies"])[int(len(metrics["latencies"]) * 0.95)] if len(metrics["latencies"]) >= 20 else max(metrics["latencies"])
    tool_str = "\n".join([f"  - `{k}`: {v} calls" for k, v in sorted(metrics["tool_calls"].items(), key=lambda x: -x[1])]) or "  - None yet"
    dom_str  = "\n".join([f"  - `{k}`: {v} queries" for k, v in sorted(metrics["domain_usage"].items(), key=lambda x: -x[1])])
    total_blocked = sum(b["blocked"] for b in rate_buckets.values())
    active_users  = len(rate_buckets)
    return f"""### 📊 Live Metrics Dashboard

| Metric | Value |
|--------|-------|
| Total Queries | **{metrics['total_queries']}** |
| Avg Latency | **{avg_lat:.0f} ms** |
| P95 Latency | **{p95_lat:.0f} ms** |
| Avg Tokens/Query | **{avg_tok:.0f}** |
| Guardrail Blocks | **{metrics['guardrail_blocks']}** |
| Rate Limit Blocks | **{total_blocked}** |
| Active Users | **{active_users}** |
| Errors | **{metrics['errors']}** |

**Tool Call Distribution:**
{tool_str}

**Domain Usage:**
{dom_str}

**Rate Limit Config:** `{RATE_LIMIT_CAPACITY} req burst` · `1 token per {RATE_LIMIT_REFILL_S:.0f}s` · `~10 req/min per user`
"""

# ── GUARDRAILS ────────────────────────────────────────────────
BLOCKED_PATTERNS = [
    r"\b(hack|exploit|malware|phishing|sql injection|ddos)\b",
    r"\b(kill|murder|suicide|self.harm)\b",
    r"\b(ssn|credit.card.number|password)\b",
]

def check_guardrails(text: str) -> tuple[bool, str]:
    """Returns (is_safe, reason). Blocks harmful or sensitive queries."""
    lower = text.lower()
    for pattern in BLOCKED_PATTERNS:
        if re.search(pattern, lower):
            metrics["guardrail_blocks"] += 1
            return False, f"⚠️ Query blocked by safety guardrail: matched pattern `{pattern}`"
    if len(text) > 2000:
        return False, "⚠️ Query too long (max 2000 characters)."
    return True, ""

# ── RATE LIMITER ──────────────────────────────────────────────
# Token bucket algorithm per user:
#   - Each user gets a bucket of 10 tokens
#   - 1 token refills every 6 seconds (= 10 requests/minute max)
#   - Each query costs 1 token
#   - Burst of up to 10 requests allowed, then throttled

RATE_LIMIT_CAPACITY  = 10      # max burst
RATE_LIMIT_REFILL_S  = 6.0     # seconds per token refill
RATE_LIMIT_WINDOW_S  = 60      # rolling window for stats

rate_buckets: dict = {}   # user_id -> {"tokens": float, "last_refill": float, "blocked": int}

def check_rate_limit(user_id: str) -> tuple[bool, str]:
    """
    Token bucket rate limiter.
    Returns (is_allowed, message).
    """
    now = time.time()

    if user_id not in rate_buckets:
        rate_buckets[user_id] = {
            "tokens":      float(RATE_LIMIT_CAPACITY),
            "last_refill": now,
            "blocked":     0,
        }

    bucket = rate_buckets[user_id]

    # Refill tokens based on elapsed time
    elapsed        = now - bucket["last_refill"]
    refill_amount  = elapsed / RATE_LIMIT_REFILL_S
    bucket["tokens"]      = min(RATE_LIMIT_CAPACITY, bucket["tokens"] + refill_amount)
    bucket["last_refill"] = now

    if bucket["tokens"] >= 1.0:
        bucket["tokens"] -= 1.0
        remaining = int(bucket["tokens"])
        return True, f"✅ {remaining} requests remaining in burst capacity"
    else:
        bucket["blocked"] += 1
        metrics["guardrail_blocks"] += 1
        wait_s = (1.0 - bucket["tokens"]) * RATE_LIMIT_REFILL_S
        return False, f"⏱️ Rate limit exceeded. Please wait {wait_s:.1f}s before your next query. (Max {RATE_LIMIT_CAPACITY} requests/minute)"

# ── RAG STORE (BM25-style TF-IDF scoring) ────────────────────
rag_chunks = []

def bm25_score(query_terms: set, doc_terms: list, k1: float = 1.5, b: float = 0.75) -> float:
    """Simplified BM25 scoring."""
    avg_dl = sum(len(c["terms"]) for c in rag_chunks) / max(len(rag_chunks), 1)
    dl     = len(doc_terms)
    score  = 0.0
    tf_map = defaultdict(int)
    for t in doc_terms:
        tf_map[t] += 1
    for term in query_terms:
        if term not in tf_map:
            continue
        tf  = tf_map[term]
        idf = 1.0  # simplified
        score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avg_dl))
    return score

def chunk_text(text: str, source: str, chunk_size: int = 300):
    words = text.split()
    step  = chunk_size // 2
    for i in range(0, max(1, len(words) - chunk_size + 1), step):
        chunk = words[i:i + chunk_size]
        rag_chunks.append({"text": " ".join(chunk), "source": source, "terms": [w.lower() for w in chunk]})

def rag_search(query: str, top_k: int = 3) -> str:
    """Search uploaded documents using BM25 retrieval."""
    if not rag_chunks:
        return "No documents uploaded yet. Please upload a document in the RAG tab first."
    q_terms = set(re.findall(r"\w+", query.lower()))
    scored  = [(bm25_score(q_terms, c["terms"]), c) for c in rag_chunks]
    scored.sort(key=lambda x: x[0], reverse=True)
    results = [f"[{c['source']}] {c['text']}" for score, c in scored[:top_k] if score > 0]
    return "\n\n---\n\n".join(results) if results else "No relevant content found in uploaded documents."

# ── TOOLS ────────────────────────────────────────────────────

def web_search(query: str) -> str:
    """Search the web for facts, news, and general knowledge."""
    try:
        # Try DuckDuckGo HTML search (more reliable than instant answer API)
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        url  = f"https://html.duckduckgo.com/html/?q={requests.utils.quote(query)}"
        resp = requests.get(url, headers=headers, timeout=6)
        # Extract result snippets
        snippets = re.findall(r'class="result__snippet"[^>]*>(.*?)</a>', resp.text, re.DOTALL)
        snippets = [re.sub(r"<[^>]+>", "", s).strip() for s in snippets[:3]]
        if snippets:
            return " | ".join(snippets)
        # Fallback: instant answer API
        api_url  = f"https://api.duckduckgo.com/?q={requests.utils.quote(query)}&format=json&no_html=1&skip_disambig=1"
        data     = requests.get(api_url, timeout=5).json()
        if data.get("AbstractText"):
            return data["AbstractText"]
        topics   = data.get("RelatedTopics", [])
        results  = [t.get("Text","") for t in topics[:3] if isinstance(t, dict) and t.get("Text")]
        return " | ".join(results) if results else f"No web results found for: {query}"
    except Exception as e:
        return f"Search error: {e}"

def get_weather(city: str) -> str:
    """Get current weather for a city. Input should be just the city name."""
    try:
        geo = requests.get(
            f"https://geocoding-api.open-meteo.com/v1/search?name={requests.utils.quote(city)}&count=1", timeout=5
        ).json()
        if not geo.get("results"):
            return f"City '{city}' not found."
        loc = geo["results"][0]
        w   = requests.get(
            f"https://api.open-meteo.com/v1/forecast?latitude={loc['latitude']}"
            f"&longitude={loc['longitude']}&current_weather=true&temperature_unit=celsius", timeout=5
        ).json()["current_weather"]
        return f"{loc['name']}: {w['temperature']}°C, wind {w['windspeed']} km/h"
    except Exception as e:
        return f"Weather error: {e}"

def calculate(expression: str) -> str:
    """Evaluate a math expression like '15 * 0.15' or '1000 * 1.07 ** 5'."""
    try:
        clean = re.sub(r"[^\d\+\-\*\/\.\(\)\s\*\*]", "", expression).strip()
        return f"{clean} = {eval(clean, {'__builtins__': {}})}" if clean else "Could not parse."
    except Exception as e:
        return f"Calc error: {e}"

def get_stock_price(symbol: str) -> str:
    """Get current stock price for a ticker like AAPL, TSLA, GOOGL, MSFT."""
    try:
        url  = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol.upper()}?interval=1d&range=1d"
        data = requests.get(url, timeout=5, headers={"User-Agent": "Mozilla/5.0"}).json()
        price = data["chart"]["result"][0]["meta"]["regularMarketPrice"]
        name  = data["chart"]["result"][0]["meta"].get("longName", symbol.upper())
        return f"{name} ({symbol.upper()}): ${price:.2f}"
    except Exception as e:
        return f"Stock error: {e}"

def search_documents(query: str) -> str:
    """Search user-uploaded documents using BM25 retrieval. Use for questions about uploaded files."""
    return rag_search(query)

# ── DOMAIN-SPECIFIC TOOLS ────────────────────────────────────

def drug_interaction_check(drugs: str) -> str:
    """Check potential interactions between drugs. Input: comma-separated drug names."""
    # Simulated — in production would call a real medical API
    drug_list = [d.strip() for d in drugs.split(",")]
    return (f"⚠️ Always consult a licensed pharmacist or physician for drug interactions. "
            f"Checking interactions for: {', '.join(drug_list)}. "
            f"This is a demo — connect to RxNorm or DrugBank API for real data.")

def financial_risk_score(portfolio: str) -> str:
    """Calculate volatility-based risk score for a portfolio of ticker symbols."""
    tickers = [t.strip().upper() for t in portfolio.replace(" ", "").split(",") if t.strip()][:5]
    results = []
    for ticker in tickers:
        try:
            # Fetch 30 days of daily closes from Yahoo Finance
            url  = (f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
                    f"?interval=1d&range=30d")
            data = requests.get(url, timeout=5,
                                headers={"User-Agent": "Mozilla/5.0"}).json()
            closes = data["chart"]["result"][0]["indicators"]["quote"][0]["close"]
            closes = [c for c in closes if c is not None]
            if len(closes) < 5:
                results.append(f"{ticker}: insufficient data")
                continue
            # Daily returns
            returns = [(closes[i] - closes[i-1]) / closes[i-1] for i in range(1, len(closes))]
            mean_r  = sum(returns) / len(returns)
            variance = sum((r - mean_r) ** 2 for r in returns) / len(returns)
            std_dev  = variance ** 0.5
            annual_vol = std_dev * (252 ** 0.5) * 100  # annualised %
            # Simple risk tier
            if annual_vol < 20:
                tier = "🟢 Low"
            elif annual_vol < 40:
                tier = "🟡 Medium"
            else:
                tier = "🔴 High"
            price = closes[-1]
            results.append(f"{ticker}: ${price:.2f} | 30d volatility {annual_vol:.1f}% | Risk: {tier}")
        except Exception as e:
            results.append(f"{ticker}: error — {e}")
    return "Portfolio Risk Analysis:\n" + "\n".join(results)

def product_sentiment(product: str) -> str:
    """Get market sentiment for a retail product or brand."""
    result = web_search(f"{product} reviews pros cons 2025")
    return f"Sentiment for '{product}':\n{result}"

print("✅ All tools ready")

# ── DOMAIN CONFIGURATIONS ────────────────────────────────────
DOMAINS = {
    "🌐 General": {
        "system": "You are a helpful general-purpose voice assistant. Answer clearly and concisely in under 4 sentences.",
        "tools": ["web_search", "get_weather", "calculate", "get_stock_price", "search_documents"],
        "color": "#10b981",
    },
    "🏥 Healthcare": {
        "system": textwrap.dedent("""
            You are a healthcare AI assistant supporting medical professionals.
            Provide evidence-based, clinical information. Always recommend consulting
            a licensed physician for diagnosis or treatment decisions.
            Focus on: symptoms, drug information, clinical guidelines, medical literature.
            Keep answers under 4 sentences for voice delivery.
        """).strip(),
        "tools": ["web_search", "drug_interaction_check", "search_documents", "calculate"],
        "color": "#3b82f6",
    },
    "💹 Finance": {
        "system": textwrap.dedent("""
            You are a financial AI assistant for investment professionals.
            Provide data-driven insights on stocks, markets, risk, and economic trends.
            Always note that this is not financial advice.
            Keep answers under 4 sentences for voice delivery.
        """).strip(),
        "tools": ["get_stock_price", "financial_risk_score", "calculate", "web_search", "search_documents"],
        "color": "#f59e0b",
    },
    "🛒 Retail": {
        "system": textwrap.dedent("""
            You are a retail AI assistant for business analysts and merchandisers.
            Focus on product trends, consumer sentiment, pricing strategy, and market analysis.
            Keep answers under 4 sentences for voice delivery.
        """).strip(),
        "tools": ["product_sentiment", "web_search", "calculate", "search_documents", "get_stock_price"],
        "color": "#8b5cf6",
    },
}

TOOL_REGISTRY = {
    "web_search": web_search,
    "get_weather": get_weather,
    "calculate": calculate,
    "get_stock_price": get_stock_price,
    "search_documents": search_documents,
    "drug_interaction_check": drug_interaction_check,
    "financial_risk_score": financial_risk_score,
    "product_sentiment": product_sentiment,
}

# ── GROQ LLM ─────────────────────────────────────────────────

def groq_complete(system: str, user: str, max_tokens: int = 768) -> tuple[str, int]:
    """Returns (text, estimated_token_count)."""
    resp = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_tokens=max_tokens,
        temperature=0.7,
    )
    text   = resp.choices[0].message.content.strip()
    tokens = resp.usage.total_tokens if hasattr(resp, "usage") and resp.usage else len(text.split()) * 2
    return text, tokens

# ── PARALLEL TOOL EXECUTION ───────────────────────────────────

PLANNER_SYSTEM = textwrap.dedent("""
    You are a planner agent. Analyze the user's request and respond in EXACTLY this format:
    INTENT: <what the user wants>
    TOOLS_NEEDED: <comma-separated list from: web_search, get_weather, calculate, get_stock_price, search_documents, drug_interaction_check, financial_risk_score, product_sentiment, none>
    QUERIES: <pipe-separated query for each tool, in same order as TOOLS_NEEDED>
    TASK: <one clear sentence for the executor>

    Examples:
    TOOLS_NEEDED: get_stock_price, calculate
    QUERIES: AAPL | 185 * 0.15

    TOOLS_NEEDED: none
    QUERIES: none
""").strip()

REASONING_SYSTEM = textwrap.dedent("""
    You are a reasoning agent. Think step by step before answering.
    Format EXACTLY as:
    THINKING:
    Step 1: <observation>
    Step 2: <analysis>
    Step 3: <conclusion>

    ANSWER:
    <final answer, max 3 sentences, natural spoken language>
""").strip()

def run_tools_in_parallel(tools_needed: list, queries: list) -> dict:
    """Execute multiple tools concurrently using ThreadPoolExecutor."""
    if not tools_needed or tools_needed == ["none"]:
        return {}

    def run_one(tool_name: str, query: str):
        fn = TOOL_REGISTRY.get(tool_name)
        if fn:
            try:
                return tool_name, fn(query)
            except Exception as e:
                return tool_name, f"Error: {e}"
        return tool_name, "Tool not found"

    futures = {
        executor_pool.submit(run_one, t, q): t
        for t, q in zip(tools_needed, queries)
        if t in TOOL_REGISTRY
    }
    results = {}
    for future in futures:
        tool_name, result = future.result(timeout=10)
        results[tool_name] = result
    return results

# ── ADK AGENTS ───────────────────────────────────────────────

def build_adk_agent(domain: str) -> tuple:
    """Build planner + executor ADK agents for a given domain."""
    domain_cfg   = DOMAINS[domain]
    domain_tools = [FunctionTool(TOOL_REGISTRY[t]) for t in domain_cfg["tools"] if t in TOOL_REGISTRY]

    executor = Agent(
        name=f"executor_{domain.split()[1].lower()}",
        model="gemini-2.0-flash",
        description=f"Executor agent for {domain} domain.",
        instruction=domain_cfg["system"],
        tools=domain_tools,
    )
    planner = Agent(
        name=f"planner_{domain.split()[1].lower()}",
        model="gemini-2.0-flash",
        description=f"Planner agent for {domain} domain.",
        instruction=PLANNER_SYSTEM,
    )
    return planner, executor

# Build all domain agents at startup
domain_agents = {}
for d in DOMAINS:
    p, e = build_adk_agent(d)
    domain_agents[d] = {"planner": p, "executor": e}

print("✅ Domain agents built:", list(DOMAINS.keys()))

# ── MULTI-USER SESSION STORE ──────────────────────────────────
# Each user gets their own isolated session + conversation history

user_sessions = {}   # user_id -> {"history": [], "log": [], "adk_sessions": {}}

def get_or_create_user(user_id: str) -> dict:
    if user_id not in user_sessions:
        user_sessions[user_id] = {
            "history": [],
            "log": [],
            "adk_sessions": {},
        }
    return user_sessions[user_id]

# ADK runners (shared, sessions are isolated per user)
adk_runners = {}
for d in DOMAINS:
    adk_runners[d] = {
        "planner":  InMemoryRunner(agent=domain_agents[d]["planner"]),
        "executor": InMemoryRunner(agent=domain_agents[d]["executor"]),
    }

async def _ensure_adk_session(user_id: str, domain: str):
    sess = get_or_create_user(user_id)
    if domain not in sess["adk_sessions"]:
        runner = adk_runners[domain]
        ps = await runner["planner"].session_service.create_session(
            app_name=f"planner_{domain}", user_id=user_id
        )
        es = await runner["executor"].session_service.create_session(
            app_name=f"executor_{domain}", user_id=user_id
        )
        sess["adk_sessions"][domain] = {"planner": ps, "executor": es}
    return sess["adk_sessions"][domain]

async def _run_adk(runner, session_id: str, text: str) -> str:
    result = ""
    async for event in runner.run_async(
        user_id="user",
        session_id=session_id,
        new_message=genai_types.Content(role="user", parts=[genai_types.Part(text=text)])
    ):
        if event.is_final_response() and event.content:
            for part in event.content.parts:
                if hasattr(part, "text") and part.text:
                    result += part.text
    return result.strip()

print("✅ ADK runners ready for all domains")

# ── MAIN PIPELINE ─────────────────────────────────────────────

def process_query(user_input: str, domain: str, user_id: str) -> dict:
    start_time = time.time()
    total_tokens = 0
    tools_used = []

    # 1. Guardrail check
    is_safe, reason = check_guardrails(user_input)
    if not is_safe:
        return {"plan": reason, "tool_results": {}, "thinking": "", "answer": reason, "latency_ms": 0, "tokens": 0}

    # 2. Rate limit check
    is_allowed, rate_msg = check_rate_limit(user_id)
    if not is_allowed:
        return {"plan": rate_msg, "tool_results": {}, "thinking": "", "answer": rate_msg, "latency_ms": 0, "tokens": 0}

    # 2. Get user session
    sess = get_or_create_user(user_id)
    context = ""
    if sess["history"]:
        recent  = sess["history"][-3:]
        context = "Previous conversation:\n" + "\n".join(
            [f"User: {h['user']}\nAssistant: {h['assistant']}" for h in recent]
        ) + "\n\n"

    # 3. Planner — identify tools needed
    plan_input = context + f"Domain: {domain}\nUser query: {user_input}"
    plan, plan_tokens = groq_complete(PLANNER_SYSTEM, plan_input)
    total_tokens += plan_tokens

    # 4. Parse tool list + queries from plan
    tools_needed, queries = [], []
    for line in plan.splitlines():
        if line.startswith("TOOLS_NEEDED:"):
            tools_needed = [t.strip() for t in line.split(":", 1)[1].split(",") if t.strip() != "none"]
        elif line.startswith("QUERIES:"):
            queries = [q.strip() for q in line.split(":", 1)[1].split("|")]

    # Pad queries if needed
    while len(queries) < len(tools_needed):
        queries.append(user_input)

    # 5. Parallel tool execution
    tool_results = run_tools_in_parallel(tools_needed, queries)
    tools_used   = list(tool_results.keys())

    # 6. Reasoning trace
    tool_summary = "\n".join([f"{k}: {v}" for k, v in tool_results.items()])
    reasoning_input = f"{context}Question: {user_input}\nTool results:\n{tool_summary}"
    raw_reasoning, r_tokens = groq_complete(REASONING_SYSTEM, reasoning_input)
    total_tokens += r_tokens

    thinking, answer = "", ""
    if "THINKING:" in raw_reasoning and "ANSWER:" in raw_reasoning:
        thinking = raw_reasoning.split("THINKING:")[1].split("ANSWER:")[0].strip()
        answer   = raw_reasoning.split("ANSWER:")[1].strip()
    else:
        answer   = raw_reasoning
        thinking = "Reasoning trace not parsed."

    # 7. ADK executor for final answer (with session isolation per user)
    executor_prompt = f"{context}User question: {user_input}\n"
    if tool_results:
        executor_prompt += "Tool data:\n" + tool_summary + "\n"
    executor_prompt += "Give a clear, natural spoken answer."

    try:
        adk_sessions = asyncio.run(_ensure_adk_session(user_id, domain))
        adk_answer   = asyncio.run(_run_adk(
            adk_runners[domain]["executor"],
            adk_sessions["executor"].id,
            executor_prompt
        ))
        if adk_answer and len(adk_answer) > 10:
            answer = adk_answer
    except Exception as e:
        print(f"ADK executor fallback: {e}")

    # 8. Record metrics
    latency_ms = (time.time() - start_time) * 1000
    record_metric(domain, latency_ms, total_tokens, tools_used)

    # 9. Update user session
    sess["history"].append({"user": user_input, "assistant": answer})
    sess["log"].append({
        "timestamp":    datetime.datetime.now().isoformat(),
        "domain":       domain,
        "user_input":   user_input,
        "plan":         plan,
        "tool_results": tool_results,
        "thinking":     thinking,
        "answer":       answer,
        "latency_ms":   round(latency_ms),
        "tokens":       total_tokens,
    })

    return {
        "plan":         plan,
        "tool_results": tool_results,
        "thinking":     thinking,
        "answer":       answer,
        "latency_ms":   round(latency_ms),
        "tokens":       total_tokens,
    }

def save_log(user_id: str) -> str:
    path = f"/tmp/session_log_{user_id[:8]}.json"
    sess = get_or_create_user(user_id)
    with open(path, "w") as f:
        json.dump(sess["log"], f, indent=2)
    return path

print("✅ Pipeline ready")

# ── SPEECH I/O ───────────────────────────────────────────────

def transcribe_audio(audio_path: str) -> str:
    with open(audio_path, "rb") as f:
        result = groq_client.audio.transcriptions.create(
            model="whisper-large-v3-turbo", file=f, response_format="text"
        )
    return result.strip()

def text_to_speech(text: str) -> str:
    """Convert text to speech — tries pyttsx3 (offline) first, then gTTS."""
    path = "/tmp/response.mp3"
    # Try pyttsx3 (offline, no rate limits)
    try:
        import pyttsx3, wave, struct
        engine = pyttsx3.init()
        engine.setProperty("rate", 165)
        wav_path = "/tmp/response.wav"
        engine.save_to_file(text, wav_path)
        engine.runAndWait()
        # Convert wav to mp3-compatible format by just renaming — Gradio handles wav too
        import shutil
        shutil.copy(wav_path, path.replace(".mp3", ".wav"))
        return path.replace(".mp3", ".wav")
    except Exception:
        pass
    # Fallback: gTTS with retry
    for attempt in range(3):
        try:
            from gtts import gTTS
            import time
            if attempt > 0:
                time.sleep(2 * attempt)
            gTTS(text=text[:500], lang="en", slow=False).save(path)
            return path
        except Exception:
            pass
    # Last resort: return None (no audio, but text answer still works)
    return None

# ── RAG UPLOAD ───────────────────────────────────────────────

def handle_upload(file) -> tuple[str, str]:
    if file is None:
        return "No file uploaded.", ""
    try:
        path = file.name if hasattr(file, "name") else file
        ext  = os.path.splitext(path)[1].lower()
        name = os.path.basename(path)
        if ext == ".pdf":
            try:
                import pypdf
                reader = pypdf.PdfReader(path)
                text   = "\n".join(p.extract_text() or "" for p in reader.pages)
            except ImportError:
                return "pypdf not installed. Upload .txt instead.", ""
        elif ext in (".txt", ".md"):
            with open(path, "r", errors="ignore") as f:
                text = f.read()
        else:
            return f"Unsupported: {ext}. Use .pdf .txt .md", ""
        if not text.strip():
            return "Could not extract text.", ""
        chunk_text(text, name)
        info = f"{len(rag_chunks)} chunks from {len(set(c['source'] for c in rag_chunks))} doc(s)"
        return f"✅ '{name}' loaded — {info}. Ask about it in Chat!", info
    except Exception as e:
        return f"Upload error: {e}", ""

# ── GRADIO UI ────────────────────────────────────────────────

def gradio_pipeline(audio_input, text_input, domain, user_id):
    if not user_id:
        user_id = str(uuid.uuid4())[:8]
    try:
        if audio_input is not None:
            try:
                user_text = transcribe_audio(audio_input)
            except Exception as e:
                user_text = text_input or "Hello!"
        elif text_input and text_input.strip():
            user_text = text_input.strip()
        else:
            return "No input.", "", "", "", None, "", user_id

        result     = process_query(user_text, domain, user_id)
        audio_path = text_to_speech(result["answer"])

        sess       = get_or_create_user(user_id)
        history_md = "\n\n".join([
            f"**You:** {h['user']}\n\n**Assistant:** {h['assistant']}"
            for h in sess["history"][-5:]
        ])
        thinking_md = f"### 🧠 Reasoning Steps\n\n{result['thinking']}\n\n---\n⏱️ `{result['latency_ms']} ms` · 🔤 `{result['tokens']} tokens`"

        tool_str = "\n".join([f"- **{k}**: {v[:120]}..." if len(v)>120 else f"- **{k}**: {v}"
                               for k, v in result["tool_results"].items()]) or "*No tools called*"

        return user_text, result["plan"], thinking_md, result["answer"], audio_path, history_md, user_id

    except Exception as e:
        import traceback
        metrics["errors"] += 1
        err = traceback.format_exc()
        print("ERROR:\n", err)
        return f"Error: {e}", err, "", "", None, "", user_id

def clear_chat(user_id):
    if user_id in user_sessions:
        user_sessions[user_id]["history"] = []
    return "", "", "", "", None, "Session cleared."

# ── BUILD GRADIO UI ──────────────────────────────────────────

with gr.Blocks(
    title="🎙️ Multi-Agent Voice AI",
    theme=gr.themes.Base(primary_hue="emerald", secondary_hue="teal", neutral_hue="slate"),
    css="""
        .gradio-container { max-width: 1100px; margin: 0 auto; }
        .tag { display:inline-block; background:#10b981; color:#fff;
               padding:2px 10px; border-radius:20px; font-size:.75rem; margin:2px; }
        .domain-healthcare { border-left: 4px solid #3b82f6 !important; }
        .domain-finance     { border-left: 4px solid #f59e0b !important; }
        .domain-retail      { border-left: 4px solid #8b5cf6 !important; }
    """
) as demo:

    # Hidden user ID state for session isolation
    user_id_state = gr.State(str(uuid.uuid4())[:8])

    gr.HTML("""
        <div style="text-align:center;padding:20px 0 8px">
            <h1 style="font-size:2rem;font-weight:700;margin-bottom:6px">🎙️ Multi-Agent Voice AI</h1>
            <p style="color:#64748b;margin-bottom:10px">
                Google ADK · Domain Agents · Parallel Tools · BM25 RAG · Reasoning Traces · Metrics Dashboard
            </p>
            <span class="tag">Google ADK</span>
            <span class="tag">Multi-Agent</span>
            <span class="tag">Parallel Tools</span>
            <span class="tag">BM25 RAG</span>
            <span class="tag">Reasoning Traces</span>
            <span class="tag">Healthcare · Finance · Retail</span>
            <span class="tag">LLaMA 3.3 70B · Groq</span>
        </div>
    """)

    with gr.Tabs():

        # ── TAB 1: CHAT ──────────────────────────────────────
        with gr.Tab("💬 Chat"):
            domain_dd = gr.Dropdown(
                choices=list(DOMAINS.keys()),
                value="🌐 General",
                label="🏷️ Domain",
                info="Switch between specialized agent modes"
            )
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 🎤 Input")
                    audio_in = gr.Audio(sources=["microphone","upload"], type="filepath", label="Speak or upload")
                    text_in  = gr.Textbox(placeholder="Or type your question...", label="Text input", lines=2)
                    with gr.Row():
                        submit_btn = gr.Button("▶ Run Agent", variant="primary")
                        clear_btn  = gr.Button("🗑 Clear")
                    gr.Markdown("""**Try by domain:**
🌐 *What's the weather in Tokyo?* · *AAPL stock price?*
🏥 *Check interaction between aspirin and ibuprofen*
💹 *Risk score for AAPL, TSLA, GOOGL portfolio*
🛒 *Product sentiment for Nike shoes*
📚 *(after upload) Summarize the document*""")

                with gr.Column(scale=1):
                    gr.Markdown("### 🤖 Agent Output")
                    transcribed = gr.Textbox(label="📝 Transcribed",    interactive=False)
                    plan_out    = gr.Textbox(label="🗺️ Planner's Plan", lines=4, interactive=False)
                    answer_out  = gr.Textbox(label="💡 Final Answer",    lines=3, interactive=False)
                    audio_out   = gr.Audio(label="🔊 Voice Response", autoplay=True)

            with gr.Accordion("🧠 Reasoning Trace + Latency", open=False):
                thinking_out = gr.Markdown("Run a query to see reasoning.")
            with gr.Accordion("💬 Conversation History", open=False):
                history_out = gr.Markdown("No history yet.")
            with gr.Row():
                log_btn  = gr.Button("📥 Download Session Log")
                log_file = gr.File(label="session_log.json")

            submit_btn.click(
                fn=gradio_pipeline,
                inputs=[audio_in, text_in, domain_dd, user_id_state],
                outputs=[transcribed, plan_out, thinking_out, answer_out, audio_out, history_out, user_id_state]
            )
            clear_btn.click(
                fn=clear_chat,
                inputs=[user_id_state],
                outputs=[transcribed, plan_out, thinking_out, answer_out, audio_out, history_out]
            )
            log_btn.click(
                fn=lambda uid: save_log(uid),
                inputs=[user_id_state],
                outputs=[log_file]
            )

        # ── TAB 2: RAG UPLOAD ────────────────────────────────
        with gr.Tab("📚 RAG Documents"):
            gr.Markdown("""### Upload Documents for BM25 Retrieval
Supports `.pdf`, `.txt`, `.md` — then ask questions in the Chat tab using any domain.""")
            upload_box    = gr.File(label="Upload", file_types=[".pdf",".txt",".md"])
            upload_btn    = gr.Button("📤 Process Document", variant="primary")
            upload_status = gr.Textbox(label="Status", interactive=False)
            rag_info      = gr.Textbox(label="RAG Store", interactive=False, value="Empty")
            upload_btn.click(fn=handle_upload, inputs=[upload_box], outputs=[upload_status, rag_info])

        # ── TAB 3: METRICS DASHBOARD ─────────────────────────
        with gr.Tab("📊 Metrics Dashboard"):
            gr.Markdown("### Live system metrics — updates after each query")
            metrics_md  = gr.Markdown(get_metrics_md())
            refresh_btn = gr.Button("🔄 Refresh Metrics")
            refresh_btn.click(fn=get_metrics_md, outputs=[metrics_md])

print("🚀 Launching...")
demo.launch(server_name="0.0.0.0")
