"""
Portfolio Allocator API
Run:  uvicorn main:app --reload
Docs: http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import time, os, json, hashlib, warnings, io
import requests
import yfinance as yf
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

warnings.filterwarnings("ignore")

app = FastAPI(
    title="Portfolio Allocator",
    description="Enter a budget → get an aggressive portfolio allocation.",
    version="1.0.0",
)

# ─── Cache ────────────────────────────────────────────────────────────────────

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
CACHE_TTL = 24 * 3600

def _cp(name): return os.path.join(CACHE_DIR, name.replace("/", "_"))
def _fresh(p):
    try: return (time.time() - os.path.getmtime(p)) < CACHE_TTL
    except: return False

def cache_get_json(name):
    os.makedirs(CACHE_DIR, exist_ok=True)
    p = _cp(name + ".json")
    if _fresh(p):
        try: return json.load(open(p))
        except: pass
    return None

def cache_set_json(name, obj):
    os.makedirs(CACHE_DIR, exist_ok=True)
    try: json.dump(obj, open(_cp(name + ".json"), "w"))
    except: pass

def cache_get_df(name):
    os.makedirs(CACHE_DIR, exist_ok=True)
    p = _cp(name + ".parquet")
    if _fresh(p):
        try: return pd.read_parquet(p)
        except: pass
    return None

def cache_set_df(name, df):
    os.makedirs(CACHE_DIR, exist_ok=True)
    try: df.to_parquet(_cp(name + ".parquet"))
    except: pass

# ─── Universe ─────────────────────────────────────────────────────────────────

def get_sp500_sectors():
    cached = cache_get_json("sp500_sectors")
    if cached: return cached
    try:
        r = requests.get(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            headers={"User-Agent": "Mozilla/5.0"}, timeout=10
        )
        # Wrap HTML in StringIO so pandas doesn't misinterpret it as a file path
        df = pd.read_html(io.StringIO(r.text))[0]
        df["Symbol"] = df["Symbol"].str.replace(".", "-", regex=False)
        result = df.set_index("Symbol")["GICS Sector"].to_dict()
        cache_set_json("sp500_sectors", result)
        return result
    except Exception as e:
        raise RuntimeError(f"Could not fetch S&P 500 data: {e}")

def get_top_tickers_by_mcap(tickers: list, n: int) -> list:
    h = hashlib.md5(str(sorted(tickers)).encode()).hexdigest()
    cached = cache_get_json(f"mcap_{n}_{h}")
    if cached: return cached

    print(f"Fetching market caps for {len(tickers)} tickers...")
    caps = {}
    for i, t in enumerate(tickers):
        try:
            mc = yf.Ticker(t).info.get("marketCap")
            if mc and mc > 0: caps[t] = mc
        except: pass
        time.sleep(0.1)
        if (i + 1) % 25 == 0: print(f"  ...{i+1}/{len(tickers)}")

    picks = [t for t, _ in sorted(caps.items(), key=lambda x: x[1], reverse=True)[:n]]
    cache_set_json(f"mcap_{n}_{h}", picks)
    return picks

def get_top_cryptos(n: int) -> list:
    cached = cache_get_json(f"crypto_{n}")
    if cached: return cached
    try:
        r = requests.get(
            "https://api.coingecko.com/api/v3/coins/markets",
            params={"vs_currency": "usd", "order": "market_cap_desc", "per_page": n, "page": 1},
            timeout=10,
        )
        picks = [f"{c['symbol'].upper()}-USD" for c in r.json()]
        cache_set_json(f"crypto_{n}", picks)
        return picks
    except:
        return ["BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD"][:n]

def build_aggressive_universe(num_stocks=20, num_crypto=5) -> list:
    sectors = get_sp500_sectors()
    target = ["Information Technology", "Consumer Discretionary", "Communication Services"]
    candidates = [t for t, s in sectors.items() if s in target]
    stocks = get_top_tickers_by_mcap(candidates, num_stocks)
    cryptos = get_top_cryptos(num_crypto)
    return stocks + cryptos

# ─── Prices & Returns ─────────────────────────────────────────────────────────

def get_prices(tickers: list, start: str, end: str) -> pd.DataFrame:
    frames = []
    for t in tickers:
        cached = cache_get_df(f"px_{t}")
        if cached is None:
            raw = yf.download(t, start=start, end=end, progress=False)
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            if not raw.empty:
                cache_set_df(f"px_{t}", raw)
                cached = raw
        if cached is not None and not cached.empty:
            col = "Adj Close" if "Adj Close" in cached.columns else "Close"
            frames.append(cached[col].rename(t))

    if not frames: return pd.DataFrame()
    prices = pd.concat(frames, axis=1).sort_index()
    prices = prices.loc[start:end].dropna(axis=1, how="all")
    return prices

# ─── Covariance (EWMA — matches aggressive risk profile) ─────────────────────

def cov_ewma(returns: pd.DataFrame, lam=0.94) -> pd.DataFrame:
    rets = returns.dropna()
    w = np.array([(1 - lam) * (lam ** i) for i in range(len(rets)-1, -1, -1)])
    w /= w.sum()
    dm = (rets - rets.mean()).values
    cov = sum(w[i] * np.outer(dm[i], dm[i]) for i in range(len(w)))
    return pd.DataFrame(cov, index=returns.columns, columns=returns.columns)

# ─── HRP ──────────────────────────────────────────────────────────────────────

def hrp(returns: pd.DataFrame, cov: pd.DataFrame) -> pd.Series:
    tickers = returns.columns.tolist()
    corr = returns.corr().values
    dist = np.sqrt(0.5 * (1 - corr))
    link = linkage(squareform(dist, checks=False), method="single")

    n = link.shape[0] + 1
    def _items(node):
        if node < n: return [int(node)]
        return _items(int(link[node-n, 0])) + _items(int(link[node-n, 1]))
    sorted_idx = _items(2*n - 2)

    def _cluster_var(items):
        sub = cov.values[np.ix_(items, items)]
        d = np.diag(sub).copy(); d[d == 0] = 1e-8
        ivp = (1/d) / (1/d).sum()
        return float(ivp @ sub @ ivp)

    w = pd.Series(1.0, index=sorted_idx)
    def _bisect(items):
        if len(items) == 1: return
        mid = len(items) // 2
        L, R = items[:mid], items[mid:]
        vL, vR = _cluster_var(L), _cluster_var(R)
        aL = 0.5 if (vL + vR) == 0 else 1 - vL / (vL + vR)
        w.loc[L] *= aL; w.loc[R] *= (1 - aL)
        _bisect(L); _bisect(R)
    _bisect(list(sorted_idx))

    weights = pd.Series(w.values, index=[tickers[i] for i in sorted_idx])
    weights /= weights.sum()
    return weights.reindex(tickers).fillna(0.0)

# ─── MVO (Max-Sharpe) ─────────────────────────────────────────────────────────

def mvo_max_sharpe(exp_ret: pd.Series, cov: pd.DataFrame, rf=0.02) -> pd.Series:
    n = len(exp_ret)
    def neg_sharpe(w):
        ret = w @ exp_ret.values
        vol = np.sqrt(w @ cov.values @ w)
        return -(ret - rf) / vol if vol > 0 else -np.inf
    res = minimize(neg_sharpe, [1/n]*n,
                   method="SLSQP",
                   bounds=[(0, 1)]*n,
                   constraints={"type": "eq", "fun": lambda w: w.sum() - 1})
    if not res.success:
        return pd.Series(1/n, index=exp_ret.index)
    return pd.Series(res.x, index=exp_ret.index)

# ─── Request / Response Models ────────────────────────────────────────────────

class AllocateRequest(BaseModel):
    budget: float = Field(..., gt=0, description="Total investment budget in USD")
    strategy: str = Field("hrp", description="Weighting strategy: 'hrp' (default) or 'mvo'")
    start: str = Field("2023-01-01", description="History start date YYYY-MM-DD")
    end: str = Field("2025-01-01", description="History end date YYYY-MM-DD")

    model_config = {
        "json_schema_extra": {
            "example": {
                "budget": 50000,
                "strategy": "hrp",
                "start": "2023-01-01",
                "end": "2025-01-01"
            }
        }
    }

class Holding(BaseModel):
    ticker: str
    weight_pct: float        # e.g. 8.34
    allocation_usd: float    # e.g. 4170.00
    latest_price: float
    units_to_buy: float

class AllocateResponse(BaseModel):
    budget: float
    strategy: str
    risk_profile: str
    total_holdings: int
    holdings: list[Holding]

# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health", tags=["Meta"])
def health():
    return {"status": "ok"}


@app.post("/allocate", response_model=AllocateResponse, tags=["Allocation"],
          summary="Build an aggressive portfolio allocation")
def allocate(req: AllocateRequest):
    """
    Provide a **budget** in USD and receive a fully-weighted aggressive portfolio.

    The universe is built from the largest-cap stocks in **Technology,
    Consumer Discretionary & Communication Services**, plus the top 5 cryptos
    by market cap.

    **Strategies**
    - `hrp` — Hierarchical Risk Parity (diversified, cluster-aware)
    - `mvo` — Mean-Variance Optimization (max Sharpe ratio)

    Results are cached for 24 hours to avoid redundant data fetches.
    """
    if req.strategy not in ("hrp", "mvo"):
        raise HTTPException(status_code=400, detail="strategy must be 'hrp' or 'mvo'")

    # 1. Build universe
    try:
        universe = build_aggressive_universe(num_stocks=20, num_crypto=5)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Universe build failed: {e}")

    # 2. Fetch prices
    prices = get_prices(universe, req.start, req.end)
    if prices.shape[1] < 2:
        raise HTTPException(status_code=422,
                            detail="Not enough price data. Try a wider date range.")

    # 3. Compute weights
    returns = prices.pct_change().dropna()
    cov = cov_ewma(returns)   # EWMA suits the aggressive profile

    if req.strategy == "hrp":
        weights = hrp(returns, cov)
    else:
        exp_ret = (1 + returns.mean()) ** 252 - 1
        ok = cov.index.intersection(exp_ret.index)
        weights = mvo_max_sharpe(exp_ret[ok], cov.loc[ok, ok])
        weights = weights.reindex(prices.columns).fillna(0.0)

    weights = weights / weights.sum()   # normalise to exactly 1

    # 4. Build response
    latest = prices.iloc[-1]
    holdings = []
    for ticker in weights.index:
        w = float(weights[ticker])
        if w < 1e-6: continue
        price = float(latest.get(ticker, 0))
        alloc = round(w * req.budget, 2)
        units = round(alloc / price, 6) if price > 0 else 0.0
        holdings.append(Holding(
            ticker=ticker,
            weight_pct=round(w * 100, 2),
            allocation_usd=alloc,
            latest_price=round(price, 4),
            units_to_buy=units,
        ))

    holdings.sort(key=lambda h: h.weight_pct, reverse=True)

    return AllocateResponse(
        budget=req.budget,
        strategy=req.strategy,
        risk_profile="aggressive",
        total_holdings=len(holdings),
        holdings=holdings,
    )
    
