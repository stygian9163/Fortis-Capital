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
from typing import Literal

warnings.filterwarnings("ignore")

app = FastAPI(
    title="Portfolio Allocator",
    description="Enter a budget → get a portfolio allocation + performance metrics.",
    version="1.0.0",
)

# ─── Filters / Guards ─────────────────────────────────────────────────────────

# Stablecoins (and near-stable pegs) tend to dominate minimum-variance style allocators
# like HRP. Since this API is explicitly "aggressive", we exclude them from the crypto
# universe and also drop any near-zero-vol series before optimization.
STABLECOIN_TICKERS = {
    "USDT-USD", "USDC-USD", "DAI-USD", "TUSD-USD", "BUSD-USD", "USDP-USD",
    "FDUSD-USD", "PYUSD-USD", "USDE-USD", "FRAX-USD", "LUSD-USD",
}

# Daily return std-dev below this is effectively “cash-like” for our purposes.
MIN_DAILY_VOL = 5e-4  # 0.05% daily vol

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
    if cached:
        # Older cached results may include stablecoins; filter defensively.
        cached = [t for t in cached if t not in STABLECOIN_TICKERS]
        return cached[:n]
    try:
        r = requests.get(
            "https://api.coingecko.com/api/v3/coins/markets",
            # Pull extra to allow filtering out stablecoins.
            params={"vs_currency": "usd", "order": "market_cap_desc", "per_page": max(n * 3, 25), "page": 1},
            timeout=10,
        )
        picks = [f"{c['symbol'].upper()}-USD" for c in r.json()]
        picks = [t for t in picks if t not in STABLECOIN_TICKERS]
        picks = picks[:n]
        cache_set_json(f"crypto_{n}", picks)
        return picks
    except:
        # Fallback list intentionally excludes stablecoins
        return ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", "ADA-USD", "DOGE-USD"][:n]

def build_universe(sectors: list[str], num_stocks: int, num_crypto: int) -> list:
    """
    Build the trading universe for a given risk profile.

    - Filters S&P 500 by the requested GICS sectors.
    - Selects top `num_stocks` by market cap from that filtered set.
    - Adds top `num_crypto` cryptos by market cap (excluding stablecoins).
    """
    sector_filters = set(sectors)
    sp500_sectors = get_sp500_sectors()  # {symbol -> sector}

    # Keep only tickers whose sector is in the requested list.
    candidates = [ticker for ticker, sector in sp500_sectors.items() if sector in sector_filters]

    stocks = get_top_tickers_by_mcap(candidates, num_stocks) if candidates else []
    cryptos = get_top_cryptos(num_crypto) if num_crypto > 0 else []
    return stocks + cryptos

RISK_PROFILES: dict[str, dict] = {
    "conservative": {
        "sectors": ["Consumer Staples", "Health Care", "Utilities", "Real Estate"],
        "num_stocks": 25,
        "num_crypto": 0,
        "cov": "ledoit_wolf",
    },
    "moderate": {
        "sectors": ["Information Technology", "Health Care", "Industrials", "Financials", "Consumer Staples"],
        "num_stocks": 25,
        "num_crypto": 2,
        "cov": "ledoit_wolf",
    },
    "aggressive": {
        "sectors": ["Information Technology", "Consumer Discretionary", "Communication Services"],
        "num_stocks": 20,
        "num_crypto": 5,
        "cov": "ewma",
    },
}

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

def cov_ledoit_wolf(returns: pd.DataFrame) -> pd.DataFrame:
    lw = LedoitWolf().fit(returns.dropna().values)
    return pd.DataFrame(lw.covariance_, index=returns.columns, columns=returns.columns)

def compute_performance_metrics(
    portfolio_returns: pd.Series,
    rf_annual: float = 0.02,
    periods_per_year: int = 252,
) -> dict:
    r = portfolio_returns.dropna()
    if r.empty:
        return {
            "rf_annual": float(rf_annual),
            "periods_per_year": int(periods_per_year),
            "n_periods": 0,
        }

    equity = (1 + r).cumprod()
    running_max = equity.cummax()
    dd = equity / running_max - 1.0
    max_dd = float(dd.min())

    total_return = float(equity.iloc[-1] - 1.0)
    ann_return = float((1.0 + total_return) ** (periods_per_year / len(r)) - 1.0) if len(r) > 0 else 0.0
    ann_vol = float(r.std(ddof=1) * np.sqrt(periods_per_year)) if len(r) > 1 else 0.0

    sharpe = float((ann_return - rf_annual) / ann_vol) if ann_vol > 0 else None

    downside = r[r < 0]
    downside_vol = float(downside.std(ddof=1) * np.sqrt(periods_per_year)) if len(downside) > 1 else 0.0
    sortino = float((ann_return - rf_annual) / downside_vol) if downside_vol > 0 else None

    calmar = float(ann_return / abs(max_dd)) if max_dd < 0 else None

    return {
        "rf_annual": float(rf_annual),
        "periods_per_year": int(periods_per_year),
        "n_periods": int(len(r)),
        "total_return": total_return,
        "annualized_return": ann_return,
        "annualized_volatility": ann_vol,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": max_dd,
        "calmar_ratio": calmar,
        "best_period": float(r.max()),
        "worst_period": float(r.min()),
        "positive_periods_pct": float((r.gt(0).sum() / len(r)) * 100.0),
    }

# ─── HRP ──────────────────────────────────────────────────────────────────────

def hrp(returns: pd.DataFrame, cov: pd.DataFrame) -> pd.Series:
    """
    Hierarchical Risk Parity (HRP) portfolio weights.

    - Uses correlation-based distance + hierarchical clustering.
    - Applies recursive bisection with inverse-variance weights inside clusters.
    - Includes numerical guards for near-singular covariance matrices.
    """
    tickers = list(returns.columns)

    # Basic guards / cleaning
    returns = (
        returns[tickers]
        .replace([np.inf, -np.inf], np.nan)
        .dropna(how="all")
    )
    if len(tickers) == 0:
        return pd.Series(dtype=float)
    if len(tickers) == 1:
        # Trivial 100% allocation if only one asset
        return pd.Series({tickers[0]: 1.0})

    # --- Correlation & distance matrix ----------------------------------------
    corr_df = returns.corr().fillna(0.0)
    np.fill_diagonal(corr_df.values, 1.0)
    corr = corr_df.values
    dist = np.sqrt(0.5 * (1.0 - corr))

    # Condensed distance form for linkage
    condensed = squareform(dist, checks=False)
    link = linkage(condensed, method="single")

    # --- Quasi-diagonalization: get leaf order -------------------------------
    n = link.shape[0] + 1

    def _get_cluster_items(node: int):
        if node < n:
            return [int(node)]
        left = int(link[node - n, 0])
        right = int(link[node - n, 1])
        return _get_cluster_items(left) + _get_cluster_items(right)

    sorted_idx = _get_cluster_items(2 * n - 2)

    # Ensure covariance is aligned to tickers and well-behaved
    cov = cov.reindex(index=tickers, columns=tickers)
    cov_values = cov.fillna(0.0).values

    def _cluster_var(items):
        sub = cov_values[np.ix_(items, items)]
        diag = np.diag(sub).copy()
        # Guard against zeros / negative variance from numerical issues
        diag[diag <= 0] = 1e-8
        ivp = 1.0 / diag
        ivp /= ivp.sum()
        w = ivp.reshape(-1, 1)
        var = float(w.T @ sub @ w)
        if not np.isfinite(var):
            return 0.0
        return var

    # --- Recursive bisection --------------------------------------------------
    w = pd.Series(1.0, index=sorted_idx, dtype=float)

    def _bisect(items):
        if len(items) == 1:
            return
        split = len(items) // 2
        L, R = items[:split], items[split:]
        vL, vR = _cluster_var(L), _cluster_var(R)
        tot = vL + vR
        if tot <= 0 or not np.isfinite(tot):
            aL = 0.5
        else:
            aL = 1.0 - vL / tot
        w.loc[L] *= aL
        w.loc[R] *= (1.0 - aL)
        _bisect(L)
        _bisect(R)

    _bisect(list(sorted_idx))

    weights = pd.Series(w.values, index=[tickers[i] for i in sorted_idx])
    weights /= weights.sum() if weights.sum() != 0 else 1.0
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
    risk_profile: Literal["conservative", "moderate", "aggressive"] = Field(
        "aggressive",
        description="Portfolio style. Affects universe construction + covariance model.",
    )
    start: str = Field("2023-01-01", description="History start date YYYY-MM-DD")
    end: str = Field("2025-01-01", description="History end date YYYY-MM-DD")
    rf_annual: float = Field(0.02, ge=0.0, le=0.25, description="Annual risk-free rate used for Sharpe/Sortino")

    model_config = {
        "json_schema_extra": {
            "example": {
                "budget": 50000,
                "strategy": "hrp",
                "risk_profile": "aggressive",
                "start": "2023-01-01",
                "end": "2025-01-01",
                "rf_annual": 0.02,
            }
        }
    }

class Holding(BaseModel):
    ticker: str
    weight_pct: float        # e.g. 8.34
    allocation_usd: float    # e.g. 4170.00
    latest_price: float
    units_to_buy: float

class PerformanceMetrics(BaseModel):
    rf_annual: float
    periods_per_year: int
    n_periods: int
    total_return: float | None = None
    annualized_return: float | None = None
    annualized_volatility: float | None = None
    sharpe_ratio: float | None = None
    sortino_ratio: float | None = None
    max_drawdown: float | None = None
    calmar_ratio: float | None = None
    best_period: float | None = None
    worst_period: float | None = None
    positive_periods_pct: float | None = None

class AllocateResponse(BaseModel):
    budget: float
    strategy: str
    risk_profile: str
    total_holdings: int
    holdings: list[Holding]
    performance: PerformanceMetrics

# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health", tags=["Meta"])
def health():
    return {"status": "ok"}


@app.post("/allocate", response_model=AllocateResponse, tags=["Allocation"],
          summary="Build a portfolio allocation")
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
        cfg = RISK_PROFILES.get(req.risk_profile)
        if not cfg:
            raise ValueError("Invalid risk_profile")
        universe = build_universe(
            sectors=cfg["sectors"],
            num_stocks=int(cfg["num_stocks"]),
            num_crypto=int(cfg["num_crypto"]),
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Universe build failed: {e}")

    # 2. Fetch prices
    prices = get_prices(universe, req.start, req.end)
    if prices.shape[1] < 2:
        raise HTTPException(status_code=422,
                            detail="Not enough price data. Try a wider date range.")

    # 3. Compute weights
    returns = prices.pct_change().dropna()

    # Drop “cash-like” / degenerate series that can dominate HRP (e.g. stablecoins).
    vol = returns.std(numeric_only=True)
    keep = vol[vol >= MIN_DAILY_VOL].index
    returns = returns[keep]
    prices = prices[keep]
    if returns.shape[1] < 2:
        raise HTTPException(
            status_code=422,
            detail="Not enough non-stable/non-degenerate assets with price data. Try a wider date range.",
        )













    cov_model = cfg.get("cov", "ewma")
    if cov_model == "ledoit_wolf":
        cov = cov_ledoit_wolf(returns)
    else:
        cov = cov_ewma(returns)

    if req.strategy == "hrp":
        weights = hrp(returns, cov)
    else:
        exp_ret = (1 + returns.mean()) ** 252 - 1
        ok = cov.index.intersection(exp_ret.index)
        weights = mvo_max_sharpe(exp_ret[ok], cov.loc[ok, ok])
        weights = weights.reindex(prices.columns).fillna(0.0)

    weights = weights / weights.sum()   # normalise to exactly 1

    # 3b. Performance metrics from in-sample portfolio returns
    port_rets = (returns * weights.reindex(returns.columns).fillna(0.0)).sum(axis=1)
    perf = compute_performance_metrics(port_rets, rf_annual=req.rf_annual)

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
        risk_profile=req.risk_profile,
        total_holdings=len(holdings),
        holdings=holdings,
        performance=PerformanceMetrics(**perf),
    )
    
