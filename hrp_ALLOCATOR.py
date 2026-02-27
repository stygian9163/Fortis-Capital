import time
import os
import json
from datetime import datetime, timedelta
import math
import requests
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from sklearn.covariance import LedoitWolf
from arch import arch_model
import backtrader as bt
import matplotlib.pyplot as plt

# -----------------------
# 0) Lightweight caching (1-day TTL)
# -----------------------
CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
CACHE_TTL_SECONDS = 24 * 60 * 60  # 1 day

def _ensure_cache_dir():
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
    except Exception:
        pass

def _is_fresh(path: str) -> bool:
    try:
        mtime = os.path.getmtime(path)
        return (time.time() - mtime) < CACHE_TTL_SECONDS
    except Exception:
        return False

def _cache_path(name: str) -> str:
    safe = name.replace('/', '_')
    return os.path.join(CACHE_DIR, safe)

def cache_get_json(name: str):
    _ensure_cache_dir()
    path = _cache_path(name + ".json")
    if _is_fresh(path):
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except Exception:
            return None
    return None

def cache_set_json(name: str, obj):
    _ensure_cache_dir()
    path = _cache_path(name + ".json")
    try:
        with open(path, 'w') as f:
            json.dump(obj, f)
    except Exception:
        pass

def cache_get_df(name: str):
    _ensure_cache_dir()
    path = _cache_path(name + ".parquet")
    if _is_fresh(path):
        try:
            return pd.read_parquet(path)
        except Exception:
            # fall back to csv if parquet not available
            path_csv = _cache_path(name + ".csv")
            if _is_fresh(path_csv):
                try:
                    return pd.read_csv(path_csv, index_col=0, parse_dates=True)
                except Exception:
                    return None
    return None

def cache_set_df(name: str, df: pd.DataFrame):
    _ensure_cache_dir()
    # Try parquet first, fallback to csv (handles environments w/o pyarrow)
    path_parquet = _cache_path(name + ".parquet")
    try:
        df.to_parquet(path_parquet)
        return
    except Exception:
        pass
    path_csv = _cache_path(name + ".csv")
    try:
        df.to_csv(path_csv)
    except Exception:
        pass

# -----------------------
# 1) Universe selection
# -----------------------
def get_sp500_tickers():
    """
    Get S&P 500 tickers from a reliable source like a GitHub repository.
    """
    cache_key = "sp500_tickers"
    cached = cache_get_json(cache_key)
    if cached:
        return cached
    url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
    try:
        df = pd.read_csv(url)
        tickers = df["Symbol"].str.replace('.', '-', regex=False).tolist()
        cache_set_json(cache_key, tickers)
        return tickers
    except Exception as e:
        print(f"Could not fetch S&P 500 tickers from GitHub: {e}")
        # Fallback to a smaller, hardcoded list if fetching fails
        return ["AAPL", "MSFT", "GOOG", "AMZN", "META"]


def get_top_stocks_by_mcap(n=5, tickers=None, sleep_per_call=0.12):
    """
    Return top-n stocks by market cap from provided tickers (default: S&P500).
    Uses yfinance ticker.info['marketCap'] where available.
    Note: yfinance calls can be slow and rate-limited; this function throttles calls lightly.
    """
    cache_key = f"top_stocks_mcap_{n}"
    cached = cache_get_json(cache_key)
    if cached:
        return cached
    if tickers is None:
        tickers = get_sp500_tickers()
    caps = {}
    # Shuffle tickers to distribute the load and potential rate limits
    import random
    random.shuffle(tickers)

    for t in tickers:
        try:
            info = yf.Ticker(t).info
            mc = info.get("marketCap", None)
            if mc and mc > 0:
                caps[t] = mc
        except Exception:
            # skip tickers that fail; continue
            pass
        time.sleep(sleep_per_call)
    # sort and pick top n
    top = sorted(caps.items(), key=lambda x: x[1], reverse=True)[:n]
    picks = [t for t, _ in top]
    cache_set_json(cache_key, picks)
    return picks

def get_top_cryptos_by_mcap(n=10):
    """
    Get top n cryptos (ids/tickers usable by yfinance like 'BTC-USD') by market cap from CoinGecko.
    Returns symbols in the form 'BTC-USD', 'ETH-USD' where possible.
    """
    cache_key = f"top_crypto_{n}"
    cached = cache_get_json(cache_key)
    if cached:
        return cached
    # Query CoinGecko for coins markets
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {"vs_currency":"usd", "order":"market_cap_desc", "per_page": n, "page":1, "sparkline": False}
    try:
        r = requests.get(url, params=params, timeout=10)
        data = r.json()
        results = []
        for coin in data:
            symbol = coin.get("symbol", "").upper()
            # Use yfinance convention: e.g. BTC-USD
            results.append(f"{symbol}-USD")
        cache_set_json(cache_key, results)
        return results
    except Exception:
        # Fallback
        return ["BTC-USD", "ETH-USD"][:n]

def build_universe(num_stocks=30, num_crypto=2):
    """
    Returns a combined list of tickers automatically selected:
      - top num_stocks by market cap from S&P500
      - top num_crypto cryptos by market cap from CoinGecko
    """
    print("Fetching S&P500 tickers and market caps (may take ~10-60s depending on n)...")
    sp5 = get_sp500_tickers()
    top_stocks = get_top_stocks_by_mcap(num_stocks, tickers=sp5)
    print(f"Selected top {len(top_stocks)} stocks.")
    print("Fetching top cryptos from CoinGecko...")
    top_crypto = get_top_cryptos_by_mcap(num_crypto)
    print(f"Selected cryptos: {top_crypto}")
    universe = top_stocks + top_crypto
    return universe

# -----------------------
# 2) Data + returns
# -----------------------
def get_prices(tickers, start, end, interval="1d"):
    """Download prices with per-ticker caching and 1-day TTL."""
    # If list-like, try individual cache per ticker and then join
    if isinstance(tickers, (list, tuple, np.ndarray)):
        price_frames = []
        for t in tickers:
            cache_key = f"prices_{t}_{interval}"
            cached_df = cache_get_df(cache_key)
            if cached_df is None:
                df_t = yf.download(t, start=start, end=end, interval=interval, progress=False)
                if isinstance(df_t.columns, pd.MultiIndex):
                    # yfinance can return multiindex even for single ticker
                    df_t.columns = df_t.columns.get_level_values(0)
                if not df_t.empty:
                    cache_set_df(cache_key, df_t)
                    cached_df = df_t
            if cached_df is not None and not cached_df.empty:
                # We will use Adj Close if available else Close
                series = cached_df["Adj Close"] if "Adj Close" in cached_df.columns else cached_df["Close"]
                price_frames.append(series.rename(t))
        if not price_frames:
            return pd.DataFrame()
        prices = pd.concat(price_frames, axis=1).sort_index()
        # Trim by requested date range
        prices = prices.loc[(prices.index >= pd.to_datetime(start)) & (prices.index <= pd.to_datetime(end))]
        # Drop columns with entirely NaN
        prices = prices.dropna(axis=1, how="all")
        prices = prices.loc[:, ~prices.isna().all()]
        return prices
    # Fallback for string/single ticker or other inputs
    df = yf.download(tickers, start=start, end=end, interval=interval, progress=False)
    if "Adj Close" in df.columns:
        prices = df["Adj Close"]
    else:
        prices = df["Close"]
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(tickers[0])
    # Drop columns with entirely NaN
    prices = prices.dropna(axis=1, how="all")
    # Align columns: some tickers may be missing; keep intersecting tickers
    prices = prices.loc[:, ~prices.isna().all()]
    return prices

def compute_returns(prices):
    return prices.pct_change().dropna()

# -----------------------
# 3) Risk model implementations
# -----------------------
def cov_ledoit_wolf(returns):
    """Calculates covariance using Ledoit-Wolf shrinkage."""
    lw = LedoitWolf().fit(returns.dropna().values)
    return lw.covariance_

def cov_ewma(returns, lambda_decay=0.94):
    """
    Calculates exponentially-weighted covariance (EWMA).
    Gives more weight to recent returns.
    """
    rets = returns.dropna()
    w = np.array([(1 - lambda_decay) * (lambda_decay ** i) for i in range(len(rets)-1, -1, -1)])
    w = w / w.sum()
    demeaned = rets - rets.mean()
    cov = np.zeros((demeaned.shape[1], demeaned.shape[1]))
    for i in range(len(w)):
        row = demeaned.values[i].reshape(-1, 1)
        cov += w[i] * (row @ row.T)
    return cov

def build_covariance(returns, method="ledoit_wolf"):
    """
    Builds a covariance matrix for the given returns.
    method in {"ledoit_wolf", "ewma"}
    """
    method_str = str(method).lower()
    
    if method_str == "ewma":
        return cov_ewma(returns)
    else:  # Default to ledoit_wolf
        return cov_ledoit_wolf(returns)

# -----------------------
# 4) HRP core (uses provided covariance)
# -----------------------
def correl_distance_from_corr(corr):
    return np.sqrt(0.5 * (1 - corr))

def quasi_diag(link):
    n = link.shape[0] + 1
    def get_cluster_items(node):
        if node < n:
            return [int(node)]
        left = int(link[node - n, 0])
        right = int(link[node - n, 1])
        return get_cluster_items(left) + get_cluster_items(right)
    return get_cluster_items(2*n - 2)

def get_cluster_var(cov, items):
    sub_cov = cov[np.ix_(items, items)]
    # Guard: if diagonal has zeros, add small epsilon
    diag = np.diag(sub_cov).copy()
    diag[diag == 0] = 1e-8
    ivp = 1.0 / diag
    ivp = ivp / ivp.sum()
    w = ivp.reshape(-1, 1)
    result = w.T @ sub_cov @ w
    return float(result.item())

def recursive_bisection(cov, sorted_indices):
    w = pd.Series(1.0, index=sorted_indices, dtype=float)
    def _rec_bisect(items):
        if len(items) == 1:
            return
        split = len(items) // 2
        left = items[:split]
        right = items[split:]
        left_var = get_cluster_var(cov, left)
        right_var = get_cluster_var(cov, right)
        # Protect against zero division
        if left_var + right_var == 0:
            alloc_left = 0.5
        else:
            alloc_left = 1.0 - left_var / (left_var + right_var)
        w.loc[left] *= alloc_left
        w.loc[right] *= (1.0 - alloc_left)
        _rec_bisect(left)
        _rec_bisect(right)
    _rec_bisect(list(sorted_indices))
    return w

def hrp_with_cov(returns, cov_matrix):
    """
    HRP given returns DataFrame and a covariance matrix (np.array aligned to returns.columns).
    Returns pd.Series of weights indexed by tickers.
    """
    tickers = returns.columns.tolist()
    corr = returns.corr().values
    dist = correl_distance_from_corr(corr)
    # Condensed distance for linkage
    condensed = squareform(dist, checks=False)
    link = linkage(condensed, method="single")
    sorted_idx = quasi_diag(link)
    ordered_tickers = [tickers[i] for i in sorted_idx]
    ordered_indices = [tickers.index(t) for t in ordered_tickers]
    weights_ordered = recursive_bisection(cov_matrix, ordered_indices)
    weights = pd.Series(weights_ordered.values, index=ordered_tickers)
    weights = weights / weights.sum()
    # Reindex to original order
    weights = weights.reindex(tickers).fillna(0.0)
    return weights

# -----------------------
# 5) Budget allocator wrapping everything
# -----------------------
def allocate_budget_dynamic(budget,
                            start="2023-01-01",
                            end="2025-01-01",
                            num_stocks=30,
                            num_crypto=2,
                            risk_method="ledoit_wolf",
                            rebalance=False):
    """
    Full pipeline:
      - build universe dynamically
      - fetch prices, compute returns
      - build covariance using chosen method
      - run HRP and convert weights -> dollar allocations -> units to buy
    """
    universe = build_universe(num_stocks=num_stocks, num_crypto=num_crypto)
    print(f"Universe size: {len(universe)}")
    prices = get_prices(universe, start, end)
    if prices.shape[1] < 2:
        raise RuntimeError("Insufficient tickers with data; reduce num_stocks or check date range.")
    returns = compute_returns(prices)

    print(f"Building covariance using method: {risk_method}")
    cov = build_covariance(returns, method=risk_method)

    weights = hrp_with_cov(returns, cov)
    latest_prices = prices.iloc[-1].reindex(weights.index)
    allocations = (weights * budget).fillna(0.0)
    units = allocations / latest_prices.replace(0, np.nan)

    out = pd.DataFrame({
        "Weight": weights,
        "Latest Price": latest_prices,
        "Allocation ($)": allocations,
        "Units to Buy": units
    })
    # Clean formatting
    out = out.fillna(0.0).round(6)
    return out

# -----------------------
# 6) Backtrader Integration
# -----------------------
class HRPStrategy(bt.Strategy):
    """
    Backtrader strategy that implements HRP allocation with periodic rebalancing.
    """
    params = (
        ('rebalance_freq', 30),  # Rebalance every 30 days
        ('risk_method', 'ledoit_wolf'),
        ('lookback_period', 252),  # 1 year of data for covariance estimation
    )
    
    def __init__(self):
        self.tickers = [d._name for d in self.datas]
        self.rebalance_counter = 0
        self.weights = None
        self.last_rebalance = None
        
    def next(self):
        """Called for each new bar"""
        self.rebalance_counter += 1
        
        # Check if it's time to rebalance
        if self.rebalance_counter >= self.params.rebalance_freq:
            self.rebalance_portfolio()
            self.rebalance_counter = 0
    
    def rebalance_portfolio(self):
        """Rebalance portfolio using HRP allocation"""
        try:
            # Get historical data for the lookback period
            if len(self.data) < self.params.lookback_period:
                return
            
            # Collect price data for all assets
            price_data = {}
            for i, data in enumerate(self.datas):
                if len(data) >= self.params.lookback_period:
                    # Get historical prices
                    hist_prices = [data.close[-j] for j in range(self.params.lookback_period, 0, -1)]
                    price_data[self.tickers[i]] = hist_prices
            
            if len(price_data) < 2:
                return
            
            # Convert to DataFrame
            prices_df = pd.DataFrame(price_data)
            returns_df = prices_df.pct_change().dropna()
            
            if returns_df.empty or len(returns_df) < 50:
                return
            
            # Build covariance matrix
            cov_matrix = build_covariance(returns_df, method=self.params.risk_method)
            
            # Get HRP weights
            self.weights = hrp_with_cov(returns_df, cov_matrix)
            
            # Calculate target positions
            total_value = self.broker.getvalue()
            
            for i, ticker in enumerate(self.tickers):
                if ticker in self.weights.index:
                    target_weight = self.weights[ticker]
                    target_value = total_value * target_weight
                    current_price = self.datas[i].close[0]
                    
                    if current_price > 0:
                        target_shares = target_value / current_price
                        current_shares = self.getposition(self.datas[i]).size
                        
                        # Calculate order size
                        order_size = target_shares - current_shares
                        
                        if abs(order_size) > 0.01:  # Only trade if significant difference
                            if order_size > 0:
                                self.buy(data=self.datas[i], size=abs(order_size))
                            else:
                                self.sell(data=self.datas[i], size=abs(order_size))
            
            self.last_rebalance = self.data.datetime.date(0)
            
        except Exception as e:
            print(f"Error in rebalancing: {e}")

def create_data_feeds(tickers, start_date, end_date):
    """
    Create backtrader data feeds for given tickers and date range.
    """
    data_feeds = []
    
    for ticker in tickers:
        try:
            # Use cached OHLCV
            cache_key = f"ohlcv_{ticker}_1d"
            data = cache_get_df(cache_key)
            if data is None:
                data = yf.download(ticker, start=start_date, end=end_date, progress=False)
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)
                if not data.empty:
                    cache_set_df(cache_key, data)
            
            if data.empty:
                print(f"No data for {ticker}")
                continue
                
            # Create backtrader data feed
            bt_data = bt.feeds.PandasData(
                dataname=data,
                datetime=None,  # Use index
                open='Open',
                high='High',
                low='Low',
                close='Close',
                volume='Volume',
                openinterest=None
            )
            bt_data._name = ticker  # Set name for identification
            data_feeds.append(bt_data)
            
        except Exception as e:
            print(f"Error creating data feed for {ticker}: {e}")
            continue
    
    return data_feeds

def run_backtrader_simulation(tickers, start_date, end_date, initial_cash=100000, 
                             rebalance_freq=30, risk_method='ledoit_wolf'):
    """
    Run backtrader simulation with HRP strategy.
    """
    # Create Cerebro engine
    cerebro = bt.Cerebro()
    
    # Add strategy
    cerebro.addstrategy(HRPStrategy, 
                       rebalance_freq=rebalance_freq,
                       risk_method=risk_method)
    
    # Create and add data feeds
    data_feeds = create_data_feeds(tickers, start_date, end_date)
    
    if not data_feeds:
        print("No valid data feeds created!")
        return None
    
    for data in data_feeds:
        cerebro.adddata(data)
    
    # Set initial cash
    cerebro.broker.setcash(initial_cash)
    
    # Set commission (0.1% per trade)
    cerebro.broker.setcommission(commission=0.001)
    
    # Add analyzers
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    
    print(f"Starting Portfolio Value: ${cerebro.broker.getvalue():.2f}")
    
    # Run the simulation
    results = cerebro.run()
    
    # Get final portfolio value
    final_value = cerebro.broker.getvalue()
    print(f"Final Portfolio Value: ${final_value:.2f}")
    
    # Print performance metrics
    strat = results[0]
    sharpe = strat.analyzers.sharpe.get_analysis()
    returns = strat.analyzers.returns.get_analysis()
    drawdown = strat.analyzers.drawdown.get_analysis()
    trades = strat.analyzers.trades.get_analysis()
    
    print(f"\nPerformance Metrics:")
    print(f"Sharpe Ratio: {sharpe.get('sharperatio', 'N/A'):.4f}")
    print(f"Total Return: {returns.get('rtot', 0)*100:.2f}%")
    print(f"Max Drawdown: {drawdown.get('max', {}).get('drawdown', 0):.2f}%")
    print(f"Total Trades: {trades.get('total', {}).get('total', 0)}")
    
    return cerebro, results

def plot_backtrader_results(cerebro, save_plot=True, plot_filename='hrp_backtest.png'):
    """
    Plot backtrader results.
    """
    try:
        # Create the plot
        fig = cerebro.plot(style='candlestick', barup='green', bardown='red')
        
        if save_plot:
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            print(f"Plot saved as {plot_filename}")
        
        plt.show()
        
    except Exception as e:
        print(f"Error creating plot: {e}")

# -----------------------
# 7) Example usage
# -----------------------
if __name__ == "__main__":
    # Example 1: Static allocation (original functionality)
    print("=" * 60)
    print("HRP ALLOCATOR - STATIC ALLOCATION")
    print("=" * 60)
    budget = 10000
    print("Running dynamic allocation (this may take ~30-90s)...")
    df = allocate_budget_dynamic(budget,
                                 start="2024-01-01",
                                 end="2025-01-01",
                                 num_stocks=20,
                                 num_crypto=2,
                                 risk_method="ledoit_wolf")
    print(df)
    
    print("\n" + "=" * 60)
    print("HRP ALLOCATOR - BACKTRADER SIMULATION")
    print("=" * 60)
    
    # Example 2: Backtrader simulation
    print("Running backtrader simulation...")
    
    # Define a smaller universe for faster simulation
    test_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "BTC-USD", "ETH-USD"]
    
    try:
        # Run the simulation
        cerebro, results = run_backtrader_simulation(
            tickers=test_tickers,
            start_date="2023-01-01",
            end_date="2024-12-31",
            initial_cash=100000,
            rebalance_freq=30,  # Monthly rebalancing
            risk_method='ledoit_wolf'
        )
        
        if cerebro and results:
            # Plot results
            print("\nGenerating performance plots...")
            plot_backtrader_results(cerebro, save_plot=True, plot_filename='hrp_backtest_results.png')
            
    except Exception as e:
        print(f"Error running backtrader simulation: {e}")
        print("Make sure backtrader is installed: pip install backtrader")