import pandas as pd
import numpy as np
import yfinance as yf

# ==========================================
# 1. DATA LAYER
# ==========================================
class MarketDataLoader:
    def __init__(self, tickers: list, start_date: str, end_date: str):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.raw_data = None
        self.clean_data = None
        self.daily_returns = None
        self.log_returns = None

    def fetch_data(self) -> None:
        try:
            data = yf.download(self.tickers, start=self.start_date, end=self.end_date)['Close']
        except Exception as e:
            raise ConnectionError(f"Connection error with Yahoo Finance API: {e}")
            
        if data.empty:
            raise ValueError("Download error. No historical data returned for the specified tickers.")
            
        if isinstance(data, pd.Series) or len(self.tickers) == 1:
            data = data.to_frame(name=self.tickers[0])
            
        missing_tickers = [t for t in self.tickers if t not in data.columns]
        if missing_tickers:
            raise ValueError(f"Incomplete series data for tickers: {', '.join(missing_tickers)}.")
            
        self.raw_data = data

    def sanitize_data(self) -> None:
        if self.raw_data is None:
            self.fetch_data()
            
        data = self.raw_data.copy()
        data.index = pd.to_datetime(data.index).tz_localize(None)
        
        valid_counts = data.count()
        data_clean = data.ffill().dropna()
        data_clean = data_clean.replace(0.0, np.nan).dropna()
        
        if data_clean.empty or len(data_clean) < 10:
            shortest_asset = valid_counts.idxmin()
            shortest_days = valid_counts.min()
            raise ValueError(
                f"Matrix overlap failure. The asset '{shortest_asset}' only has {shortest_days} valid trading days in the selected time horizon. "
                f"All assets must have an overlapping history of at least 10 days to compute cross-sectional covariance matrices. "
                f"Adjust your time horizon to match the IPO date of '{shortest_asset}' or remove it from the allocation vector."
            )

        self.clean_data = data_clean
        
        dr = data_clean.pct_change()
        self.daily_returns = dr.replace([np.inf, -np.inf], np.nan).dropna()
        
        lr = np.log(data_clean / data_clean.shift(1))
        self.log_returns = lr.replace([np.inf, -np.inf], np.nan).dropna()


# ==========================================
# 2. RISK LAYER
# ==========================================
class RiskModeler:
    def __init__(self, data_loader: MarketDataLoader, weights: np.ndarray, risk_free_rate: float = 0.02):
        self.data = data_loader.clean_data
        self.daily_returns = data_loader.daily_returns
        self.log_returns = data_loader.log_returns
        self.weights = weights
        self.rf_rate = risk_free_rate
        self.trading_days = 252
        
        self.port_daily_returns = None
        self.cumulative_returns = None
        self.metrics = {}
        self.matrices = {}
        
    def _simulate_monthly_rebalancing(self) -> None:
        portfolio_daily_returns_list = []
        active_weights = self.weights.copy()
        dates = self.daily_returns.index
        asset_ret_matrix = self.daily_returns.values
        
        for i in range(len(dates)):
            day_ret_vector = asset_ret_matrix[i]
            day_port_return = np.sum(active_weights * day_ret_vector)
            portfolio_daily_returns_list.append(day_port_return)
            
            active_weights = active_weights * (1 + day_ret_vector)
            sum_active = np.sum(active_weights)
            if sum_active > 0:
                active_weights /= sum_active
                
            if i < len(dates) - 1:
                if dates[i].month != dates[i+1].month:
                    active_weights = self.weights.copy()
                    
        self.port_daily_returns = pd.Series(portfolio_daily_returns_list, index=dates)
        self.cumulative_returns = (1 + self.port_daily_returns).cumprod()

    def compute_core_metrics(self) -> None:
        if self.port_daily_returns is None:
            self._simulate_monthly_rebalancing()
            
        cagr = (1 + self.port_daily_returns.mean()) ** self.trading_days - 1
        volatility = self.port_daily_returns.std() * np.sqrt(self.trading_days)
        sharpe = (cagr - self.rf_rate) / volatility if volatility > 0 else 0
        
        neg_ret = self.port_daily_returns[self.port_daily_returns < 0]
        downside_dev = neg_ret.std() * np.sqrt(self.trading_days)
        sortino = (cagr - self.rf_rate) / downside_dev if downside_dev > 0 else 0
        
        peak = self.cumulative_returns.cummax()
        drawdown = (self.cumulative_returns - peak) / peak
        max_dd = drawdown.min() if not drawdown.empty else 0.0
        
        var_95 = np.percentile(self.port_daily_returns, 5)
        cvar_95 = self.port_daily_returns[self.port_daily_returns <= var_95].mean()
        cvar_95 = cvar_95 if not np.isnan(cvar_95) else 0.0
        
        hhi = float(np.sum(self.weights ** 2))
        
        self.metrics.update({
            "CAGR": cagr, "Volatility": volatility, "Sharpe": sharpe,
            "Sortino": sortino, "Max DD": max_dd,
            "VaR 95": var_95, "CVaR 95": cvar_95, "HHI": hhi
        })
        self.matrices["drawdown"] = drawdown

    def compute_matrices_and_pca(self) -> None:
        corr_matrix = self.log_returns.corr(method="pearson").fillna(0)
        self.matrices["corr"] = corr_matrix
        
        if len(corr_matrix) > 1:
            max_corr = corr_matrix.where(~np.eye(corr_matrix.shape[0], dtype=bool)).max().max()
            self.metrics["Max Corr"] = max_corr if not np.isnan(max_corr) else 1.0
        else:
            self.metrics["Max Corr"] = 1.0
            
        eigenvalues, _ = np.linalg.eig(corr_matrix)
        eigenvalues = np.sort(eigenvalues)[::-1]
        sum_eig = np.sum(eigenvalues)
        sum_sq_eig = np.sum(eigenvalues**2)
        self.metrics["N_eff"] = (sum_eig**2) / sum_sq_eig if sum_sq_eig > 0 else 1.0

    def compute_stress_tests(self) -> dict:
        scenarios = {
            "COVID-19 Liquidity Contraction (Feb-Mar 2020)": ('2020-02-19', '2020-03-23'),
            "Inflationary Regime Shift (Jan-Oct 2022)": ('2022-01-03', '2022-10-12'),
            "Post-Pause Yield Compression (Oct-Dec 2023)": ('2023-10-27', '2023-12-29')
        }
        results = {}
        for name, (start, end) in scenarios.items():
            try:
                sub_period = self.port_daily_returns.loc[start:end]
                if not sub_period.empty:
                    geometric_return = np.prod(1 + sub_period) - 1
                    results[name] = float(geometric_return)
                else:
                    results[name] = None
            except Exception:
                results[name] = None
        return results

    def generate_algorithmic_audit(self) -> list:
        m = self.metrics
        nom_count = len(self.data.columns)
        audit_ledger = []
        
        audit_ledger.append({
            "Audit Parameter": "Spectral Factor Orthogonality (PCA)",
            "Institutional Threshold": f"N_eff >= {nom_count * 0.60:.2f}",
            "Empirical Observation": f"{m['N_eff']:.2f}",
            "State": "FAIL" if m['N_eff'] < nom_count * 0.6 else "PASS",
            "Algorithmic Directive": "Divest highly collinear nodes to expand independent principal components." if m['N_eff'] < nom_count * 0.6 else "Orthogonal factor distribution validated."
        })
        
        audit_ledger.append({
            "Audit Parameter": "Mean-Variance Efficiency",
            "Institutional Threshold": "Sharpe >= 0.50",
            "Empirical Observation": f"{m['Sharpe']:.2f}",
            "State": "FAIL" if m['Sharpe'] < 0.5 else "PASS",
            "Algorithmic Directive": "Execute variance trimming to neutralize excessive volatility drag." if m['Sharpe'] < 0.5 else "Risk premium extraction is mathematically robust."
        })
        
        audit_ledger.append({
            "Audit Parameter": "Tail Loss Exposure",
            "Institutional Threshold": "Max DD >= -0.25",
            "Empirical Observation": f"{m['Max DD']:.2f}",
            "State": "FAIL" if m['Max DD'] < -0.25 else "PASS",
            "Algorithmic Directive": "Integrate convexity buffers (duration/volatility) to truncate left-tail distribution." if m['Max DD'] < -0.25 else "Maximum drawdown boundaries preserved."
        })

        audit_ledger.append({
            "Audit Parameter": "Pairwise Linear Independence",
            "Institutional Threshold": "Max Corr <= 0.85",
            "Empirical Observation": f"{m['Max Corr']:.2f}",
            "State": "FAIL" if m['Max Corr'] > 0.85 else "PASS",
            "Algorithmic Directive": "Liquidate redundant asset structures exhibiting severe collinearity." if m['Max Corr'] > 0.85 else "Cross-sectional independence verified."
        })
        
        audit_ledger.append({
            "Audit Parameter": "Capital Allocation Density (HHI)",
            "Institutional Threshold": "HHI <= 0.15",
            "Empirical Observation": f"{m['HHI']:.3f}",
            "State": "FAIL" if m['HHI'] > 0.15 else "PASS",
            "Algorithmic Directive": "Enforce strict position limits to dilute idiosyncratic risk concentration." if m['HHI'] > 0.15 else "Weight distribution density is mathematically balanced."
        })
        
        return audit_ledger


# ==========================================
# 3. OPTIMIZATION LAYER
# ==========================================
class OptimizationEngine:
    def __init__(self, data_loader: MarketDataLoader, risk_free_rate: float = 0.02):
        self.daily_returns = data_loader.daily_returns
        self.rf_rate = risk_free_rate
        self.trading_days = 252

    def compute_efficient_frontier(self, num_assets: int, num_portfolios: int = 2000) -> np.ndarray:
        if num_assets <= 1:
            return None
            
        results = np.zeros((3, num_portfolios))
        mean_returns = self.daily_returns.mean() * self.trading_days
        cov_matrix = self.daily_returns.cov().fillna(0) * self.trading_days
        
        for i in range(num_portfolios):
            rand_weights = np.random.random(num_assets)
            rand_weights /= np.sum(rand_weights)
            p_return = np.sum(mean_returns * rand_weights)
            p_std = np.sqrt(np.dot(rand_weights.T, np.dot(cov_matrix, rand_weights)))
            results[0,i] = p_return
            results[1,i] = p_std
            results[2,i] = (p_return - self.rf_rate) / p_std if p_std > 0 else 0
            
        return results


# ==========================================
# 4. ORCHESTRATOR 
# ==========================================
def run_portfolio_analysis(tickers: list, weights: np.ndarray, start_date: str, end_date: str, rf_rate: float) -> dict:
    loader = MarketDataLoader(tickers, start_date, end_date)
    loader.sanitize_data()
    
    risk_model = RiskModeler(loader, weights, risk_free_rate=rf_rate)
    risk_model.compute_core_metrics()
    risk_model.compute_matrices_and_pca()
    
    optimizer = OptimizationEngine(loader, risk_free_rate=rf_rate)
    ef_results = optimizer.compute_efficient_frontier(len(weights))
    
    return {
        "returns": risk_model.port_daily_returns,
        "cum_returns": risk_model.cumulative_returns,
        "drawdown": risk_model.matrices["drawdown"],
        "metrics": risk_model.metrics,
        "matrices": {"corr": risk_model.matrices["corr"]},
        "efficient_frontier": ef_results,
        "stress_tests": risk_model.compute_stress_tests(),
        "audit_ledger": risk_model.generate_algorithmic_audit()
    }
