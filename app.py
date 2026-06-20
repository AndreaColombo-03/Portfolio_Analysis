import streamlit as st
import pandas as pd
import numpy as np
import datetime
import plotly.express as px
import plotly.graph_objects as go

from core.quant_engine import run_portfolio_analysis

st.set_page_config(page_title="Institutional Portfolio Sandbox", layout="wide")

# --- UI/UX: SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.header("Global Parameters")
    st.markdown("Define the mathematical estimation window and macroeconomic baseline constraints.")
    
    time_option = st.radio(
        "Historical Horizon",
        ["1 Year", "3 Years", "5 Years", "10 Years", "Custom Date Range"],
        index=2
    )

    end_date_dt = datetime.date.today()
    if time_option == "1 Year":
        start_date_dt = end_date_dt - datetime.timedelta(days=365)
    elif time_option == "3 Years":
        start_date_dt = end_date_dt - datetime.timedelta(days=365 * 3)
    elif time_option == "5 Years":
        start_date_dt = end_date_dt - datetime.timedelta(days=365 * 5)
    elif time_option == "10 Years":
        start_date_dt = end_date_dt - datetime.timedelta(days=365 * 10)
    else:
        start_date_dt = st.date_input("Start Date", datetime.date(2019, 1, 1))
        end_date_dt = st.date_input("End Date", datetime.date(2024, 1, 1))

    start_date_str = start_date_dt.strftime("%Y-%m-%d")
    end_date_str = end_date_dt.strftime("%Y-%m-%d")
    
    st.markdown("---")
    st.subheader("Financial Constraints")
    initial_capital = st.number_input("Initial Capital Base (USD)", min_value=1000, max_value=100000000, value=100000, step=10000)
    rf_rate_input = st.slider("Risk-Free Benchmark (%)", min_value=0.0, max_value=10.0, value=2.0, step=0.1)
    rf_rate = rf_rate_input / 100.0
    
    st.markdown("---")
    st.markdown("**System Status:** Engine Ready.")

# --- UI/UX: MAIN DASHBOARD ---
st.title("Institutional Portfolio Sandbox")
st.markdown("Construct a multi-asset target vector and execute cross-sectional econometric risk factor diagnostics.")

if "portfolio" not in st.session_state:
    st.session_state.portfolio = pd.DataFrame([{"Ticker": "", "Weight (%)": 0.0}])

st.subheader("Asset Allocation Target Vector")

edited_df = st.data_editor(
    st.session_state.portfolio,
    num_rows="dynamic",
    column_config={
        "Ticker": st.column_config.TextColumn("Ticker (Yahoo Finance)", required=True),
        "Weight (%)": st.column_config.NumberColumn("Target Weight (%)", min_value=0.0, max_value=100.0, format="%.2f")
    },
    use_container_width=True,
    hide_index=True
)

total_weight = round(edited_df["Weight (%)"].sum(), 2)
st.progress(min(total_weight / 100.0, 1.0), text=f"Total Allocated Weight: {total_weight}% / 100.00%")

if total_weight != 100.0:
    st.warning("The portfolio asset weights must sum exactly to 100.0% to execute analytics.")

is_ready = (total_weight == 100.0)

if st.button("Execute Quantitative Analysis", disabled=not is_ready, type="primary"):
    portfolio_clean = edited_df.copy()
    portfolio_clean["Ticker"] = portfolio_clean["Ticker"].astype(str).str.strip().str.upper()
    portfolio_clean = portfolio_clean[(portfolio_clean["Ticker"] != "") & (portfolio_clean["Weight (%)"] > 0)]
    
    tickers = portfolio_clean["Ticker"].tolist()
    weights = portfolio_clean["Weight (%)"].values / 100.0
    
    if len(tickers) < 2:
        st.error("Execution halted. Minimum requirement of 2 valid tickers to compute cross-sectional covariance matrices.")
        st.stop()

    with st.spinner("Executing Econometric Modeling and Factor Extractions..."):
        try:
            results = run_portfolio_analysis(tickers, weights, start_date_str, end_date_str, rf_rate)
            m = results["metrics"]
            
            st.markdown("---")
            
            tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
                "Summary", 
                "Risk Matrix", 
                "Asymmetry Diagnostics", 
                "Efficient Frontier",
                "Algorithmic Auditing",
                "Regime Stress Testing",
                "Strategic Action Plan"
            ])
            
            # --- TAB 1: SUMMARY ---
            with tab1:
                st.subheader("Tier 1 - Core Performance Metrics")
                col1, col2, col3, col4 = st.columns(4)
                
                expected_return_usd = initial_capital * m['CAGR']
                absolute_dd_usd = initial_capital * m['Max DD']
                
                col1.metric("Compound Annual Growth Rate", f"{m['CAGR']*100:.2f}%", delta=f"${expected_return_usd:,.0f} Annually")
                col2.metric("Annualized Standard Deviation", f"{m['Volatility']*100:.2f}%", delta_color="inverse")
                col3.metric(f"Sharpe Ratio (Rf={rf_rate_input}%)", f"{m['Sharpe']:.2f}")
                col4.metric("Maximum Drawdown", f"{m['Max DD']*100:.2f}%", delta=f"${absolute_dd_usd:,.0f} Capital Risk", delta_color="inverse")
                
                fig_eq = px.line(results["cum_returns"], title="Simulated Portfolio Equity Line (Monthly Rebalanced)")
                fig_eq.update_layout(showlegend=False, yaxis_title="Capital Multiplier Factor", xaxis_title="Estimation Timeline")
                fig_eq.update_xaxes(rangeslider_visible=True, rangeselector=dict(buttons=list([
                    dict(count=1, label="1Y", step="year", stepmode="backward"),
                    dict(count=3, label="3Y", step="year", stepmode="backward"),
                    dict(count=5, label="5Y", step="year", stepmode="backward"),
                    dict(count=10, label="10Y", step="year", stepmode="backward"),
                    dict(step="all", label="MAX")
                ])))
                st.plotly_chart(fig_eq, use_container_width=True)

                st.markdown("#### Empirical Diagnostics")
                st.markdown(f"The empirical evaluation of the portfolio's geometric compounding trajectory yields a Sharpe ratio of **{m['Sharpe']:.2f}**. Assuming weak-sense stationarity and constant variance within the estimation window, this metric quantifies the precise risk premium extracted per unit of assumed volatility. A coefficient significantly below the $0.50$ institutional boundary indicates that the absolute yield is structurally degraded by uncompensated variance, necessitating immediate vector reconfiguration to suppress volatility drag. Conversely, an equilibrium state $\ge 1.00$ mathematically validates the orthogonalization of systemic risk factors and optimal capital allocation.")

                with st.expander("Read Institutional Theory & Utility Guide"):
                    st.markdown("""
                    * **THE VISUAL:** The time-series trajectory maps the continuous cumulative compounding of a base unit of capital over the specified historical estimation window. The interface features an interactive Range Selector to isolate specific chronological regimes.
                    * **THE THEORY:** The Sharpe Ratio evaluates the expected excess geometric return scaled per unit of standard deviation risk. A ratio below 0.50 signifies that the absolute return generated fails to mathematically compensate for the variance penalty exacted on the capital base.
                    * **THE UTILITY:** This serves as the absolute baseline efficiency filter. If the system flags a Sharpe Ratio below 0.50 or a Maximum Drawdown breaching institutional limits, the allocation vector is architecturally compromised and requires reconfiguration.
                    """)

            # --- TAB 2: RISK MATRIX ---
            with tab2:
                st.subheader("Tier 2 - Dimensionality and Linear Dependence")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Nominal Assets", len(tickers))
                c2.metric("Effective Asset Count (N_eff)", f"{m['N_eff']:.2f}")
                
                max_corr_val = results['matrices']['corr'].where(~np.eye(results['matrices']['corr'].shape[0], dtype=bool)).max().max() if len(tickers) > 1 else 1.0
                c3.metric("Maximum Pairwise Correlation", f"{max_corr_val:.2f}", delta_color="inverse")
                
                daily_var_usd = initial_capital * m['VaR 95']
                c4.metric("Daily VaR (95%)", f"{m['VaR 95']*100:.2f}%", delta=f"${daily_var_usd:,.0f}", delta_color="inverse")
                
                fig_corr = px.imshow(results["matrices"]["corr"], text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r", title="Symmetric Pairwise Pearson Correlation Matrix")
                st.plotly_chart(fig_corr, use_container_width=True)

                st.markdown("#### Empirical Diagnostics")
                st.markdown(f"Spectral decomposition of the empirical covariance matrix extracts the eigenvalues ($\lambda_i$) necessary to compute the intrinsic dimensionality of the allocation, resulting in an Effective Asset Count (N_eff) of **{m['N_eff']:.2f}** against a nominal count of **{len(tickers)}**. An extreme deviation between nominal parameters and N_eff proves severe systemic concentration, indicating that the first principal eigenvector explains a disproportionate percentage of total variance. The maximum pairwise Pearson coefficient observed is **{max_corr_val:.2f}**, which establishes the upper boundary of linear dependence. Coefficients exceeding $0.85$ conclusively identify redundant factor exposures and overlapping beta clustering.")

                with st.expander("Read Institutional Theory & Utility Guide"):
                    st.markdown("""
                    * **THE VISUAL:** The symmetric heatmap array visualizes the cross-sectional Pearson correlation coefficients. Deep blue matrices signal perfect positive co-movement, while deep red identifies structural inverse correlation.
                    * **THE THEORY:** The Principal Component Analysis (PCA) spectral decomposition generates an 'Effective Asset Count' (N_eff). If assets share high collinearity, the PCA collapses the dimensionality, revealing the true number of independent macroeconomic variables driving the portfolio.
                    * **THE UTILITY:** Inspect the correlation boundaries. Any coefficient exceeding 0.85 represents mathematical capital inefficiency ('false diversification'). Liquidate redundant nodes and reallocate toward independent macro factors exhibiting correlation coefficients below 0.30.
                    """)

            # --- TAB 3: ASYMMETRY DIAGNOSTICS ---
            with tab3:
                st.subheader("Tier 3 - Non-Gaussian Risk Metrics and Loss Trajectories")
                c1, c2, c3 = st.columns(3)
                c1.metric(f"Sortino Ratio (Rf={rf_rate_input}%)", f"{m['Sortino']:.2f}")
                c2.metric("Expected Shortfall (CVaR 95%)", f"{m['CVaR 95']*100:.2f}%", delta_color="inverse")
                
                absolute_cvar_usd = initial_capital * m['CVaR 95']
                c3.metric("Tail Event Expectation (USD)", f"${absolute_cvar_usd:,.0f}", delta_color="inverse")
                
                fig_dd = px.area(results["drawdown"], title="Portfolio Underwater Plot (Peak-to-Trough Retracement Matrix)", color_discrete_sequence=['rgba(214, 39, 40, 0.8)'])
                fig_dd.update_layout(showlegend=False, yaxis_title="Drawdown Realization (%)", xaxis_title="Estimation Timeline")
                fig_dd.update_yaxes(tickformat=".1%")
                st.plotly_chart(fig_dd, use_container_width=True)

                st.markdown("#### Empirical Diagnostics")
                st.markdown(f"The left-tail properties of the return distribution exhibit a 95% Conditional Value at Risk (CVaR) of **{m['CVaR 95']*100:.2f}%**, translating to an absolute mean capital depletion of **\${absolute_cvar_usd:,.0f}** upon breaching the VaR threshold. This econometric integral isolates the expected shortfall within the extreme negative domain, explicitly bypassing standard Gaussian probability assumptions. A deeply skewed CVaR relative to baseline volatility confirms severe leptokurtosis (fat-tailed distribution), leaving the capital base exposed to high-velocity asymmetric downside accelerations during regime breaks.")

                with st.expander("Read Institutional Theory & Utility Guide"):
                    st.markdown("""
                    * **THE VISUAL:** The 'Underwater Plot' exclusively isolates periods of capital distress. The shaded topology quantifies the absolute geometric severity of historical drawdowns and visually demonstrates the precise temporal duration required to regenerate lost capital.
                    * **THE THEORY:** The Sortino Ratio adjusts for non-Gaussian asymmetry by substituting total variance with the downside semi-variance integral. Expected Shortfall (CVaR) calculates the conditional expectation of loss explicitly constrained within the worst 5% domain of the probability distribution.
                    * **THE UTILITY:** Extended multi-year recovery channels represent severe opportunity costs. A portfolio demonstrating strong upside expectancy but severe downside skew (deep CVaR metrics) mandates the immediate implementation of tail-risk hedging protocols or trailing stop mechanics.
                    """)

            # --- TAB 4: EFFICIENT FRONTIER ---
            with tab4:
                st.subheader("Tier 4 - Modern Portfolio Theory Convex Optimization")
                ef_data = results["efficient_frontier"]
                
                if ef_data is not None:
                    max_sharpe_idx = np.argmax(ef_data[2, :])
                    msr_return = ef_data[0, max_sharpe_idx]
                    msr_vol = ef_data[1, max_sharpe_idx]
                    msr_sharpe = ef_data[2, max_sharpe_idx]
                    
                    user_return = m['CAGR']
                    user_vol = m['Volatility']
                    user_sharpe = m['Sharpe']
                    
                    fig_ef = go.Figure()
                    fig_ef.add_trace(go.Scatter(
                        x=ef_data[1,:], y=ef_data[0,:], mode='markers',
                        marker=dict(size=4, color=ef_data[2,:], colorscale='Viridis', showscale=True, colorbar=dict(title="Sharpe Ratio")),
                        name="Simulated Allocation Cloud", hoverinfo='skip'
                    ))
                    fig_ef.add_trace(go.Scatter(
                        x=[user_vol], y=[user_return], mode='markers+text',
                        marker=dict(color='black', size=16, symbol='star-open-dot', line=dict(color='black', width=2)),
                        text=["Target Vector"], textposition="top center", name="Target Vector"
                    ))
                    fig_ef.update_layout(xaxis_title="Annualized Portfolio Volatility (σ)", yaxis_title="Expected Portfolio Return (CAGR)", title="Efficient Frontier Simulation Cloud")
                    st.plotly_chart(fig_ef, use_container_width=True)

                    st.markdown("#### Empirical Diagnostics")
                    delta_sharpe = msr_sharpe - user_sharpe
                    st.markdown(f"The simulation plots the target vector against the empirical Markowitz efficient boundary. The current allocation dictates a Sharpe ratio of **{user_sharpe:.2f}**, producing a differential of **{delta_sharpe:.2f}** against the maximum simulated tangency portfolio ({msr_sharpe:.2f}). An excessive Euclidean displacement toward the interior of the simulation cloud confirms the absorption of uncompensated idiosyncratic risk. Optimization requires the iterative execution of mean-variance reweighting algorithms to align the capital distribution tightly with the upper-left boundary locus.")

                with st.expander("Read Institutional Theory & Utility Guide"):
                    st.markdown("""
                    * **THE VISUAL:** The coordinate space maps 2,000 distinct allocation vectors constructed via random Monte Carlo permutations. The extreme upper-left geometric boundary constitutes the empirical Efficient Frontier curve.
                    * **THE THEORY:** Systematic algorithmic weighting demonstrates that a portfolio's aggregate variance can be minimized for any specified expected return target purely through covariance blending. Interior coordinates designate structurally inferior allocations burdened with uncompensated risk.
                    * **THE UTILITY:** Analyze the Euclidean displacement between your target vector and the boundary locus. Capital must be systematically redistributed from high-variance components toward the optimization boundary configuration to maximize theoretical absolute yield.
                    """)

            # --- TAB 5: ALGORITHMIC AUDITING ---
            with tab5:
                st.subheader("Algorithmic Compliance Ledger")
                st.markdown("Deterministic, rules-based validation of structural matrix bounds. Identifies hidden architectural vulnerabilities independent of human cognitive bias.")
                
                # Convert list of dicts to a Pandas DataFrame for a clean, professional grid look
                audit_data = results["audit_ledger"]
                df_audit = pd.DataFrame(audit_data)
                
                # Display without colors, focusing on pure data presentation
                st.dataframe(df_audit, use_container_width=True, hide_index=True)

                st.markdown("#### Empirical Diagnostics")
                st.markdown("The ledger applies rigid parameter thresholds to the allocation output. Metrics such as the Herfindahl-Hirschman Index (HHI) structurally verify capital density and identify asymmetric concentration risk. Failure to satisfy these strictly defined mathematical boundaries prior to physical execution constitutes a critical breach of institutional risk frameworks.")

            # --- TAB 6: REGIME STRESS TESTING ---
            with tab6:
                st.subheader("Historical Macro Regime Shock Simulation")
                stress_data = {k: v for k, v in results["stress_tests"].items() if v is not None}
                
                if stress_data:
                    df_stress = pd.DataFrame(list(stress_data.items()), columns=["Macroeconomic Regime Matrix", "Compound Geometric Return"])
                    
                    df_stress["Absolute Impact (USD)"] = initial_capital * df_stress["Compound Geometric Return"]
                    df_stress["Performance (%)"] = df_stress["Compound Geometric Return"] * 100
                    
                    fig_stress = px.bar(
                        df_stress, 
                        x="Macroeconomic Regime Matrix", 
                        y="Absolute Impact (USD)", 
                        title="Absolute Capital Depreciation Trajectory (USD) Under Historical Shocks",
                        text=df_stress["Absolute Impact (USD)"].apply(lambda x: f"${x:,.0f}")
                    )
                    fig_stress.update_layout(showlegend=False, yaxis_title="Capital Fluctuation (USD)", xaxis_title="Historical Regime Slice")
                    fig_stress.update_traces(textposition='outside')
                    st.plotly_chart(fig_stress, use_container_width=True)

                    st.markdown("#### Empirical Diagnostics")
                    st.markdown("The simulation overrides stationary standard deviation assumptions by evaluating out-of-sample systemic fragility. By forcing the static vector through localized periods of acute liquidity contraction and persistent inflationary regime shifts, it exposes the breakdown of conditional correlations. Uniform negative geometric destruction across uncorrelated historical scenarios mathematically proves an over-reliance on general equity beta, requiring the integration of orthogonal convexity buffers to ensure survival during sudden volatility expansions.")
                else:
                    st.warning("Insufficient historical time-series length to perform valid chronological slicing for the specified macro events.")

            # --- TAB 7: STRATEGIC ACTION PLAN ---
            with tab7:
                st.subheader("Structural Recommendations Based on Full Test Execution")
                st.markdown("Algorithmic directives ordered by immediate systemic priority to optimize matrix efficiency, neutralize tail risk, and compress volatility drag.")
                
                critical_advice = []
                optimization_advice = []
                
                if m['Max DD'] < -0.25 or m['CVaR 95'] < -0.05:
                    critical_advice.append("Integrate convexity buffers. The historical drawdown profile breaches safety limits. Allocate 10% to 15% of total capital to negatively correlated safe-haven assets (e.g., long-duration sovereign vectors) to serve as a structural shock absorber.")
                
                if m['N_eff'] < len(tickers) * 0.6 or max_corr_val > 0.85:
                    critical_advice.append("Liquidate collinear components. Severe false diversification detected. Identify the asset pair with a correlation coefficient $>0.85$. Liquidate the component exhibiting inferior mean-variance metrics and deploy liquidity into completely orthogonal factors.")
                
                if m['Sharpe'] < 0.8:
                    optimization_advice.append("Execute mean-variance trimming. The current vector absorbs excessive volatility without corresponding yield. Systematically extract weight from high-variance components and redistribute capital into stable, low-beta assets to align with the empirical Markowitz boundary.")
                
                if m['HHI'] > 0.15:
                    optimization_advice.append("Enforce strict distribution constraints. The weight vector displays highly asymmetric density. Enforce a hard cap rule to ensure no single constituent dominates the variance budget. Dilute concentrated nodes to mathematically restrict the HHI below 0.15.")

                col_a, col_b = st.columns(2)
                
                with col_a:
                    st.markdown("#### Priority 1: Critical Exposures")
                    if critical_advice:
                        for i, advice in enumerate(critical_advice, 1):
                            st.info(f"**{i}.** {advice}")
                    else:
                        st.info("No critical systemic exposures detected.")

                with col_b:
                    st.markdown("#### Priority 2: Yield Optimization")
                    if optimization_advice:
                        for i, advice in enumerate(optimization_advice, 1):
                            st.info(f"**{i}.** {advice}")
                    else:
                        st.info("No secondary optimization required. Locus is highly efficient.")

                st.markdown("---")
                with st.expander("Quantitative Glossary & Methodological Definitions"):
                    st.markdown("""
                    * **Compound Annual Growth Rate (CAGR):** The geometric mean return over the annualized time continuum, indicating the constant rate of compounding required to achieve the terminal portfolio value.
                    * **Annualized Standard Deviation:** The square root of the second central moment of the daily returns distribution. Deployed as the primary standard metric for systemic dispersion and risk density.
                    * **Sharpe Ratio:** Evaluated as expected excess return divided by the standard deviation. Calculates the geometric risk premium generated per unit of aggregate portfolio variance.
                    * **Sortino Ratio:** Formulated similarly to Sharpe, but represents the standard deviation measured exclusively over the negative return semi-domain. Eliminates penalization for positive asymmetric variance.
                    * **Maximum Drawdown (Max DD):** The maximum cumulative geometric contraction measured from the highest historical peak coordinate to the subsequent nadir coordinate.
                    * **Expected Shortfall (CVaR):** The mathematically derived expectation of realized loss specifically within the lowest 5% quantile of the probability distribution tail.
                    * **Effective Asset Count (N_eff):** Processed relying on the eigenvalues extracted from the principal components. Determines the authentic statistical dimensionality of the framework.
                    * **Herfindahl-Hirschman Index (HHI):** Quantifies the absolute density of portfolio weight concentration.
                    * **Pearson Correlation Coefficient:** Defines the covariance of discrete vector components scaled by the product of their respective standard deviations.
                    * **Modern Portfolio Theory (Markowitz):** A foundational mathematical paradigm demonstrating that asset diversification can minimize aggregate portfolio variance for a fixed expected return constraint.
                    """)
                
        except Exception as e:
            st.error(f"Critical execution error in computation loop: {e}")