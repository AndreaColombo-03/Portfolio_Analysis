# Institutional Portfolio Sandbox & Quantitative Risk Engine

## Overview
This project is a production-grade **Quantitative Portfolio Engine** and interactive web dashboard. It is designed to model multi-asset portfolios and execute rigorous cross-sectional econometric risk factor diagnostics, simulating the operational standards required by financial regulatory bodies and quantitative research divisions.

## Key Econometric Capabilities
The engine provides a programmatic framework for portfolio assessment, focusing on:
* **Dimensionality & Factor Orthogonality:** Principal Component Analysis (PCA) to derive the 'Effective Asset Count' (N_eff), identifying if assets provide independent risk premia or redundant factor exposure.
* **Non-Gaussian Risk Analysis:** Calculation of **Sortino Ratios** and **Conditional Value at Risk (CVaR 95%)** to quantify exposure to fat-tailed distribution shocks that standard deviation ignores.
* **Convex Optimization:** Integration of **Markowitz Modern Portfolio Theory (MPT)** to simulate the Efficient Frontier and map the portfolio's Euclidean distance from the tangency locus.
* **Macroeconomic Stress Testing:** Evaluation of portfolio resilience against verified historical liquidity contractions and inflationary shocks.
* **Automated Institutional Auditing:** A rules-based compliance ledger that evaluates allocations against structural thresholds (e.g., Herfindahl-Hirschman Index for concentration, pairwise correlation limits).

## Live Demo
You can test the engine directly in your browser:
👉 **[Launch Institutional Portfolio Sandbox](INSERISCI_QUI_IL_LINK_CHE_TI_DARA_STREAMLIT)**

## Project Architecture
The system employs a strict separation between computational logic and the user interface:
- `/core`: Houses the modular `quant_engine.py` library, implementing the object-oriented backend for econometric modeling.
- `app.py`: The Streamlit-based frontend, managing user state and high-fidelity financial visualization.
- `requirements.txt`: Environment dependencies for production deployment.

## Technical Methodology
The engine operates by ingesting raw time-series data, applying forward-fill sanitization protocols, and performing a spectral decomposition of the covariance matrix. This methodology ensures that all metrics—from the Sortino ratio to the tail-risk projections—are calculated using robust, non-parametric statistical techniques.

## Local Execution
To run this project locally on your machine:
1. Clone the repository: `git clone https://github.com/AndreaColombo-03/Portfolio_Analysis.git`
2. Install the dependencies: `pip install -r requirements.txt`
3. Launch the sandbox: `streamlit run app.py`

---
*Disclaimer: This tool is designed for educational and analytical purposes. It does not constitute financial advice or investment recommendations.*
