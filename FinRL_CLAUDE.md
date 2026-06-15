# Claude Project Context: FinRL

This file provides codebase context for the **FinRL** repository, used as part of an AI trading-knowledge project.

## Project Overview
* **Name**: FinRL (Financial Reinforcement Learning)
* **Language**: Python
* **Type**: Open-source deep reinforcement learning (DRL) framework for quantitative finance
* **Purpose**: Provide a full pipeline — data, environment, agent, training, backtesting — to build DRL trading agents for stocks, crypto, and portfolio allocation.
* **Maintainer**: AI4Finance Foundation
* **License**: MIT (verify per sub-repo; some companion repos differ)

## Core System Boundaries
* **Three-layer design**: Market Environments (Gym-style), DRL Agents (via Stable-Baselines3, ElegantRL, RLlib), and Applications (stock/crypto trading, portfolio allocation).
* **Data sources**: Yahoo Finance, Alpaca, CCXT/Binance (crypto), WRDS (academic data).
* **Supported algorithms**: A2C, PPO, DDPG, TD3, SAC, DQN — accessed through a unified `DRLAgent` wrapper.
* **Companion repos**: `FinRL-Meta` (market simulators/data), `FinRL-Trading` (deployment examples), `FinGPT` (LLM-based, separate project).

## Repository Structure & Key Directories
* `/finrl/` - Core library
  * `/finrl/meta/` - Market environments (Gym envs) and data processors for different markets/brokers
  * `/finrl/agents/` - Wrappers around Stable-Baselines3 / ElegantRL / RLlib algorithms
  * `/finrl/applications/` - Pre-built application environments (stock_trading, cryptocurrency_trading, portfolio_allocation, high_frequency_trading)
  * `/finrl/config.py`, `/finrl/config_tickers.py` - Default config and ticker lists (e.g. DOW_30_TICKER)
* `/examples/` - Jupyter notebooks demonstrating end-to-end pipelines (data download → train → backtest → plot)
* `/unit_tests/` - Tests showing minimal working usage

## Standard Operational Commands
* **Install**: `pip install --upgrade git+https://github.com/AI4Finance-Foundation/FinRL.git` (or `pip install finrl` for PyPI releases — check version freshness)
* **Run example notebook**: open notebooks under `/examples/` in Jupyter

## Standard Python API Usage
```python
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent

# 1. Download & preprocess data
df = YahooDownloader(start_date="2020-01-01", end_date="2023-01-01",
                      ticker_list=["AAPL", "MSFT"]).fetch_data()

# 2. Create training environment
env = StockTradingEnv(df=df, ...)

# 3. Train an agent
agent = DRLAgent(env=env)
model = agent.get_model("ppo")
trained_model = agent.train_model(model=model, tb_log_name="ppo", total_timesteps=50000)
```

## AI Assistant Guidelines for this Project
1. **Pipeline order**: always follow data → environment → agent → train → backtest/evaluate; don't skip preprocessing steps (technical indicators, turbulence index, normalization).
2. **Environment config**: highlight required env parameters (initial amount, transaction cost, state/action space dimensions, reward scaling) since misconfiguration silently breaks training.
3. **Algorithm choice**: when unsure, suggest PPO or A2C as stable defaults for continuous action spaces (portfolio weights/position sizes).
4. **Crypto vs stock envs**: clarify that `cryptocurrency_trading` environment expects CCXT/Binance-style OHLCV data, while `stock_trading` expects Yahoo Finance-style data — don't mix formats.
5. **Backtest realism**: remind users DRL backtests are prone to overfitting; recommend walk-forward validation and out-of-sample testing before any live deployment.
6. **No financial advice**: FinRL is a research/education framework — results are not investment recommendations.
