# Agentic Football Match Predictor ⚽🤖

A hybrid sports prediction engine that enforces a strict boundary between deterministic mathematical modeling and generative AI. 

This system uses pure mathematics (Elo ratings, Poisson distributions, Dixon-Coles correction) to calculate match probabilities, and securely wraps that engine in a constrained LLM layer to handle natural language querying and result narration. Nothing in the control flow or math is left to AI reasoning.

## Features

* **Rigorous Mathematical Core**: 
  * Baseline team strength calculated via goal-difference-weighted Elo.
  * Scoreline probabilities generated using a Poisson distribution with a Dixon-Coles correction for low-score dependencies.
  * Dynamic in-play expected goals (xG) adjustments based on remaining time and live momentum differentials.
* **Agentic LLM Surface**: Exactly two LLM calls control the user experience:
  1. Structured-data extraction of team names and intent from natural language.
  2. Sports-commentator style narration of the strictly calculated probabilities.
* **Live Score Integration**: Fetches real-time match data via the `football-data.org` API.
* **Backtesting Harness**: Includes an isolated backtester to prove the predictive value of the Elo module against historical datasets before applying live-layer code.

## Architecture

* `prediction_engine.py`: Pure, deterministic match-prediction math. Zero LLM calls.
* `elo_backtester.py`: Harness to validate the Elo module against a `.csv` of historical results.
* `llm_pipeline.py`: The LLM agent layer. Parses user queries into JSON and narrates the deterministic output.
* `match_predictor.py`: The interactive command-line interface for pre-match predictions and live updates.

## Getting Started

### Prerequisites
* Python 3.8+
* [Gemini API Key](https://aistudio.google.com/app/apikey)
* [Football-Data.org API Key](https://www.football-data.org/) (Optional, for live scores)

### Installation

1. Clone the repository:
   ```bash
   git clone [https://github.com/yourusername/football-predictor.git](https://github.com/yourusername/football-predictor.git)
   cd football-predictor
