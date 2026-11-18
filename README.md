# Flight Price Intelligence System

An end-to-end ML system that predicts flight prices, advises **when to buy**, and suggests alternative
routes. It covers the whole pipeline: web scraping → data cleaning → feature engineering → calibrated
regression → decision logic → recommendations → Flask web app.

Built for the *Advanced Python* module at Berliner Hochschule für Technik (BHT), winter semester 2025/26.
**Authors:** Amor Miladi, Salwa El Machhour

## What it does

- **Predicts the price** of a flight for a given route and departure date
- **Recommends buy or wait** – simulates the price curve for the next 7 days and finds the best booking window
- **Suggests alternatives** – nearby airports, cheaper layover routes and personalised destinations

## Architecture

```
Data acquisition (Selenium) → Cleaning → Feature engineering → Enrichment
                                                  │
                                                  ▼
                       Model 1: calibrated price regression (scikit-learn)
                                                  │
                         ┌────────────────────────┴───────────────────────┐
                         ▼                                                ▼
         Model 2: buy/wait decision                     Model 3: hybrid recommender
         (simulated price curves, confidence,           (content-based + popularity,
          trend analysis, departure penalty)             diversity re-ranking, SQLite history)
                         └────────────────────────┬───────────────────────┘
                                                  ▼
                                    Flask REST API + web interface
```

## Dataset

- ~2 GB of **longitudinal** price data scraped from Google Flights: the same routes were scraped on
  many consecutive days for fixed departure dates, so the data shows how prices move as departure approaches
- Features: days until departure, route distance (Haversine), hub and competition scores, route
  popularity, number of stops, flight duration, seasonality

## Results

Model 1 evaluated on a held-out test set (prices in EUR):

| Metric | Raw model | After calibration |
|--------|-----------|-------------------|
| MAE    | `TODO`    | `TODO`            |
| RMSE   | `TODO`    | `TODO`            |
| R²     | `TODO`    | `TODO`            |

The calibration layer is a linear correction (scale + offset) fitted on 3,000 predictions from the
running API (`backend-api/scripts/calibrate_model1_api.py`).

## Repository structure

```
cheapflight-scraper-main/
  scraper/              Selenium scrapers (multi_route_scraper.py, scrape_google_flights.py)
  scripts/cleaning/     de-duplication, type fixes, removing missing prices
  scripts/features/     days-left and non-stop features
  scripts/enrichment/   airport, route and continent enrichment → master training dataset
ml-workspace/
  scripts/merge_parts.py      merge the dataset parts
  scripts/model1.py           train the price model (prints MAE / RMSE / R²)
  scripts/model2_training.py  generate buy/wait advice
backend-api/
  app.py                Flask app (http://127.0.0.1:5001)
  services/             model1, model2, model3 and recommender logic
  scripts/calibrate_model1_api.py   fit the calibration layer
  tests/                smoke and API tests
```

## How to run

> **Note:** the dataset (~2 GB) and trained model files are not in the repository (see `.gitignore`).
> To run the app you first need to scrape the data and train the model, or ask the authors for the files.

```bash
pip install -r requirements.txt

# 1. Collect data (optional, takes a long time)
python cheapflight-scraper-main/scraper/multi_route_scraper.py

# 2. Train the price model (expects master_training_dataset.csv in the working directory)
cd ml-workspace/scripts
python model1.py               # saves model1_pipeline_fast.pkl and model1_metadata_fast.pkl
# copy both .pkl files to backend-api/models/

# 3. Start the web app
cd ../../backend-api
python app.py                  # open http://127.0.0.1:5001

# 4. (Optional) Fit the calibration layer while the app is running
python scripts/calibrate_model1_api.py --data path/to/master_training_dataset.csv
```

## Tech stack

| Area | Tools |
|------|-------|
| Data collection | Python, Selenium |
| Data processing | pandas, NumPy |
| Machine learning | scikit-learn |
| Backend | Flask, REST API, SQLite |
| Tools | Git, Linux |
