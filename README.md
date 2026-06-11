# Flight Price Intelligence System

An end-to-end AI-based flight price prediction and recommendation system built with Python. The system collects longitudinal flight pricing data, trains a calibrated regression model, and provides actionable booking recommendations.

## What it does

- **Predicts flight prices** for a given route and departure date
- **Recommends when to buy** — simulates price curves over the next 7 days and identifies the optimal booking window
- **Suggests alternatives** — nearby airports, cheaper layover routes, and personalized destination recommendations
- Deployed as a **Flask REST API** with a web interface

## System architecture

The project is organized into 5 layers:

```
Data Acquisition → Preprocessing → Price Prediction (Model 1)
                                → Decision Logic  (Model 2)
                                → Recommendations (Model 3)
```

- **Model 1** — Calibrated regression pipeline (scikit-learn) predicting prices from temporal, geographic, and structural route features
- **Model 2** — Decision module computing optimal waiting period via simulated price curves, confidence scoring, trend analysis, and departure proximity penalties
- **Model 3** — Hybrid recommendation engine combining content-based filtering, popularity ranking, and diversity-aware re-ranking; SQLite for user history and personalization

## Tech stack

| Area | Tools |
|------|-------|
| Data collection | Python, Selenium, Google Flights |
| Data processing | pandas, NumPy, scikit-learn |
| Machine learning | scikit-learn regression pipeline, calibration layer |
| Backend | Flask, REST API, SQLite |
| Infrastructure | Git, Linux |

## Dataset

- ~2GB of longitudinal flight pricing data collected via automated scraping
- Routes scraped over multiple consecutive booking days for fixed departure dates — enabling modeling of price dynamics over time
- Features: days until departure, route distance (Haversine), hub scores, competition scores, route popularity, number of stops, flight duration, seasonality

## How to run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) Rebuild the dataset
python multi_route_scraper.py

# 3. Train the model
python train_model.py

# 4. Start the Flask server
python app.py

# 5. Open in browser
http://localhost:5000
```

## Project context

Built as part of the **Advanced Python** module at Berliner Hochschule für Technik (BHT), Winter Semester 2025/2026.

**Authors:** Amor Miladi, Salwa El Machhour
