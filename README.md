Stuff+ Model

## Running it

```bash
python -m pip install -r requirements.txt

python main.py train      # train the model from the season tables -> model/artifacts/
python main.py score      # score each season's raw pitches -> pitches_<season>_scored
python main.py profiles   # aggregate scored pitches into cards + leaderboards (JSON)
python main.py live       # start the live in-season update loop

./start.sh                # run the Flask web app locally on http://localhost:5001
gunicorn app:app          # serve the web app (see Procfile for the Railway command)
```

## Layout

| Path | Responsibility |
|------|----------------|
| `model/prob_resid.py` | the model: features, swing softmax, GB/air contact head, run values |
| `model/train.py` | training run — fits and saves all inference artifacts |
| `model/predict.py` | inference — loads the model, turns pitches into grades |
| `features/engineering.py` | shape-feature engineering + induced (Magnus) components |
| `main.py` | CLI: `train` / `score` / `profiles` / `live` |
| `app.py` | Flask web app (leaderboards, pitcher cards, editor) |
| `profiles/` | pitcher-card and leaderboard builders |
| `scraper/`, `live/` | data ingestion and the live update loop |
