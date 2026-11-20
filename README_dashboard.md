# SportsBet Research Dashboard (Streamlit)

## Run locally

```bash
pip install -r requirements_dashboard.txt
streamlit run sportsbet_dashboard.py
```

The dashboard reads `/workspace/artifacts/logs/SportsBet_Research_Log.jsonl` and shows:
- KPI cards (settled count, 1X2 / Over 2.5 / BTTS hit rates)
- **Rolling 20-match 1X2 hit rate chart**
- **Calibration chart** (predicted vs actual)
- Recent settled matches
- Upcoming / unsettled matches
- Hit rate by competition
- **Sidebar filters** for competition and date range

It auto-refreshes every 30 seconds.

## Notes
- New entries appended to the JSONL appear on the next refresh.
- Settled matches are those with `_SETTLED` in `event_id` or a non-null `hit_1x2`.
- Calibration uses the first probability value in `model_probs` (typically home win).
