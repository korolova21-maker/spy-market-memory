# SPY Market Memory — Streamlit MVP

This is a first research prototype for a SPY-focused Market Memory / Signal Intelligence tool.

It answers:

- What does the current SPY state resemble historically?
- What usually happened next after similar states?
- Is the current environment better for continuation, pullback-buying, scalp-only trading, or avoiding chase entries?

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Community Cloud

1. Create a GitHub repository.
2. Upload:
   - app.py
   - requirements.txt
   - README.md
3. Go to Streamlit Community Cloud.
4. Deploy the repo and select app.py as the entrypoint.

## MVP limitations

- Daily data only.
- SPY is the main analysed ticker.
- QQQ and VIX are used as context features.
- Signal Forensics is only a placeholder until TradingView signal dates are imported.
- No financial advice. Research/education only.