
# FCN Pricer

A Streamlit app for pricing a worst-of FCN or similar multi-underlying structure using Yahoo Finance data.

## Files
- `fcn_pricer_app.py` - Streamlit app
- `fcn_pricer_template.xlsx` - optional input template
- `requirements.txt` - Python dependencies

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run fcn_pricer_app.py
```

## Deploy to Streamlit Community Cloud
1. Put these files in a GitHub repository.
2. Set the main file to `fcn_pricer_app.py`.
3. Deploy from the Streamlit Cloud dashboard.
