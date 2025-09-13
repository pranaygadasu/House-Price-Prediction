# House Price Prediction — Starter

## Quickstart
1) Create & activate a virtualenv, then install:
   ```bash
   pip install -r requirements.txt
   ```
2) Download Kaggle "House Prices: Advanced Regression Techniques" and put `train.csv` into `data/`.
3) Open the notebooks in `notebooks/` to run EDA and training.
4) After training saves `models/house_price_pipeline.joblib`, run the app:
   ```bash
   cd app
   streamlit run app.py
   ```

## Structure
```
house-price-ml/
├─ data/              # put raw CSVs here
├─ models/            # saved model/pipeline
├─ notebooks/         # EDA + experiments
├─ src/               # reusable code
├─ app/               # Streamlit app
├─ requirements.txt
└─ README.md
```
