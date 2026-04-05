# 🏏 IPL 2026 Winner Prediction — AI/ML Project

## Project Structure
```
ipl2026/
├── data/
│   └── data_generator.py       # Synthetic IPL dataset generator (2008–2025)
├── models/
│   └── train_models.py         # Model training: LR, RF, XGBoost, NN
├── utils/
│   ├── feature_engineering.py  # All feature calculations
│   └── evaluation.py           # Metrics & confusion matrix
├── notebooks/
│   └── ipl_analysis.py         # Full EDA + analysis notebook-style script
├── dashboard/
│   └── app.py                  # Streamlit dashboard
├── predict_2026.py             # MAIN prediction script — run this!
└── requirements.txt            # All dependencies
```

## Quick Start
```bash
pip install -r requirements.txt
python predict_2026.py
```

## Dashboard
```bash
streamlit run dashboard/app.py
```
