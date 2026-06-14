## Dataset
Hotel Booking Demand Dataset (2015–2017)

Jumlah data: 119.390 record
Target:
- is_canceled

## Model yang Digunakan
- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost
- LightGBM

Model terbaik yang digunakan pada deployment adalah Random Forest.

## Deployment
Link Streamlit:
https://hotel-booking-cancellation-n2lte6jyusat2mtxbb4gmn.streamlit.app/

## Cara Menjalankan Secara Lokal
1. Install dependencies
```bash
pip install -r requirements.txt

2. Jalankan Streamlit
```bash
streamlit run app_streamlit.py

## Struktur Proyek
hotel_predictor/
├── app_streamlit.py
├── requirements.txt
├── best_model.pkl
├── scaler.pkl
├── feature_names.pkl
├── model_metadata.json
├── README.md
└── notebooks/
    └── UAS_DS_FIXED.ipynb

Cheyza Amanda Syafira
S1 Teknik Komputer
Telkom University
