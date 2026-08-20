# Interpretabilitas Model Prediksi Memantau Kerentanan Pasar Saham Berbasis Faktor Makroekonomi

Repositori kode untuk skripsi dan paper (target: JAIC, SINTA 3) yang menganalisis
perbedaan pengaruh faktor makroekonomi domestik dan global terhadap return
harian IHSG (2021-2026) menggunakan Linear Regression, Random Forest, dan
XGBoost, diinterpretasikan dengan SHAP.

## Struktur Folder

```
ihsg-makro-shap/
├── data_raw/          # Data mentah hasil unduhan (Yahoo Finance, FRED, manual BI/EPU)
├── data_processed/     # Data setelah digabung, di-lag, dan disaring hari bursa
├── notebooks/          # Notebook eksplorasi (opsional, kalau kerja di Jupyter/Colab)
├── src/                 # Skrip Python per tahap (akuisisi, preprocessing, modeling, SHAP)
├── outputs/
│   ├── figures/         # Grafik SHAP, perbandingan model, dll.
│   └── tables/          # Tabel hasil (OLS, evaluasi model, ablation study)
├── config_template.py   # Template API key -- copy jadi config.py, jangan commit config.py
├── requirements.txt
└── .gitignore
```

## Setup

```bash
git clone <URL_REPO_ANDA>
cd ihsg-makro-shap
pip install -r requirements.txt
cp config_template.py config.py
# lalu isi FRED_API_KEY di config.py
```

## Urutan Eksekusi

1. `src/hari1_akuisisi_data.py` -- ambil data dari Yahoo Finance & FRED,
   plus unduh manual BI Rate, INDONIA, dan EPU Index ke `data_raw/`.
2. Penggabungan data, penyaringan hari bursa, lag, uji ADF, EDA.
3. Estimasi OLS + 3 model ML (Linear Regression, Random Forest, XGBoost)
   dengan walk-forward validation (latih 2021-2024, uji 2025-2026).
4. SHAP (LinearExplainer, TreeExplainer), SHAP jendela bergulir,
   ablation study.
5. Kompilasi hasil untuk penulisan Bab IV / paper.

## Variabel

**Domestik (3):** BI Rate, kurs USD/IDR, INDONIA
**Global (7):** Fed Funds Rate, yield obligasi pemerintah AS 10 tahun,
return S&P 500, harga minyak dunia, Dollar Index, VIX, US Economic
Policy Uncertainty Index

## Sumber Data

| Variabel | Sumber | Otomatis? |
|---|---|---|
| IHSG, kurs, S&P 500, minyak, DXY, VIX | Yahoo Finance | Ya |
| Fed Funds Rate, yield 10Y AS | FRED API | Ya |
| BI Rate, INDONIA | bi.go.id | Manual |
| EPU Index | policyuncertainty.com | Manual |
