# Sistem Machine Learning Prediksi Harga Mobil Bekas

Proyek ini menyediakan alur ML untuk kasus regresi: data mentah → EDA → training & evaluasi → REST API.

## Struktur

- `data/raw/` - tempat menaruh dataset mentah (CSV / Excel)
- `src/data_loader.py` - pembacaan data dan konfigurasi dataset
- `src/eda.py` - eksplorasi data teks dan visualisasi
- `src/train.py` - pelatihan model regresi dan evaluasi
- `src/api.py` - REST API FastAPI untuk prediksi harga
- `artifacts/` - model dan pipeline tersimpan
- `outputs/eda/` - hasil EDA
- `outputs/eval/` - grafik evaluasi dan metadata
- `config.yaml` - konfigurasi dataset, kolom target, dan model
- `requirements-api.txt` - dependensi untuk serving API dengan versi ter-pin

## Dataset

Sumber data yang digunakan berasal dari Kaggle:
- https://www.kaggle.com/datasets/sujithmandala/second-hand-car-price-prediction?resource=download

Dataset dapat ditempatkan di `data/raw/cars.csv` (atau file CSV/Excel lain yang Anda sesuaikan di `config.yaml`).

Format yang digunakan pada proyek ini adalah data mobil bekas dengan kolom seperti:
- `Brand`, `Model`, `Year`, `Kilometers_Driven`, `Fuel_Type`, `Transmission`, `Owner_Type`, `Mileage`, `Engine`, `Power`, `Seats`, `Price`

## Cara pakai

1. Letakkan dataset Anda di `data/raw/cars.csv` (atau sesuaikan path di `config.yaml`).
2. Sesuaikan `config.yaml` jika nama kolom berbeda atau ingin mengubah parameter.
3. Instal dependensi:
   ```bash
   pip install -r requirements.txt
   ```
4. Jalankan EDA:
   ```bash
   python -m src.eda
   ```
5. Latih model:
   ```bash
   python -m src.train
   ```
6. Jalankan REST API:
   ```bash
   uvicorn src.api:app --reload
   ```
7. Buat laporan PDF:
   ```bash
   python generate_pdf_report.py
   ```

## Endpoint API

- `GET /` - informasi layanan
- `GET /health` - status model
- `POST /predict` - prediksi harga mobil

Contoh request:

```bash
curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d '{"features": {"Brand": "Toyota", "Model": "Corolla", "Year": 2018, "Kilometers_Driven": 50000, "Fuel_Type": "Petrol", "Transmission": "Manual", "Owner_Type": "First", "Mileage": 15, "Engine": 1498, "Power": 108, "Seats": 5}}'
```

## Catatan

- Sistem ini fokus pada regresi harga mobil bekas.
- Model yang digunakan dapat dikembangkan lebih lanjut dengan data yang lebih besar dan fitur tambahan.
- Laporan PDF dapat dihasilkan otomatis melalui file `generate_pdf_report.py`.
