# Sistem Machine Learning Klasifikasi Teks

Proyek ini menyediakan alur ML untuk kasus NLP klasifikasi teks: data mentah → EDA → training & evaluasi → REST API.

## Struktur

- `data/raw/` - tempat menaruh dataset mentah (CSV / Excel)
- `src/data_loader.py` - pembacaan data dan konfigurasi dataset
- `src/eda.py` - eksplorasi data teks dan visualisasi
- `src/train.py` - pelatihan model teks dan evaluasi
- `src/api.py` - REST API FastAPI untuk prediksi teks
- `artifacts/` - model dan pipeline tersimpan
- `outputs/eda/` - hasil EDA
- `outputs/eval/` - grafik evaluasi dan metadata
- `config.yaml` - konfigurasi dataset, kolom teks, target, dan model
- `requirements-api.txt` - dependensi untuk serving API dengan versi ter-pin

## Dataset

Letakkan dataset Anda di `data/raw/data.csv` dengan format minimal:

- `text`: kolom teks berbahasa Indonesia
- `label`: label kelas target

Contoh:

| text | label |
|---|---|
| "Saya suka produk ini" | positif |
| "Pelayanan buruk dan lambat" | negatif |

## Cara pakai

1. Letakkan dataset Anda di `data/raw/data.csv`.
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

## Endpoint API

- `GET /` - informasi layanan
- `GET /health` - status model
- `POST /predict-teks` - prediksi teks

Contoh request:

```bash
curl -X POST "http://127.0.0.1:8000/predict-teks" -H "Content-Type: application/json" -d '{"text": "Produk ini sangat bagus"}'
```

## Catatan

- Sistem ini fokus pada klasifikasi teks bahasa Indonesia.
- Representasi teks dibuat dengan TF-IDF + n-gram.
- Penanganan negasi diterapkan pada token seperti `tidak`, `gak`, `bukan`.
