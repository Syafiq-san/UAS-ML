from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "raw" / "cars.csv"
EDA_DIR = ROOT / "outputs" / "eda"
EVAL_DIR = ROOT / "outputs" / "eval"
OUTPUT_PDF = ROOT / "reports" / "laporan_ml.pdf"


def read_metrics() -> dict:
    metrics = {}
    if (EVAL_DIR / "metrics.txt").exists():
        for line in (EVAL_DIR / "metrics.txt").read_text(encoding="utf-8").splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                metrics[key.strip()] = value.strip()
    return metrics


def build_pdf():
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("Title", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=18, leading=24, spaceAfter=12)
    heading_style = ParagraphStyle("Heading", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=12, leading=16, spaceBefore=8, spaceAfter=6)
    body_style = ParagraphStyle("Body", parent=styles["BodyText"], fontName="Helvetica", fontSize=10, leading=14)
    small_style = ParagraphStyle("Small", parent=styles["BodyText"], fontName="Helvetica", fontSize=9, leading=12)

    doc = SimpleDocTemplate(str(OUTPUT_PDF), pagesize=A4, rightMargin=2.2 * cm, leftMargin=2.2 * cm, topMargin=2.2 * cm, bottomMargin=2.2 * cm)
    story = []

    story.append(Paragraph("Laporan UAS Machine Learning End-to-End", title_style))
    story.append(Paragraph("Sistem prediksi harga mobil berbasis data tabular", body_style))
    story.append(Paragraph("Nama: [Nama Mahasiswa]", small_style))
    story.append(Paragraph("Program Studi: Informatika", small_style))
    story.append(Paragraph("Institusi: Institut Teknologi Tangerang Selatan", small_style))
    story.append(Spacer(1, 0.3 * cm))

    df = pd.read_csv(DATA_PATH)
    missing = pd.read_csv(EDA_DIR / "missing_values.csv")
    duplicate_count = pd.read_csv(EDA_DIR / "duplicates.csv").iloc[0, 0]
    metrics = read_metrics()

    story.append(Paragraph("1. Ringkasan proyek", heading_style))
    story.append(Paragraph(f"Dataset yang digunakan: {DATA_PATH.name}", body_style))
    story.append(Paragraph(f"Jumlah baris: {df.shape[0]} | Jumlah kolom: {df.shape[1]}", body_style))
    story.append(Paragraph(f"Target: {df.columns[-1]}", body_style))
    story.append(Paragraph("Tujuan: memprediksi harga mobil berdasarkan atribut seperti brand, model, tahun, jarak tempuh, mesin, tenaga, dan kapasitas penumpang.", body_style))
    story.append(Paragraph("Model yang dipakai: Random Forest Regressor dengan preprocessing numerik + kategorikal.", body_style))
    story.append(Paragraph("Sumber data: file CSV yang diunggah ke folder data/raw/cars.csv.", body_style))
    story.append(Spacer(1, 0.2 * cm))

    story.append(Paragraph("2. Hasil EDA", heading_style))
    story.append(Paragraph("Pemeriksaan wajib yang dilakukan:", body_style))
    story.append(Paragraph("- df.isna().sum()", body_style))
    story.append(Paragraph("- df.describe()", body_style))
    story.append(Paragraph("- distribusi target", body_style))
    story.append(Paragraph("- pemeriksaan duplikat", body_style))
    story.append(Paragraph(f"Jumlah duplikat ditemukan: {duplicate_count}", body_style))
    story.append(Paragraph("Hasil EDA disimpan pada folder outputs/eda dan meliputi summary.csv, missing_values.csv, duplicates.csv, serta grafik target_distribution.png.", body_style))

    if (EDA_DIR / "target_distribution.png").exists():
        story.append(Spacer(1, 0.2 * cm))
        story.append(Paragraph("Grafik 1: Distribusi target (Price)", body_style))
        img = Image(str(EDA_DIR / "target_distribution.png"), width=12 * cm, height=6 * cm)
        story.append(img)
        story.append(Spacer(1, 0.2 * cm))

    story.append(Paragraph("3. Training dan evaluasi", heading_style))
    table_data = [["Metrik", "Nilai"]]
    for key, value in metrics.items():
        table_data.append([key, value])
    table = Table(table_data, repeatRows=1)
    table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f4e79")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
        ])
    )
    story.append(table)
    story.append(Spacer(1, 0.2 * cm))

    story.append(Paragraph("4. Struktur folder proyek", heading_style))
    folder_list = [
        "data/raw/cars.csv",
        "src/data_loader.py",
        "src/eda.py",
        "src/train.py",
        "src/api.py",
        "outputs/eda",
        "outputs/eval",
        "artifacts/pipeline.joblib",
        "reports/laporan_ml.pdf",
    ]
    for item in folder_list:
        story.append(Paragraph(f"- {item}", body_style))
    story.append(Spacer(1, 0.2 * cm))

    story.append(Paragraph("5. REST API", heading_style))
    story.append(Paragraph("Endpoint yang tersedia:", body_style))
    story.append(Paragraph("- GET /health", body_style))
    story.append(Paragraph("- POST /predict", body_style))
    story.append(Paragraph("Contoh input API: field features berisi atribut mobil yang akan diprediksi.", body_style))
    story.append(Spacer(1, 0.2 * cm))

    story.append(Paragraph("6. Kesimpulan", heading_style))
    story.append(Paragraph("Sistem ML ini sudah berjalan end-to-end dari data mentah, EDA, training, evaluasi, hingga endpoint prediksi REST API.", body_style))
    story.append(Paragraph("Proses ini dapat dipertanggungjawabkan karena setiap tahapan disertai artefak dan hasil yang tersimpan di folder proyek.", body_style))

    doc.build(story)
    return OUTPUT_PDF


if __name__ == "__main__":
    path = build_pdf()
    print(f"PDF laporan dibuat di {path}")
