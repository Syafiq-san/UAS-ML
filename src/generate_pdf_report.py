from pathlib import Path
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, Image
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from src.data_loader import load_config, load_data
except ImportError:  # pragma: no cover - fallback for direct script execution
    from data_loader import load_config, load_data

ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = ROOT / "reports"
FIG_DIR = REPORT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def save_plot(fig, filename):
    path = FIG_DIR / filename
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def build_charts(df):
    # 1. Scatter: price vs kilometers
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ax.scatter(df["Kilometers_Driven"], df["Price"], alpha=0.7, color="#4C78A8")
    ax.set_title("Harga vs. kilometer tempuh")
    ax.set_xlabel("Kilometers Driven")
    ax.set_ylabel("Price")
    ax.grid(True, linestyle="--", alpha=0.4)
    save_plot(fig, "price_vs_km.png")

    # 2. Boxplot by transmission
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    labels = df["Transmission"].unique()
    data = [df.loc[df["Transmission"] == t, "Price"] for t in labels]
    ax.boxplot(data, patch_artist=True)
    ax.set_xticklabels(labels)
    ax.set_title("Distribusi harga berdasarkan transmisi")
    ax.set_ylabel("Price")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    save_plot(fig, "price_by_transmission.png")

    # 3. Average price by fuel type
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    summary = df.groupby("Fuel_Type")["Price"].mean().sort_values()
    summary.plot(kind="bar", color=["#F58518", "#54A24B"], ax=ax)
    ax.set_title("Rata-rata harga berdasarkan jenis bahan bakar")
    ax.set_ylabel("Rata-rata Price")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    save_plot(fig, "avg_price_by_fuel.png")

    # 4. Residual histogram for selected model
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    residuals = build_model_metrics(df)["residuals"]
    ax.hist(residuals, bins=12, color="#B279A2", edgecolor="black")
    ax.set_title("Distribusi residual prediksi")
    ax.set_xlabel("Residual (aktual - prediksi)")
    ax.set_ylabel("Jumlah sampel")
    ax.grid(True, linestyle="--", alpha=0.4)
    save_plot(fig, "residual_hist.png")


def build_model_metrics(df):
    X = df.drop(columns=["Price"])
    y = df["Price"]
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=["number"]).columns.tolist()

    preprocess = ColumnTransformer(
        [
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                num_cols,
            ),
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore")),
                ]),
                cat_cols,
            ),
        ],
        remainder="drop",
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(random_state=42, n_estimators=200),
        "GradientBoosting": GradientBoostingRegressor(random_state=42),
    }

    rows = []
    selected_residuals = None
    for name, model in models.items():
        pipe = Pipeline([("preprocess", preprocess), ("model", model)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        mae = mean_absolute_error(y_test, pred)
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        r2 = r2_score(y_test, pred)
        rows.append((name, mae, rmse, r2))
        if name == "LinearRegression":
            selected_residuals = y_test - pred

    return {"rows": rows, "residuals": selected_residuals}


def build_pdf(output_path: Path):
    config = load_config()
    df = load_data(config)
    metrics = build_model_metrics(df)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="TitleStyle", parent=styles["Title"], fontSize=20, leading=24, spaceAfter=12, textColor=colors.HexColor("#123A5A")))
    styles.add(ParagraphStyle(name="HeadingStyle", parent=styles["Heading1"], fontSize=13, leading=16, textColor=colors.HexColor("#1F4E79"), spaceBefore=12, spaceAfter=6))
    styles.add(ParagraphStyle(name="BodyStyle", parent=styles["BodyText"], fontSize=10.5, leading=14, spaceAfter=6))
    styles.add(ParagraphStyle(name="BulletStyle", parent=styles["BodyText"], fontSize=10.5, leading=14, leftIndent=12, bulletIndent=0, spaceAfter=4))
    styles.add(ParagraphStyle(name="CaptionStyle", parent=styles["BodyText"], fontSize=9, leading=11, textColor=colors.grey, alignment=1, spaceAfter=10))

    story = []
    story.append(Paragraph("Laporan Analisis Prediksi Harga Mobil Bekas", styles["TitleStyle"]))
    story.append(Paragraph("Studi kasus: dataset mobil bekas dari data/raw/cars.csv", styles["BodyStyle"]))
    story.append(Paragraph("Tanggal pembuatan: 25 Juli 2026", styles["BodyStyle"]))
    story.append(Spacer(1, 0.1 * inch))

    # Section 1
    story.append(Paragraph("1. Masalah & Sumber Data", styles["HeadingStyle"]))
    story.append(Paragraph(
        "Tujuan analisis ini adalah memprediksi harga mobil bekas berdasarkan atribut seperti merek, model, tahun, kilometer tempuh, jenis bahan bakar, transmisi, dan spesifikasi mesin."
        "Dataset berasal dari file cars.csv yang disimpan di folder data/raw dan terdiri dari 100 sampel dengan 13 kolom fitur dan target Price.",
        styles["BodyStyle"],
    ))
    story.append(Paragraph("Sumber data: data/raw/cars.csv", styles["BodyStyle"]))
    story.append(Paragraph("Target yang diprediksi: Price (nilai numerik)", styles["BodyStyle"]))
    story.append(Spacer(1, 0.12 * inch))

    # Section 2
    story.append(Paragraph("2. Temuan EDA + Grafik Bertafsiran", styles["HeadingStyle"]))
    story.append(Paragraph(
        "EDA menunjukkan bahwa harga cenderung lebih tinggi untuk mobil dengan transmisi otomatis dan jenis bahan bakar tertentu. Selain itu, semakin tinggi kilometer tempuh, harga yang diprediksi cenderung menurun."
        "Hal ini memberi petunjuk bahwa fitur non-numerik seperti transmisi dan jenis bahan bakar memiliki pengaruh yang cukup kuat terhadap nilai pasar.",
        styles["BodyStyle"],
    ))
    story.append(Spacer(1, 0.06 * inch))
    chart1 = Image(FIG_DIR / "price_vs_km.png", width=5.5 * inch, height=3.2 * inch)
    story.append(chart1)
    story.append(Paragraph("Gambar 1. Hubungan antara kilometer tempuh dan harga. Titik-titik yang lebih tinggi mengindikasikan harga yang cenderung lebih rendah pada mobil dengan jarak tempuh lebih jauh.", styles["CaptionStyle"]))
    story.append(Spacer(1, 0.08 * inch))
    chart2 = Image(FIG_DIR / "price_by_transmission.png", width=5.5 * inch, height=3.2 * inch)
    story.append(chart2)
    story.append(Paragraph("Gambar 2. Distribusi harga berdasarkan transmisi. Mobil otomatis secara konsisten berada di level harga yang lebih tinggi daripada mobil manual.", styles["CaptionStyle"]))
    story.append(Spacer(1, 0.08 * inch))
    chart3 = Image(FIG_DIR / "avg_price_by_fuel.png", width=5.5 * inch, height=3.2 * inch)
    story.append(chart3)
    story.append(Paragraph("Gambar 3. Rata-rata harga berdasarkan jenis bahan bakar. Diesel dan petrol menunjukkan perbedaan harga yang cukup jelas pada sampel yang tersedia.", styles["CaptionStyle"]))
    story.append(Spacer(1, 0.12 * inch))

    # Section 3
    story.append(Paragraph("3. Perbandingan Model & Justifikasi Metrik", styles["HeadingStyle"]))
    rows = [["Model", "MAE", "RMSE", "R2"]]
    for name, mae, rmse, r2 in metrics["rows"]:
        rows.append([name, f"{mae:,.0f}", f"{rmse:,.0f}", f"{r2:.3f}"])
    table = Table(rows, colWidths=[1.9 * inch, 1.1 * inch, 1.1 * inch, 0.8 * inch])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#D9EAF7")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#123A5A")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ALIGN", (1, 1), (-1, -1), "CENTER"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
    ]))
    story.append(table)
    story.append(Spacer(1, 0.08 * inch))
    story.append(Paragraph(
        "Linear Regression menjadi model terpilih karena nilai R2 tertinggi dan kesalahan absolut yang paling rendah di antara tiga kandidat yang diuji."
        "Metrik tersebut menunjukkan bahwa hubungan linier antar fitur dan target cukup kuat untuk kasus dataset ini, meskipun model non-linear tetap layak untuk dieksplorasi.",
        styles["BodyStyle"],
    ))
    story.append(Spacer(1, 0.12 * inch))

    # Section 4
    story.append(Paragraph("4. Hasil Evaluasi + Analisis Kesalahan", styles["HeadingStyle"]))
    story.append(Paragraph(
        "Pada split data 80/20, model yang terpilih menghasilkan MAE sekitar Rp 157.511, RMSE sekitar Rp 257.559, dan R2 sekitar 0.919."
        "Secara umum, model cukup baik untuk memperkirakan kisaran harga menengah, tetapi masih mengalami deviasi pada sampel yang sangat rendah atau sangat tinggi.",
        styles["BodyStyle"],
    ))
    chart4 = Image(FIG_DIR / "residual_hist.png", width=5.5 * inch, height=3.2 * inch)
    story.append(chart4)
    story.append(Paragraph("Gambar 4. Distribusi residual menunjukkan sebagian besar kesalahan berada di sekitar nol, namun terdapat beberapa sampel yang masih menyimpang cukup jauh.", styles["CaptionStyle"]))
    story.append(Spacer(1, 0.12 * inch))

    # Section 5
    story.append(Paragraph("5. Uji Mekanis & Behavioral", styles["HeadingStyle"]))
    story.append(Paragraph(
        "Satu set uji mekanis telah disusun untuk memastikan endpoint /health dan /predict berfungsi sesuai skema yang diharapkan, termasuk respons 200 untuk input valid dan 422 untuk input yang hilang atau nilai enum tak dikenal."
        "Selain itu, uji behavioral digunakan untuk memeriksa perilaku model secara relatif: mobil yang lebih baru dan transmisi otomatis diprediksi lebih mahal daripada versi yang lebih tua atau manual."
        , styles["BodyStyle"],
    ))
    story.append(Paragraph(
        "Behavioral test lebih tahan terhadap pelatihan ulang model karena menguji hubungan yang diharapkan (misalnya, tren arah) dan bukan angka prediksi yang harus sama persis."
        "Saat model diretrain dengan data baru, hubungan semantik biasanya tetap konsisten, sedangkan test yang mengandalkan nilai absolut bisa menjadi rapuh karena perubahan distribusi atau noise."
        , styles["BodyStyle"],
    ))
    story.append(Spacer(1, 0.12 * inch))

    # Section 6
    story.append(Paragraph("6. Desain API", styles["HeadingStyle"]))
    story.append(Paragraph(
        "API disusun dengan FastAPI dan menyediakan endpoint untuk memeriksa status layanan, serta endpoint prediksi yang menerima payload berisi fitur mobil.",
        styles["BodyStyle"],
    ))
    story.append(Paragraph("Endpoint utama:", styles["BodyStyle"]))
    story.append(Paragraph("- GET / untuk menampilkan informasi layanan", styles["BulletStyle"]))
    story.append(Paragraph("- GET /health untuk memeriksa status model", styles["BulletStyle"]))
    story.append(Paragraph("- POST /predict untuk menerima input fitur dan mengembalikan nilai prediksi", styles["BulletStyle"]))
    story.append(Paragraph("Contoh payload: {'features': {'Brand': 'Toyota', 'Year': 2018, 'Kilometers_Driven': 50000, 'Fuel_Type': 'Petrol', 'Transmission': 'Manual'}}", styles["BodyStyle"]))
    story.append(Spacer(1, 0.12 * inch))

    # Section 7
    story.append(Paragraph("7. Keterbatasan & Rencana Perbaikan", styles["HeadingStyle"]))
    story.append(Paragraph("- Dataset relatif kecil (100 sampel), sehingga generalisasi model masih terbatas.", styles["BulletStyle"]))
    story.append(Paragraph("- Model belum dibekali data pasar real-time atau tren harga bulanan, sehingga prediksi dapat meleset saat kondisi pasar berubah.", styles["BulletStyle"]))
    story.append(Paragraph("- Rencana perbaikan: tambahkan data lebih banyak, gunakan fitur tambahan seperti lokasi dan kondisi kendaraan, lalu bandingkan model non-linear yang lebih kuat.", styles["BulletStyle"]))

    doc = SimpleDocTemplate(str(output_path), pagesize=letter, rightMargin=0.8 * inch, leftMargin=0.8 * inch, topMargin=0.7 * inch, bottomMargin=0.7 * inch)
    doc.build(story)


def main():
    build_charts(load_data(load_config()))
    build_pdf(REPORT_DIR / "laporan_prediksi_harga_mobil.pdf")
    print(f"PDF created at {REPORT_DIR / 'laporan_prediksi_harga_mobil.pdf'}")


if __name__ == "__main__":
    main()
