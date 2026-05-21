"""
Wafer AI Inspection System
Upload -> Analyze -> Detect Defect | Dashboard | PDF Reports
"""
from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from flask import Flask, flash, jsonify, redirect, render_template, request, send_from_directory, url_for
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas as pdf_canvas
from werkzeug.utils import secure_filename

plt.switch_backend("Agg")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "images"
REPORT_FOLDER = BASE_DIR / "reports"
HEATMAP_FOLDER = BASE_DIR / "static" / "heatmaps"
GRAPH_FOLDER = BASE_DIR / "static" / "graphs"
ANALYTICS_FILE = REPORT_FOLDER / "analytics.json"
MODEL_FILE = BASE_DIR / "model" / "wafer_model.h5"
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
LABELS = ["Crack", "Scratch", "Spot", "Contamination", "Normal wafer"]
IMG_SIZE = 128

app = Flask(__name__, static_folder=str(BASE_DIR / "static"))
app.secret_key = "wafer-ai-dev-key-change-in-production"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

for folder in (UPLOAD_FOLDER, REPORT_FOLDER, HEATMAP_FOLDER, GRAPH_FOLDER, BASE_DIR / "model", BASE_DIR / "dataset"):
    folder.mkdir(parents=True, exist_ok=True)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def load_analytics() -> dict:
    if not ANALYTICS_FILE.exists():
        analytics = {
            "total_wafers": 0,
            "defect_counts": {label: 0 for label in LABELS},
            "runs": [],
        }
        save_analytics(analytics)
        return analytics

    with ANALYTICS_FILE.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    for label in LABELS:
        data.setdefault("defect_counts", {})[label] = data.get("defect_counts", {}).get(label, 0)
    data.setdefault("runs", [])
    return data


def save_analytics(analytics: dict) -> None:
    with ANALYTICS_FILE.open("w", encoding="utf-8") as handle:
        json.dump(analytics, handle, indent=2)


def update_analytics(prediction: str, confidence: float) -> dict:
    analytics = load_analytics()
    analytics["total_wafers"] += 1
    analytics["defect_counts"].setdefault(prediction, 0)
    analytics["defect_counts"][prediction] += 1
    analytics["runs"].append(
        {
            "timestamp": utc_now().isoformat(),
            "prediction": prediction,
            "confidence": round(confidence * 100, 1),
        }
    )
    analytics["runs"] = analytics["runs"][-50:]
    save_analytics(analytics)
    return analytics


@lru_cache(maxsize=1)
def get_keras_model():
    if not MODEL_FILE.exists():
        return None
    try:
        import os

        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        from tensorflow.keras.models import load_model

        model = load_model(str(MODEL_FILE), compile=False)
        logger.info("Loaded Keras model from %s", MODEL_FILE)
        return model
    except Exception as exc:
        logger.warning("Could not load model: %s", exc)
        return None


def preprocess_for_model(image: np.ndarray) -> np.ndarray:
    resized = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    batch = np.expand_dims(rgb.astype(np.float32) / 255.0, axis=0)
    return batch


def predict_with_model(image: np.ndarray) -> Optional[tuple[str, float]]:
    model = get_keras_model()
    if model is None:
        return None
    batch = preprocess_for_model(image)
    probs = model.predict(batch, verbose=0)[0]
    idx = int(np.argmax(probs))
    return LABELS[idx], float(probs[idx])


def predict_heuristic(image: np.ndarray) -> tuple[str, float]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 80, 180)
    edge_ratio = float(np.count_nonzero(edges)) / max(1, edges.size)
    std_dev = float(np.std(gray))

    if edge_ratio < 0.002 and std_dev < 25:
        label = "Normal wafer"
        confidence = 0.88
    elif edge_ratio < 0.007:
        label = "Scratch"
        confidence = min(0.92, 0.7 + edge_ratio * 25)
    elif edge_ratio < 0.014:
        label = "Spot"
        confidence = min(0.9, 0.68 + edge_ratio * 18)
    elif edge_ratio < 0.03 or std_dev > 40:
        label = "Contamination" if std_dev > 35 else "Spot"
        confidence = min(0.9, 0.65 + edge_ratio * 12)
    else:
        label = "Crack"
        confidence = min(0.95, 0.72 + edge_ratio * 8)
    return label, confidence


def predict_defect(image: np.ndarray) -> tuple[str, float, str]:
    """Returns label, confidence, method used."""
    result = predict_with_model(image)
    if result:
        return result[0], result[1], "AI Model"
    label, confidence = predict_heuristic(image)
    return label, confidence, "Computer Vision (train model for AI)"


def build_defect_mask(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 60, 160)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    defect = cv2.bitwise_or(edges, cv2.bitwise_not(thresh))
    kernel = np.ones((3, 3), np.uint8)
    defect = cv2.dilate(defect, kernel, iterations=1)
    return defect


def create_heatmap(image: np.ndarray, output_path: Path) -> str:
    mask = build_defect_mask(image)
    heat = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image, 0.65, heat, 0.35, 0)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) < 30:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)
    if not cv2.imwrite(str(output_path), overlay):
        raise IOError(f"Failed to write heatmap to {output_path}")
    return output_path.name


def _save_figure(fig, path: Path) -> None:
    canvas = FigureCanvas(fig)
    canvas.draw()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def create_dashboard_graphs(analytics: dict) -> dict:
    labels = list(analytics["defect_counts"].keys())
    values = [analytics["defect_counts"].get(label, 0) for label in labels]
    dist_path = GRAPH_FOLDER / "defect_distribution.png"

    fig, ax = plt.subplots(figsize=(6, 4))
    if sum(values) > 0:
        ax.pie(values, labels=labels, autopct="%1.0f%%", startangle=140)
        ax.set_title("Defect Type Distribution")
    else:
        ax.text(0.5, 0.5, "No scans yet", ha="center", va="center", fontsize=14)
        ax.set_title("Defect Type Distribution")
        ax.axis("off")
    _save_figure(fig, dist_path)

    runs = analytics.get("runs", [])
    accuracy_path = GRAPH_FOLDER / "confidence_trend.png"
    fig, ax = plt.subplots(figsize=(8, 4))
    if runs:
        recent = runs[-10:]
        timestamps = [run["timestamp"][11:19] if len(run["timestamp"]) > 19 else run["timestamp"] for run in recent]
        confidences = [run["confidence"] for run in recent]
        ax.plot(range(len(confidences)), confidences, marker="o", color="#2563eb", linewidth=2)
        ax.set_xticks(range(len(timestamps)))
        ax.set_xticklabels(timestamps, rotation=35, ha="right")
        ax.set_ylabel("Confidence (%)")
        ax.set_xlabel("Recent scans (time)")
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
        ax.set_title("Model Confidence Trend")
    else:
        ax.text(0.5, 0.5, "Run inspections to see trends", ha="center", va="center", fontsize=12)
        ax.set_title("Model Confidence Trend")
        ax.axis("off")
    _save_figure(fig, accuracy_path)

    bar_path = GRAPH_FOLDER / "defect_counts_bar.png"
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#ef4444", "#f97316", "#eab308", "#8b5cf6", "#22c55e"]
    ax.bar(labels, values, color=colors[: len(labels)])
    ax.set_ylabel("Count")
    ax.set_title("Defect Counts by Type")
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    _save_figure(fig, bar_path)

    return {
        "distribution": f"graphs/{dist_path.name}",
        "confidence": f"graphs/{accuracy_path.name}",
        "bar": f"graphs/{bar_path.name}",
    }


def defective_stats(analytics: dict) -> tuple[int, float]:
    total = analytics.get("total_wafers", 0)
    normal = analytics.get("defect_counts", {}).get("Normal wafer", 0)
    defective = max(0, total - normal)
    pct = (defective / total * 100) if total else 0.0
    return defective, pct


def create_pdf_report(
    report_path: Path,
    image_path: Path,
    heatmap_path: Path,
    prediction: str,
    confidence: float,
    method: str,
    analytics: dict,
) -> str:
    c = pdf_canvas.Canvas(str(report_path), pagesize=letter)
    width, height = letter
    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, height - 60, "Wafer Inspection Report")
    c.setFont("Helvetica", 12)
    c.drawString(50, height - 90, f"Generated: {utc_now().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    c.drawString(50, height - 110, f"Prediction: {prediction}")
    c.drawString(50, height - 130, f"Confidence: {confidence * 100:.1f}%")
    c.drawString(50, height - 150, f"Detection method: {method}")
    c.drawString(50, height - 170, f"Total wafers tested: {analytics['total_wafers']}")
    _, defect_pct = defective_stats(analytics)
    c.drawString(50, height - 190, f"Defective rate: {defect_pct:.1f}%")
    c.drawString(50, height - 210, "Defect distribution:")

    y = height - 230
    for label, count in analytics["defect_counts"].items():
        c.drawString(70, y, f"- {label}: {count}")
        y -= 16

    try:
        c.drawImage(ImageReader(str(image_path)), 50, 280, width=240, height=180, preserveAspectRatio=True)
        c.drawString(50, 265, "Uploaded wafer image")
        c.drawImage(ImageReader(str(heatmap_path)), 320, 280, width=240, height=180, preserveAspectRatio=True)
        c.drawString(320, 265, "Defect heatmap")
    except Exception as exc:
        logger.warning("PDF image embed failed: %s", exc)
        c.drawString(50, 280, "Images could not be embedded in the PDF.")

    c.showPage()
    c.save()
    return report_path.name


def read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Could not read image file: {path.name}")
    if image.size == 0:
        raise ValueError("Image file is empty or corrupted")
    return image


def analyze_image(image_path: Path) -> dict:
    image = read_image(image_path)
    prediction, confidence, method = predict_defect(image)
    heatmap_name = f"heatmap_{uuid.uuid4().hex}.png"
    heatmap_path = HEATMAP_FOLDER / heatmap_name
    create_heatmap(image, heatmap_path)
    analytics = update_analytics(prediction, confidence)
    graphs = create_dashboard_graphs(analytics)
    report_name = f"report_{utc_now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.pdf"
    report_path = REPORT_FOLDER / report_name
    create_pdf_report(
        report_path, image_path, heatmap_path, prediction, confidence, method, analytics
    )
    defective, defective_pct = defective_stats(analytics)

    return {
        "uploaded_image": image_path.name,
        "heatmap": heatmap_name,
        "report": report_name,
        "prediction": prediction,
        "confidence": round(confidence * 100, 1),
        "method": method,
        "analytics": analytics,
        "graphs": graphs,
        "defective_count": defective,
        "defective_pct": round(defective_pct, 1),
    }


def save_uploaded_image(file) -> str:
    filename = secure_filename(file.filename or "")
    if not filename or not allowed_file(filename):
        raise ValueError("Unsupported file format. Use JPG, PNG, BMP, or TIFF.")

    suffix = Path(filename).suffix.lower()
    target_name = f"upload_{uuid.uuid4().hex}{suffix}"
    target_path = UPLOAD_FOLDER / target_name
    file.save(str(target_path))
    if not target_path.exists() or target_path.stat().st_size == 0:
        raise ValueError("Upload failed or file is empty.")
    return target_name


@app.route("/images/<path:filename>")
def uploaded_image(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route("/reports/<path:filename>")
def download_report(filename):
    return send_from_directory(REPORT_FOLDER, filename, as_attachment=True)


@app.route("/health")
def health():
    return jsonify(
        status="ok",
        model_loaded=get_keras_model() is not None,
        model_path=str(MODEL_FILE),
    )


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    analytics = load_analytics()
    graphs = create_dashboard_graphs(analytics)
    defective, defective_pct = defective_stats(analytics)
    model_ready = MODEL_FILE.exists()

    if request.method == "POST":
        file = request.files.get("wafer_image")
        if not file or not file.filename:
            flash("Please upload a wafer image before analysis.")
            return redirect(url_for("index"))

        if not allowed_file(file.filename):
            flash("Unsupported file format. Use JPG, PNG, BMP, or TIFF.")
            return redirect(url_for("index"))

        try:
            image_name = save_uploaded_image(file)
            result = analyze_image(UPLOAD_FOLDER / image_name)
            analytics = result["analytics"]
            graphs = result["graphs"]
            defective = result["defective_count"]
            defective_pct = result["defective_pct"]
        except ValueError as exc:
            flash(str(exc))
            return redirect(url_for("index"))
        except Exception as exc:
            logger.exception("Analysis failed")
            flash(f"Unable to analyze image: {exc}")
            return redirect(url_for("index"))

    return render_template(
        "index.html",
        analytics=analytics,
        result=result,
        graphs=graphs,
        labels=LABELS,
        defective_count=defective,
        defective_pct=defective_pct,
        model_ready=model_ready,
    )


if __name__ == "__main__":
    if not MODEL_FILE.exists():
        logger.info(
            "Model not found at %s. Run: python train_model.py",
            MODEL_FILE,
        )
    app.run(host="0.0.0.0", port=5000, debug=True)
