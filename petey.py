#!/usr/bin/env python3
import io
import os
import base64
import shutil
import json
from typing import List, Dict, Tuple, Any
import cv2

from flask import Flask, request, jsonify, send_from_directory
from PIL import Image, ImageDraw
import numpy as np
import pandas as pd
from ultralytics import YOLO
import zipfile
from collections import Counter

# --- Flask App ---
app = Flask(__name__, static_folder="static", static_url_path="/")

# --- Load YOLO model ---
MODEL_PATH = os.path.expanduser("~/Downloads/finetune_beseech_continue/weights/best.pt")
yolo_model = YOLO(MODEL_PATH)

# --- Class names from the YOLO model ---
CLASS_NAMES = [yolo_model.names[i] for i in sorted(yolo_model.names.keys())] if isinstance(yolo_model.names, dict) else list(yolo_model.names)
CLASS_NAME_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}

# --------- Image Utils ---------
def pil_to_base64(pil_img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def numpy_to_pil(img_rgb: np.ndarray) -> Image.Image:
    return Image.fromarray(img_rgb)

def base64_to_numpy_image(b64_str: str) -> np.ndarray:
    # accepts raw base64 PNG/JPG bytes (no data: prefix)
    img_bytes = base64.b64decode(b64_str)
    arr = np.asarray(bytearray(img_bytes), dtype=np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Failed to decode image bytes")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb

def annotate_image_yolo(img: np.ndarray, detections: List[Dict[str, Any]]) -> Image.Image:
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)
        draw.text((x1, max(0, y1 - 10)), f"{det.get('label','') } {det.get('confidence',0):.2f}", fill=(255, 255, 0))
    return pil

def annotate_image_with_boxes(img: np.ndarray, boxes: List[Dict[str, Any]]) -> Image.Image:
    # boxes are dicts with x1,y1,x2,y2 and optional label/confidence
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    for b in boxes:
        x1, y1, x2, y2 = int(b["x1"]), int(b["y1"]), int(b["x2"]), int(b["y2"])
        draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=3)
        label = b.get("label", "")
        conf = b.get("confidence")
        text = f"{label} {conf:.2f}" if conf is not None else label
        if text.strip():
            draw.text((x1, max(0, y1 - 10)), text, fill=(255, 255, 0))
    return pil

def process_yolo(image_bytes: bytes, conf_threshold: float) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    arr = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return [], np.zeros((1,1,3), dtype=np.uint8)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    results = yolo_model.predict(img_rgb, imgsz=640, verbose=False, conf=conf_threshold)[0]
    detections = []

    img_area = img_rgb.shape[0] * img_rgb.shape[1]
    max_area = img_area * 0.15  # max 15% of image area

    raw_detections = []
    if hasattr(results, 'boxes') and results.boxes is not None:
        for box, conf, cls_id in zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls):
            x1, y1, x2, y2 = map(int, box)
            width, height = x2 - x1, y2 - y1
            area = max(0, width * height)
            raw_detections.append((x1, y1, x2, y2, width, height, area, float(conf), int(cls_id)))

    if not raw_detections:
        return [], img_rgb

    areas = [d[6] for d in raw_detections]
    mean_area = np.mean(areas)
    std_area = np.std(areas) if len(areas) > 1 else 0.0
    area_cutoff = mean_area + 2 * std_area

    for (x1, y1, x2, y2, width, height, area, conf, cls_id) in raw_detections:
        if area > max_area:
            continue
        if std_area > 0 and area > area_cutoff:
            continue
        detections.append({
            "bbox": [x1, y1, x2, y2],
            "confidence": conf,
            "label": CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id),
            "center_x": int((x1 + x2) / 2),
            "center_y": int((y1 + y2) / 2),
            "width": width,
            "height": height,
            "area": area,
        })

    return detections, img_rgb

# --------- Color counting ---------
def count_detections_by_color(img_rgb: np.ndarray, detections: List[Dict[str, Any]], color_map: Dict[str, Tuple[int,int,int]]) -> Dict[str, int]:
    counts = Counter({color: 0 for color in color_map.keys()})
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        # clamp
        h, w = img_rgb.shape[:2]
        x1c, y1c, x2c, y2c = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        crop = img_rgb[y1c:y2c, x1c:x2c]
        if crop.size == 0:
            continue
        mean_color = crop.mean(axis=(0,1))
        closest_color = min(color_map.keys(), key=lambda c: np.linalg.norm(np.array(color_map[c]) - mean_color))
        counts[closest_color] += 1
    return dict(counts)

# --------- Manual box processing helper ---------
def process_manual_boxes_from_bytes(image_bytes: bytes, boxes: List[Dict[str, Any]], user_colors: List[str]=None) -> Dict[str, Any]:
    """
    boxes: list of dicts with keys x1,y1,x2,y2 and optional label_id/label
    returns counts, annotated image b64, per-box metadata, and color counts if requested
    """
    arr = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return {"error": "Could not decode image"}
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    per_box = []
    detections = []
    h, w = img_rgb.shape[:2]
    for b in boxes:
        x1 = int(max(0, min(w, b.get("x1", 0))))
        y1 = int(max(0, min(h, b.get("y1", 0))))
        x2 = int(max(0, min(w, b.get("x2", x1+1))))
        y2 = int(max(0, min(h, b.get("y2", y1+1))))
        width = max(1, x2 - x1)
        height = max(1, y2 - y1)
        area = width * height
        label_id = b.get("label_id")
        label = b.get("label") if b.get("label") else (CLASS_NAMES[label_id] if label_id is not None and label_id < len(CLASS_NAMES) else "")
        det = {
            "bbox": [x1, y1, x2, y2],
            "width": width,
            "height": height,
            "area": area,
            "label": label,
            "label_id": label_id
        }
        detections.append(det)
        per_box.append(det)

    annotated_pil = annotate_image_with_boxes(img_rgb, [{"x1":d["bbox"][0],"y1":d["bbox"][1],"x2":d["bbox"][2],"y2":d["bbox"][3],"label":d.get("label","")} for d in detections])
    annotated_b64 = pil_to_base64(annotated_pil)

    color_counts = {}
    if user_colors:
        color_map = {
            "red": (255,0,0),
            "green": (0,255,0),
            "blue": (0,0,255),
            "yellow": (255,255,0),
            "orange": (255,165,0)
        }
        selected_map = {c: color_map[c] for c in user_colors if c in color_map}
        if selected_map:
            # transform detections to the format expected by count_detections_by_color
            dets_for_count = []
            for d in detections:
                dets_for_count.append({"bbox": d["bbox"]})
            color_counts = count_detections_by_color(img_rgb, dets_for_count, selected_map)

    return {
        "manual_count": len(detections),
        "per_box": per_box,
        "annotated_image_base64": annotated_b64,
        "color_counts": color_counts
    }

# --------- Core Processing ---------
def process_single_image(image_bytes: bytes, params: Dict[str, Any], reference_bytes: List[Tuple[str, bytes]]) -> Dict[str, Any]:
    conf_threshold = float(params.get("confidence", 0.25))
    detections, img_rgb = process_yolo(image_bytes, conf_threshold)
    annotated = annotate_image_yolo(img_rgb, detections)
    annotated_b64 = pil_to_base64(annotated)

    # also include original image bytes as base64 so frontend can edit original
    try:
        original_pil = Image.fromarray(img_rgb)
        original_b64 = pil_to_base64(original_pil)
    except Exception:
        original_b64 = annotated_b64

    color_counts = {}
    user_colors = params.get("colors")
    if user_colors:
        color_map = {
            "red": (255,0,0),
            "green": (0,255,0),
            "blue": (0,0,255),
            "yellow": (255,255,0),
            "orange": (255,165,0)
        }
        selected_map = {c: color_map[c] for c in user_colors if c in color_map}
        color_counts = count_detections_by_color(img_rgb, detections, selected_map)

    return {
        "detections": detections,
        "annotated_image_base64": annotated_b64,
        "original_image_base64": original_b64,
        "color_counts": color_counts
    }

# --------- Filesystem helpers for supplemental data ---------
BASE_SUPP = os.path.abspath("supplemental")
PENDING_IMAGES = os.path.join(BASE_SUPP, "pending", "images")
PENDING_LABELS = os.path.join(BASE_SUPP, "pending", "labels")
TRAIN_IMAGES = os.path.join(BASE_SUPP, "train", "images")
TRAIN_LABELS = os.path.join(BASE_SUPP, "train", "labels")

def ensure_dirs():
    for p in (PENDING_IMAGES, PENDING_LABELS, TRAIN_IMAGES, TRAIN_LABELS):
        os.makedirs(p, exist_ok=True)
ensure_dirs()

def sanitize_filename(name: str) -> str:
    # remove path components, spaces -> underscores, basic sanitize
    base = os.path.basename(name)
    base = base.replace(" ", "_")
    return "".join(c for c in base if c.isalnum() or c in "._-").strip()

def save_yolo_label_file(label_path: str, boxes: List[Dict[str, Any]], img_w: int, img_h: int):
    """
    Write a YOLO-format .txt: each line: class_id x_center y_center width height (normalized 0..1)
    """
    lines = []
    for b in boxes:
        cls = int(b.get("label_id", 0))
        x1, y1, x2, y2 = b["x1"], b["y1"], b["x2"], b["y2"]
        # clamp
        x1 = max(0, min(img_w, x1))
        y1 = max(0, min(img_h, y1))
        x2 = max(0, min(img_w, x2))
        y2 = max(0, min(img_h, y2))
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        x_center = (x1 + x2) / 2.0
        y_center = (y1 + y2) / 2.0
        x_center_n = x_center / img_w
        y_center_n = y_center / img_h
        w_n = bw / img_w
        h_n = bh / img_h
        lines.append(f"{cls} {x_center_n:.6f} {y_center_n:.6f} {w_n:.6f} {h_n:.6f}")
    with open(label_path, "w") as fh:
        fh.write("\n".join(lines))

# --------- Routes ---------
@app.route("/")
def root():
    # serve static/index.html from your static folder
    return send_from_directory(app.static_folder, "index.html")

@app.get("/api/classes")
def api_classes():
    return jsonify({"classes": CLASS_NAMES})

@app.post("/api/process")
def api_process():
    if "image" not in request.files:
        return jsonify({"error": "Missing image"}), 400

    image_file = request.files["image"]
    conf_threshold = request.form.get("confidence", 0.25)
    ref_files = request.files.getlist("references")
    references: List[Tuple[str, bytes]] = [(rf.filename, rf.read()) for rf in ref_files]

    color_param = request.form.get("colors")
    colors = []
    if color_param:
        try:
            colors = list(eval(color_param)) if isinstance(color_param, str) else color_param
        except:
            colors = []

    out = process_single_image(image_file.read(), {"confidence": conf_threshold, "colors": colors}, references)
    return jsonify(out)

@app.post("/api/batch")
def api_batch():
    files = request.files.getlist("images")
    if not files:
        return jsonify({"error": "No images uploaded"}), 400

    conf_threshold = request.form.get("confidence", 0.25)
    ref_files = request.files.getlist("references")
    references: List[Tuple[str, bytes]] = [(rf.filename, rf.read()) for rf in ref_files]

    color_param = request.form.get("colors")
    colors = []
    if color_param:
        try:
            colors = list(eval(color_param)) if isinstance(color_param, str) else color_param
        except:
            colors = []

    rows = []
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w") as zf:
        for f in files:
            try:
                image_bytes = f.read()
                result = process_single_image(image_bytes, {"confidence": conf_threshold, "colors": colors}, references)
                total = len(result["detections"])
                rows.append({
                    "image_name": sanitize_filename(f.filename),
                    "total_features": total,
                    "detections": result["detections"],
                    "color_counts": result["color_counts"],
                    "annotated_image_base64": result["annotated_image_base64"],
                    "original_image_base64": result.get("original_image_base64")
                })
                img_data = base64.b64decode(result["annotated_image_base64"])
                zf.writestr(sanitize_filename(f.filename), img_data)
            except Exception as e:
                rows.append({
                    "image_name": sanitize_filename(f.filename),
                    "error": str(e),
                    "detections": [],
                    "color_counts": {}
                })

    zip_buf.seek(0)
    csv_df = pd.DataFrame(rows)
    csv_bytes = csv_df.to_csv(index=False).encode("utf-8")
    csv_b64 = base64.b64encode(csv_bytes).decode("utf-8")
    zip_b64 = base64.b64encode(zip_buf.read()).decode("utf-8")

    return jsonify({
        "results": rows,
        "csv_base64": csv_b64,
        "zip_base64": zip_b64
    })

@app.post("/api/count_manual")
def api_count_manual():
    """
    Accepts JSON:
    {
      "image_b64": "...",   # base64 PNG/JPG bytes (no data: prefix)
      "boxes": [ {"x1":..,"y1":..,"x2":..,"y2":..,"label_id":0}, ... ],
      "colors": ["red","blue"]   # optional
    }
    Returns:
      manual_count, per_box, annotated_image_base64, color_counts
    """
    try:
        payload = request.get_json(force=True)
    except Exception as e:
        return jsonify({"error": "Invalid JSON payload"}), 400

    if not payload:
        return jsonify({"error": "Empty payload"}), 400

    image_b64 = payload.get("image_b64")
    boxes = payload.get("boxes", [])
    colors = payload.get("colors", None)

    if not image_b64:
        return jsonify({"error": "Missing image_b64"}), 400

    try:
        image_bytes = base64.b64decode(image_b64)
    except Exception as e:
        return jsonify({"error": "Invalid base64 image"}), 400

    try:
        resp = process_manual_boxes_from_bytes(image_bytes, boxes, user_colors=colors)
    except Exception as e:
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500

    return jsonify(resp)

@app.post("/api/save_correction")
def api_save_correction():
    """
    Expects JSON:
    {
      "image_name": "...",
      "image_b64": "....",  # base64 PNG (no data: prefix)
      "image_width": 1234,
      "image_height": 987,
      "boxes": [
         {"x1":..,"y1":..,"x2":..,"y2":..,"label_id": 0}, ...
      ],
      "colors": ["red","blue"]  # optional
    }
    Writes:
      supplemental/pending/images/<image_name>  (png)
      supplemental/pending/labels/<basename>.txt (YOLO format)
    Also returns manual counts and annotated image b64 so frontend can immediately display counts.
    """
    try:
        payload = request.get_json(force=True)
    except Exception as e:
        return jsonify({"error": "Invalid JSON payload"}), 400

    if not payload:
        return jsonify({"error": "Empty payload"}), 400

    image_name = sanitize_filename(payload.get("image_name", "unnamed.png"))
    image_b64 = payload.get("image_b64")
    image_width = int(payload.get("image_width", 0))
    image_height = int(payload.get("image_height", 0))
    boxes = payload.get("boxes", [])
    colors = payload.get("colors", None)

    if not image_b64:
        return jsonify({"error": "Missing image_b64"}), 400

    # ensure extension .png if not present
    if not os.path.splitext(image_name)[1]:
        image_name = image_name + ".png"
    elif os.path.splitext(image_name)[1].lower() not in ('.png', '.jpg', '.jpeg'):
        # force png extension for saved image
        image_name = os.path.splitext(image_name)[0] + ".png"

    try:
        image_bytes = base64.b64decode(image_b64)
    except Exception as e:
        return jsonify({"error": "Invalid base64 image"}), 400

    image_path = os.path.join(PENDING_IMAGES, image_name)
    with open(image_path, "wb") as fh:
        fh.write(image_bytes)

    # YOLO label path
    label_basename = os.path.splitext(image_name)[0] + ".txt"
    label_path = os.path.join(PENDING_LABELS, label_basename)
    try:
        if image_width <= 0 or image_height <= 0:
            # try to infer dims from saved image
            img = Image.open(io.BytesIO(image_bytes))
            image_width, image_height = img.size

        save_yolo_label_file(label_path, boxes, image_width, image_height)
    except Exception as e:
        return jsonify({"error": f"Failed to save labels: {str(e)}"}), 500

    # Also compute manual counts and annotated image for immediate frontend feedback
    try:
        manual_resp = process_manual_boxes_from_bytes(image_bytes, boxes, user_colors=colors)
    except Exception as e:
        manual_resp = {"error": f"Failed to compute manual counts: {str(e)}"}

    return jsonify({
        "saved_image": image_path,
        "saved_label_file": label_path,
        "manual_result": manual_resp
    })

@app.post("/api/finalize_supplemental")
def api_finalize_supplemental():
    """
    Move files from supplemental/pending -> supplemental/train
    """
    moved = 0
    # images
    for src_dir, dst_dir in [(PENDING_IMAGES, TRAIN_IMAGES), (PENDING_LABELS, TRAIN_LABELS)]:
        for fname in os.listdir(src_dir):
            src = os.path.join(src_dir, fname)
            dst = os.path.join(dst_dir, fname)
            try:
                shutil.move(src, dst)
                moved += 1
            except Exception as e:
                app.logger.exception(f"Could not move {src} -> {dst}: {e}")
    return jsonify({"moved": moved})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=True)
