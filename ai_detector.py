"""
High-Performance AI Vision Detector using YOLOv8 ONNX Runtime for Foscam IP Cameras.
Optimized for AMD Ryzen CPU vectorized inference without PyTorch dependencies.
"""

import io
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger("ai_detector")
logging.basicConfig(level=logging.INFO)

MODEL_PATH = Path("C:/app camara/models/yolov8n.onnx")

COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
    'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
    'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
    'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
    'teddy bear', 'hair drier', 'toothbrush'
]

CLASS_GROUPS = {
    "person": ["person"],
    "vehicle": ["car", "motorcycle", "bicycle", "bus", "truck"],
    "pet": ["cat", "dog", "bird", "horse"]
}

def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.45) -> List[int]:
    """Non-Maximum Suppression rápida en NumPy."""
    if len(boxes) == 0:
        return []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)

        inds = np.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]

    return keep

class AIDetector:
    def __init__(self, model_path: Path = MODEL_PATH):
        self.model_path = model_path
        self.session: Optional[ort.InferenceSession] = None
        self.input_name = "images"
        self.is_loaded = False
        self._load_model()

    def _load_model(self):
        if not self.model_path.exists():
            logger.error(f"Archivo de modelo YOLOv8 no encontrado en: {self.model_path}")
            return

        try:
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            opts.intra_op_num_threads = 4
            self.session = ort.InferenceSession(str(self.model_path), sess_options=opts, providers=['CPUExecutionProvider'])
            self.input_name = self.session.get_inputs()[0].name
            self.is_loaded = True
            logger.info("Modelo YOLOv8n ONNX cargado exitosamente.")
        except Exception as e:
            logger.error(f"Error cargando modelo YOLOv8n: {e}")

    def detect(
        self,
        image: Image.Image,
        conf_threshold: float = 0.55,
        allowed_classes: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Ejecuta inferencia de detección sobre una imagen PIL.
        Retorna lista de objetos detectados con coordenadas normalizadas y píxel.
        """
        if not self.is_loaded or not self.session:
            return []

        orig_w, orig_h = image.size

        # 1. Preprocesamiento: redimensionar a 640x640 y normalizar
        img_resized = image.resize((640, 640))
        arr = np.array(img_resized, dtype=np.float32) / 255.0
        blob = np.transpose(arr, (2, 0, 1))[np.newaxis, ...]

        # 2. Inferencia en ONNX Runtime
        t0 = time.time()
        outputs = self.session.run(None, {self.input_name: blob})
        inference_time_ms = round((time.time() - t0) * 1000, 1)

        # 3. Postprocesamiento de cajas candidatas: shape (8400, 84)
        out = outputs[0][0].T
        boxes_raw = out[:, :4]
        scores_raw = out[:, 4:]

        class_ids = np.argmax(scores_raw, axis=1)
        confidences = np.max(scores_raw, axis=1)

        mask = confidences >= conf_threshold
        if not np.any(mask):
            return []

        boxes_filt = boxes_raw[mask]
        conf_filt = confidences[mask]
        class_filt = class_ids[mask]

        # Convertir cajas (cx, cy, w, h) a (x1, y1, x2, y2) en escala 640
        cx = boxes_filt[:, 0]
        cy = boxes_filt[:, 1]
        w = boxes_filt[:, 2]
        h = boxes_filt[:, 3]
        x1_640 = cx - w / 2
        y1_640 = cy - h / 2
        x2_640 = cx + w / 2
        y2_640 = cy + h / 2
        boxes_xyxy_640 = np.stack([x1_640, y1_640, x2_640, y2_640], axis=1)

        # 4. Non-Maximum Suppression (NMS)
        keep_indices = nms(boxes_xyxy_640, conf_filt, iou_threshold=0.45)
        if not keep_indices:
            return []

        scale_x = orig_w / 640.0
        scale_y = orig_h / 640.0

        detections = []
        for idx in keep_indices:
            cid = int(class_filt[idx])
            cname = COCO_CLASSES[cid] if cid < len(COCO_CLASSES) else f"class_{cid}"
            confidence = float(conf_filt[idx])

            # Filtrar si se especificaron clases permitidas
            if allowed_classes and cname not in allowed_classes:
                continue

            bx1 = max(0, int(boxes_xyxy_640[idx][0] * scale_x))
            by1 = max(0, int(boxes_xyxy_640[idx][1] * scale_y))
            bx2 = min(orig_w, int(boxes_xyxy_640[idx][2] * scale_x))
            by2 = min(orig_h, int(boxes_xyxy_640[idx][3] * scale_y))

            center_x = (bx1 + bx2) // 2
            center_y = (by1 + by2) // 2
            norm_cx = round(center_x / orig_w, 3)
            norm_cy = round(center_y / orig_h, 3)

            detections.append({
                "class_id": cid,
                "class_name": cname,
                "confidence": round(confidence, 2),
                "box": [bx1, by1, bx2, by2],
                "center": (center_x, center_y),
                "norm_center": (norm_cx, norm_cy),
                "width": bx2 - bx1,
                "height": by2 - by1,
                "inference_ms": inference_time_ms
            })

        return detections

    def draw_detections(self, image: Image.Image, detections: List[Dict[str, Any]]) -> Image.Image:
        """Dibuja recuadros delimitadores y etiquetas con estética moderna en la imagen."""
        annotated = image.copy()
        draw = ImageDraw.Draw(annotated)

        for det in detections:
            x1, y1, x2, y2 = det["box"]
            cname = det["class_name"]
            conf = int(det["confidence"] * 100)

            # Color según categoría
            if cname == "person":
                color = (16, 185, 129)  # Verde esmeralda
            elif cname in CLASS_GROUPS["vehicle"]:
                color = (6, 182, 212)   # Cyan eléctrico
            elif cname in CLASS_GROUPS["pet"]:
                color = (245, 158, 11)  # Ámbar
            else:
                color = (139, 92, 246)  # Púrpura

            # Rectángulo con grosor 3
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

            # Cartel con texto
            label = f"{cname.upper()} {conf}%"
            # Estimación del tamaño del cartel
            text_w = len(label) * 8 + 10
            text_h = 18
            draw.rectangle([x1, max(0, y1 - text_h), x1 + text_w, max(0, y1)], fill=color)
            draw.text((x1 + 5, max(0, y1 - text_h) + 2), label, fill=(255, 255, 255))

        return annotated

    def annotate_and_get_jpeg(self, image: Image.Image, detections: List[Dict[str, Any]]) -> bytes:
        """Retorna la imagen anotada directamente como bytes JPEG."""
        annotated = self.draw_detections(image, detections)
        buf = io.BytesIO()
        annotated.save(buf, format="JPEG", quality=85)
        return buf.getvalue()

ai_detector = AIDetector()
