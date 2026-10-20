"""
High-Performance AI Vision Detector using YOLO (v8, 11, v10) ONNX Runtime for Foscam IP Cameras.
Optimized for AMD Ryzen CPU vectorized inference without PyTorch dependencies.
Includes Multi-Model Dynamic Switching, Letterbox scaling, Pre-NMS Class Filtering, and Smart Motion Gate.
"""

import io
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw

logger = logging.getLogger("ai_detector")
logging.basicConfig(level=logging.INFO)

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models" if (BASE_DIR / "models").exists() else Path("C:/app camara/models")
DEFAULT_MODEL_NAME = "yolov8n.onnx"
DEFAULT_MODEL_PATH = MODELS_DIR / DEFAULT_MODEL_NAME
FALLBACK_MODEL_PATH = Path("C:/app camara/models/yolov8n.onnx")
MODEL_PATH = DEFAULT_MODEL_PATH if DEFAULT_MODEL_PATH.exists() else FALLBACK_MODEL_PATH

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

CLASS_NAME_TO_ID = {name: idx for idx, name in enumerate(COCO_CLASSES)}

CLASS_GROUPS = {
    "person": ["person"],
    "vehicle": ["car", "motorcycle", "bicycle", "bus", "truck"],
    "pet": ["cat", "dog", "bird", "horse"]
}


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.45) -> List[int]:
    """Non-Maximum Suppression vectorizada en NumPy."""
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
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
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
    def __init__(self, model_path: Optional[Union[str, Path]] = None):
        self.models_dir = MODELS_DIR
        self.session: Optional[ort.InferenceSession] = None
        self.input_name = "images"
        self.is_loaded = False
        self.model_path: Path = Path(model_path) if model_path else MODEL_PATH
        self.active_model_name: str = self.model_path.name if self.model_path else DEFAULT_MODEL_NAME
        self.model_type: str = "yolo_nms"  # "yolo_nms" o "yolov10"

        # Pre-alocación de memoria para evitar garbage collection en cada frame
        self._input_blob = np.zeros((1, 3, 640, 640), dtype=np.float32)

        # Estado del Smart Motion Gate
        self._prev_motion_thumb: Optional[np.ndarray] = None
        self._last_detections: List[Dict[str, Any]] = []
        self._last_inference_ts: float = 0.0
        self._motion_pixel_diff_threshold: float = 3.2
        self._max_idle_skip_sec: float = 1.5

        self.load_model(self.model_path)

    def load_model(self, model_name_or_path: Optional[Union[str, Path]] = None) -> bool:
        """Carga o cambia dinámicamente el modelo YOLO en tiempo de ejecución."""
        if model_name_or_path:
            p = Path(model_name_or_path)
            target = p if p.is_absolute() else self.models_dir / model_name_or_path
        elif self.model_path and self.model_path.exists():
            target = self.model_path
        else:
            target = self.models_dir / DEFAULT_MODEL_NAME

        if not target.exists():
            logger.error(f"Archivo de modelo YOLO ONNX no encontrado en: {target}")
            return False

        try:
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            opts.intra_op_num_threads = 4
            opts.enable_mem_pattern = True
            opts.enable_cpu_mem_arena = True

            sess = ort.InferenceSession(
                str(target),
                sess_options=opts,
                providers=['CPUExecutionProvider']
            )

            # Auto-detectar arquitectura a partir de la firma de salida
            out_shape = sess.get_outputs()[0].shape
            if len(out_shape) == 3 and out_shape[1] == 300 and out_shape[2] == 6:
                detected_type = "yolov10"
            else:
                detected_type = "yolo_nms"

            self.session = sess
            self.model_path = target
            self.active_model_name = target.name
            self.input_name = sess.get_inputs()[0].name
            self.model_type = detected_type
            self.is_loaded = True

            logger.info(
                f"Modelo ONNX {self.active_model_name} ({self.model_type}) "
                f"cargado exitosamente desde {target}"
            )
            return True
        except Exception as e:
            logger.error(f"Error cargando modelo {target}: {e}")
            return False

    def get_available_models(self) -> List[Dict[str, Any]]:
        """Lista todos los modelos .onnx disponibles en el directorio models/."""
        models = []
        if self.models_dir.exists():
            for p in sorted(self.models_dir.glob("*.onnx")):
                size_mb = round(p.stat().st_size / (1024 * 1024), 2)
                models.append({
                    "name": p.name,
                    "path": str(p),
                    "size_mb": size_mb,
                    "is_active": (p.name == self.active_model_name)
                })
        return models

    def _has_scene_changed(self, image: Image.Image) -> bool:
        """
        Verifica en <0.1ms si hubo variación visual en la escena.
        Si no hubo movimiento y la última inferencia fue reciente (y sin objetos),
        permite ahorrar 100% del uso de CPU de ONNX.
        """
        now = time.time()
        thumb = np.asarray(
            image.resize((64, 64), Image.Resampling.NEAREST).convert("L"),
            dtype=np.int16
        )

        if self._prev_motion_thumb is None:
            self._prev_motion_thumb = thumb
            return True

        diff = float(np.mean(np.abs(thumb - self._prev_motion_thumb)))
        self._prev_motion_thumb = thumb

        # Si había objetos en el frame anterior o pasó el tiempo máximo, forzamos inferencia
        if self._last_detections or (now - self._last_inference_ts) >= self._max_idle_skip_sec:
            return True

        return diff >= self._motion_pixel_diff_threshold

    def _preprocess_letterbox(self, image: Image.Image) -> Tuple[np.ndarray, float, int, int]:
        """
        Redimensiona manteniendo la relación de aspecto original (Letterbox) a 640x640
        sobre un buffer pre-alocado para máxima precisión y velocidad.
        """
        orig_w, orig_h = image.size
        scale = min(640.0 / orig_w, 640.0 / orig_h)
        new_w, new_h = int(round(orig_w * scale)), int(round(orig_h * scale))
        pad_w = (640 - new_w) // 2
        pad_h = (640 - new_h) // 2

        if image.mode != "RGB":
            image = image.convert("RGB")

        img_resized = image.resize((new_w, new_h), Image.Resampling.BILINEAR)
        arr = np.asarray(img_resized, dtype=np.float32) * (1.0 / 255.0)

        # Rellenar con gris neutro (114/255 = 0.447) estándar de entrenamiento YOLO
        self._input_blob.fill(0.4470588)
        self._input_blob[0, :, pad_h:pad_h + new_h, pad_w:pad_w + new_w] = np.transpose(arr, (2, 0, 1))

        return self._input_blob, scale, pad_w, pad_h

    def detect(
        self,
        image: Image.Image,
        conf_threshold: float = 0.55,
        allowed_classes: Optional[List[str]] = None,
        use_motion_gate: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Ejecuta inferencia de detección sobre una imagen PIL.
        Retorna lista de objetos detectados con coordenadas normalizadas y píxel.
        """
        if not self.is_loaded or not self.session:
            return []

        # 0. Filtro rápido de escena estática cuando no hay actividad
        if use_motion_gate and not self._has_scene_changed(image):
            return self._last_detections

        orig_w, orig_h = image.size

        # 1. Preprocesamiento con Letterboxing sin distorsión
        blob, scale, pad_w, pad_h = self._preprocess_letterbox(image)
        inv_scale = 1.0 / scale

        # 2. Inferencia en ONNX Runtime
        t0 = time.time()
        outputs = self.session.run(None, {self.input_name: blob})
        self._last_inference_ts = time.time()
        inference_time_ms = round((self._last_inference_ts - t0) * 1000, 1)

        # 3. Postprocesamiento según arquitectura del modelo
        detections: List[Dict[str, Any]] = []

        if self.model_type == "yolov10":
            # Formato End-to-End NMS-Free: shape (1, 300, 6) -> [x1, y1, x2, y2, score, class_id]
            preds = outputs[0][0]
            scores = preds[:, 4]
            mask = scores >= conf_threshold
            if not np.any(mask):
                self._last_detections = []
                return []

            filt = preds[mask]
            for p in filt:
                cid = int(p[5])
                cname = COCO_CLASSES[cid] if cid < len(COCO_CLASSES) else f"class_{cid}"
                if allowed_classes and cname not in allowed_classes:
                    continue
                confidence = float(p[4])

                # Deshacer letterbox padding y escalar a dimensiones originales
                bx1 = max(0, int(round((p[0] - pad_w) * inv_scale)))
                by1 = max(0, int(round((p[1] - pad_h) * inv_scale)))
                bx2 = min(orig_w, int(round((p[2] - pad_w) * inv_scale)))
                by2 = min(orig_h, int(round((p[3] - pad_h) * inv_scale)))

                if bx2 <= bx1 or by2 <= by1:
                    continue

                center_x = (bx1 + bx2) // 2
                center_y = (by1 + by2) // 2
                detections.append({
                    "class_id": cid,
                    "class_name": cname,
                    "confidence": round(confidence, 2),
                    "box": [bx1, by1, bx2, by2],
                    "center": (center_x, center_y),
                    "norm_center": (round(center_x / orig_w, 3), round(center_y / orig_h, 3)),
                    "width": bx2 - bx1,
                    "height": by2 - by1,
                    "inference_ms": inference_time_ms,
                    "model": self.active_model_name
                })
        else:
            # Formato YOLOv8 / YOLO11: shape (1, 84, 8400)
            out = outputs[0][0].T
            boxes_raw = out[:, :4]
            scores_raw = out[:, 4:]

            # Pre-filtrado de clases ANTES de argmax/NMS para ahorrar CPU
            if allowed_classes:
                valid_cids = [CLASS_NAME_TO_ID[c] for c in allowed_classes if c in CLASS_NAME_TO_ID]
                if not valid_cids:
                    self._last_detections = []
                    return []
                scores_sub = scores_raw[:, valid_cids]
                sub_idx = np.argmax(scores_sub, axis=1)
                confidences = scores_sub[np.arange(scores_sub.shape[0]), sub_idx]
                class_ids = np.array(valid_cids, dtype=np.int32)[sub_idx]
            else:
                class_ids = np.argmax(scores_raw, axis=1)
                confidences = np.max(scores_raw, axis=1)

            mask = confidences >= conf_threshold
            if not np.any(mask):
                self._last_detections = []
                return []

            boxes_filt = boxes_raw[mask]
            conf_filt = confidences[mask]
            class_filt = class_ids[mask]

            # Convertir cajas (cx, cy, w, h) a (x1, y1, x2, y2) en espacio 640
            half_w = boxes_filt[:, 2] * 0.5
            half_h = boxes_filt[:, 3] * 0.5
            x1_640 = boxes_filt[:, 0] - half_w
            y1_640 = boxes_filt[:, 1] - half_h
            x2_640 = boxes_filt[:, 0] + half_w
            y2_640 = boxes_filt[:, 1] + half_h
            boxes_xyxy_640 = np.stack([x1_640, y1_640, x2_640, y2_640], axis=1)

            # 4. Non-Maximum Suppression (NMS)
            keep_indices = nms(boxes_xyxy_640, conf_filt, iou_threshold=0.45)
            if not keep_indices:
                self._last_detections = []
                return []

            for idx in keep_indices:
                cid = int(class_filt[idx])
                cname = COCO_CLASSES[cid] if cid < len(COCO_CLASSES) else f"class_{cid}"
                confidence = float(conf_filt[idx])

                # Deshacer letterbox padding y escalar a dimensiones originales
                bx1 = max(0, int(round((boxes_xyxy_640[idx][0] - pad_w) * inv_scale)))
                by1 = max(0, int(round((boxes_xyxy_640[idx][1] - pad_h) * inv_scale)))
                bx2 = min(orig_w, int(round((boxes_xyxy_640[idx][2] - pad_w) * inv_scale)))
                by2 = min(orig_h, int(round((boxes_xyxy_640[idx][3] - pad_h) * inv_scale)))

                if bx2 <= bx1 or by2 <= by1:
                    continue

                center_x = (bx1 + bx2) // 2
                center_y = (by1 + by2) // 2
                detections.append({
                    "class_id": cid,
                    "class_name": cname,
                    "confidence": round(confidence, 2),
                    "box": [bx1, by1, bx2, by2],
                    "center": (center_x, center_y),
                    "norm_center": (round(center_x / orig_w, 3), round(center_y / orig_h, 3)),
                    "width": bx2 - bx1,
                    "height": by2 - by1,
                    "inference_ms": inference_time_ms,
                    "model": self.active_model_name
                })

        self._last_detections = detections
        return detections

    def draw_detections(self, image: Image.Image, detections: List[Dict[str, Any]]) -> Image.Image:
        """Dibuja recuadros delimitadores y etiquetas con estética moderna en la imagen."""
        annotated = image.copy()
        draw = ImageDraw.Draw(annotated)

        for det in detections:
            x1, y1, x2, y2 = det["box"]
            cname = det["class_name"]
            conf = int(det["confidence"] * 100)

            if cname == "person":
                color = (16, 185, 129)
            elif cname in CLASS_GROUPS["vehicle"]:
                color = (6, 182, 212)
            elif cname in CLASS_GROUPS["pet"]:
                color = (245, 158, 11)
            else:
                color = (139, 92, 246)

            # Si es el blanco fijado en tracking, destacar con borde más grueso
            border_width = 4 if det.get("is_locked") else 2
            draw.rectangle([x1, y1, x2, y2], outline=color, width=border_width)

            lock_badge = " [LOCKED]" if det.get("is_locked") else ""
            track_badge = f"#{det['track_id']} " if "track_id" in det else ""
            label = f"{track_badge}{cname.upper()} {conf}%{lock_badge}"

            text_w = len(label) * 8 + 10
            text_h = 18
            y_top = max(0, y1 - text_h)
            draw.rectangle([x1, y_top, x1 + text_w, max(text_h, y1)], fill=color)
            draw.text((x1 + 5, y_top + 2), label, fill=(255, 255, 255))

        return annotated

    def annotate_and_get_jpeg(self, image: Image.Image, detections: List[Dict[str, Any]]) -> bytes:
        """Retorna la imagen anotada directamente como bytes JPEG optimizados."""
        annotated = self.draw_detections(image, detections)
        buf = io.BytesIO()
        annotated.save(buf, format="JPEG", quality=85, subsampling=1)
        return buf.getvalue()


ai_detector = AIDetector()
