"""
AI Vision & Autonomous PTZ Motor Auto-Tracking Controller for Foscam FI9821W V2.
Features ByteTrack-style IoU Multi-Object Tracking in pure NumPy and Persistent Target Locking.
"""

import io
import json
import time
import logging
import threading
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
from PIL import Image

from ai_detector import ai_detector, CLASS_GROUPS
from telegram_alerts import telegram_notifier

logger = logging.getLogger("ai_tracker")
logging.basicConfig(level=logging.INFO)

CONFIG_PATH = Path("C:/app camara/ai_tracker_config.json")


def box_iou(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """Calcula matriz de IoU entre dos conjuntos de cajas [N, 4] y [M, 4]."""
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)

    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    xx1 = np.maximum(boxes1[:, None, 0], boxes2[None, :, 0])
    yy1 = np.maximum(boxes1[:, None, 1], boxes2[None, :, 1])
    xx2 = np.minimum(boxes1[:, None, 2], boxes2[None, :, 2])
    yy2 = np.minimum(boxes1[:, None, 3], boxes2[None, :, 3])

    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    inter = w * h

    union = area1[:, None] + area2[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def match_boxes(boxes_tracks: np.ndarray, boxes_dets: np.ndarray, iou_thresh: float = 0.35):
    """Asociación voraz vectorizada entre cajas predichas y detecciones."""
    if len(boxes_tracks) == 0 or len(boxes_dets) == 0:
        return [], list(range(len(boxes_tracks))), list(range(len(boxes_dets)))

    ious = box_iou(boxes_tracks, boxes_dets)
    matched_tracks = set()
    matched_dets = set()
    matches = []

    # Ordenar por IoU descendente
    indices = np.dstack(np.unravel_index(np.argsort(-ious, axis=None), ious.shape))[0]
    for r, c in indices:
        if ious[r, c] < iou_thresh:
            break
        if r in matched_tracks or c in matched_dets:
            continue
        matched_tracks.add(r)
        matched_dets.add(c)
        matches.append((r, c))

    unmatched_tracks = [i for i in range(len(boxes_tracks)) if i not in matched_tracks]
    unmatched_dets = [j for j in range(len(boxes_dets)) if j not in matched_dets]
    return matches, unmatched_tracks, unmatched_dets


class TrackedObject:
    """Representa una trayectoria individual con predicción de velocidad y persistencia."""
    _next_id = 1

    def __init__(self, box: List[int], score: float, class_name: str, class_id: int):
        self.track_id = TrackedObject._next_id
        TrackedObject._next_id += 1
        self.box = np.array(box, dtype=np.float32)
        self.score = float(score)
        self.class_name = class_name
        self.class_id = class_id
        self.velocity = np.zeros(4, dtype=np.float32)
        self.hits = 1
        self.age = 1
        self.time_since_update = 0

    def predict(self) -> np.ndarray:
        """Predice la posición de la caja en el siguiente frame según su velocidad."""
        pred_box = self.box + self.velocity * 0.6
        return pred_box

    def update(self, new_box: List[int], new_score: float):
        """Actualiza la caja y recalcula el vector de velocidad con filtro de suavizado."""
        new_box_arr = np.array(new_box, dtype=np.float32)
        # Velocidad estimada
        delta = new_box_arr - self.box
        self.velocity = 0.5 * self.velocity + 0.5 * delta
        self.box = new_box_arr
        self.score = float(new_score)
        self.hits += 1
        self.age += 1
        self.time_since_update = 0

    def mark_missed(self):
        self.age += 1
        self.time_since_update += 1


class ByteTracker:
    """
    Algoritmo ByteTrack de asociación en 2 etapas (High-score y Low-score)
    100% en NumPy puro sin PyTorch ni dependencias externas.
    """
    def __init__(self, max_lost_frames: int = 14, iou_threshold: float = 0.35):
        self.tracks: List[TrackedObject] = []
        self.max_lost_frames = max_lost_frames
        self.iou_threshold = iou_threshold

    def update(self, detections: List[Dict[str, Any]], orig_w: int, orig_h: int) -> List[Dict[str, Any]]:
        # Predecir posiciones estimadas de todos los tracks existentes
        for t in self.tracks:
            t.mark_missed()

        if not detections:
            # Purgar tracks viejos
            self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_lost_frames]
            return []

        # Separar detecciones en Alta confianza (>= 0.45) y Baja confianza (< 0.45)
        high_dets = []
        low_dets = []
        for d in detections:
            if d["confidence"] >= 0.45:
                high_dets.append(d)
            else:
                low_dets.append(d)

        active_tracks = [t for t in self.tracks if t.time_since_update <= self.max_lost_frames]
        pred_boxes = np.array([t.predict() for t in active_tracks], dtype=np.float32) if active_tracks else np.empty((0, 4))

        # --- ETAPA 1: Asociar tracks activos con detecciones de alta confianza ---
        high_boxes = np.array([d["box"] for d in high_dets], dtype=np.float32) if high_dets else np.empty((0, 4))
        matches1, un_tracks1, un_high_dets = match_boxes(pred_boxes, high_boxes, self.iou_threshold)

        for trk_idx, det_idx in matches1:
            active_tracks[trk_idx].update(high_dets[det_idx]["box"], high_dets[det_idx]["confidence"])

        # --- ETAPA 2: Asociar tracks restantes no emparejados con detecciones de baja confianza ---
        # (Clave de ByteTrack: recupera objetos temporalmente ocluidos o borrosos)
        rem_tracks = [active_tracks[i] for i in un_tracks1]
        rem_pred_boxes = np.array([t.predict() for t in rem_tracks], dtype=np.float32) if rem_tracks else np.empty((0, 4))
        low_boxes = np.array([d["box"] for d in low_dets], dtype=np.float32) if low_dets else np.empty((0, 4))

        matches2, un_tracks2, _ = match_boxes(rem_pred_boxes, low_boxes, iou_thresh=0.22)
        for trk_idx, det_idx in matches2:
            rem_tracks[trk_idx].update(low_dets[det_idx]["box"], low_dets[det_idx]["confidence"])

        # Inicializar nuevos tracks para detecciones de alta confianza no emparejadas
        for det_idx in un_high_dets:
            det = high_dets[det_idx]
            new_track = TrackedObject(det["box"], det["confidence"], det["class_name"], det["class_id"])
            self.tracks.append(new_track)

        # Filtrar tracks vivos y activos (confirmados al menos con 2 hits)
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_lost_frames]

        tracked_results = []
        for t in self.tracks:
            if t.time_since_update == 0 and t.hits >= 2:
                b = t.box.astype(int)
                bx1, by1, bx2, by2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])
                cx = (bx1 + bx2) // 2
                cy = (by1 + by2) // 2
                tracked_results.append({
                    "track_id": t.track_id,
                    "class_id": t.class_id,
                    "class_name": t.class_name,
                    "confidence": round(t.score, 2),
                    "box": [bx1, by1, bx2, by2],
                    "center": (cx, cy),
                    "norm_center": (round(cx / orig_w, 3), round(cy / orig_h, 3)),
                    "width": bx2 - bx1,
                    "height": by2 - by1,
                    "hits": t.hits
                })

        return tracked_results


class AITrackerController:
    def __init__(self, camera=None, stream_worker=None):
        self.camera = camera
        self.stream_worker = stream_worker
        self.lock = threading.Lock()
        
        # Opciones configurables
        self.ai_enabled = True
        self.tracking_enabled = False
        self.target_classes = ["person", "vehicle"]
        self.min_confidence = 0.60
        self.return_home_enabled = True
        self.home_preset = "center"
        self.deadzone_x = 0.18   # Margen horizontal seguro (18%)
        self.deadzone_y = 0.22   # Margen vertical seguro (22%)
        self.model_name = "yolov8n.onnx"
        
        # Motor ByteTrack y Fijación de Blanco (Target Locking)
        self.byte_tracker = ByteTracker(max_lost_frames=12, iou_threshold=0.35)
        self.locked_track_id: Optional[int] = None
        self.locked_track_lost_time: float = 0.0
        self._smoothed_target = None

        # Estado en tiempo real
        self.is_running = False
        self.last_target_time = 0
        self.last_motor_command_time = 0
        self.returned_home = True
        self.current_detections: List[Dict[str, Any]] = []
        self.last_annotated_jpeg: Optional[bytes] = None
        self.fps = 0.0
        self.inference_ms = 0.0

        self.load_config()

        # Iniciar hilo de seguimiento autónomo
        self.worker_thread = threading.Thread(target=self._tracking_loop, daemon=True)
        self.worker_thread.start()

    def set_camera_and_worker(self, camera, stream_worker):
        self.camera = camera
        self.stream_worker = stream_worker

    def load_config(self):
        if CONFIG_PATH.exists():
            try:
                with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.ai_enabled = data.get("ai_enabled", True)
                    self.tracking_enabled = data.get("tracking_enabled", False)
                    self.target_classes = data.get("target_classes", ["person", "vehicle"])
                    self.min_confidence = data.get("min_confidence", 0.60)
                    self.return_home_enabled = data.get("return_home_enabled", True)
                    self.home_preset = data.get("home_preset", "center")
                    self.deadzone_x = data.get("deadzone_x", 0.18)
                    self.deadzone_y = data.get("deadzone_y", 0.22)
                    self.model_name = data.get("model_name", "yolov8n.onnx")
                    if self.model_name != ai_detector.active_model_name:
                        ai_detector.load_model(self.model_name)
            except Exception as e:
                logger.error(f"Error cargando ai_tracker_config.json: {e}")

    def save_config(self):
        with self.lock:
            data = {
                "ai_enabled": self.ai_enabled,
                "tracking_enabled": self.tracking_enabled,
                "target_classes": self.target_classes,
                "min_confidence": self.min_confidence,
                "return_home_enabled": self.return_home_enabled,
                "home_preset": self.home_preset,
                "deadzone_x": self.deadzone_x,
                "deadzone_y": self.deadzone_y,
                "model_name": ai_detector.active_model_name
            }
            try:
                with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
            except Exception as e:
                logger.error(f"Error guardando ai_tracker_config.json: {e}")

    def switch_model(self, model_name: str) -> bool:
        """Cambia el modelo YOLO activo en caliente y guarda la configuración."""
        with self.lock:
            success = ai_detector.load_model(model_name)
            if success:
                self.model_name = ai_detector.active_model_name
                data = {
                    "ai_enabled": self.ai_enabled,
                    "tracking_enabled": self.tracking_enabled,
                    "target_classes": self.target_classes,
                    "min_confidence": self.min_confidence,
                    "return_home_enabled": self.return_home_enabled,
                    "home_preset": self.home_preset,
                    "deadzone_x": self.deadzone_x,
                    "deadzone_y": self.deadzone_y,
                    "model_name": self.model_name
                }
                try:
                    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                        json.dump(data, f, indent=2)
                except Exception as e:
                    logger.error(f"Error guardando ai_tracker_config.json tras cambio de modelo: {e}")
            return success

    def get_allowed_classes(self) -> List[str]:
        classes = []
        for grp in self.target_classes:
            classes.extend(CLASS_GROUPS.get(grp, [grp]))
        return classes

    def _tracking_loop(self):
        logger.info("Iniciando bucle de IA con ByteTrack y Target Locking...")
        self.is_running = True
        frame_count = 0
        fps_start = time.time()

        while True:
            try:
                if not self.ai_enabled:
                    time.sleep(0.5)
                    continue

                # 1. Obtener fotograma más reciente (prioridad local de baja latencia)
                jpeg_bytes = None
                if self.stream_worker:
                    jpeg_bytes = self.stream_worker.get_latest_frame()

                if not jpeg_bytes and self.camera:
                    jpeg_bytes = self.camera.get_immediate_frame()

                if not jpeg_bytes:
                    time.sleep(0.1)
                    continue

                # 2. Cargar imagen con Pillow
                try:
                    img = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
                except Exception:
                    time.sleep(0.05)
                    continue

                orig_w, orig_h = img.size

                # 3. Detección YOLOv8 con Letterboxing y Smart Motion Gate
                allowed = self.get_allowed_classes()
                raw_detections = ai_detector.detect(img, conf_threshold=self.min_confidence, allowed_classes=allowed)

                # 4. Actualizar trayectorias con ByteTrack (asigna track_id único)
                tracked_objects = self.byte_tracker.update(raw_detections, orig_w, orig_h)
                
                # Si ByteTrack no ha consolidado hits todavía pero hay raw_detections, usamos raw con ID provisional
                active_objects = tracked_objects if tracked_objects else raw_detections
                self.current_detections = active_objects

                now = time.time()
                frame_count += 1
                if (now - fps_start) >= 2.0:
                    self.fps = round(frame_count / (now - fps_start), 1)
                    frame_count = 0
                    fps_start = now

                if active_objects:
                    self.inference_ms = active_objects[0].get("inference_ms", 0.0)
                    self.last_target_time = now
                    self.returned_home = False

                    # 5. Lógica de Target Locking y seguimiento PTZ
                    target = None
                    if self.tracking_enabled and self.camera:
                        target = self._process_auto_tracking(active_objects)

                    # Marcar cuál objeto tiene el blanco fijado (Target Lock)
                    if target and "track_id" in target:
                        for obj in active_objects:
                            obj["is_locked"] = (obj.get("track_id") == target["track_id"])

                    # Generar imagen anotada
                    annotated_jpeg = ai_detector.annotate_and_get_jpeg(img, active_objects)
                    self.last_annotated_jpeg = annotated_jpeg

                    # 6. Disparar alerta en Telegram si corresponde
                    telegram_notifier.trigger_ai_detection_alert(annotated_jpeg, active_objects)

                else:
                    self._smoothed_target = None
                    self.locked_track_id = None
                    self.locked_track_lost_time = 0.0

                    # Sin detecciones: comprobar si debe regresar a la posición de guardia
                    if self.tracking_enabled and self.return_home_enabled and not self.returned_home:
                        if (now - self.last_target_time) > 12.0:  # 12 segundos sin presencia
                            logger.info("Escena despejada por >12s. Regresando a Posición de Guardia...")
                            self._return_home()
                            self.returned_home = True

                # Sampling de 4 a 6 FPS (150ms) óptimo para no fatigar el motor
                time.sleep(0.15)

            except Exception as e:
                logger.error(f"Error en bucle de IA Tracker: {e}")
                time.sleep(0.5)

    def _select_target(self, objects: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Selecciona y mantiene bloqueado el objetivo actual (Target Lock).
        Evita que la cámara titubee si dos o más personas se cruzan en la escena.
        """
        now = time.time()

        # 1. Si ya tenemos un blanco fijado, buscarlo activamente por su track_id
        if self.locked_track_id is not None:
            for obj in objects:
                if obj.get("track_id") == self.locked_track_id:
                    self.locked_track_lost_time = 0.0
                    return obj

            # Si el blanco fijado no se vio en este fotograma, retenerlo hasta 1.8 segundos
            if self.locked_track_lost_time == 0.0:
                self.locked_track_lost_time = now

            if (now - self.locked_track_lost_time) < 1.8:
                # Mantener el lock temporalmente (no saltar bruscamente a otra persona)
                return None
            else:
                # Blanco definitivamente perdido, liberar lock para adquirir uno nuevo
                logger.info(f"Blanco #{self.locked_track_id} perdido. Re-adquiriendo nuevo objetivo...")
                self.locked_track_id = None
                self.locked_track_lost_time = 0.0
                self._smoothed_target = None

        # 2. Si no hay blanco fijado, elegir el mejor candidato (prioridad: persona más centrada)
        persons = [o for o in objects if o["class_name"] == "person"]
        candidates = persons if persons else objects

        if not candidates:
            return None

        # Elegir el candidato más cercano al centro del fotograma
        best_candidate = min(
            candidates,
            key=lambda c: abs(c["norm_center"][0] - 0.5) + abs(c["norm_center"][1] - 0.5)
        )

        if "track_id" in best_candidate:
            self.locked_track_id = best_candidate["track_id"]
            logger.info(f"🎯 BLANCO FIJADO: ID #{self.locked_track_id} ({best_candidate['class_name'].upper()})")

        return best_candidate

    def _process_auto_tracking(self, objects: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Calcula el error del objetivo fijado y ejecuta micropulsos proporcionales calibrados."""
        now = time.time()

        target = self._select_target(objects)
        if not target:
            return None

        # Rate-limiter: al menos 650ms entre comandos mecánicos para estabilización de imagen
        if (now - self.last_motor_command_time) < 0.65:
            return target

        raw_cx, raw_cy = target["norm_center"]

        # Filtro de Suavizado Exponencial (EMA, alpha=0.35)
        if self._smoothed_target is None:
            self._smoothed_target = [raw_cx, raw_cy]
        else:
            self._smoothed_target[0] = 0.35 * raw_cx + 0.65 * self._smoothed_target[0]
            self._smoothed_target[1] = 0.35 * raw_cy + 0.65 * self._smoothed_target[1]

        smooth_cx, smooth_cy = self._smoothed_target
        err_x = smooth_cx - 0.5   # Negativo: Izquierda, Positivo: Derecha
        err_y = smooth_cy - 0.5   # Negativo: Arriba, Positivo: Abajo

        moved = False
        track_label = f"#{target.get('track_id', '?')}"

        # Corrección Horizontal (Pan) proporcional fina
        if abs(err_x) > self.deadzone_x:
            direction = "left" if err_x < 0 else "right"
            excess_x = abs(err_x) - self.deadzone_x
            # Micro-pulso proporcional fino de 40ms a 105ms (0.7° a 1.9° a velocidad 0)
            duration = int(40 + min(65, excess_x * 160))
            self.camera.ptz_step(direction, duration_ms=duration, is_tracking=True)
            moved = True
            logger.info(f"Auto-Tracking PAN {track_label} -> {direction.upper()} (err: {err_x:.2f}, dur: {duration}ms)")

        # Corrección Vertical (Tilt) si el error persiste
        if not moved and abs(err_y) > self.deadzone_y:
            direction = "up" if err_y < 0 else "down"
            excess_y = abs(err_y) - self.deadzone_y
            # Micro-pulso proporcional fino de 35ms a 85ms
            duration = int(35 + min(50, excess_y * 140))
            self.camera.ptz_step(direction, duration_ms=duration, is_tracking=True)
            moved = True
            logger.info(f"Auto-Tracking TILT {track_label} -> {direction.upper()} (err: {err_y:.2f}, dur: {duration}ms)")

        if moved:
            self.last_motor_command_time = time.time()

        return target

    def _return_home(self):
        if not self.camera:
            return
        if self.home_preset == "center":
            self.camera.ptz_reset()
        else:
            self.camera.cgi_request("ptzGotoPresetPoint", {"name": self.home_preset})

    def get_status(self) -> Dict[str, Any]:
        return {
            "ai_enabled": self.ai_enabled,
            "tracking_enabled": self.tracking_enabled,
            "target_classes": self.target_classes,
            "min_confidence": self.min_confidence,
            "return_home_enabled": self.return_home_enabled,
            "home_preset": self.home_preset,
            "fps": self.fps,
            "inference_ms": self.inference_ms,
            "detections_count": len(self.current_detections),
            "detections": self.current_detections,
            "locked_track_id": self.locked_track_id,
            "model_name": ai_detector.active_model_name,
            "model_type": ai_detector.model_type,
            "available_models": ai_detector.get_available_models(),
            "pan": getattr(self.camera, "pan_angle", 0.0) if self.camera else 0.0,
            "tilt": getattr(self.camera, "tilt_angle", 0.0) if self.camera else 0.0,
            "telegram": {
                "enabled": telegram_notifier.enabled,
                "configured": bool(telegram_notifier.bot_token and telegram_notifier.chat_id),
                "is_silenced": telegram_notifier.is_silenced(),
                "cooldown_seconds": telegram_notifier.cooldown_seconds
            }
        }


ai_tracker = AITrackerController()
