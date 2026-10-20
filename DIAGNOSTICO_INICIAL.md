# Diagnóstico Técnico de Arquitectura: Foscam Control Center

**Fecha:** 2026-10-19  
**Repositorio:** `foscam-control-center`  
**Estado:** Inspección Inicial (Solo Lectura)  

---

## 1. Stack de `app.py`

* **Framework Web & API:**  
  * **FastAPI** ejecutándose sobre el servidor ASGI **Uvicorn**.
  * No utiliza interfaces gráficas de escritorio (Tkinter, PyQt ni OpenCV `imshow`).
  * Expone endpoints REST, WebSockets binarios de baja latencia (`/ws/stream`, `/ws/talk`) y respuestas de streaming HTTP multipart MJPEG (`StreamingResponse`).
  * Monta y sirve la aplicación web frontend estática directamente en el directorio `/static`.

* **Captura y Distribución de Video:**  
  * **No utiliza** `cv2.VideoCapture` ni polling HTTP CGI continuo para el video en vivo.
  * La captura se gestiona a través de la clase `RTSPStreamWorker`, la cual levanta un subproceso nativo de **FFmpeg** conectado por TCP (`-rtsp_transport tcp`) al flujo RTSP primario de la cámara:  
    `rtsp://admin:@192.168.1.13:88/videoMain`
  * **Parámetros de baja latencia de FFmpeg:**  
    `-fflags nobuffer -flags low_delay -avioflags direct -fflags +discardcorrupt -probesize 32768 -analyzeduration 0 -f mjpeg -r 30`
  * La salida se transcodifica en tiempo real a MJPEG enviada a stdout `PIPE` sin buffering intermedio, entregando fotogramas a los clientes web vía WebSocket con latencias inferiores a 50 ms.
  * Mantiene soporte secundario de captura estática vía CGI HTTP (`/cgi-bin/CGIProxy.fcgi?cmd=snapPicture2`) únicamente como respaldo si el flujo RTSP se interrumpe.

---

## 2. Estado de `ai_detector.py`

* **Librerías Importadas:**
  ```python
  import io
  import logging
  from pathlib import Path
  import time
  from typing import Any, Dict, List, Optional, Tuple

  import numpy as np
  import onnxruntime as ort
  from PIL import Image, ImageDraw, ImageFont
  ```

* **Dependencias de Inferencia:**  
  * Emplea exclusivamente **`onnxruntime` puro** (`CPUExecutionProvider`) optimizado con `GraphOptimizationLevel.ORT_ENABLE_ALL` y 4 hilos de ejecución CPU.
  * **NO importa ni carga `ultralytics` ni `torch`**, eliminando gigabytes de sobrecarga en memoria y dependencias pesadas.
  * Las etapas de preprocesamiento (redimensionamiento a 640x640 y normalización a `float32 [0.0, 1.0]`) y postprocesamiento (cálculo de coordenadas relativas, filtrado de umbral de confianza y Non-Maximum Suppression `nms`) están programadas de forma nativa en NumPy.
  * Detección basada en el modelo COCO de 80 clases (`yolov8n.onnx`), categorizado internamente en grupos: `person`, `vehicle` y `pet`.

---

## 3. Estado de `ai_tracker.py`

* **Arquitectura de Ejecución:**  
  * Se ejecuta de forma autónoma en un hilo de fondo desacoplado (`threading.Thread` en `_tracking_loop`), muestreando fotogramas a una tasa controlada de 4 a 6 FPS (150 ms de pausa entre inferencias) para evitar sobrecalentamiento del CPU.

* **Envío de Comandos PTZ:**  
  * Los comandos no se ejecutan como peticiones HTTP síncronas bloqueantes. Se canalizan mediante `camera.ptz_step(...)`, que despacha un hilo trabajador independiente.
  * A nivel de transporte, utiliza **sockets TCP directos (`socket.SOCK_STREAM`)** con HTTP 1.1 single-socket pipelining (`ptz_pulse_precise`):
    1. Abre una conexión TCP limpia hacia el puerto 88 de la cámara.
    2. Envía el comando de inicio de movimiento (`GET /cgi-bin/CGIProxy.fcgi?cmd=...`).
    3. Espera la duración exacta del micropulso en milisegundos mediante `self.ptz_abort_event.wait(timeout=dur_sec)`.
    4. Envía de inmediato el comando de frenado (`ptzStopRun`) por el mismo socket sin realizar un nuevo handshake de red ni esperar el payload de respuesta HTTP.
  * Esto reduce el jitter de red de ~450 ms a una precisión mecánica de $\pm 0.5$ ms.

* **Control de Cooldown, Deadzone y Estabilización:**  
  * **Cooldown:** Posee un limitador estricto de tasa que bloquea nuevos comandos si han transcurrido menos de **650 ms** desde el último pulso de motor (`(now - self.last_motor_command_time) < 0.65`).
  * **Deadzone:** Márgenes muertos configurables independientes:  
    * Horizontal: `deadzone_x` (16% a 20%).
    * Vertical: `deadzone_y` (22% a 25%).
  * **Filtro Suavizador EMA:** Implementa un promedio móvil exponencial ($\alpha = 0.35$) sobre el centro del objetivo detectado, eliminando movimientos bruscos o sacudidas producidas por el balanceo corporal de las personas.
  * **Micropulsos Proporcionales:** Duraciones calibradas a velocidad de motor 0:
    * Pan: 40 ms a 105 ms ($0.7^\circ$ a $1.9^\circ$).
    * Tilt: 35 ms a 85 ms ($0.5^\circ$ a $1.2^\circ$).
  * **Retorno de Guardia:** Si transcurren más de 12 segundos sin detecciones continuas, la cámara retorna automáticamente a su posición central o preset de vigilancia configurado.

---

## 4. Estado de `gdrive_sync.py`

* **Librerías de Google Utilizadas:**  
  * Utiliza **`google-api-python-client`** (`googleapiclient.discovery.build`, `MediaFileUpload`).
  * Utiliza **`google-auth`** (`google.oauth2.service_account`, `google.oauth2.credentials`).
  * **No utiliza `PyDrive`** ni librerías obsoletas de sincronización.

* **Tipos de Credenciales Esperadas:**  
  * `gdrive_config.json` contiene únicamente parámetros de estado y metadatos (`folder_id`, `folder_name`, `auto_upload`, `delete_after_upload`, `user_email`).
  * Las credenciales de acceso se cargan desde el directorio `credentials/` bajo la siguiente prioridad:
    1. **Service Account:** `service_account.json` (recomendado para servidores desatendidos sin navegador).
    2. **Token OAuth2 de Usuario:** `token.json` (credenciales de usuario autenticado con auto-renovación vía refresh token).
    3. **OAuth Client Secrets:** `credentials.json` (flujo de consentimiento web/desktop).

* **Concurrencia y Comportamiento del Hilo:**  
  * Las subidas de video a la nube **nunca bloquean el hilo principal ni la API de FastAPI**.
  * La clase `RecorderManager` (en `recorder.py`) administra una cola de subida (`upload_queue`) y despacha cada archivo en un **hilo demonio separado** (`threading.Thread(target=self._upload_worker, args=(next_file,), daemon=True)`).
  * La transferencia se realiza mediante subida reanudable (resumable upload) en fragmentos de 5 MB (`chunksize=5 * 1024 * 1024`).

---

## 5. Código Fuente Completo: `ai_detector.py`

```python
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
```
