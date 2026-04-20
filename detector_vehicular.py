"""
================================================================================
TECNOLÓGICO NACIONAL DE MÉXICO — INSTITUTO TECNOLÓGICO DE OAXACA
Ingeniería en Sistemas Computacionales
Taller de Investigación II

Título del proyecto:
    Sistema de semáforos inteligentes basado en detección vehicular para
    optimizar el flujo de tráfico en Oaxaca de Juárez, Oaxaca.

Módulo:
    detector_vehicular.py — Subsistema de procesamiento (Capa 2)
    Detección vehicular en tiempo real con YOLOv5 + OpenCV

Autores:
    Delgado Molina Karla Rocío
    Martínez Martínez Jesús Alexander
    Zarate Matus Ángel Adrián

Catedrática:
    Pérez López Otilia

Correspondencia con el Plan de Trabajo (Capítulo III, sección 3.3.1):
    ┌─────────────────┬────────────────────────────────────────────┬──────────────────────┐
    │ Actividad       │ Descripción                                │ Periodo              │
    ├─────────────────┼────────────────────────────────────────────┼──────────────────────┤
    │ Actividad 8     │ Diseñar pipeline de pre-procesamiento      │ 12/04/26 – 19/04/26  │
    │                 │ de imágenes con OpenCV  [COMPLETADA]       │                      │
    ├─────────────────┼────────────────────────────────────────────┼──────────────────────┤
    │ Actividad 9     │ Especificar configuración y parámetros     │ 20/04/26 – 27/04/26  │
    │  ← ACTUAL       │ de YOLOv5 para detección vehicular         │                      │
    └─────────────────┴────────────────────────────────────────────┴──────────────────────┘

Metodología SCRUM — Fase 2: Modelado, Diseño y Refinamiento (Sprints de Desarrollo)
    Sprint actual: Modelado y Refinamiento de Algoritmos (Sprints Finales)
    Fuente: Protocolo de Investigación, sección 2.8.3.2

Flujo completo del subsistema de procesamiento (Capa 2):

    ┌─────────────────────────────────────────────────────────────────┐
    │              SUBSISTEMA DE PROCESAMIENTO (Capa 2)               │
    │                                                                 │
    │  Cámara IP                                                      │
    │     │                                                           │
    │     ▼                                                           │
    │  [pipeline_preprocesamiento.py]  ← Actividad 8 (completada)    │
    │     │  frame BGR crudo → tensor RGB 640×640 normalizado         │
    │     ▼                                                           │
    │  [detector_vehicular.py]         ← Actividad 9 (este módulo)   │
    │     │  tensor → detecciones → conteo por carril                 │
    │     ▼                                                           │
    │  Motor de decisión               ← Actividad 10                 │
    │     │  conteo → tiempos de ciclo semafórico adaptativo          │
    │     ▼                                                           │
    │  Controlador semafórico                                         │
    └─────────────────────────────────────────────────────────────────┘

INSTALACIÓN (ejecutar en tu venv antes de usar este módulo):
    # 1. PyTorch CPU (si no tienes GPU NVIDIA)
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

    # 2. YOLOv5 vía ultralytics (recomendado, más estable)
    pip install ultralytics

    # 3. Dependencias auxiliares
    pip install scipy

    # Verificar instalación:
    python -c "import torch, ultralytics; print('OK')"

Referencias principales:
    - San Miguel, S. (2024). Sistema de coordinación de semáforos inteligentes
      con algoritmos de inteligencia artificial. Revista ConCiencia Joven, 2, 32-38.
    - Ultralytics (2024). Comprehensive Guide to Ultralytics YOLOv5.
      https://docs.ultralytics.com/yolov5/
    - Roboflow (2020). What is YOLOv5? A Guide for Beginners.
      https://blog.roboflow.com/yolov5-improvements-and-evaluation/
    - ScienceDirect (2024). YOLOv5 — an overview.

Versión:   1.0.0
Fecha:     Abril 2026
================================================================================
"""

import cv2
import numpy as np
import time
import logging
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Optional

# ---------------------------------------------------------------------------
# Importaciones opcionales — el módulo informa claramente si faltan
# ---------------------------------------------------------------------------
try:
    import torch
    TORCH_OK = True
except ImportError:
    TORCH_OK = False

try:
    from ultralytics import YOLO
    ULTRALYTICS_OK = True
except ImportError:
    ULTRALYTICS_OK = False

# Pipeline de la Actividad 8 (debe estar en el mismo directorio)
try:
    from pipeline_preprocesamiento import (
        PipelinePreprocesamiento,
        ConfigPipeline,
        generar_video_sintetico,
    )
    PIPELINE_OK = True
except ImportError:
    PIPELINE_OK = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("detector_vehicular")


# ===========================================================================
# SECCIÓN 1 — CLASES COCO QUE SE CONSIDERAN "VEHÍCULO"
#
# YOLOv5 entrenado con COCO detecta 80 clases. Para este sistema solo
# interesan las clases vehiculares relevantes para tráfico urbano en Oaxaca.
# Referencia: sección 2.5.1 del protocolo — Tecnologías de detección vehicular.
# ===========================================================================

# Índices de clases COCO correspondientes a vehículos
CLASES_VEHICULO = {
    2:  "automóvil",
    3:  "motocicleta",
    5:  "autobús",
    7:  "camión",
    1:  "bicicleta",      # incluida por movilidad urbana
}

# Colores BGR para visualización (uno por tipo de vehículo)
COLORES_CLASE = {
    2: (30, 144, 255),   # automóvil  — azul
    3: (0, 200, 100),    # motocicleta — verde
    5: (0, 80, 255),     # autobús    — naranja
    7: (50, 50, 200),    # camión     — rojo oscuro
    1: (200, 180, 0),    # bicicleta  — cyan
}


# ===========================================================================
# SECCIÓN 2 — CONFIGURACIÓN DEL DETECTOR
# Parámetros operativos YOLOv5 según protocolo, sección 2.3.6.
# ===========================================================================

@dataclass
class ConfigDetector:
    """
    Parámetros operativos del detector vehicular YOLOv5.

    Atributos
    ----------
    modelo_yolo : str
        Variante del modelo YOLOv5.
        Opciones: 'yolov5n' (nano), 'yolov5s' (small), 'yolov5m' (medium).
        Para CPU sin GPU: usar 'yolov5n' o 'yolov5s'.
        Para GPU NVIDIA: usar 'yolov5m' o 'yolov5l'.
    confianza_minima : float
        Umbral de confianza mínima para aceptar una detección (0.0–1.0).
        Detecciones por debajo se descartan. Recomendado: 0.45.
    iou_umbral : float
        Umbral IoU para Non-Maximum Suppression (NMS).
        Controla superposición máxima entre bounding boxes. Recomendado: 0.45.
    tam_imagen : int
        Tamaño de entrada al modelo (debe coincidir con pipeline). Estándar: 640.
    dispositivo : str
        'cpu' o 'cuda' (GPU). Se detecta automáticamente.
    mostrar_ventana : bool
        Si True, muestra ventana en tiempo real con bounding boxes.
    guardar_video : bool
        Si True, guarda el video anotado como archivo MP4.
    ruta_video_salida : str
        Ruta del video anotado de salida.
    lineas_conteo : list[dict]
        Líneas virtuales para conteo de vehículos por carril.
        Cada línea es un dict con 'nombre', 'y' (posición vertical) y 'color'.
    """
    modelo_yolo: str = "yolov5s"
    confianza_minima: float = 0.45
    iou_umbral: float = 0.45
    tam_imagen: int = 640
    dispositivo: str = "cuda" if (TORCH_OK and torch.cuda.is_available()) else "cpu"
    mostrar_ventana: bool = True
    guardar_video: bool = True
    ruta_video_salida: str = "deteccion_vehicular.mp4"

    # Líneas virtuales de conteo (en coordenadas del frame 640×640)
    # Cada línea horizontal divide los carriles de la intersección
    lineas_conteo: list = field(default_factory=lambda: [
        {"nombre": "Carril norte",  "y": 160, "color": (0, 255, 255)},
        {"nombre": "Carril centro", "y": 320, "color": (255, 165, 0)},
        {"nombre": "Carril sur",    "y": 480, "color": (0, 165, 255)},
    ])


# ===========================================================================
# SECCIÓN 3 — RESULTADO DE DETECCIÓN POR FRAME
# Estructura de datos que resume lo detectado en un fotograma.
# Alimentará el motor de decisión semafórico (Actividad 10).
# ===========================================================================

@dataclass
class ResultadoDeteccion:
    """
    Resultado del procesamiento de un fotograma por el detector YOLOv5.

    Esta estructura es la salida del detector y la entrada del motor de
    decisión semafórico (Actividad 10). Contiene toda la información
    necesaria para calcular los tiempos de ciclo adaptativos.

    Atributos
    ----------
    n_frame : int
        Número secuencial del fotograma procesado.
    timestamp : float
        Marca de tiempo UNIX del procesamiento.
    total_vehiculos : int
        Conteo total de vehículos detectados en el frame.
    conteo_por_tipo : dict[str, int]
        Vehículos por categoría (automóvil, motocicleta, autobús, camión).
    conteo_por_carril : dict[str, int]
        Vehículos agrupados por carril (norte, centro, sur).
    detecciones_raw : list[dict]
        Lista de detecciones individuales con bbox, confianza y clase.
    latencia_ms : float
        Tiempo de inferencia YOLOv5 para este frame en milisegundos.
    nivel_congestion : str
        Clasificación del nivel de congestión: 'libre', 'moderado', 'congestionado'.
    """
    n_frame: int = 0
    timestamp: float = 0.0
    total_vehiculos: int = 0
    conteo_por_tipo: dict = field(default_factory=dict)
    conteo_por_carril: dict = field(default_factory=dict)
    detecciones_raw: list = field(default_factory=list)
    latencia_ms: float = 0.0
    nivel_congestion: str = "libre"

    def __str__(self) -> str:
        return (
            f"Frame {self.n_frame:04d} | "
            f"Vehículos: {self.total_vehiculos:2d} | "
            f"Congestión: {self.nivel_congestion:12s} | "
            f"Latencia: {self.latencia_ms:.1f} ms | "
            f"Tipos: {self.conteo_por_tipo}"
        )


# ===========================================================================
# SECCIÓN 4 — DETECTOR VEHICULAR
# Clase principal que integra YOLOv5 con el pipeline de la Actividad 8.
# Corresponde a la Actividad 9 del plan de trabajo (20/04/26–27/04/26).
# ===========================================================================

class DetectorVehicular:
    """
    Detector de vehículos en tiempo real basado en YOLOv5 + OpenCV.

    Integra el pipeline de pre-procesamiento (Actividad 8) con el modelo
    YOLOv5 para detectar, clasificar y contar vehículos en cada fotograma.
    Genera resultados estructurados (ResultadoDeteccion) que alimentarán
    al motor de decisión semafórico en la Actividad 10.

    Parámetros operativos clave (protocolo, sección 2.3.6):
      - Confianza mínima: 0.45 (balance precisión/recall para tráfico urbano)
      - IoU NMS: 0.45 (evita detecciones duplicadas en vehículos cercanos)
      - Variante: yolov5s (140 fps en GPU; ~15 fps en CPU)
      - Clases: automóvil, motocicleta, autobús, camión, bicicleta

    Ejemplo de uso
    --------------
    >>> cfg_pipeline  = ConfigPipeline(rtsp_url="video.mp4")
    >>> cfg_detector  = ConfigDetector(modelo_yolo="yolov5s")
    >>> detector = DetectorVehicular(cfg_pipeline, cfg_detector)
    >>> detector.ejecutar()
    """

    def __init__(self, cfg_pipeline: ConfigPipeline, cfg_detector: ConfigDetector):
        self.cfg_p = cfg_pipeline
        self.cfg_d = cfg_detector
        self.modelo = None
        self.pipeline = None
        self.writer_salida = None

        # Métricas acumuladas de la sesión
        self.metricas_sesion = {
            "frames_totales": 0,
            "vehiculos_detectados_total": 0,
            "latencia_acum_ms": 0.0,
            "conteo_por_tipo_sesion": defaultdict(int),
            "maximo_vehiculos_frame": 0,
        }

    # ------------------------------------------------------------------
    # Ciclo de vida
    # ------------------------------------------------------------------

    def iniciar(self) -> None:
        """Carga el modelo YOLOv5 e inicia el pipeline de captura."""
        self._verificar_dependencias()
        self._cargar_modelo()
        self.pipeline = PipelinePreprocesamiento(self.cfg_p)
        self.pipeline.iniciar()
        logger.info(
            f"Detector iniciado — Modelo: {self.cfg_d.modelo_yolo} | "
            f"Dispositivo: {self.cfg_d.dispositivo.upper()} | "
            f"Confianza mín.: {self.cfg_d.confianza_minima}"
        )

    def detener(self) -> None:
        """Libera recursos y muestra métricas finales de la sesión."""
        if self.pipeline:
            self.pipeline.detener()
        if self.writer_salida:
            self.writer_salida.release()
        cv2.destroyAllWindows()
        self._imprimir_metricas_sesion()

    def _verificar_dependencias(self) -> None:
        """Verifica que PyTorch y Ultralytics estén instalados."""
        errores = []
        if not TORCH_OK:
            errores.append(
                "PyTorch no instalado.\n"
                "  → pip install torch torchvision "
                "--index-url https://download.pytorch.org/whl/cpu"
            )
        if not ULTRALYTICS_OK:
            errores.append(
                "Ultralytics no instalado.\n"
                "  → pip install ultralytics"
            )
        if not PIPELINE_OK:
            errores.append(
                "pipeline_preprocesamiento.py no encontrado.\n"
                "  → Asegúrate de que esté en el mismo directorio."
            )
        if errores:
            msg = "\n\nDependencias faltantes:\n" + "\n".join(errores)
            raise ImportError(msg)

    def _cargar_modelo(self) -> None:
        """
        Descarga y carga el modelo YOLOv5 usando Ultralytics.

        La primera ejecución descarga los pesos (~14 MB para yolov5s)
        automáticamente desde los servidores de Ultralytics. Las
        ejecuciones siguientes usan la caché local (~/.cache/ultralytics).
        """
        logger.info(f"Cargando modelo {self.cfg_d.modelo_yolo}...")
        try:
            self.modelo = YOLO(f"{self.cfg_d.modelo_yolo}.pt")
            logger.info(f"Modelo {self.cfg_d.modelo_yolo} cargado correctamente.")
        except Exception as e:
            raise RuntimeError(
                f"No se pudo cargar el modelo '{self.cfg_d.modelo_yolo}'.\n"
                f"Error: {e}\n"
                f"Verifica tu conexión a internet para la primera descarga."
            )

    # ------------------------------------------------------------------
    # Inferencia
    # ------------------------------------------------------------------

    def _inferir(self, frame_bgr_orig: np.ndarray, n_frame: int) -> ResultadoDeteccion:
        """
        Ejecuta YOLOv5 sobre un frame y retorna el resultado estructurado.

        Parámetros
        ----------
        frame_bgr_orig : np.ndarray
            Frame BGR original (sin pre-procesar) para visualización.
            El modelo recibe su propia copia redimensionada internamente.
        n_frame : int
            Número secuencial del frame.

        Retorna
        -------
        ResultadoDeteccion
            Conteo, clasificación y metadata del frame analizado.
        """
        t0 = time.perf_counter()

        # YOLOv5 vía Ultralytics acepta directamente el frame BGR de OpenCV
        resultados = self.modelo.predict(
            source=frame_bgr_orig,
            conf=self.cfg_d.confianza_minima,
            iou=self.cfg_d.iou_umbral,
            imgsz=self.cfg_d.tam_imagen,
            device=self.cfg_d.dispositivo,
            verbose=False,
            classes=list(CLASES_VEHICULO.keys()),
        )

        latencia_ms = (time.perf_counter() - t0) * 1000

        # Extraer detecciones del resultado
        detecciones = []
        conteo_tipo = defaultdict(int)
        conteo_carril = defaultdict(int)

        h, w = frame_bgr_orig.shape[:2]

        for r in resultados:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                if cls_id not in CLASES_VEHICULO:
                    continue

                conf   = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = (x1 + x2) // 2   # centro X del vehículo
                cy = (y1 + y2) // 2   # centro Y del vehículo
                nombre_clase = CLASES_VEHICULO[cls_id]

                detecciones.append({
                    "clase_id":  cls_id,
                    "clase":     nombre_clase,
                    "confianza": round(conf, 3),
                    "bbox":      (x1, y1, x2, y2),
                    "centro":    (cx, cy),
                })

                conteo_tipo[nombre_clase] += 1

                # Asignar al carril por posición vertical (cy)
                carril = self._asignar_carril(cy, h)
                conteo_carril[carril] += 1

        total = len(detecciones)
        nivel = self._clasificar_congestion(total)

        return ResultadoDeteccion(
            n_frame=n_frame,
            timestamp=time.time(),
            total_vehiculos=total,
            conteo_por_tipo=dict(conteo_tipo),
            conteo_por_carril=dict(conteo_carril),
            detecciones_raw=detecciones,
            latencia_ms=round(latencia_ms, 2),
            nivel_congestion=nivel,
        )

    def _asignar_carril(self, cy: int, alto_frame: int) -> str:
        """
        Asigna un vehículo a un carril según la posición vertical de su centro.

        Divide el frame en tres zonas horizontales iguales:
          - Zona superior (0 – 33%)   → Carril norte
          - Zona media   (33% – 66%)  → Carril centro
          - Zona inferior(66% – 100%) → Carril sur

        Parámetros
        ----------
        cy : int
            Coordenada Y del centro del bounding box.
        alto_frame : int
            Alto total del frame en píxeles.

        Retorna
        -------
        str
            Nombre del carril ('norte', 'centro' o 'sur').
        """
        tercio = alto_frame // 3
        if cy < tercio:
            return "norte"
        elif cy < 2 * tercio:
            return "centro"
        else:
            return "sur"

    def _clasificar_congestion(self, n_vehiculos: int) -> str:
        """
        Clasifica el nivel de congestión según el conteo vehicular del frame.

        Umbrales definidos según parámetros operativos del protocolo
        (sección 2.3.6) y adaptados a intersecciones de Oaxaca de Juárez.

        Parámetros
        ----------
        n_vehiculos : int
            Número de vehículos detectados en el frame actual.

        Retorna
        -------
        str
            'libre' (0–2), 'moderado' (3–6) o 'congestionado' (7+).
        """
        if n_vehiculos <= 2:
            return "libre"
        elif n_vehiculos <= 6:
            return "moderado"
        else:
            return "congestionado"

    # ------------------------------------------------------------------
    # Visualización
    # ------------------------------------------------------------------

    def _anotar_frame(
        self,
        frame_bgr: np.ndarray,
        resultado: ResultadoDeteccion,
    ) -> np.ndarray:
        """
        Dibuja bounding boxes, etiquetas, líneas de carril y HUD sobre el frame.

        Parámetros
        ----------
        frame_bgr : np.ndarray
            Frame BGR original de la cámara.
        resultado : ResultadoDeteccion
            Resultado de la inferencia para este frame.

        Retorna
        -------
        np.ndarray
            Frame anotado listo para mostrar o guardar.
        """
        frame = frame_bgr.copy()
        h, w = frame.shape[:2]

        # ── Líneas de carril ──
        for linea in self.cfg_d.lineas_conteo:
            y_linea = int(linea["y"] * h / 640)
            cv2.line(frame, (0, y_linea), (w, y_linea), linea["color"], 1)
            cv2.putText(frame, linea["nombre"],
                        (4, y_linea - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.38, linea["color"], 1)

        # ── Bounding boxes y etiquetas ──
        for det in resultado.detecciones_raw:
            x1, y1, x2, y2 = det["bbox"]
            cls_id = det["clase_id"]
            color  = COLORES_CLASE.get(cls_id, (200, 200, 200))
            etiq   = f"{det['clase']} {det['confianza']:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Fondo de etiqueta
            (tw, th), _ = cv2.getTextSize(etiq, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(frame, etiq,
                        (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (255, 255, 255), 1)

            # Punto central
            cv2.circle(frame, det["centro"], 3, color, -1)

        # ── HUD superior ──
        color_nivel = {
            "libre":         (0, 200, 80),
            "moderado":      (0, 165, 255),
            "congestionado": (0, 0, 220),
        }.get(resultado.nivel_congestion, (200, 200, 200))

        panel_h = 70
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, panel_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        cv2.putText(frame,
                    f"Vehiculos: {resultado.total_vehiculos}  |  "
                    f"Congestion: {resultado.nivel_congestion.upper()}  |  "
                    f"Frame: {resultado.n_frame}",
                    (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color_nivel, 1)

        cv2.putText(frame,
                    f"Lat. YOLOv5: {resultado.latencia_ms:.1f} ms  |  "
                    f"Tipos: {resultado.conteo_por_tipo}",
                    (8, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

        cv2.putText(frame,
                    f"Carriles: {resultado.conteo_por_carril}  |  "
                    f"ITO — Semaforos Inteligentes",
                    (8, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (130, 130, 130), 1)

        return frame

    # ------------------------------------------------------------------
    # Ejecución principal
    # ------------------------------------------------------------------

    def ejecutar(self) -> None:
        """
        Bucle principal: captura → pre-procesamiento → inferencia → visualización.

        Conecta el pipeline de la Actividad 8 con YOLOv5 (Actividad 9) y
        muestra el resultado anotado en tiempo real. Los resultados
        estructurados (ResultadoDeteccion) se imprimen en consola y están
        listos para conectar al motor de decisión semafórico (Actividad 10).

        Controles en ventana:
            q — salir
            p — pausar / reanudar
            s — guardar captura del frame actual como PNG
        """
        if not self.pipeline:
            raise RuntimeError("Llama a iniciar() antes de ejecutar().")

        captura = cv2.VideoCapture(self.cfg_p.rtsp_url)
        if not captura.isOpened():
            raise IOError(f"No se pudo abrir: {self.cfg_p.rtsp_url}")

        fps_src   = captura.get(cv2.CAP_PROP_FPS) or 25
        w_src     = int(captura.get(cv2.CAP_PROP_FRAME_WIDTH))
        h_src     = int(captura.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if self.cfg_d.guardar_video:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer_salida = cv2.VideoWriter(
                self.cfg_d.ruta_video_salida, fourcc, fps_src, (w_src, h_src)
            )
            logger.info(f"Grabando video anotado en: {self.cfg_d.ruta_video_salida}")

        logger.info("Iniciando detección — presiona 'q' para salir, 'p' para pausar.")

        n_frame  = 0
        pausado  = False

        try:
            while True:
                if not pausado:
                    ret, frame_bgr = captura.read()
                    if not ret:
                        logger.info("Fin del video.")
                        break

                    n_frame += 1

                    # ── Inferencia YOLOv5 ──────────────────────────────
                    resultado = self._inferir(frame_bgr, n_frame)

                    # ── Actualizar métricas de sesión ──────────────────
                    self._actualizar_metricas(resultado)

                    # ── Log en consola ─────────────────────────────────
                    logger.info(str(resultado))

                    # ── Anotar frame para visualización ───────────────
                    frame_anotado = self._anotar_frame(frame_bgr, resultado)

                    if self.cfg_d.guardar_video and self.writer_salida:
                        self.writer_salida.write(frame_anotado)

                # ── Mostrar ventana ────────────────────────────────────
                if self.cfg_d.mostrar_ventana:
                    # if not pausado:
                    #     cv2.imshow(
                    #         "Deteccion vehicular YOLOv5 — ITO Semaforos Inteligentes",
                    #         frame_anotado,
                    #     )
                    if frame_anotado is not None:
                        cv2.imshow(
                            "Deteccion vehicular YOLOv5 — ITO Semaforos Inteligentes",
                            frame_anotado,
                        )
                    tecla = cv2.waitKey(1) & 0xFF
                    if tecla == ord("q"):
                        logger.info("Salida por usuario.")
                        break
                    elif tecla == ord("p"):
                        pausado = not pausado
                        logger.info("PAUSADO" if pausado else "REANUDADO")
                    elif tecla == ord("s") and not pausado:
                        nombre_cap = f"captura_frame_{n_frame:04d}.png"
                        cv2.imwrite(nombre_cap, frame_anotado)
                        logger.info(f"Captura guardada: {nombre_cap}")

        except KeyboardInterrupt:
            logger.info("Interrupción por teclado.")
        finally:
            captura.release()
            self.detener()

    def _actualizar_metricas(self, resultado: ResultadoDeteccion) -> None:
        """Acumula métricas de la sesión para el reporte final."""
        m = self.metricas_sesion
        m["frames_totales"] += 1
        m["vehiculos_detectados_total"] += resultado.total_vehiculos
        m["latencia_acum_ms"] += resultado.latencia_ms
        m["maximo_vehiculos_frame"] = max(
            m["maximo_vehiculos_frame"], resultado.total_vehiculos
        )
        for tipo, cnt in resultado.conteo_por_tipo.items():
            m["conteo_por_tipo_sesion"][tipo] += cnt

    def _imprimir_metricas_sesion(self) -> None:
        """Imprime el resumen estadístico de la sesión al terminar."""
        m = self.metricas_sesion
        n = max(m["frames_totales"], 1)
        logger.info("=" * 60)
        logger.info("RESUMEN DE SESIÓN — Actividad 9")
        logger.info(f"  Frames procesados       : {m['frames_totales']}")
        logger.info(f"  Vehículos detectados    : {m['vehiculos_detectados_total']}")
        logger.info(f"  Promedio veh/frame      : {m['vehiculos_detectados_total']/n:.2f}")
        logger.info(f"  Máximo en un frame      : {m['maximo_vehiculos_frame']}")
        logger.info(f"  Latencia YOLOv5 prom.   : {m['latencia_acum_ms']/n:.1f} ms")
        logger.info(f"  Conteo por tipo         : {dict(m['conteo_por_tipo_sesion'])}")
        logger.info("=" * 60)


# ===========================================================================
# SECCIÓN 5 — PUNTO DE ENTRADA
#
# Uso:
#   python detector_vehicular.py <video.mp4>    — detecta en video existente
#   python detector_vehicular.py --sintetico    — genera video sintético y detecta
#   python detector_vehicular.py               — muestra ayuda e instrucciones
# ===========================================================================

if __name__ == "__main__":
    import sys

    AYUDA = """
Uso:
  python detector_vehicular.py <ruta_video.mp4>
      Ejecuta detección YOLOv5 sobre un video existente.

  python detector_vehicular.py --sintetico
      Genera un video sintético y ejecuta detección sobre él.
      Útil para probar sin video real (Actividad 9, sprint de validación).

  python detector_vehicular.py --noventana <video.mp4>
      Corre en modo headless (sin GUI), solo log en consola.
      Útil en servidores sin pantalla.

Instalación requerida (ejecutar primero en tu venv):
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
  pip install ultralytics

Controles en ventana:
  q — salir
  p — pausar / reanudar
  s — guardar captura PNG del frame actual
"""

    args = sys.argv[1:]

    if not args:
        print(AYUDA)
        sys.exit(0)

    # Modo headless
    headless = "--noventana" in args
    args = [a for a in args if a != "--noventana"]

    if args[0] == "--sintetico":
        if not PIPELINE_OK:
            print("Error: pipeline_preprocesamiento.py no encontrado en este directorio.")
            sys.exit(1)
        ruta_video = generar_video_sintetico("test_deteccion.mp4", n_frames=200)
    else:
        ruta_video = args[0]

    cfg_pipeline = ConfigPipeline(rtsp_url=ruta_video)
    cfg_detector = ConfigDetector(
        modelo_yolo="yolov5s",
        mostrar_ventana=not headless,
        guardar_video=True,
        ruta_video_salida="deteccion_vehicular.mp4",
    )

    detector = DetectorVehicular(cfg_pipeline, cfg_detector)

    try:
        detector.iniciar()
        detector.ejecutar()
    except ImportError as e:
        print(f"\n{e}\n")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error inesperado: {e}")
        raise