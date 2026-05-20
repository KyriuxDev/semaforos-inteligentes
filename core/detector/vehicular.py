"""
core/detector/vehicular.py — Detector vehicular en tiempo real con YOLOv5.

Implementa la Actividad 9 del plan de trabajo (20/04/26 – 27/04/26):
"Especificar configuración y parámetros de YOLOv5 para detección vehicular".

Recibe frames BGR del pipeline (Actividad 8), ejecuta inferencia YOLOv5 con
los parámetros operativos de ConfigDetector y produce ResultadoDeteccion con:
  · Conteo de vehículos por tipo y por carril.
  · Nivel de congestión clasificado en tres bandas.
  · Tiempo de verde recomendado por carril (fórmula adaptativa).

La salida (ResultadoDeteccion) es la entrada del motor de decisión (Act. 10).

Referencias:
    - San Miguel, S. (2024). Revista ConCiencia Joven, 2, 32-38.
    - ScienceDirect (2024). YOLOv5 — an overview.
    - Blogs ETSII URJC (2025). Sistema de semáforos inteligente.
    - Ultralytics (2024). Comprehensive Guide to Ultralytics YOLOv5.
    - Protocolo de Investigación, sección 2.3.6 — Parámetros operativos.
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Dict, List, Optional

import cv2
import numpy as np

from config.settings import (
    CLASES_VEHICULO,
    COLORES_CLASE,
    UMBRAL_CONGESTION_LIBRE,
    UMBRAL_CONGESTION_MODERADA,
    ConfigDetector,
    ConfigPipeline,
)
from models.schemas import ResultadoDeteccion
from utils.logger import get_logger

logger = get_logger("core.detector.vehicular")

# ---------------------------------------------------------------------------
# Importaciones opcionales
# ---------------------------------------------------------------------------
try:
    from ultralytics import YOLO
    _ULTRALYTICS_OK = True
except ImportError:
    _ULTRALYTICS_OK = False

try:
    import torch
    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False


# ===========================================================================
# DetectorVehicular
# ===========================================================================

class DetectorVehicular:
    """
    Detector de vehículos en tiempo real basado en YOLOv5 + OpenCV.

    Integra el pipeline de pre-procesamiento (Actividad 8) con el modelo
    YOLOv5, configurado según la Actividad 9, y emite ResultadoDeteccion
    listos para el motor de decisión semafórico (Actividad 10).

    Parámetros operativos (protocolo, sección 2.3.6):
      · Modelo:      yolov5s (balance velocidad/precisión)
      · Confianza:   0.45
      · IoU NMS:     0.45
      · Imagen:      640 × 640 px
      · t_verde:     max(12, min(60, n_veh × 4)) segundos/carril

    Ejemplo de uso
    --------------
    >>> from config import ConfigPipeline, ConfigDetector
    >>> from core.detector.vehicular import DetectorVehicular
    >>> detector = DetectorVehicular(ConfigPipeline(rtsp_url="data/video.mp4"),
    ...                              ConfigDetector())
    >>> detector.iniciar()
    >>> detector.ejecutar()
    """

    def __init__(self, cfg_pipeline: ConfigPipeline, cfg_detector: ConfigDetector):
        self.cfg_p  = cfg_pipeline
        self.cfg_d  = cfg_detector
        self.modelo: Optional[object] = None
        self.writer: Optional[cv2.VideoWriter] = None

        self._metricas: Dict = {
            "frames_totales":             0,
            "vehiculos_detectados_total": 0,
            "latencia_acum_ms":           0.0,
            "maximo_vehiculos_frame":     0,
            "conteo_por_tipo_sesion":     defaultdict(int),
            "frames_por_nivel": {
                "libre": 0, "moderado": 0, "congestionado": 0,
            },
        }

    # ------------------------------------------------------------------
    # Ciclo de vida
    # ------------------------------------------------------------------

    def iniciar(self) -> None:
        """Verifica dependencias y carga el modelo YOLOv5."""
        self._verificar_dependencias()
        self._cargar_modelo()
        logger.info(
            f"Detector listo — modelo={self.cfg_d.modelo_yolo} | "
            f"dispositivo={self.cfg_d.dispositivo.upper()} | "
            f"conf={self.cfg_d.confianza_minima} | "
            f"iou={self.cfg_d.iou_umbral}"
        )
        logger.info(
            f"Temporización adaptativa — "
            f"base={self.cfg_d.tiempo_base_por_vehiculo_s} s/veh | "
            f"mín={self.cfg_d.tiempo_verde_minimo_s} s | "
            f"máx={self.cfg_d.tiempo_verde_maximo_s} s"
        )

    def detener(self) -> None:
        """Libera recursos y muestra el resumen de la sesión."""
        if self.writer:
            self.writer.release()
        cv2.destroyAllWindows()
        self._log_metricas()

    def _verificar_dependencias(self) -> None:
        errores = []
        if not _TORCH_OK:
            errores.append(
                "PyTorch no encontrado.\n"
                "  pip install torch torchvision "
                "--index-url https://download.pytorch.org/whl/cpu"
            )
        if not _ULTRALYTICS_OK:
            errores.append(
                "Ultralytics no encontrado.\n  pip install ultralytics"
            )
        if errores:
            raise ImportError("\nDependencias faltantes:\n" + "\n".join(errores))

    def _cargar_modelo(self) -> None:
        logger.info(f"Cargando {self.cfg_d.modelo_yolo}...")
        try:
            self.modelo = YOLO(f"{self.cfg_d.modelo_yolo}.pt")
            logger.info(f"{self.cfg_d.modelo_yolo} cargado correctamente.")
        except Exception as exc:
            raise RuntimeError(
                f"No se pudo cargar '{self.cfg_d.modelo_yolo}': {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Lógica de clasificación
    # ------------------------------------------------------------------

    def _asignar_carril(self, cx: int, ancho: int) -> str:
        tercio = ancho // 3
        if cx < tercio:     return "norte"
        if cx < 2*tercio:   return "centro"
        return "sur"

    def _clasificar_congestion(self, n: int) -> str:
        """
        Clasifica el nivel de congestión del frame.
        · libre         → 0–2 vehículos
        · moderado      → 3–6 vehículos
        · congestionado → 7+  vehículos
        Protocolo, sección 2.2.7; San Miguel (2024).
        """
        if n <= UMBRAL_CONGESTION_LIBRE:
            return "libre"
        if n <= UMBRAL_CONGESTION_MODERADA:
            return "moderado"
        return "congestionado"

    def _calcular_tiempo_verde(self, n_veh: int) -> float:
        """
        Tiempo de verde recomendado para un carril con n_veh vehículos.
        Fórmula: max(t_mín, min(t_máx, n_veh × t_base))
        Referencia: Blogs ETSII URJC (2025); protocolo, sec. 2.3.5–2.3.6.
        """
        return max(
            self.cfg_d.tiempo_verde_minimo_s,
            min(
                self.cfg_d.tiempo_verde_maximo_s,
                n_veh * self.cfg_d.tiempo_base_por_vehiculo_s,
            ),
        )

    # ------------------------------------------------------------------
    # Inferencia
    # ------------------------------------------------------------------

    def inferir(self, frame_bgr: np.ndarray, n_frame: int) -> ResultadoDeteccion:
        """
        Ejecuta YOLOv5 sobre un frame y retorna el ResultadoDeteccion.

        Parámetros
        ----------
        frame_bgr : np.ndarray
            Frame BGR capturado por el pipeline.
        n_frame : int
            Número secuencial del fotograma.

        Retorna
        -------
        ResultadoDeteccion
        """
        t0 = time.perf_counter()

        resultados = self.modelo.predict(
            source=frame_bgr,
            conf=self.cfg_d.confianza_minima,
            iou=self.cfg_d.iou_umbral,
            imgsz=self.cfg_d.tam_imagen,
            device=self.cfg_d.dispositivo,
            verbose=False,
            classes=list(CLASES_VEHICULO.keys()),
        )

        latencia_ms   = (time.perf_counter() - t0) * 1000
        detecciones:  List[dict]       = []
        conteo_tipo:  Dict[str, int]   = defaultdict(int)
        conteo_carril: Dict[str, int]  = defaultdict(int)
        h, w = frame_bgr.shape[:2]

        for r in resultados:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                if cls_id not in CLASES_VEHICULO:
                    continue
                conf       = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy     = (x1 + x2) // 2, (y1 + y2) // 2
                nombre_cls = CLASES_VEHICULO[cls_id]
                carril = self._asignar_carril(cx, w)

                detecciones.append({
                    "clase_id":  cls_id,
                    "clase":     nombre_cls,
                    "confianza": round(conf, 3),
                    "bbox":      (x1, y1, x2, y2),
                    "centro":    (cx, cy),
                    "carril":    carril,
                })
                conteo_tipo[nombre_cls] += 1
                conteo_carril[carril]   += 1

        total         = len(detecciones)
        nivel         = self._clasificar_congestion(total)
        tiempos_verde = {
            c: self._calcular_tiempo_verde(conteo_carril.get(c, 0))
            for c in ("norte", "centro", "sur")
        }

        return ResultadoDeteccion(
            n_frame=n_frame,
            timestamp=time.time(),
            total_vehiculos=total,
            conteo_por_tipo=dict(conteo_tipo),
            conteo_por_carril=dict(conteo_carril),
            tiempo_verde_recomendado_s=tiempos_verde,
            detecciones_raw=detecciones,
            latencia_ms=round(latencia_ms, 2),
            nivel_congestion=nivel,
        )

    # ------------------------------------------------------------------
    # Visualización
    # ------------------------------------------------------------------

    def anotar_frame(self, frame_bgr: np.ndarray, res: ResultadoDeteccion) -> np.ndarray:
        """Dibuja bounding boxes, líneas de carril y HUD sobre el frame."""
        frame = frame_bgr.copy()
        h, w  = frame.shape[:2]

        # # Líneas virtuales de carril
        # for linea in self.cfg_d.lineas_conteo:
        #     y_px = int(linea["y"] * h / self.cfg_d.tam_imagen)
        #     cv2.line(frame, (0, y_px), (w, y_px), linea["color"], 1)
        #     cv2.putText(frame, linea["nombre"], (4, y_px - 5),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.38, linea["color"], 1)

        # Bounding boxes
        for det in res.detecciones_raw:
            x1, y1, x2, y2 = det["bbox"]
            color = COLORES_CLASE.get(det["clase_id"], (200, 200, 200))
            etiq  = f"{det['clase']} {det['confianza']:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            (tw, th), _ = cv2.getTextSize(etiq, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(frame, etiq, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            cv2.circle(frame, det["centro"], 3, color, -1)

        # HUD
        color_nv = {"libre": (0, 200, 80), "moderado": (0, 165, 255),
                    "congestionado": (0, 0, 220)}.get(res.nivel_congestion, (200, 200, 200))
        overlay  = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 82), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        cv2.putText(frame,
                    f"Vehiculos: {res.total_vehiculos}  |  "
                    f"Congestion: {res.nivel_congestion.upper()}  |  "
                    f"Frame: {res.n_frame}",
                    (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color_nv, 1)
        cv2.putText(frame,
                    f"Lat. YOLOv5: {res.latencia_ms:.1f} ms  |  Tipos: {res.conteo_por_tipo}",
                    (8, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)
        t_str = "  ".join(f"{c}: {t:.0f}s" for c, t in res.tiempo_verde_recomendado_s.items())
        cv2.putText(frame, f"t_verde → {t_str}",
                    (8, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 220, 255), 1)
        cv2.putText(frame, "ITO — Sistema Semaforos Inteligentes | Act. 9: YOLOv5",
                    (8, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.37, (110, 110, 110), 1)
        return frame

    # ------------------------------------------------------------------
    # Ejecución principal
    # ------------------------------------------------------------------

    def ejecutar(self) -> None:
        """
        Bucle principal: captura → inferencia → visualización.
        Controles: q = salir | p = pausar | s = captura PNG.
        """
        captura = cv2.VideoCapture(self.cfg_p.rtsp_url)
        if not captura.isOpened():
            raise IOError(f"No se pudo abrir: {self.cfg_p.rtsp_url}")

        fps_src = captura.get(cv2.CAP_PROP_FPS) or 25
        w_src   = int(captura.get(cv2.CAP_PROP_FRAME_WIDTH))
        h_src   = int(captura.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if self.cfg_d.guardar_video:
            fourcc     = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(
                self.cfg_d.ruta_video_salida, fourcc, fps_src, (w_src, h_src)
            )
            logger.info(f"Grabando video → {self.cfg_d.ruta_video_salida}")

        logger.info("Detección iniciada — q: salir | p: pausar | s: captura PNG")

        n_frame       = 0
        pausado       = False
        frame_anotado: Optional[np.ndarray] = None

        try:
            while True:
                if not pausado:
                    ret, frame_bgr = captura.read()
                    if not ret:
                        logger.info("Fin de video o stream interrumpido.")
                        break

                    n_frame  += 1
                    resultado = self.inferir(frame_bgr, n_frame)
                    self._actualizar_metricas(resultado)
                    logger.info(str(resultado))

                    frame_anotado = self.anotar_frame(frame_bgr, resultado)
                    if self.cfg_d.guardar_video and self.writer:
                        self.writer.write(frame_anotado)

                if self.cfg_d.mostrar_ventana and frame_anotado is not None:
                    cv2.imshow("Deteccion YOLOv5 — ITO Semaforos Inteligentes", frame_anotado)
                    tecla = cv2.waitKey(1) & 0xFF
                    if tecla == ord("q"):
                        logger.info("Salida por usuario.")
                        break
                    elif tecla == ord("p"):
                        pausado = not pausado
                        logger.info("PAUSADO" if pausado else "REANUDADO")
                    elif tecla == ord("s") and not pausado:
                        nombre = f"data/captura_{n_frame:05d}.png"
                        cv2.imwrite(nombre, frame_anotado)
                        logger.info(f"Captura guardada: {nombre}")

        except KeyboardInterrupt:
            logger.info("Interrupción por teclado.")
        finally:
            captura.release()
            self.detener()

    # ------------------------------------------------------------------
    # Métricas internas
    # ------------------------------------------------------------------

    def _actualizar_metricas(self, res: ResultadoDeteccion) -> None:
        m = self._metricas
        m["frames_totales"]             += 1
        m["vehiculos_detectados_total"] += res.total_vehiculos
        m["latencia_acum_ms"]           += res.latencia_ms
        m["maximo_vehiculos_frame"]      = max(m["maximo_vehiculos_frame"], res.total_vehiculos)
        m["frames_por_nivel"][res.nivel_congestion] += 1
        for tipo, cnt in res.conteo_por_tipo.items():
            m["conteo_por_tipo_sesion"][tipo] += cnt

    def _log_metricas(self) -> None:
        m = self._metricas
        n = max(m["frames_totales"], 1)
        logger.info("=" * 68)
        logger.info("RESUMEN — Actividad 9: Configuración y parámetros YOLOv5")
        logger.info(f"  Modelo / dispositivo       : {self.cfg_d.modelo_yolo} / {self.cfg_d.dispositivo.upper()}")
        logger.info(f"  Confianza / IoU NMS        : {self.cfg_d.confianza_minima} / {self.cfg_d.iou_umbral}")
        logger.info("-" * 68)
        logger.info(f"  Frames procesados          : {m['frames_totales']}")
        logger.info(f"  Vehículos detectados total : {m['vehiculos_detectados_total']}")
        logger.info(f"  Promedio veh/frame         : {m['vehiculos_detectados_total']/n:.2f}")
        logger.info(f"  Máximo en un frame         : {m['maximo_vehiculos_frame']}")
        logger.info(f"  Latencia YOLOv5 promedio   : {m['latencia_acum_ms']/n:.1f} ms")
        logger.info(f"  Conteo por tipo            : {dict(m['conteo_por_tipo_sesion'])}")
        logger.info(f"  Frames por nivel           : {m['frames_por_nivel']}")
        logger.info("=" * 68)
