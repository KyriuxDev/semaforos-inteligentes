"""
core/pipeline/preprocesamiento.py — Pipeline de pre-procesamiento de imágenes.

Implementa las 7 etapas de pre-procesamiento descritas en la Actividad 8
del plan de trabajo (12/04/26 – 19/04/26):

  Etapa 1 — Captura y decodificación     (cv2.VideoCapture)
  Etapa 2 — Recorte ROI y redimensionado (letterbox)
  Etapa 3 — Conversión BGR→RGB y norm.   (/255.0)
  Etapa 4 — Mejora de contraste CLAHE    (condicional por luminancia)
  Etapa 5 — Reducción de ruido gaussiano (condicional por luminancia)
  Etapa 6 — Conversión a tensor PyTorch  (NCHW)
  Etapa 7 — Validación de calidad        (negro, congelado, borroso)

Latencia objetivo: ≤ 8 ms/frame @ 25 fps.
Protocolo de Investigación, sección 2.4.2 y 2.8.3.2.

Referencias:
    - Ultralytics (2024). YOLOv5 preprocessing — letterbox function.
    - CertiDevs (2025). OpenCV: biblioteca Python para procesamiento de imágenes.
    - OpenWebinars (2024). OpenCV: Introducción y su rol en la visión por computadora.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Optional

import cv2
import numpy as np

from config.settings import ConfigPipeline
from utils.logger import get_logger

logger = get_logger("core.pipeline.preprocesamiento")

# ---------------------------------------------------------------------------
# Importaciones opcionales
# ---------------------------------------------------------------------------
try:
    import torch
    _TORCH_DISPONIBLE = True
except ImportError:
    _TORCH_DISPONIBLE = False

try:
    from skimage.metrics import structural_similarity as ssim
    _SKIMAGE_DISPONIBLE = True
except ImportError:
    _SKIMAGE_DISPONIBLE = False


# ===========================================================================
# Funciones auxiliares de procesamiento de imagen
# ===========================================================================

def aplicar_roi(frame: np.ndarray, poligono: Optional[list]) -> np.ndarray:
    """Recorta el frame al polígono ROI; retorna el frame sin cambios si None."""
    if poligono is None:
        return frame
    mascara = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mascara, [np.array(poligono, dtype=np.int32)], 255)
    return cv2.bitwise_and(frame, frame, mask=mascara)


def redimensionar_letterbox(
    frame: np.ndarray,
    tam_destino: tuple,
    color_relleno: tuple = (114, 114, 114),
) -> np.ndarray:
    """
    Redimensiona conservando la relación de aspecto (letterboxing).
    Estándar YOLOv5; relleno gris 114 en los bordes.
    Ultralytics (2024).
    """
    h_orig, w_orig = frame.shape[:2]
    w_dest, h_dest = tam_destino
    escala   = min(w_dest / w_orig, h_dest / h_orig)
    w_nuevo  = int(w_orig * escala)
    h_nuevo  = int(h_orig * escala)
    f_redim  = cv2.resize(frame, (w_nuevo, h_nuevo), interpolation=cv2.INTER_LINEAR)
    pt       = (h_dest - h_nuevo) // 2
    pb       = h_dest - h_nuevo - pt
    pl       = (w_dest - w_nuevo) // 2
    pr       = w_dest - w_nuevo - pl
    return cv2.copyMakeBorder(f_redim, pt, pb, pl, pr, cv2.BORDER_CONSTANT, value=color_relleno)


def calcular_luminancia_media(frame_rgb: np.ndarray) -> float:
    """Retorna la luminancia media del canal L en espacio LAB."""
    u8  = (frame_rgb * 255).astype(np.uint8)
    lab = cv2.cvtColor(cv2.cvtColor(u8, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2LAB)
    return float(lab[:, :, 0].mean())


def aplicar_clahe(frame_rgb: np.ndarray, clip_limit: float, tile_grid: tuple) -> np.ndarray:
    """CLAHE sobre el canal L para mejorar contraste en condiciones adversas."""
    u8  = (frame_rgb * 255).astype(np.uint8)
    lab = cv2.cvtColor(cv2.cvtColor(u8, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(cv2.cvtColor(lab, cv2.COLOR_LAB2BGR), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def aplicar_reduccion_ruido(frame_rgb: np.ndarray, kernel: tuple) -> np.ndarray:
    """Filtro gaussiano para eliminar ruido de alta frecuencia."""
    u8 = (frame_rgb * 255).astype(np.uint8)
    return cv2.GaussianBlur(u8, kernel, sigmaX=0).astype(np.float32) / 255.0


def frame_a_tensor(frame_rgb: np.ndarray, dispositivo: str, batch_size: int = 1):
    """
    Convierte frame NumPy HWC → tensor PyTorch NCHW.
    Si PyTorch no está disponible retorna arreglo NumPy CHW.
    """
    chw = np.ascontiguousarray(frame_rgb.transpose(2, 0, 1))
    if not _TORCH_DISPONIBLE:
        logger.warning("PyTorch no disponible — retornando NumPy CHW.")
        return chw
    tensor = torch.from_numpy(chw).float().unsqueeze(0)
    try:
        tensor = tensor.to(dispositivo)
    except RuntimeError as exc:
        logger.warning(f"No se pudo mover tensor a '{dispositivo}': {exc}. Usando CPU.")
        tensor = tensor.to("cpu")
    return tensor


# ===========================================================================
# Validador de calidad (Etapa 7)
# ===========================================================================

class ValidadorCalidad:
    """
    Detecta frames corruptos antes de la inferencia:
      · Nivel de negro demasiado bajo (frame negro/corrupto)
      · SSIM > umbral (stream congelado)
      · Varianza del Laplaciano baja (frame borroso)

    Ante un frame inválido retorna el último frame válido del buffer FIFO.
    Protocolo, sección 2.5.2 — Detección en tiempo real.
    """

    def __init__(self, cfg: ConfigPipeline):
        self.cfg             = cfg
        self.buffer_validos  = deque(maxlen=cfg.tam_buffer_fallback)
        self.frame_anterior: Optional[np.ndarray] = None
        self.conteo_rechazados = 0

    def validar(self, frame_rgb_f32: np.ndarray) -> np.ndarray:
        u8 = (frame_rgb_f32 * 255).astype(np.uint8)

        if float(u8.mean()) < self.cfg.umbral_frame_negro:
            self.conteo_rechazados += 1
            logger.warning("Frame rechazado: nivel medio demasiado bajo (negro).")
            return self._fallback(frame_rgb_f32)

        if _SKIMAGE_DISPONIBLE and self.frame_anterior is not None:
            sim = ssim(u8, self.frame_anterior, channel_axis=2, data_range=255)
            if sim > self.cfg.umbral_ssim_congelado:
                self.conteo_rechazados += 1
                logger.warning("Frame rechazado: SSIM alto (stream congelado).")
                return self._fallback(frame_rgb_f32)

        gris = cv2.cvtColor(u8, cv2.COLOR_RGB2GRAY)
        if cv2.Laplacian(gris, cv2.CV_64F).var() < self.cfg.umbral_var_laplaciano:
            self.conteo_rechazados += 1
            logger.warning("Frame rechazado: varianza Laplaciano baja (borroso).")
            return self._fallback(frame_rgb_f32)

        self.buffer_validos.append(frame_rgb_f32.copy())
        self.frame_anterior = u8.copy()
        return frame_rgb_f32

    def _fallback(self, frame_original: np.ndarray) -> np.ndarray:
        if self.buffer_validos:
            logger.info("Usando frame de fallback del buffer.")
            return self.buffer_validos[-1]
        logger.warning("Buffer vacío — retornando frame sin validar.")
        return frame_original


# ===========================================================================
# Pipeline principal
# ===========================================================================

class PipelinePreprocesamiento:
    """
    Orquesta las 7 etapas de pre-procesamiento para cada fotograma.

    Ejemplo de uso
    --------------
    >>> from config import ConfigPipeline
    >>> from core.pipeline.preprocesamiento import PipelinePreprocesamiento
    >>> cfg = ConfigPipeline(rtsp_url="data/video.mp4")
    >>> pipeline = PipelinePreprocesamiento(cfg)
    >>> pipeline.iniciar()
    >>> for tensor in pipeline.generar_tensores():
    ...     resultado = modelo_yolo(tensor)
    >>> pipeline.detener()
    """

    def __init__(self, cfg: ConfigPipeline):
        self.cfg      = cfg
        self.captura: Optional[cv2.VideoCapture] = None
        self.validador = ValidadorCalidad(cfg)
        self._activo   = False
        self.metricas  = {
            "frames_procesados": 0,
            "tiempo_total_ms":   0.0,
        }

    def iniciar(self) -> None:
        logger.info(f"Iniciando captura: {self.cfg.rtsp_url}")
        self.captura = cv2.VideoCapture(self.cfg.rtsp_url)
        if not self.captura.isOpened():
            raise IOError(f"No se pudo abrir: {self.cfg.rtsp_url}")
        self.captura.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.captura.set(cv2.CAP_PROP_FPS, self.cfg.fps_objetivo)
        w   = int(self.captura.get(cv2.CAP_PROP_FRAME_WIDTH))
        h   = int(self.captura.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.captura.get(cv2.CAP_PROP_FPS)
        logger.info(f"Captura activa — {w}×{h} px | {fps:.1f} fps | {self.cfg.dispositivo_torch.upper()}")
        self._activo = True

    def detener(self) -> None:
        self._activo = False
        if self.captura and self.captura.isOpened():
            self.captura.release()
            logger.info("Captura liberada.")
        n = max(self.metricas["frames_procesados"], 1)
        logger.info(
            f"Pipeline — Frames: {self.metricas['frames_procesados']} | "
            f"Rechazados: {self.validador.conteo_rechazados} | "
            f"Lat. prom.: {self.metricas['tiempo_total_ms']/n:.2f} ms"
        )

    def procesar_frame(self, frame_bgr: np.ndarray):
        """Aplica etapas 2–7 y retorna tensor listo para YOLOv5."""
        t0 = time.perf_counter()

        frame = aplicar_roi(frame_bgr, self.cfg.roi_poligono)
        frame = redimensionar_letterbox(frame, self.cfg.tam_entrada_modelo)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        l_mean = calcular_luminancia_media(frame)

        if l_mean < self.cfg.umbral_luz_baja:
            frame = aplicar_clahe(frame, self.cfg.clahe_clip_limit, self.cfg.clahe_tile_grid)
            logger.debug(f"CLAHE activado — L_mean={l_mean:.1f}")

        if l_mean <= self.cfg.umbral_luz_alta:
            frame = aplicar_reduccion_ruido(frame, self.cfg.kernel_blur)

        frame  = self.validador.validar(frame)
        tensor = frame_a_tensor(frame, self.cfg.dispositivo_torch, self.cfg.batch_size)

        lat_ms = (time.perf_counter() - t0) * 1000
        self.metricas["frames_procesados"] += 1
        self.metricas["tiempo_total_ms"]   += lat_ms
        logger.debug(f"Frame {self.metricas['frames_procesados']} | L={l_mean:.1f} | {lat_ms:.2f} ms")

        return tensor

    def generar_tensores(self):
        """Generador: captura → procesa → yield tensor por frame."""
        if not self._activo or self.captura is None:
            raise RuntimeError("Llama a iniciar() primero.")
        logger.info("Generando tensores...")
        while self._activo:
            ret, frame_bgr = self.captura.read()
            if not ret:
                logger.warning("Frame no disponible — reintentando...")
                time.sleep(1.0 / self.cfg.fps_objetivo)
                continue
            yield self.procesar_frame(frame_bgr)
