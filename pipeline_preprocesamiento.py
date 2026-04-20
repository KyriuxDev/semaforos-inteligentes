"""
================================================================================
TECNOLÓGICO NACIONAL DE MÉXICO — INSTITUTO TECNOLÓGICO DE OAXACA
Ingeniería en Sistemas Computacionales
Taller de Investigación II

Título del proyecto:
    Sistema de semáforos inteligentes basado en detección vehicular para
    optimizar el flujo de tráfico en Oaxaca de Juárez, Oaxaca.

Módulo:
    pipeline_preprocesamiento.py — Subsistema de procesamiento (Capa 2)
    Pipeline de pre-procesamiento de imágenes con OpenCV

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
    │                 │ de imágenes con OpenCV                     │                      │
    ├─────────────────┼────────────────────────────────────────────┼──────────────────────┤
    │ Actividad 9     │ Especificar configuración y parámetros     │ 20/04/26 – 27/04/26  │
    │ (parcial)       │ de YOLOv5 para detección vehicular         │                      │
    └─────────────────┴────────────────────────────────────────────┴──────────────────────┘

Metodología SCRUM — Fase 2: Modelado, Diseño y Refinamiento (Sprints de Desarrollo)
    Sprint actual: Diseño de arquitectura y pipeline (Sprints Intermedios)
    Fuente: Protocolo de Investigación, sección 2.8.3.2

Referencias principales:
    - San Miguel, S. (2024). Sistema de coordinación de semáforos inteligentes
      con algoritmos de inteligencia artificial. Revista ConCiencia Joven, 2, 32-38.
    - Monterrey Cañas et al. (2020). Diseño de un sistema de semaforización
      inteligente para controlar flujo vehicular a partir de procesamiento de imágenes.
    - Ultralytics (2024). Comprehensive Guide to Ultralytics YOLOv5.
      https://docs.ultralytics.com/yolov5/
    - OpenWebinars (2024). OpenCV: Introducción y su rol en la visión por computadora.

Dependencias requeridas:
    pip install opencv-python>=4.8.0
    pip install numpy>=1.24.0
    pip install torch>=2.0.0 torchvision>=0.15.0  (con soporte CUDA)
    pip install scikit-image>=0.21.0               (para SSIM en validación)

Versión:   1.0.0
Fecha:     Abril 2026
================================================================================
"""

import cv2
import numpy as np
import time
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

# scikit-image es opcional; solo se usa en la validación de calidad (Etapa 7)
try:
    from skimage.metrics import structural_similarity as ssim
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False

# torch es opcional en este módulo; se importa solo para la conversión a tensor (Etapa 6)
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# ---------------------------------------------------------------------------
# Configuración del logger
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("pipeline_preprocesamiento")


# ===========================================================================
# SECCIÓN 1 — CONFIGURACIÓN
# Centraliza todos los parámetros ajustables del pipeline.
# Corresponde a: sección 2.3.6 "Parámetros operativos" del marco teórico.
# ===========================================================================

@dataclass
class ConfigPipeline:
    """
    Parámetros operativos del pipeline de pre-procesamiento.

    Todos los valores tienen defaults basados en las especificaciones del
    protocolo de investigación y en los requisitos técnicos de YOLOv5
    (Ultralytics, 2024).

    Atributos
    ----------
    rtsp_url : str
        URL del stream RTSP de la cámara IP de la intersección.
    roi_poligono : list[tuple[int,int]]
        Vértices del polígono que delimita la Región de Interés (ROI).
        Configurable por intersección; por defecto cubre el fotograma completo.
    tam_entrada_modelo : tuple[int,int]
        Resolución objetivo para YOLOv5: (ancho, alto) en píxeles.
        Valor estándar: 640 × 640 px.
    fps_objetivo : int
        Cadencia de captura deseada. Debe coincidir con la cámara IP (≥ 25 fps).
    umbral_luz_baja : float
        Luminancia media del canal L (LAB) por debajo de la cual se activa CLAHE.
        Rango: 0–255. Valor recomendado: 80.
    umbral_luz_alta : float
        Luminancia media por encima de la cual se omite el filtro de ruido.
        Rango: 0–255. Valor recomendado: 120.
    clahe_clip_limit : float
        Límite de recorte para CLAHE. Controla la amplificación máxima de contraste.
        Valores altos amplifican ruido. Recomendado: 2.0.
    clahe_tile_grid : tuple[int,int]
        Tamaño de los bloques locales para CLAHE. Recomendado: (8, 8).
    kernel_blur : tuple[int,int]
        Tamaño del kernel gaussiano para reducción de ruido. Recomendado: (3, 3).
    tam_buffer_fallback : int
        Número de frames válidos a conservar como respaldo ante frames corruptos.
    umbral_var_laplaciano : float
        Varianza mínima del Laplaciano para considerar un frame nítido.
        Frames con varianza menor se descartan. Recomendado: 50.0.
    umbral_frame_negro : float
        Nivel medio de píxel por debajo del cual el frame se considera negro/corrupto.
    umbral_ssim_congelado : float
        Similitud estructural (SSIM) por encima de la cual el frame se considera
        congelado (idéntico al anterior). Recomendado: 0.999.
    dispositivo_torch : str
        Dispositivo de cómputo para el tensor: 'cuda' (GPU) o 'cpu'.
    batch_size : int
        Número de frames agrupados en un batch para inferencia simultánea.
        Rango recomendado: 1–4 según cantidad de cámaras.
    """

    # --- Fuente de video ---
    rtsp_url: str = "rtsp://192.168.1.100:554/stream"

    # --- ROI (Región de Interés) ---
    # None = usar el fotograma completo (modo de prueba)
    roi_poligono: Optional[list] = None

    # --- Redimensionado ---
    tam_entrada_modelo: tuple = (640, 640)  # px — estándar YOLOv5

    # --- Captura ---
    fps_objetivo: int = 25

    # --- Umbrales de iluminación ---
    umbral_luz_baja: float = 80.0   # activa CLAHE
    umbral_luz_alta: float = 120.0  # desactiva filtro de ruido

    # --- CLAHE (Etapa 4) ---
    clahe_clip_limit: float = 2.0
    clahe_tile_grid: tuple = (8, 8)

    # --- Filtro de ruido (Etapa 5) ---
    kernel_blur: tuple = (3, 3)

    # --- Validación de calidad (Etapa 7) ---
    tam_buffer_fallback: int = 3
    umbral_var_laplaciano: float = 50.0
    umbral_frame_negro: float = 5.0
    umbral_ssim_congelado: float = 0.999

    # --- Tensor y batch (Etapa 6) ---
    dispositivo_torch: str = "cuda" if (TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu"
    batch_size: int = 1


# ===========================================================================
# SECCIÓN 2 — FUNCIONES AUXILIARES
# Implementan operaciones atómicas reutilizables por el pipeline principal.
# ===========================================================================

def aplicar_roi(frame: np.ndarray, poligono: Optional[list]) -> np.ndarray:
    """
    Recorta el frame a la Región de Interés definida por un polígono.

    Si no se especifica polígono, devuelve el frame sin modificar.
    El polígono debe cubrir al menos los carriles vehiculares de la
    intersección, excluyendo cielo, edificios y zonas peatonales que
    no aporten información al conteo vehicular.

    Parámetros
    ----------
    frame : np.ndarray
        Frame BGR de entrada (H × W × 3, uint8).
    poligono : list[tuple[int,int]] | None
        Lista de vértices (x, y) del polígono ROI en coordenadas de píxel.
        Ejemplo: [(100, 200), (540, 200), (540, 800), (100, 800)]

    Retorna
    -------
    np.ndarray
        Frame recortado con máscara negra fuera del ROI (misma forma que entrada).

    Referencias
    -----------
    Protocolo, sección 2.8.3.2 — Diseño de pipeline de pre-procesamiento.
    """
    if poligono is None:
        return frame

    mascara = np.zeros(frame.shape[:2], dtype=np.uint8)
    pts = np.array(poligono, dtype=np.int32)
    cv2.fillPoly(mascara, [pts], 255)
    return cv2.bitwise_and(frame, frame, mask=mascara)


def redimensionar_letterbox(
    frame: np.ndarray,
    tam_destino: tuple,
    color_relleno: tuple = (114, 114, 114),
) -> np.ndarray:
    """
    Redimensiona el frame conservando la relación de aspecto (letterboxing).

    El letterboxing agrega bandas de color neutro (gris 114) en los bordes
    para alcanzar el tamaño cuadrado requerido por YOLOv5 sin distorsionar
    los vehículos, lo que podría afectar negativamente la precisión del modelo.

    Parámetros
    ----------
    frame : np.ndarray
        Frame BGR de entrada.
    tam_destino : tuple[int, int]
        (ancho, alto) en píxeles del frame de salida.
    color_relleno : tuple[int, int, int]
        Color BGR del relleno. Por convención YOLOv5: (114, 114, 114).

    Retorna
    -------
    np.ndarray
        Frame redimensionado con letterbox, forma (alto, ancho, 3).

    Referencias
    -----------
    Ultralytics (2024). YOLOv5 preprocessing — letterbox function.
    """
    h_orig, w_orig = frame.shape[:2]
    w_dest, h_dest = tam_destino

    escala = min(w_dest / w_orig, h_dest / h_orig)
    w_nuevo = int(w_orig * escala)
    h_nuevo = int(h_orig * escala)

    frame_redim = cv2.resize(frame, (w_nuevo, h_nuevo), interpolation=cv2.INTER_LINEAR)

    # Calcular padding simétrico
    pad_top = (h_dest - h_nuevo) // 2
    pad_bottom = h_dest - h_nuevo - pad_top
    pad_left = (w_dest - w_nuevo) // 2
    pad_right = w_dest - w_nuevo - pad_left

    frame_lb = cv2.copyMakeBorder(
        frame_redim,
        pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT,
        value=color_relleno,
    )
    return frame_lb


def calcular_luminancia_media(frame_rgb: np.ndarray) -> float:
    """
    Calcula la luminancia media del canal L en el espacio de color LAB.

    Se utiliza como indicador del nivel de iluminación de la escena para
    decidir condicionalmente la activación de CLAHE (baja iluminación)
    y el filtro de ruido (alta iluminación).

    Parámetros
    ----------
    frame_rgb : np.ndarray
        Frame en espacio RGB, float32, rango [0, 1].

    Retorna
    -------
    float
        Luminancia media en escala 0–255.
    """
    frame_uint8 = (frame_rgb * 255).astype(np.uint8)
    frame_bgr = cv2.cvtColor(frame_uint8, cv2.COLOR_RGB2BGR)
    frame_lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    return float(frame_lab[:, :, 0].mean())


def aplicar_clahe(frame_rgb: np.ndarray, clip_limit: float, tile_grid: tuple) -> np.ndarray:
    """
    Aplica CLAHE (Contrast Limited Adaptive Histogram Equalization) al canal L.

    La ecualización adaptativa local mejora la visibilidad de vehículos en
    condiciones de contraluz (faros), niebla o lluvia intensa, frecuentes en
    Oaxaca durante la temporada de lluvias (mayo–octubre).

    Solo se modifica el canal de luminancia (L) en el espacio LAB, preservando
    la crominancia (A y B) para evitar alteraciones de color.

    Parámetros
    ----------
    frame_rgb : np.ndarray
        Frame RGB float32, rango [0, 1].
    clip_limit : float
        Límite de recorte del histograma. Recomendado: 2.0.
    tile_grid : tuple[int, int]
        Tamaño de los bloques locales. Recomendado: (8, 8).

    Retorna
    -------
    np.ndarray
        Frame RGB float32 con contraste mejorado, rango [0, 1].

    Referencias
    -----------
    CertiDevs (2025). OpenCV: biblioteca Python para procesamiento de imágenes.
    Protocolo de investigación, sección 2.4.2 — Procesamiento de imágenes.
    """
    frame_uint8 = (frame_rgb * 255).astype(np.uint8)
    frame_bgr = cv2.cvtColor(frame_uint8, cv2.COLOR_RGB2BGR)
    frame_lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    frame_lab[:, :, 0] = clahe.apply(frame_lab[:, :, 0])

    frame_bgr_mejorado = cv2.cvtColor(frame_lab, cv2.COLOR_LAB2BGR)
    frame_rgb_mejorado = cv2.cvtColor(frame_bgr_mejorado, cv2.COLOR_BGR2RGB)
    return frame_rgb_mejorado.astype(np.float32) / 255.0


def aplicar_reduccion_ruido(frame_rgb: np.ndarray, kernel: tuple) -> np.ndarray:
    """
    Aplica un filtro gaussiano para eliminar ruido de alta frecuencia.

    Reduce artefactos de compresión H.265 (blockiness) y ruido electrónico
    del sensor CMOS en condiciones de baja iluminación. El kernel 3×3 es
    suficiente para eliminar ruido sin degradar los bordes vehiculares que
    YOLOv5 utiliza para localizar bounding boxes.

    Parámetros
    ----------
    frame_rgb : np.ndarray
        Frame RGB float32, rango [0, 1].
    kernel : tuple[int, int]
        Tamaño del kernel gaussiano. Recomendado: (3, 3).

    Retorna
    -------
    np.ndarray
        Frame suavizado, misma forma y dtype que la entrada.

    Referencias
    -----------
    CertiDevs (2025). OpenCV — filtros de suavizado.
    Protocolo de investigación, sección 2.4.2 — Procesamiento de imágenes.
    """
    frame_uint8 = (frame_rgb * 255).astype(np.uint8)
    frame_suavizado = cv2.GaussianBlur(frame_uint8, kernel, sigmaX=0)
    return frame_suavizado.astype(np.float32) / 255.0


def frame_a_tensor(frame_rgb: np.ndarray, dispositivo: str, batch_size: int = 1):
    """
    Convierte un frame NumPy al tensor NCHW requerido por PyTorch/YOLOv5.

    La conversión incluye:
      1. Transposición HWC → CHW (Height×Width×Channels → Channels×Height×Width)
      2. Garantía de contiguidad en memoria (necesaria para operaciones CUDA)
      3. Expansión de dimensión de batch (N=1 por defecto)
      4. Transferencia a dispositivo GPU (VRAM) si está disponible

    Parámetros
    ----------
    frame_rgb : np.ndarray
        Frame RGB float32 (H × W × 3), rango [0, 1].
    dispositivo : str
        'cuda' para GPU o 'cpu' para procesamiento en CPU.
    batch_size : int
        Número de frames en el batch. Normalmente 1 por cámara.

    Retorna
    -------
    torch.Tensor | np.ndarray
        Tensor NCHW float32 en el dispositivo especificado.
        Si PyTorch no está disponible, retorna el arreglo NumPy CHW.

    Referencias
    -----------
    Ultralytics (2024). YOLOv5 — model inference pipeline.
    Protocolo de investigación, sección 2.6.2.1 — YOLOv5.
    """
    # HWC → CHW
    frame_chw = np.ascontiguousarray(frame_rgb.transpose(2, 0, 1))

    if not TORCH_AVAILABLE:
        logger.warning("PyTorch no disponible — retornando NumPy CHW (modo degradado).")
        return frame_chw

    tensor = torch.from_numpy(frame_chw).float()
    tensor = tensor.unsqueeze(0)  # añade dimensión de batch: (1, C, H, W)

    try:
        tensor = tensor.to(dispositivo)
    except RuntimeError as e:
        logger.warning(f"No se pudo mover tensor a '{dispositivo}': {e}. Usando CPU.")
        tensor = tensor.to("cpu")

    return tensor


# ===========================================================================
# SECCIÓN 3 — VALIDACIÓN DE CALIDAD (Etapa 7)
# Detecta frames corruptos o degradados antes de la inferencia.
# Corresponde a la estrategia de resiliencia del sistema descrita en la
# sección 2.5.2 del protocolo (detección en tiempo real).
# ===========================================================================

class ValidadorCalidad:
    """
    Módulo de control de calidad de frames antes de inferencia YOLOv5.

    Implementa tres verificaciones independientes:
      - Detección de frame negro/corrupto (nivel medio de píxel)
      - Detección de frame congelado (SSIM con el frame anterior)
      - Detección de frame borroso (varianza del Laplaciano)

    Si un frame no supera alguna verificación, se usa el último frame
    válido almacenado en el buffer FIFO como fallback.

    Atributos
    ----------
    cfg : ConfigPipeline
        Configuración con umbrales de validación.
    buffer_validos : deque
        Buffer circular con los últimos N frames válidos.
    frame_anterior : np.ndarray | None
        Último frame procesado (para comparación SSIM).
    conteo_rechazados : int
        Acumulador de frames rechazados (para diagnóstico).
    """

    def __init__(self, cfg: ConfigPipeline):
        self.cfg = cfg
        self.buffer_validos: deque = deque(maxlen=cfg.tam_buffer_fallback)
        self.frame_anterior: Optional[np.ndarray] = None
        self.conteo_rechazados: int = 0

    def _es_negro(self, frame: np.ndarray) -> bool:
        """Retorna True si el frame está mayoritariamente negro (corrupto)."""
        return float(frame.mean()) < self.cfg.umbral_frame_negro

    def _esta_congelado(self, frame: np.ndarray) -> bool:
        """
        Retorna True si el frame es casi idéntico al anterior (stream congelado).
        Requiere scikit-image. Si no está disponible, omite esta verificación.
        """
        if not SKIMAGE_AVAILABLE or self.frame_anterior is None:
            return False
        similitud = ssim(frame, self.frame_anterior, channel_axis=2, data_range=255)
        return similitud > self.cfg.umbral_ssim_congelado

    def _esta_borroso(self, frame: np.ndarray) -> bool:
        """
        Retorna True si la varianza del Laplaciano es inferior al umbral.
        Frames muy borrosos reducen la precisión de detección de YOLOv5.
        """
        gris = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        varianza = cv2.Laplacian(gris, cv2.CV_64F).var()
        return varianza < self.cfg.umbral_var_laplaciano

    def validar(self, frame_rgb_f32: np.ndarray) -> np.ndarray:
        """
        Valida el frame y retorna un frame confiable para inferencia.

        Si el frame pasa todas las verificaciones, se agrega al buffer
        de fallback y se retorna. Si falla alguna, se retorna el frame
        válido más reciente del buffer. Si el buffer está vacío (inicio
        del sistema), se retorna el frame original con una advertencia.

        Parámetros
        ----------
        frame_rgb_f32 : np.ndarray
            Frame RGB float32 [0, 1] post-procesamiento.

        Retorna
        -------
        np.ndarray
            Frame validado (o fallback), misma forma y dtype.
        """
        frame_uint8 = (frame_rgb_f32 * 255).astype(np.uint8)

        # — Verificación 1: frame negro
        if self._es_negro(frame_uint8):
            self.conteo_rechazados += 1
            logger.warning("Frame rechazado: nivel medio de píxel demasiado bajo (frame negro).")
            return self._obtener_fallback(frame_rgb_f32)

        # — Verificación 2: frame congelado
        if self._esta_congelado(frame_uint8):
            self.conteo_rechazados += 1
            logger.warning("Frame rechazado: SSIM > umbral (stream posiblemente congelado).")
            return self._obtener_fallback(frame_rgb_f32)

        # — Verificación 3: frame borroso
        if self._esta_borroso(frame_uint8):
            self.conteo_rechazados += 1
            logger.warning("Frame rechazado: varianza del Laplaciano baja (frame borroso/corrupto).")
            return self._obtener_fallback(frame_rgb_f32)

        # Frame válido — actualizar buffer y referencia SSIM
        self.buffer_validos.append(frame_rgb_f32.copy())
        self.frame_anterior = frame_uint8.copy()
        return frame_rgb_f32

    def _obtener_fallback(self, frame_original: np.ndarray) -> np.ndarray:
        """Retorna el último frame válido del buffer, o el original si el buffer está vacío."""
        if self.buffer_validos:
            logger.info("Usando frame de fallback del buffer.")
            return self.buffer_validos[-1]
        logger.warning("Buffer de fallback vacío — retornando frame original sin validar.")
        return frame_original


# ===========================================================================
# SECCIÓN 4 — PIPELINE PRINCIPAL
# Orquesta las 7 etapas en secuencia para cada fotograma.
# Corresponde a la Actividad 8 del plan de trabajo (12/04/26 – 19/04/26).
# ===========================================================================

class PipelinePreprocesamiento:
    """
    Pipeline completo de pre-procesamiento de imágenes para el sistema
    de semáforos inteligentes basado en detección vehicular (YOLOv5).

    Implementa las 7 etapas secuenciales descritas en el protocolo de
    investigación (Actividad 8, Fase 2 SCRUM):

      Etapa 1 — Captura y decodificación     (cv2.VideoCapture)
      Etapa 2 — Recorte ROI y redimensionado (cv2.resize + letterbox)
      Etapa 3 — Conversión y normalización   (cvtColor + /255.0)
      Etapa 4 — Mejora de contraste CLAHE    (cv2.createCLAHE) [condicional]
      Etapa 5 — Reducción de ruido           (cv2.GaussianBlur) [condicional]
      Etapa 6 — Conversión a tensor CUDA     (torch.from_numpy + .to('cuda'))
      Etapa 7 — Validación de calidad        (ValidadorCalidad)

    Latencia objetivo total: ≤ 8 ms por fotograma @ 25 fps.

    Parámetros
    ----------
    cfg : ConfigPipeline
        Configuración completa del pipeline.

    Ejemplo de uso
    --------------
    >>> cfg = ConfigPipeline(rtsp_url="rtsp://192.168.1.100:554/stream")
    >>> pipeline = PipelinePreprocesamiento(cfg)
    >>> pipeline.iniciar()
    >>> try:
    ...     for tensor in pipeline.generar_tensores():
    ...         resultado = modelo_yolov5(tensor)  # inferencia
    ... finally:
    ...     pipeline.detener()
    """

    def __init__(self, cfg: ConfigPipeline):
        self.cfg = cfg
        self.captura: Optional[cv2.VideoCapture] = None
        self.validador = ValidadorCalidad(cfg)
        self._activo = False
        self.metricas = {
            "frames_procesados": 0,
            "frames_rechazados": 0,
            "tiempo_total_ms": 0.0,
        }

    # ------------------------------------------------------------------
    # Ciclo de vida
    # ------------------------------------------------------------------

    def iniciar(self) -> None:
        """
        Abre la fuente de video y configura la captura.

        Etapa 1 — Captura y decodificación.
        Configura el buffer mínimo (1 frame) para reducir latencia y
        asegura la cadencia de captura en fps_objetivo.
        """
        logger.info(f"Iniciando captura: {self.cfg.rtsp_url}")
        self.captura = cv2.VideoCapture(self.cfg.rtsp_url)

        if not self.captura.isOpened():
            raise IOError(
                f"No se pudo abrir la fuente de video: {self.cfg.rtsp_url}\n"
                "Verifique la URL RTSP, la red y las credenciales de la cámara."
            )

        # Minimizar latencia de buffer
        self.captura.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.captura.set(cv2.CAP_PROP_FPS, self.cfg.fps_objetivo)

        ancho = int(self.captura.get(cv2.CAP_PROP_FRAME_WIDTH))
        alto = int(self.captura.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_real = self.captura.get(cv2.CAP_PROP_FPS)

        logger.info(f"Captura activa — Resolución: {ancho}×{alto} px | FPS: {fps_real:.1f}")
        logger.info(f"Dispositivo de inferencia: {self.cfg.dispositivo_torch.upper()}")
        self._activo = True

    def detener(self) -> None:
        """Libera la captura y registra métricas finales."""
        self._activo = False
        if self.captura and self.captura.isOpened():
            self.captura.release()
            logger.info("Captura liberada.")

        if self.metricas["frames_procesados"] > 0:
            lat_prom = self.metricas["tiempo_total_ms"] / self.metricas["frames_procesados"]
            logger.info(
                f"Métricas finales — "
                f"Frames procesados: {self.metricas['frames_procesados']} | "
                f"Rechazados: {self.validador.conteo_rechazados} | "
                f"Latencia promedio: {lat_prom:.2f} ms"
            )

    # ------------------------------------------------------------------
    # Procesamiento de un fotograma
    # ------------------------------------------------------------------

    def procesar_frame(self, frame_bgr: np.ndarray):
        """
        Aplica las etapas 2 a 7 del pipeline sobre un fotograma BGR.

        Parámetros
        ----------
        frame_bgr : np.ndarray
            Frame BGR uint8 capturado en la Etapa 1.

        Retorna
        -------
        torch.Tensor | np.ndarray
            Tensor NCHW float32 listo para inferencia con YOLOv5.
        """
        t0 = time.perf_counter()

        # ── Etapa 2: Recorte ROI y redimensionado ──────────────────────
        frame_roi = aplicar_roi(frame_bgr, self.cfg.roi_poligono)
        frame_640 = redimensionar_letterbox(frame_roi, self.cfg.tam_entrada_modelo)

        # ── Etapa 3: Conversión BGR→RGB y normalización [0,1] ──────────
        frame_rgb = cv2.cvtColor(frame_640, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # ── Indicador de iluminación (determina etapas condicionales) ──
        l_mean = calcular_luminancia_media(frame_rgb)

        # ── Etapa 4: CLAHE (solo en baja iluminación) ──────────────────
        if l_mean < self.cfg.umbral_luz_baja:
            frame_rgb = aplicar_clahe(
                frame_rgb,
                self.cfg.clahe_clip_limit,
                self.cfg.clahe_tile_grid,
            )
            logger.debug(f"CLAHE activado — L_mean={l_mean:.1f}")

        # ── Etapa 5: Reducción de ruido (omitida en luz diurna clara) ──
        if l_mean <= self.cfg.umbral_luz_alta:
            frame_rgb = aplicar_reduccion_ruido(frame_rgb, self.cfg.kernel_blur)

        # ── Etapa 7: Validación de calidad y fallback ──────────────────
        frame_rgb = self.validador.validar(frame_rgb)

        # ── Etapa 6: Conversión a tensor CUDA ──────────────────────────
        tensor = frame_a_tensor(frame_rgb, self.cfg.dispositivo_torch, self.cfg.batch_size)

        # Registrar métricas
        latencia_ms = (time.perf_counter() - t0) * 1000
        self.metricas["frames_procesados"] += 1
        self.metricas["tiempo_total_ms"] += latencia_ms

        logger.debug(
            f"Frame {self.metricas['frames_procesados']} — "
            f"L_mean={l_mean:.1f} | Latencia={latencia_ms:.2f} ms"
        )

        return tensor

    # ------------------------------------------------------------------
    # Generador de tensores (interfaz principal)
    # ------------------------------------------------------------------

    def generar_tensores(self):
        """
        Generador que captura frames continuamente y los entrega como tensores.

        Itera sobre el stream de video mientras el pipeline esté activo,
        capturando cada fotograma (Etapa 1) y procesándolo con las etapas
        2 a 7. Cada iteración entrega un tensor listo para inferencia YOLOv5.

        Yields
        ------
        torch.Tensor | np.ndarray
            Tensor NCHW float32 por fotograma.

        Notas
        -----
        Para detener el generador, llamar a `pipeline.detener()` o
        interrumpir con KeyboardInterrupt.
        """
        if not self._activo or self.captura is None:
            raise RuntimeError("El pipeline no está iniciado. Llame primero a iniciar().")

        logger.info("Iniciando generación de tensores...")

        while self._activo:
            ret, frame_bgr = self.captura.read()

            if not ret:
                logger.warning("Frame no disponible — reintentando...")
                time.sleep(1.0 / self.cfg.fps_objetivo)
                continue

            yield self.procesar_frame(frame_bgr)


# ===========================================================================
# SECCIÓN 5 — MODO DE PRUEBA CON ARCHIVO DE VIDEO LOCAL
# Permite validar el pipeline sin conexión a cámara IP, usando un archivo
# .mp4 o .avi grabado previamente en campo (aforos vehiculares, Actividad 3).
# Corresponde a la Actividad 11 del plan de trabajo (simulación, 06/05/26).
# ===========================================================================

def ejecutar_prueba_local(ruta_video: str, cfg: Optional[ConfigPipeline] = None) -> None:
    """
    Ejecuta el pipeline sobre un archivo de video local para validación.

    Muestra cada tensor procesado como imagen en pantalla y registra
    métricas de latencia por frame. Útil durante la Actividad 11
    (simulación y optimización de parámetros operativos).

    Parámetros
    ----------
    ruta_video : str
        Ruta al archivo de video (.mp4, .avi, etc.).
    cfg : ConfigPipeline | None
        Configuración personalizada. Si es None, usa defaults.

    Ejemplo
    -------
    >>> ejecutar_prueba_local("video_interseccion_5senores.mp4")
    """
    if cfg is None:
        cfg = ConfigPipeline(rtsp_url=ruta_video)
    else:
        cfg.rtsp_url = ruta_video

    pipeline = PipelinePreprocesamiento(cfg)

    try:
        pipeline.iniciar()
        logger.info("Modo prueba local — presione 'q' para salir.")

        for tensor in pipeline.generar_tensores():
            # Reconstruir imagen para visualización (CHW → HWC, float→uint8)
            if TORCH_AVAILABLE and hasattr(tensor, "cpu"):
                frame_vis = tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
            else:
                frame_vis = tensor.transpose(1, 2, 0)

            frame_vis = (frame_vis * 255).astype(np.uint8)
            frame_bgr_vis = cv2.cvtColor(frame_vis, cv2.COLOR_RGB2BGR)

            # Sobreponer métricas en pantalla
            n = pipeline.metricas["frames_procesados"]
            lat = pipeline.metricas["tiempo_total_ms"] / max(n, 1)
            cv2.putText(
                frame_bgr_vis,
                f"Frame: {n} | Lat: {lat:.1f} ms | Dev: {cfg.dispositivo_torch.upper()}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1,
            )

            cv2.imshow("Pipeline pre-procesamiento — ITO Semaforos Inteligentes", frame_bgr_vis)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                logger.info("Salida manual por usuario.")
                break

    except IOError as e:
        logger.error(f"Error de captura: {e}")
    except KeyboardInterrupt:
        logger.info("Interrupción por teclado.")
    finally:
        pipeline.detener()
        cv2.destroyAllWindows()


# ===========================================================================
# SECCIÓN 6 — GENERADOR DE VIDEO SINTÉTICO DE PRUEBA
# Crea un MP4 con vehículos simulados para validar el pipeline sin necesidad
# de cámara IP ni video externo. Útil durante la Actividad 8 y la Actividad 11
# (simulación y optimización de parámetros, 06/05/26–13/05/26).
# ===========================================================================

def generar_video_sintetico(
    ruta_salida: str = "test_trafico.mp4",
    n_frames: int = 300,
    resolucion: tuple = (1280, 720),
    fps: int = 25,
) -> str:
    """
    Genera un video MP4 sintético con vehículos simulados en movimiento.

    Simula tres vehículos con trayectorias, colores y velocidades distintas
    sobre una calzada gris. Incluye carril, líneas de intersección, número
    de frame y timestamp para facilitar la depuración visual del pipeline.

    Parámetros
    ----------
    ruta_salida : str
        Ruta del archivo MP4 a generar.
    n_frames : int
        Número total de fotogramas (duración = n_frames / fps segundos).
    resolucion : tuple[int, int]
        (ancho, alto) del video en píxeles.
    fps : int
        Fotogramas por segundo del video generado.

    Retorna
    -------
    str
        Ruta del archivo generado.

    Ejemplo
    -------
    >>> ruta = generar_video_sintetico("mi_prueba.mp4", n_frames=150)
    >>> ejecutar_prueba_local(ruta)
    """
    ancho, alto = resolucion
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(ruta_salida, fourcc, fps, (ancho, alto))

    if not writer.isOpened():
        raise IOError(f"No se pudo crear el archivo de video: {ruta_salida}")

    logger.info(f"Generando video sintético: {ruta_salida} ({n_frames} frames @ {fps} fps)")

    for i in range(n_frames):
        # Fondo: asfalto gris con degradado sutil de carretera
        frame = np.full((alto, ancho, 3), 100, dtype=np.uint8)

        # Carriles (líneas blancas)
        y_carril_sup = alto // 3
        y_carril_inf = 2 * alto // 3
        cv2.line(frame, (0, y_carril_sup), (ancho, y_carril_sup), (200, 200, 200), 2)
        cv2.line(frame, (0, y_carril_inf), (ancho, y_carril_inf), (200, 200, 200), 2)

        # Línea central punteada
        for x in range(0, ancho, 60):
            cv2.line(frame, (x, alto // 2), (x + 30, alto // 2), (220, 220, 0), 2)

        # Línea de pare (intersección simulada)
        cv2.line(frame, (ancho // 2, 0), (ancho // 2, alto), (255, 255, 255), 1)
        cv2.putText(frame, "INTERSECCION", (ancho // 2 - 80, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # ── Vehículo 1: sedan azul, carril superior, izq → der ──
        x1 = (80 + i * 3) % (ancho + 120) - 120
        y1_t, y1_b = y_carril_sup + 10, y_carril_sup + 55
        cv2.rectangle(frame, (x1, y1_t), (x1 + 110, y1_b), (30, 144, 255), -1)
        cv2.rectangle(frame, (x1 + 15, y1_t - 18), (x1 + 95, y1_t), (100, 180, 255), -1)
        cv2.circle(frame, (x1 + 20, y1_b), 10, (40, 40, 40), -1)
        cv2.circle(frame, (x1 + 90, y1_b), 10, (40, 40, 40), -1)

        # ── Vehículo 2: camioneta naranja, carril central, izq → der (más lento) ──
        x2 = (300 + i * 2) % (ancho + 140) - 140
        y2_t, y2_b = alto // 2 - 30, alto // 2 + 30
        cv2.rectangle(frame, (x2, y2_t), (x2 + 140, y2_b), (0, 120, 255), -1)
        cv2.rectangle(frame, (x2 + 10, y2_t - 22), (x2 + 130, y2_t), (50, 160, 255), -1)
        cv2.circle(frame, (x2 + 25, y2_b), 12, (40, 40, 40), -1)
        cv2.circle(frame, (x2 + 115, y2_b), 12, (40, 40, 40), -1)

        # ── Vehículo 3: sedan verde, carril inferior, der → izq (sentido contrario) ──
        x3 = ancho - ((150 + i * 3) % (ancho + 120))
        y3_t, y3_b = y_carril_inf + 10, y_carril_inf + 55
        cv2.rectangle(frame, (x3, y3_t), (x3 + 110, y3_b), (50, 205, 50), -1)
        cv2.rectangle(frame, (x3 + 15, y3_t - 18), (x3 + 95, y3_t), (100, 230, 100), -1)
        cv2.circle(frame, (x3 + 20, y3_b), 10, (40, 40, 40), -1)
        cv2.circle(frame, (x3 + 90, y3_b), 10, (40, 40, 40), -1)

        # ── HUD de información ──
        segundos = i / fps
        cv2.rectangle(frame, (0, 0), (340, 52), (30, 30, 30), -1)
        cv2.putText(frame, f"Frame: {i+1}/{n_frames}  |  t={segundos:.2f}s",
                    (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 120), 1)
        cv2.putText(frame, "ITO — Sistema semaforos inteligentes [PRUEBA]",
                    (8, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

        writer.write(frame)

    writer.release()
    logger.info(f"Video sintético generado: {ruta_salida} "
                f"({n_frames / fps:.1f}s, {resolucion[0]}x{resolucion[1]}px)")
    return ruta_salida


# ===========================================================================
# SECCIÓN 7 — PUNTO DE ENTRADA
#
# Subcomandos disponibles:
#   python pipeline_preprocesamiento.py <video.mp4>   — procesa video existente
#   python pipeline_preprocesamiento.py --sintetico    — genera video y lo procesa
#   python pipeline_preprocesamiento.py --generar      — solo genera el video
#   python pipeline_preprocesamiento.py               — muestra ayuda
# ===========================================================================

if __name__ == "__main__":
    import sys

    AYUDA = """
Uso:
  python pipeline_preprocesamiento.py <ruta_video.mp4>
      Procesa un video existente (archivo local o stream RTSP).

  python pipeline_preprocesamiento.py --sintetico
      Genera un video sintético de prueba y lo procesa inmediatamente.
      Útil para validar el pipeline sin video externo (Actividad 8).

  python pipeline_preprocesamiento.py --generar [salida.mp4]
      Solo genera el video sintético sin procesarlo.
      Por defecto guarda como 'test_trafico.mp4'.

  python pipeline_preprocesamiento.py --sintetico --frames 500
      Genera un video sintético con 500 frames y lo procesa.

Ejemplos:
  python pipeline_preprocesamiento.py video.mp4
  python pipeline_preprocesamiento.py --sintetico
  python pipeline_preprocesamiento.py --generar mi_prueba.mp4
  python pipeline_preprocesamiento.py --sintetico --frames 150

Para integrar con YOLOv5 (Actividad 9):
  from pipeline_preprocesamiento import PipelinePreprocesamiento, ConfigPipeline
  cfg = ConfigPipeline(rtsp_url="rtsp://192.168.1.100:554/stream")
  modelo = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True)
  pipeline = PipelinePreprocesamiento(cfg)
  pipeline.iniciar()
  for tensor in pipeline.generar_tensores():
      resultados = modelo(tensor)
"""

    args = sys.argv[1:]

    # Extraer --frames N si está presente
    n_frames = 300
    if "--frames" in args:
        idx = args.index("--frames")
        try:
            n_frames = int(args[idx + 1])
            args = [a for j, a in enumerate(args) if j not in (idx, idx + 1)]
        except (IndexError, ValueError):
            print("Error: --frames requiere un número entero. Usando 300.")

    if not args:
        print(AYUDA)

    elif args[0] == "--sintetico":
        # Generar video sintético y procesarlo de inmediato
        ruta = generar_video_sintetico("test_trafico.mp4", n_frames=n_frames)
        ejecutar_prueba_local(ruta)

    elif args[0] == "--generar":
        # Solo generar el video, sin procesarlo
        nombre = args[1] if len(args) > 1 else "test_trafico.mp4"
        ruta = generar_video_sintetico(nombre, n_frames=n_frames)
        print(f"\nVideo listo: {ruta}")
        print(f"Para procesarlo: python pipeline_preprocesamiento.py {ruta}\n")

    else:
        # Procesar video indicado como argumento
        ejecutar_prueba_local(args[0])