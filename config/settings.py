"""
config/settings.py — Configuraciones y constantes del sistema.

Centraliza todos los parámetros operativos del proyecto para facilitar
el ajuste sin necesidad de modificar los módulos de lógica.

Secciones:
  A — Clases COCO vehiculares
  B — Umbrales de congestión
  C — Temporización semafórica adaptativa
  D — ConfigPipeline   (Actividad 8)
  E — ConfigDetector   (Actividad 9)
  F — ConfigMotor      (Actividad 10)

Referencias:
    - Blogs ETSII URJC (2025). Sistema de semáforos inteligente.
    - San Miguel, S. (2024). Revista ConCiencia Joven, 2, 32-38.
    - Ultralytics (2024). Comprehensive Guide to Ultralytics YOLOv5.
    - ScienceDirect (2024). YOLOv5 — an overview.
    - Protocolo de Investigación, secciones 2.3.4–2.3.7.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Detección de disponibilidad de PyTorch (sin importación obligatoria)
# ---------------------------------------------------------------------------
try:
    import torch
    _TORCH_OK = True
    _CUDA_DISPONIBLE = torch.cuda.is_available()
except ImportError:
    _TORCH_OK = False
    _CUDA_DISPONIBLE = False


# ===========================================================================
# SECCIÓN A — CLASES COCO VEHICULARES
#
# YOLOv5/COCO detecta 80 clases; solo se retienen las vehiculares
# relevantes para el tráfico urbano de Oaxaca de Juárez.
# Protocolo, sección 2.5.1 — Tecnologías de detección vehicular.
# ===========================================================================

CLASES_VEHICULO: Dict[int, str] = {
    2: "automóvil",
    3: "motocicleta",
    5: "autobús",
    7: "camión",
    1: "bicicleta",
    0: "peatón",
}

# Colores BGR para visualización en OpenCV
COLORES_CLASE: Dict[int, tuple] = {
    2: (30,  144, 255),   # automóvil   — azul dodger
    3: (0,   200, 100),   # motocicleta — verde
    5: (0,   80,  255),   # autobús     — naranja oscuro
    7: (50,  50,  200),   # camión      — rojo oscuro
    1: (200, 200, 0),     # bicicleta   — cian
    0: (180, 0,   255),   # ← peatón — morado
}


# ===========================================================================
# SECCIÓN B — UMBRALES DE CONGESTIÓN
#
# Bandas operacionales para clasificar el estado de la intersección.
# Protocolo, sección 2.2.7 — Patrones de congestión.
# San Miguel (2024): evaluación constante de niveles de congestión.
# ===========================================================================

UMBRAL_CONGESTION_LIBRE:    int = 2   # 0–2  vehículos → libre
UMBRAL_CONGESTION_MODERADA: int = 6   # 3–6  vehículos → moderado
#                                      # 7+   vehículos → congestionado


# ===========================================================================
# SECCIÓN C — TEMPORIZACIÓN SEMAFÓRICA ADAPTATIVA
#
# Fórmula: t_verde = max(t_mín, min(t_máx, n_veh × t_base))
# Referencia: Blogs ETSII URJC (2025); protocolo, sec. 2.3.5–2.3.6.
# ===========================================================================

TIEMPO_BASE_POR_VEHICULO_S: float = 4.0   # s/vehículo
TIEMPO_VERDE_MINIMO_S:      float = 12.0  # mínimo garantizado (equidad)
TIEMPO_VERDE_MAXIMO_S:      float = 60.0  # máximo (evitar esperas largas)


# ===========================================================================
# SECCIÓN D — ConfigPipeline
#
# Parámetros del pipeline de pre-procesamiento de imágenes (Actividad 8).
# Protocolo, sección 2.8.3.2 — Diseño de arquitectura y pipeline.
# ===========================================================================

@dataclass
class ConfigPipeline:
    """
    Parámetros operativos del pipeline de pre-procesamiento (Actividad 8).

    Atributos
    ----------
    rtsp_url : str
        URL del stream RTSP o ruta a un archivo de video local.
    roi_poligono : list[tuple] | None
        Vértices del polígono ROI. None = frame completo.
    tam_entrada_modelo : tuple[int, int]
        Resolución objetivo (ancho, alto) para YOLOv5. Estándar: 640×640.
    fps_objetivo : int
        Cadencia de captura deseada (≥ 25 fps según protocolo, sec. 2.3.6).
    umbral_luz_baja : float
        Luminancia media del canal L (LAB) bajo la cual se activa CLAHE.
    umbral_luz_alta : float
        Luminancia media sobre la cual se omite el filtro de ruido.
    clahe_clip_limit : float
        Límite de recorte para CLAHE. Recomendado: 2.0.
    clahe_tile_grid : tuple[int, int]
        Tamaño de bloques locales para CLAHE. Recomendado: (8, 8).
    kernel_blur : tuple[int, int]
        Kernel gaussiano para reducción de ruido. Recomendado: (3, 3).
    tam_buffer_fallback : int
        Frames válidos almacenados como respaldo ante frames corruptos.
    umbral_var_laplaciano : float
        Varianza mínima del Laplaciano para considerar un frame nítido.
    umbral_frame_negro : float
        Nivel medio de píxel bajo el cual el frame se considera corrupto.
    umbral_ssim_congelado : float
        SSIM sobre el cual el frame se considera congelado.
    dispositivo_torch : str
        'cuda' (GPU) o 'cpu'. Se detecta automáticamente.
    batch_size : int
        Frames por batch de inferencia (1–4).
    """
    rtsp_url:              str   = "rtsp://192.168.1.100:554/stream"
    roi_poligono:          Optional[list] = None
    tam_entrada_modelo:    tuple = (640, 640)
    fps_objetivo:          int   = 25
    umbral_luz_baja:       float = 80.0
    umbral_luz_alta:       float = 120.0
    clahe_clip_limit:      float = 2.0
    clahe_tile_grid:       tuple = (8, 8)
    kernel_blur:           tuple = (3, 3)
    tam_buffer_fallback:   int   = 3
    umbral_var_laplaciano: float = 50.0
    umbral_frame_negro:    float = 5.0
    umbral_ssim_congelado: float = 0.999
    dispositivo_torch:     str   = "cuda" if _CUDA_DISPONIBLE else "cpu"
    batch_size:            int   = 1


# ===========================================================================
# SECCIÓN E — ConfigDetector
#
# Parámetros operativos del detector vehicular YOLOv5 (Actividad 9).
# Protocolo, sección 2.3.6 — Parámetros operativos.
# ===========================================================================

@dataclass
class ConfigDetector:
    """
    Parámetros operativos del detector vehicular YOLOv5 (Actividad 9).

    Justificación de los valores por defecto
    -----------------------------------------
    modelo_yolo = 'yolov5s':
        Balance velocidad/precisión óptimo para el proyecto. ScienceDirect
        (2024) reporta 140 fps en GPU RTX 3060; en CPU mantiene ≤ 70 ms/frame,
        suficiente para el ciclo semafórico mínimo de 12 s. San Miguel (2024).

    confianza_minima = 0.45:
        Minimiza falsos negativos sin incrementar falsos positivos en
        escenarios de tráfico urbano denso. Protocolo, sec. 2.3.6.

    iou_umbral = 0.45:
        Evita detecciones duplicadas en vehículos cercanos (caravanas,
        paradas de autobús) sin fusionar vehículos distintos. Sec. 2.6.2.1.

    Atributos — Temporización semafórica
    -------------------------------------
    Implementan la fórmula: t_verde = max(t_mín, min(t_máx, n_veh × t_base))
    Referencia: Blogs ETSII URJC (2025); protocolo, sec. 2.3.5–2.3.6.
    """
    # ── Modelo ──────────────────────────────────────────────────────────────
    modelo_yolo:       str   = "yolov5s"
    confianza_minima:  float = 0.45
    iou_umbral:        float = 0.45
    tam_imagen:        int   = 640
    dispositivo:       str   = "cuda" if _CUDA_DISPONIBLE else "cpu"

    # ── Salida ──────────────────────────────────────────────────────────────
    mostrar_ventana:   bool  = True
    guardar_video:     bool  = True
    ruta_video_salida: str   = "data/deteccion_vehicular.mp4"

    # # ── Líneas virtuales de conteo por carril (frame 640×640) ───────────────
    # lineas_conteo: List[dict] = field(default_factory=lambda: [
    #     {"nombre": "Carril norte",  "y": 160, "color": (0,   255, 255)},
    #     {"nombre": "Carril centro", "y": 320, "color": (255, 165, 0)},
    #     {"nombre": "Carril sur",    "y": 480, "color": (0,   165, 255)},
    # ])

    # ── Temporización semafórica adaptativa ─────────────────────────────────
    tiempo_base_por_vehiculo_s: float = TIEMPO_BASE_POR_VEHICULO_S
    tiempo_verde_minimo_s:      float = TIEMPO_VERDE_MINIMO_S
    tiempo_verde_maximo_s:      float = TIEMPO_VERDE_MAXIMO_S


# ===========================================================================
# SECCIÓN F — ConfigMotor
#
# Parámetros del motor de decisión semafórica (Actividad 10).
# Protocolo, secciones 2.3.4–2.3.7 — Coordinación dinámica, ciclos
# adaptativos, parámetros operativos y mecanismos de priorización.
# ===========================================================================

@dataclass
class ConfigMotor:
    """
    Parámetros del motor de decisión semafórica (Actividad 10).

    Atributos — Tiempos de transición
    ----------------------------------
    tiempo_amarillo_s : float
        Duración de la fase amarilla (advertencia de cambio). Valor: 3 s.
        Calculado para velocidades de aproximación de 40–50 km/h típicas
        en intersecciones urbanas de Oaxaca. Protocolo, sección 2.2.4.
    tiempo_todo_rojo_s : float
        Duración del intervalo todo-rojo entre fases (despeje de intersección).
        Valor: 2 s. Garantiza que no haya vehículos en la zona de conflicto
        al iniciar la siguiente fase. Protocolo, sección 2.2.4.

    Atributos — Límites de ciclo
    ----------------------------
    ciclo_min_s : float
        Duración mínima del ciclo completo (todas las fases). Valor: 45 s.
        Por debajo de este umbral las pérdidas por transición superan la
        ganancia adaptativa. Protocolo, sección 2.2.4 (método de Webster).
    ciclo_max_s : float
        Duración máxima del ciclo. Valor: 180 s.
        Ciclos más largos generan tiempos de espera inaceptables (> 90 s
        por fase) que incrementan frustración y emisiones contaminantes.
        Protocolo, sección 2.2.5 — Tiempos de espera.

    Atributos — Priorización de autobús
    -------------------------------------
    bonus_autobus_s : float
        Segundos adicionales de verde cuando se detecta ≥ 1 autobús de
        transporte público en un carril. Valor: 8 s.
        Implementa la priorización de transporte colectivo descrita en
        protocolo sección 2.3.7. ILUNION (2024) reporta que esta estrategia
        redujo el tiempo de viaje en la línea E de Seattle en ~6 minutos y
        aumentó pasajeros un 35 %.
    max_autobuses_bonus : int
        Número máximo de autobuses que acumulan bonus (cap anti-monopolio).
        Valor: 2. Evita que un carril acapare todo el ciclo.

    Atributos — Robustez y fallback
    --------------------------------
    max_frames_sin_deteccion : int
        Número máximo de frames consecutivos sin ResultadoDeteccion antes
        de activar el modo fallback de tiempo fijo. Valor: 10 frames.
        A 25 fps equivale a 0.4 s sin datos del detector.
        Protocolo, sección 2.5.2 — Detección en tiempo real.
    tiempos_fallback_s : dict[str, float]
        Tiempos de verde fijos aplicados en modo fallback.
        Corresponden a los tiempos promedio medidos en los aforos
        vehiculares de la Actividad 3 (intersecciones OAX-01/02/03).
        Por defecto: 30 s para todos los carriles (ciclo simétrico).

    Atributos — Historial
    ----------------------
    tam_historial : int
        Número de decisiones recientes almacenadas en memoria para
        análisis estadístico en la Actividad 11 (Simulación).
        Valor: 100 decisiones.
    """
    # ── Transiciones ────────────────────────────────────────────────────────
    tiempo_amarillo_s:   float = 3.0
    tiempo_todo_rojo_s:  float = 2.0

    # ── Límites de ciclo ─────────────────────────────────────────────────────
    ciclo_min_s:         float = 45.0
    ciclo_max_s:         float = 180.0

    # ── Priorización autobús ─────────────────────────────────────────────────
    bonus_autobus_s:     float = 8.0
    max_autobuses_bonus: int   = 2

    # ── Robustez / fallback ──────────────────────────────────────────────────
    max_frames_sin_deteccion: int = 10
    tiempos_fallback_s: Dict[str, float] = field(default_factory=lambda: {
        "norte":  30.0,
        "centro": 30.0,
        "sur":    30.0,
    })

    # ── Historial ────────────────────────────────────────────────────────────
    tam_historial: int = 100
