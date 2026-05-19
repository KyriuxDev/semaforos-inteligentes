"""
utils/video.py — Utilidades de video para pruebas y validación.

Contiene el generador de video sintético utilizado durante la Actividad 8
(validación del pipeline de pre-procesamiento) y la Actividad 11
(simulación y optimización de parámetros, 06/05/26 – 13/05/26).

Protocolo de Investigación, sección 2.8.3.2.
"""

import cv2
import numpy as np
from utils.logger import get_logger

logger = get_logger("utils.video")


def generar_video_sintetico(
    ruta_salida: str = "data/test_trafico.mp4",
    n_frames: int = 300,
    resolucion: tuple = (1280, 720),
    fps: int = 25,
) -> str:
    """
    Genera un video MP4 sintético con vehículos simulados en movimiento.

    Crea tres vehículos con trayectorias, colores y velocidades distintas
    sobre una calzada simulada. Incluye líneas de carril, marcador de
    intersección, número de frame y timestamp para depuración visual.

    Utilidad
    --------
    Permite validar el pipeline y el detector sin necesidad de video real
    de las intersecciones OAX-01, OAX-02 u OAX-03.
    Protocolo, sección 2.8.3.2 — Actividad 11 (Simulación).

    Parámetros
    ----------
    ruta_salida : str
        Ruta del archivo MP4 a generar.
    n_frames : int
        Número de fotogramas (duración = n_frames / fps segundos).
    resolucion : tuple[int, int]
        (ancho, alto) del video en píxeles.
    fps : int
        Fotogramas por segundo.

    Retorna
    -------
    str
        Ruta del archivo generado.
    """
    ancho, alto = resolucion
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(ruta_salida, fourcc, fps, (ancho, alto))

    if not writer.isOpened():
        raise IOError(f"No se pudo crear el archivo de video: {ruta_salida}")

    logger.info(f"Generando video sintético → {ruta_salida} ({n_frames} frames @ {fps} fps)")

    y_sup = alto // 3
    y_inf = 2 * alto // 3

    for i in range(n_frames):
        frame = np.full((alto, ancho, 3), 100, dtype=np.uint8)

        # Líneas de carril
        cv2.line(frame, (0, y_sup), (ancho, y_sup), (200, 200, 200), 2)
        cv2.line(frame, (0, y_inf), (ancho, y_inf), (200, 200, 200), 2)
        for x in range(0, ancho, 60):
            cv2.line(frame, (x, alto // 2), (x + 30, alto // 2), (220, 220, 0), 2)

        # Línea de intersección
        cv2.line(frame, (ancho // 2, 0), (ancho // 2, alto), (255, 255, 255), 1)
        cv2.putText(frame, "INTERSECCION", (ancho // 2 - 80, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Vehículo 1 — sedan azul, carril norte, izq→der
        x1 = (80 + i * 3) % (ancho + 120) - 120
        cv2.rectangle(frame, (x1, y_sup + 10), (x1 + 110, y_sup + 55), (30, 144, 255), -1)
        cv2.rectangle(frame, (x1 + 15, y_sup - 8), (x1 + 95, y_sup + 10), (100, 180, 255), -1)
        cv2.circle(frame, (x1 + 20, y_sup + 55), 10, (40, 40, 40), -1)
        cv2.circle(frame, (x1 + 90, y_sup + 55), 10, (40, 40, 40), -1)

        # Vehículo 2 — camioneta naranja, carril centro, más lento
        x2 = (300 + i * 2) % (ancho + 140) - 140
        yc = alto // 2
        cv2.rectangle(frame, (x2, yc - 30), (x2 + 140, yc + 30), (0, 120, 255), -1)
        cv2.rectangle(frame, (x2 + 10, yc - 52), (x2 + 130, yc - 30), (50, 160, 255), -1)
        cv2.circle(frame, (x2 + 25, yc + 30), 12, (40, 40, 40), -1)
        cv2.circle(frame, (x2 + 115, yc + 30), 12, (40, 40, 40), -1)

        # Vehículo 3 — sedan verde, carril sur, der→izq
        x3 = ancho - ((150 + i * 3) % (ancho + 120))
        cv2.rectangle(frame, (x3, y_inf + 10), (x3 + 110, y_inf + 55), (50, 205, 50), -1)
        cv2.rectangle(frame, (x3 + 15, y_inf - 8), (x3 + 95, y_inf + 10), (100, 230, 100), -1)
        cv2.circle(frame, (x3 + 20, y_inf + 55), 10, (40, 40, 40), -1)
        cv2.circle(frame, (x3 + 90, y_inf + 55), 10, (40, 40, 40), -1)

        # HUD
        t = i / fps
        cv2.rectangle(frame, (0, 0), (360, 52), (30, 30, 30), -1)
        cv2.putText(frame, f"Frame: {i+1}/{n_frames}  |  t={t:.2f}s",
                    (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 120), 1)
        cv2.putText(frame, "ITO — Sistema semaforos inteligentes [PRUEBA]",
                    (8, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

        writer.write(frame)

    writer.release()
    logger.info(
        f"Video sintético listo: {ruta_salida} "
        f"({n_frames/fps:.1f}s, {ancho}×{alto}px)"
    )
    return ruta_salida
