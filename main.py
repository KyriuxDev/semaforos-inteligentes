"""
main.py — Punto de entrada del sistema de semáforos inteligentes.

Orquesta los tres subsistemas del Subsistema de Procesamiento (Capa 2):
  · Pipeline de pre-procesamiento  (core/pipeline/  — Actividad 8)
  · Detector vehicular YOLOv5      (core/detector/  — Actividad 9)
  · Motor de decisión semafórica   (core/decision/  — Actividad 10)

Uso
---
  python main.py <video.mp4>           Detección sobre video existente.
  python main.py --sintetico           Genera video sintético y lo procesa.
  python main.py --noventana <video>   Modo headless (sin GUI).
  python main.py --ayuda               Muestra esta pantalla.

Protocolo de Investigación, sección 2.8.3.2.
ITO — Ingeniería en Sistemas Computacionales | Taller de Investigación II
"""

from __future__ import annotations

import sys

from config.settings import ConfigDetector, ConfigPipeline
from core.detector.vehicular import DetectorVehicular
from utils.logger import configurar_logging, get_logger
from utils.video import generar_video_sintetico

configurar_logging()
logger = get_logger("main")

AYUDA = """
╔══════════════════════════════════════════════════════════════════════╗
║  TECNOLÓGICO NACIONAL DE MÉXICO — INSTITUTO TECNOLÓGICO DE OAXACA  ║
║  Sistema de Semáforos Inteligentes — Subsistema de Procesamiento    ║
╠══════════════════════════════════════════════════════════════════════╣
║  Uso:                                                               ║
║    python main.py <ruta_video.mp4>                                  ║
║        Detección YOLOv5 sobre un video existente.                   ║
║                                                                     ║
║    python main.py --sintetico                                       ║
║        Genera video sintético de prueba y ejecuta detección.        ║
║                                                                     ║
║    python main.py --noventana <ruta_video.mp4>                      ║
║        Modo headless: solo log en consola, sin GUI.                 ║
║                                                                     ║
║  Controles en ventana:                                              ║
║    q — salir   |   p — pausar/reanudar   |   s — captura PNG        ║
║                                                                     ║
║  Parámetros YOLOv5 (protocolo, sec. 2.3.6):                        ║
║    Modelo: yolov5s  |  Confianza: 0.45  |  IoU NMS: 0.45           ║
║    t_base: 4 s/veh  |  t_mín: 12 s     |  t_máx: 60 s             ║
╚══════════════════════════════════════════════════════════════════════╝
"""


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]

    if not args or "--ayuda" in args or "-h" in args:
        print(AYUDA)
        return 0

    headless = "--noventana" in args
    args     = [a for a in args if a != "--noventana"]

    if args[0] == "--sintetico":
        logger.info("Generando video sintético de prueba...")
        ruta_video = generar_video_sintetico(
            ruta_salida="data/test_trafico.mp4",
            n_frames=200,
        )
    else:
        ruta_video = args[0]

    cfg_pipeline = ConfigPipeline(rtsp_url=ruta_video)
    cfg_detector = ConfigDetector(
        modelo_yolo="yolov5s",
        mostrar_ventana=not headless,
        guardar_video=True,
        ruta_video_salida="data/deteccion_vehicular.mp4",
    )

    detector = DetectorVehicular(cfg_pipeline, cfg_detector)

    try:
        detector.iniciar()
        detector.ejecutar()
    except ImportError as exc:
        logger.error(str(exc))
        return 1
    except Exception as exc:
        logger.error(f"Error inesperado: {exc}", exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
