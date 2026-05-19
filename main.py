"""
main.py — Punto de entrada del sistema de semáforos inteligentes.

Orquesta los tres subsistemas del Subsistema de Procesamiento (Capa 2):
  · Pipeline de pre-procesamiento  (core/pipeline/  — Actividad 8)
  · Detector vehicular YOLOv5      (core/detector/  — Actividad 9)
  · Motor de decisión semafórica   (core/decision/  — Actividad 10)

Uso
---
  python main.py <video.mp4>           Detección + decisión sobre video.
  python main.py --sintetico           Genera video sintético y lo procesa.
  python main.py --noventana <video>   Modo headless (sin GUI).
  python main.py --ayuda               Muestra esta pantalla.

Protocolo de Investigación, sección 2.8.3.2.
ITO — Ingeniería en Sistemas Computacionales | Taller de Investigación II
"""

from __future__ import annotations

import os
import sys
import warnings

# Silenciar advertencias de Qt/OpenCV en entornos Wayland (Debian/Ubuntu)
# No afectan el funcionamiento del sistema; son ruido del entorno gráfico.
os.environ.setdefault("QT_LOGGING_RULES", "*.debug=false;qt.qpa.*=false")
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

# Silenciar el PRO TIP de Ultralytics sobre yolov5su
warnings.filterwarnings("ignore", category=UserWarning, module="ultralytics")

from config.settings import ConfigDetector, ConfigMotor, ConfigPipeline
from core.decision.motor import MotorDecision
from core.detector.vehicular import DetectorVehicular
from models.schemas import ResultadoDeteccion
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
║    python main.py --sintetico                                       ║
║    python main.py --noventana <ruta_video.mp4>                      ║
║                                                                     ║
║  Controles en ventana:                                              ║
║    q — salir   |   p — pausar/reanudar   |   s — captura PNG        ║
║                                                                     ║
║  Parámetros YOLOv5 (protocolo, sec. 2.3.6):                        ║
║    Modelo: yolov5s  |  Confianza: 0.45  |  IoU NMS: 0.45           ║
║    t_base: 4 s/veh  |  t_mín: 12 s     |  t_máx: 60 s             ║
║                                                                     ║
║  Parámetros motor (protocolo, sec. 2.3.4–2.3.7):                   ║
║    Amarillo: 3 s  |  Todo-rojo: 2 s  |  Bonus bus: 8 s             ║
║    Ciclo mín: 45 s  |  Ciclo máx: 180 s                            ║
╚══════════════════════════════════════════════════════════════════════╝
"""


def _bucle_con_motor(
    detector: DetectorVehicular,
    motor: MotorDecision,
    cfg_pipeline: ConfigPipeline,
    cfg_detector: ConfigDetector,
) -> None:
    """
    Bucle principal que encadena detector → motor en cada frame.

    DetectorVehicular.ejecutar() es suficiente para visualización standalone.
    Este bucle alternativo permite que la DecisionSemaforica del motor sea
    procesada en cada frame (p. ej. para logging extendido o controlador real).
    """
    import cv2
    from typing import Optional
    import numpy as np

    captura = cv2.VideoCapture(cfg_pipeline.rtsp_url)
    if not captura.isOpened():
        raise IOError(f"No se pudo abrir: {cfg_pipeline.rtsp_url}")

    fps_src = captura.get(cv2.CAP_PROP_FPS) or 25
    w_src   = int(captura.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_src   = int(captura.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer: Optional[cv2.VideoWriter] = None
    if cfg_detector.guardar_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(cfg_detector.ruta_video_salida, fourcc, fps_src, (w_src, h_src))
        logger.info(f"Grabando → {cfg_detector.ruta_video_salida}")

    logger.info("Bucle detector+motor iniciado — q: salir | p: pausar | s: captura")

    n_frame       = 0
    pausado       = False
    frame_anotado: Optional[np.ndarray] = None

    try:
        while True:
            if not pausado:
                ret, frame_bgr = captura.read()
                if not ret:
                    logger.info("Fin de video.")
                    break

                n_frame += 1

                # ── Actividad 9: inferencia del detector ───────────────────
                resultado: ResultadoDeteccion = detector.inferir(frame_bgr, n_frame)
                detector._actualizar_metricas(resultado)

                # ── Actividad 10: motor de decisión ────────────────────────
                decision = motor.decidir(resultado)

                # Log consolidado cada 25 frames (≈ 1 línea/s a 25 fps)
                if n_frame % 25 == 0:
                    logger.info(
                        f"[{n_frame:05d}] "
                        f"veh={resultado.total_vehiculos} "
                        f"({resultado.nivel_congestion}) | "
                        f"ciclo={decision.ciclo_total_s:.0f}s "
                        f"prior={decision.fase_prioritaria} "
                        f"[{decision.motivo_prioridad}] | "
                        f"lat={resultado.latencia_ms:.0f}ms"
                    )

                frame_anotado = detector.anotar_frame(frame_bgr, resultado)
                if writer:
                    writer.write(frame_anotado)

            if cfg_detector.mostrar_ventana and frame_anotado is not None:
                cv2.imshow("ITO — Semáforos Inteligentes (Act. 9+10)", frame_anotado)
                tecla = cv2.waitKey(1) & 0xFF
                if tecla == ord("q"):
                    break
                elif tecla == ord("p"):
                    pausado = not pausado
                    logger.info("PAUSADO" if pausado else "REANUDADO")
                elif tecla == ord("s") and not pausado:
                    nombre = f"data/captura_{n_frame:05d}.png"
                    cv2.imwrite(nombre, frame_anotado)
                    logger.info(f"Captura: {nombre}")

    except KeyboardInterrupt:
        logger.info("Interrupción por teclado.")
    finally:
        captura.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        detector._log_metricas()
        print(motor.resumen_sesion())


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
    cfg_motor = ConfigMotor()

    detector = DetectorVehicular(cfg_pipeline, cfg_detector)
    motor    = MotorDecision(cfg_motor)

    try:
        detector.iniciar()
        _bucle_con_motor(detector, motor, cfg_pipeline, cfg_detector)
    except ImportError as exc:
        logger.error(str(exc))
        return 1
    except Exception as exc:
        logger.error(f"Error inesperado: {exc}", exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
