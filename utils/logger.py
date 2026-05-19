"""
utils/logger.py — Configuración centralizada del sistema de logging.

Todos los módulos del proyecto obtienen su logger a través de get_logger()
para garantizar un formato uniforme y un único punto de configuración.
"""

import logging


def configurar_logging(nivel: int = logging.INFO) -> None:
    """Configura el handler raíz una sola vez al arrancar la aplicación."""
    logging.basicConfig(
        level=nivel,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def get_logger(nombre: str) -> logging.Logger:
    """
    Retorna un logger con el nombre dado.

    Parámetros
    ----------
    nombre : str
        Identificador del módulo, p. ej. 'core.detector.vehicular'.

    Retorna
    -------
    logging.Logger
    """
    return logging.getLogger(nombre)
