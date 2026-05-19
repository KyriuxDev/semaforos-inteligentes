# config/__init__.py
from config.settings import (
    ConfigPipeline,
    ConfigDetector,
    CLASES_VEHICULO,
    COLORES_CLASE,
    UMBRAL_CONGESTION_LIBRE,
    UMBRAL_CONGESTION_MODERADA,
    TIEMPO_BASE_POR_VEHICULO_S,
    TIEMPO_VERDE_MINIMO_S,
    TIEMPO_VERDE_MAXIMO_S,
)

__all__ = [
    "ConfigPipeline",
    "ConfigDetector",
    "CLASES_VEHICULO",
    "COLORES_CLASE",
    "UMBRAL_CONGESTION_LIBRE",
    "UMBRAL_CONGESTION_MODERADA",
    "TIEMPO_BASE_POR_VEHICULO_S",
    "TIEMPO_VERDE_MINIMO_S",
    "TIEMPO_VERDE_MAXIMO_S",
]
