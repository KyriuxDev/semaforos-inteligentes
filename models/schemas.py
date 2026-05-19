"""
models/schemas.py — Estructuras de datos compartidas entre capas del sistema.

Define los modelos de datos (dataclasses) que viajan entre los módulos:
  · ResultadoDeteccion  → salida del detector (Act. 9), entrada del motor (Act. 10)
  · DecisionSemaforica  → salida del motor de decisión (Act. 10), entrada del
                          controlador semafórico.

Al centralizar los schemas aquí ningún módulo de lógica importa de otro
módulo de lógica directamente, evitando dependencias circulares.

Protocolo de Investigación, sección 2.8.3.2 — Actividades 9 y 10.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


# ===========================================================================
# ResultadoDeteccion
#
# Interfaz de salida del DetectorVehicular (Actividad 9) y de entrada del
# MotorDecision (Actividad 10).
#
# El campo tiempo_verde_recomendado_s implementa la fórmula adaptativa:
#     t_verde = max(t_mín, min(t_máx, n_veh × t_base))
# Referencia: Blogs ETSII URJC (2025); protocolo, sec. 2.3.5–2.3.6.
# ===========================================================================

@dataclass
class ResultadoDeteccion:
    """
    Resultado del procesamiento de un fotograma por el detector YOLOv5.

    Atributos
    ----------
    n_frame : int
        Número secuencial del fotograma.
    timestamp : float
        Marca de tiempo UNIX del procesamiento.
    total_vehiculos : int
        Conteo total de vehículos en el frame.
    conteo_por_tipo : dict[str, int]
        Vehículos por categoría (automóvil, motocicleta, autobús, camión,
        bicicleta). Insumo para priorización por tipo (protocolo, sec. 2.3.7).
    conteo_por_carril : dict[str, int]
        Vehículos por carril ('norte', 'centro', 'sur').
    tiempo_verde_recomendado_s : dict[str, float]
        Tiempo de verde recomendado por carril (segundos).
        Propuesta del detector hacia el motor de decisión.
    detecciones_raw : list[dict]
        Detecciones individuales con bbox, confianza, clase y carril.
    latencia_ms : float
        Latencia de inferencia YOLOv5 en milisegundos.
    nivel_congestion : str
        'libre' | 'moderado' | 'congestionado'.
        Protocolo, sección 2.2.7 — Patrones de congestión.
    """
    n_frame:                    int   = 0
    timestamp:                  float = 0.0
    total_vehiculos:            int   = 0
    conteo_por_tipo:            Dict[str, int]   = field(default_factory=dict)
    conteo_por_carril:          Dict[str, int]   = field(default_factory=dict)
    tiempo_verde_recomendado_s: Dict[str, float] = field(default_factory=dict)
    detecciones_raw:            List[dict]        = field(default_factory=list)
    latencia_ms:                float = 0.0
    nivel_congestion:           str   = "libre"

    def __str__(self) -> str:
        t = {c: f"{v:.0f}s" for c, v in self.tiempo_verde_recomendado_s.items()}
        return (
            f"Frame {self.n_frame:05d} | "
            f"Veh: {self.total_vehiculos:2d} | "
            f"Congest.: {self.nivel_congestion:12s} | "
            f"t_verde: {t} | "
            f"Lat.: {self.latencia_ms:.1f} ms"
        )


# ===========================================================================
# DecisionSemaforica
#
# Interfaz de salida del MotorDecision (Actividad 10) y de entrada del
# controlador semafórico.  Se define aquí aunque el motor aún no está
# implementado para que el equipo pueda programar contra la interfaz.
#
# Protocolo, sección 2.3.4–2.3.5 — Coordinación dinámica y ciclos
# adaptativos.
# ===========================================================================

@dataclass
class DecisionSemaforica:
    """
    Decisión de temporización semafórica generada por el motor de decisión.

    Atributos
    ----------
    timestamp : float
        Marca de tiempo UNIX de la decisión.
    tiempos_verde_s : dict[str, float]
        Tiempo de verde asignado a cada fase/carril (segundos).
    fase_prioritaria : str
        Carril o fase que recibe mayor tiempo de verde en este ciclo.
    nivel_congestion_global : str
        Nivel de congestión de la intersección en este ciclo.
    ciclo_total_s : float
        Duración total del ciclo semafórico (suma de todas las fases).
    origen : str
        Modo de generación: 'adaptativo' | 'tiempo_fijo' | 'emergencia'.
    """
    timestamp:               float = 0.0
    tiempos_verde_s:         Dict[str, float] = field(default_factory=dict)
    fase_prioritaria:        str   = ""
    nivel_congestion_global: str   = "libre"
    ciclo_total_s:           float = 0.0
    origen:                  str   = "adaptativo"
