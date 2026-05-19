"""
models/schemas.py — Estructuras de datos compartidas entre capas del sistema.

Define los modelos de datos que viajan entre módulos:
  · ResultadoDeteccion  → salida Act. 9, entrada Act. 10
  · FaseSemaforica      → temporización completa de un carril en un ciclo
  · DecisionSemaforica  → salida Act. 10, entrada controlador semafórico

Ningún módulo de lógica importa de otro módulo de lógica directamente;
todos importan desde aquí, evitando dependencias circulares.

Protocolo de Investigación, secciones 2.2.4, 2.3.4–2.3.7.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


# ===========================================================================
# ResultadoDeteccion  (Actividad 9 → Actividad 10)
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
        Conteo total de vehículos detectados en el frame.
    conteo_por_tipo : dict[str, int]
        Vehículos por categoría. Insumo de priorización por tipo
        (protocolo, sec. 2.3.7).
    conteo_por_carril : dict[str, int]
        Vehículos por carril ('norte', 'centro', 'sur').
    tiempo_verde_recomendado_s : dict[str, float]
        Propuesta de tiempo de verde por carril del detector.
        Fórmula: max(t_mín, min(t_máx, n_veh × t_base)).
    detecciones_raw : list[dict]
        Detecciones individuales con bbox, confianza, clase y carril.
        Permite al motor calcular buses_por_carril sin re-inferencia.
    latencia_ms : float
        Latencia de inferencia YOLOv5 en milisegundos.
    nivel_congestion : str
        'libre' | 'moderado' | 'congestionado'.
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
# FaseSemaforica  (componente de DecisionSemaforica)
#
# Temporización completa de un carril dentro de un ciclo.
# Protocolo, sección 2.2.4 — Ciclos semafóricos.
# ===========================================================================

@dataclass
class FaseSemaforica:
    """
    Temporización completa de un carril para un ciclo semafórico.

    Atributos
    ----------
    carril : str
        Identificador: 'norte', 'centro' o 'sur'.
    tiempo_verde_s : float
        Segundos de verde efectivo asignados por el motor de decisión.
    tiempo_amarillo_s : float
        Fase amarilla (advertencia). Típico: 3 s.
    tiempo_todo_rojo_s : float
        Intervalo todo-rojo (despeje de intersección). Típico: 2 s.
    duracion_total_s : float
        verde + amarillo + todo-rojo. Se calcula automáticamente si es 0.
    prioridad_autobus : bool
        True si el verde incluye bonus por autobús detectado.
        Protocolo, sección 2.3.7.
    n_vehiculos : int
        Vehículos detectados en el carril (referencia del frame).
    n_autobuses : int
        Autobuses detectados en el carril (referencia).
    """
    carril:             str   = ""
    tiempo_verde_s:     float = 0.0
    tiempo_amarillo_s:  float = 3.0
    tiempo_todo_rojo_s: float = 2.0
    duracion_total_s:   float = 0.0
    prioridad_autobus:  bool  = False
    n_vehiculos:        int   = 0
    n_autobuses:        int   = 0

    def __post_init__(self) -> None:
        if self.duracion_total_s == 0.0:
            self.duracion_total_s = (
                self.tiempo_verde_s + self.tiempo_amarillo_s + self.tiempo_todo_rojo_s
            )

    def __str__(self) -> str:
        bus = " [BUS]" if self.prioridad_autobus else ""
        return (
            f"{self.carril:6s}{bus}: "
            f"verde={self.tiempo_verde_s:.0f}s "
            f"amarillo={self.tiempo_amarillo_s:.0f}s "
            f"todo-rojo={self.tiempo_todo_rojo_s:.0f}s "
            f"(total={self.duracion_total_s:.0f}s) "
            f"| veh={self.n_vehiculos} bus={self.n_autobuses}"
        )


# ===========================================================================
# DecisionSemaforica  (Actividad 10 → Controlador semafórico)
#
# Salida completa del MotorDecision.
# Protocolo, secciones 2.3.4–2.3.7.
# ===========================================================================

@dataclass
class DecisionSemaforica:
    """
    Decisión de temporización generada por el motor de decisión semafórica.

    Atributos
    ----------
    timestamp : float
        Marca de tiempo UNIX de la decisión.
    fases : list[FaseSemaforica]
        Lista ordenada de fases del ciclo. El controlador las ejecuta
        en el orden de la lista.
    tiempos_verde_s : dict[str, float]
        Resumen de tiempos de verde por carril (acceso rápido).
    fase_prioritaria : str
        Carril que recibe mayor tiempo de verde en este ciclo.
    nivel_congestion_global : str
        Nivel de congestión de la intersección en este ciclo.
    ciclo_total_s : float
        Duración total del ciclo (verde + amarillo + todo-rojo de todas
        las fases).
    origen : str
        'adaptativo' | 'tiempo_fijo' | 'emergencia'.
    motivo_prioridad : str
        Razón de la priorización: 'congestion_norte', 'autobus_centro',
        'tiempo_fijo_fallback', etc. Para logs y análisis (Actividad 11).
    n_frame_origen : int
        Frame de detección que originó esta decisión.
    """
    timestamp:               float = 0.0
    fases:                   List[FaseSemaforica] = field(default_factory=list)
    tiempos_verde_s:         Dict[str, float]     = field(default_factory=dict)
    fase_prioritaria:        str   = ""
    nivel_congestion_global: str   = "libre"
    ciclo_total_s:           float = 0.0
    origen:                  str   = "adaptativo"
    motivo_prioridad:        str   = ""
    n_frame_origen:          int   = 0

    def __str__(self) -> str:
        return (
            f"Decisión | origen={self.origen:11s} | "
            f"ciclo={self.ciclo_total_s:.0f}s | "
            f"prioritaria={self.fase_prioritaria} | "
            f"motivo={self.motivo_prioridad}"
        )
