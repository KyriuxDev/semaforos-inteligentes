"""
core/decision/motor.py — Motor de decisión semafórica (Actividad 10).

Corresponde a la Actividad 10 del plan de trabajo:
"Diseñar la lógica de coordinación semafórica dinámica".

Estado actual: INTERFAZ DEFINIDA — implementación pendiente (Actividad 10).
El equipo puede programar contra esta interfaz desde ya.

La clase MotorDecision recibe ResultadoDeteccion del detector (Act. 9) y
produce DecisionSemaforica para el controlador semafórico.

Estrategia de decisión planificada (protocolo, sección 2.3.4–2.3.7):
  1. Priorización por congestión: más tiempo al carril más cargado.
  2. Equidad mínima: ningún carril recibe menos de t_mín segundos.
  3. Prioridad especial: autobuses y vehículos de emergencia.
  4. Modo de fallback: tiempo fijo si se pierde conexión con el detector.

Referencias:
    - San Miguel (2024). Revista ConCiencia Joven, 2, 32-38.
    - Protocolo de Investigación, sección 2.3.4–2.3.7.
    - Blogs ETSII URJC (2025). Sistema de semáforos inteligente.
"""

from __future__ import annotations

import time
from typing import List

from models.schemas import DecisionSemaforica, ResultadoDeteccion
from utils.logger import get_logger

logger = get_logger("core.decision.motor")


class MotorDecision:
    """
    Motor de decisión semafórica adaptativa.

    Consume ResultadoDeteccion del DetectorVehicular y produce
    DecisionSemaforica para el controlador semafórico.

    Estado: interfaz definida — lógica completa en Actividad 10.

    Ejemplo de uso (cuando esté implementado)
    ------------------------------------------
    >>> from core.decision.motor import MotorDecision
    >>> motor = MotorDecision()
    >>> decision = motor.decidir(resultado_deteccion)
    >>> controlador.aplicar(decision)
    """

    def decidir(self, resultado: ResultadoDeteccion) -> DecisionSemaforica:
        """
        Genera una DecisionSemaforica a partir de un ResultadoDeteccion.

        Actualmente aplica directamente los tiempos recomendados por el
        detector como decisión base. La lógica de priorización multi-criterio
        se implementará en la Actividad 10.

        Parámetros
        ----------
        resultado : ResultadoDeteccion
            Salida del DetectorVehicular para un frame.

        Retorna
        -------
        DecisionSemaforica
            Decisión de temporización para el controlador semafórico.
        """
        tiempos = resultado.tiempo_verde_recomendado_s

        # Fase prioritaria: carril con mayor tiempo de verde recomendado
        fase_prioritaria = max(tiempos, key=lambda c: tiempos[c]) if tiempos else ""
        ciclo_total      = sum(tiempos.values())

        decision = DecisionSemaforica(
            timestamp=time.time(),
            tiempos_verde_s=tiempos,
            fase_prioritaria=fase_prioritaria,
            nivel_congestion_global=resultado.nivel_congestion,
            ciclo_total_s=ciclo_total,
            origen="adaptativo",
        )

        logger.debug(
            f"Decisión — prioritaria={fase_prioritaria} | "
            f"ciclo={ciclo_total:.0f}s | origen={decision.origen}"
        )
        return decision

    def procesar_lote(
        self, resultados: List[ResultadoDeteccion]
    ) -> List[DecisionSemaforica]:
        """
        Procesa una lista de ResultadoDeteccion (un resultado por carril/cámara).

        TODO (Actividad 10): Implementar lógica de coordinación entre
        múltiples intersecciones (protocolo, sección 2.3.4).

        Parámetros
        ----------
        resultados : list[ResultadoDeteccion]
            Un resultado por cámara/intersección.

        Retorna
        -------
        list[DecisionSemaforica]
        """
        return [self.decidir(r) for r in resultados]
