"""
core/decision/motor.py — Motor de decisión semafórica adaptativa.

Implementa la Actividad 10 del plan de trabajo:
"Diseñar la lógica de coordinación semafórica dinámica".

Recibe ResultadoDeteccion del DetectorVehicular (Actividad 9) y produce
DecisionSemaforica para el controlador semafórico, aplicando tres capas
de lógica en orden de prioridad decreciente:

  Capa 1 — FALLBACK (máxima prioridad)
      Si el detector lleva más de max_frames_sin_deteccion frames sin
      entregar datos, se activa el modo tiempo_fijo con los parámetros
      de la Actividad 3 (aforos en campo).
      Protocolo, sección 2.5.2 — Detección en tiempo real.

  Capa 2 — PRIORIZACIÓN DE AUTOBÚS
      Si se detecta ≥ 1 autobús en un carril, ese carril recibe un
      bonus de tiempo (bonus_autobus_s). Implementa la priorización de
      transporte público descrita en la sección 2.3.7 del protocolo.
      ILUNION (2024): Seattle redujo tiempo de viaje 6 min, +35% pasajeros.

  Capa 3 — BALANCEO DE CICLO
      Los tiempos de verde (base + bonus) se escalan para que el ciclo
      total se mantenga dentro de [ciclo_min_s, ciclo_max_s].
      Si el ciclo escalado aún supera el máximo, se recortan los carriles
      menos congestionados primero (política de equidad).
      Protocolo, sección 2.2.4 — Ciclos semafóricos (método de Webster).

Flujo principal:
    ResultadoDeteccion
        → _autobuses_por_carril()       extrae buses de detecciones_raw
        → _aplicar_bonus_autobus()      suma bonus si hay buses
        → _balancear_ciclo()            escala para respetar límites
        → _construir_fases()            empaqueta en FaseSemaforica
        → DecisionSemaforica            entrega al controlador

Historial:
    MotorDecision mantiene un deque de las últimas tam_historial decisiones.
    Accesible vía motor.historial para análisis en la Actividad 11.

Referencias:
    - San Miguel (2024). Revista ConCiencia Joven, 2, 32-38.
    - ILUNION (2024). Semáforos inteligentes: Innovaciones para ciudades modernas.
    - Protocolo de Investigación, secciones 2.3.4–2.3.7.
    - Blogs ETSII URJC (2025). Sistema de semáforos inteligente.
"""

from __future__ import annotations

import time
from collections import deque, defaultdict
from typing import Deque, Dict, List, Optional, Tuple

from config.settings import ConfigMotor
from models.schemas import DecisionSemaforica, FaseSemaforica, ResultadoDeteccion
from utils.logger import get_logger

logger = get_logger("core.decision.motor")

# Orden estándar de ejecución de fases (norte → centro → sur)
ORDEN_FASES: List[str] = ["norte", "centro", "sur"]


# ===========================================================================
# MotorDecision
# ===========================================================================

class MotorDecision:
    """
    Motor de decisión semafórica adaptativa (Actividad 10).

    Transforma ResultadoDeteccion del detector en DecisionSemaforica
    aplicando priorización de autobús, balanceo de ciclo y fallback
    a tiempo fijo ante pérdida de señal del detector.

    Parámetros
    ----------
    cfg : ConfigMotor
        Configuración operativa del motor (ver config/settings.py).

    Atributos públicos
    ------------------
    historial : deque[DecisionSemaforica]
        Últimas cfg.tam_historial decisiones. Útil para análisis en
        la Actividad 11 (Simulación y optimización).

    Ejemplo de uso
    --------------
    >>> from config import ConfigMotor
    >>> from core.decision.motor import MotorDecision
    >>> motor = MotorDecision(ConfigMotor())
    >>> decision = motor.decidir(resultado)
    >>> print(decision)
    >>> print(motor.resumen_sesion())
    """

    def __init__(self, cfg: Optional[ConfigMotor] = None) -> None:
        self.cfg = cfg or ConfigMotor()
        self.historial: Deque[DecisionSemaforica] = deque(maxlen=self.cfg.tam_historial)
        self._frames_sin_deteccion: int = 0
        self._stats = {
            "decisiones_totales":   0,
            "decisiones_adaptativas": 0,
            "decisiones_fallback":  0,
            "activaciones_bus":     0,
            "ciclo_acum_s":         0.0,
        }

    # ------------------------------------------------------------------
    # 1. API pública
    # ------------------------------------------------------------------

    def decidir(self, resultado: ResultadoDeteccion) -> DecisionSemaforica:
        """
        Genera una DecisionSemaforica a partir de un ResultadoDeteccion.

        Aplica las tres capas de lógica en orden:
          1. Fallback si no hay detección válida.
          2. Bonus de autobús por carril.
          3. Balanceo del ciclo total dentro de los límites configurados.

        Parámetros
        ----------
        resultado : ResultadoDeteccion
            Salida del DetectorVehicular para el frame actual.

        Retorna
        -------
        DecisionSemaforica
        """
        # ── Capa 1: ¿hay datos válidos del detector? ───────────────────────
        if resultado.total_vehiculos == 0 and resultado.n_frame == 0:
            self._frames_sin_deteccion += 1
        else:
            self._frames_sin_deteccion = 0

        if self._frames_sin_deteccion >= self.cfg.max_frames_sin_deteccion:
            logger.warning(
                f"Sin detecciones por {self._frames_sin_deteccion} frames — "
                "activando modo tiempo_fijo."
            )
            return self._decision_fallback(resultado.n_frame)

        # ── Capa 2: aplicar bonus de autobús ───────────────────────────────
        buses_por_carril = self._autobuses_por_carril(resultado.detecciones_raw)
        tiempos, motivos_bus = self._aplicar_bonus_autobus(
            resultado.tiempo_verde_recomendado_s, buses_por_carril
        )

        # ── Capa 3: balancear ciclo ────────────────────────────────────────
        tiempos = self._balancear_ciclo(tiempos, resultado.conteo_por_carril)

        # ── Construir fases y decisión final ──────────────────────────────
        fases            = self._construir_fases(tiempos, buses_por_carril, resultado)
        fase_prioritaria = max(tiempos, key=lambda c: tiempos[c])
        ciclo_total      = sum(f.duracion_total_s for f in fases)
        motivo           = self._motivo_prioridad(
            fase_prioritaria, resultado, buses_por_carril, motivos_bus
        )

        decision = DecisionSemaforica(
            timestamp=time.time(),
            fases=fases,
            tiempos_verde_s={f.carril: f.tiempo_verde_s for f in fases},
            fase_prioritaria=fase_prioritaria,
            nivel_congestion_global=resultado.nivel_congestion,
            ciclo_total_s=round(ciclo_total, 1),
            origen="adaptativo",
            motivo_prioridad=motivo,
            n_frame_origen=resultado.n_frame,
        )

        self._registrar(decision, adaptativa=True)
        logger.debug(str(decision))
        for fase in fases:
            logger.debug(f"  {fase}")

        return decision

    def notificar_sin_deteccion(self) -> DecisionSemaforica:
        """
        Llamar cuando el detector no entrega ResultadoDeteccion en el frame
        actual (p. ej. frame inválido descartado por el validador del pipeline).
        Incrementa el contador de frames sin detección y aplica fallback si
        se supera el umbral.
        """
        self._frames_sin_deteccion += 1
        if self._frames_sin_deteccion >= self.cfg.max_frames_sin_deteccion:
            return self._decision_fallback(n_frame=0)
        return self._decision_fallback(n_frame=0)

    # ------------------------------------------------------------------
    # 2. Capa de fallback (tiempo fijo)
    # ------------------------------------------------------------------

    def _decision_fallback(self, n_frame: int) -> DecisionSemaforica:
        """
        Genera una decisión de tiempo fijo usando tiempos_fallback_s.

        Se activa cuando el detector lleva más de max_frames_sin_deteccion
        sin entregar datos. Los tiempos provienen de los aforos vehiculares
        de la Actividad 3 del plan de trabajo.
        Protocolo, sección 2.5.2 — Detección en tiempo real.
        """
        tiempos = self.cfg.tiempos_fallback_s
        fases   = [
            FaseSemaforica(
                carril=carril,
                tiempo_verde_s=t,
                tiempo_amarillo_s=self.cfg.tiempo_amarillo_s,
                tiempo_todo_rojo_s=self.cfg.tiempo_todo_rojo_s,
                n_vehiculos=0,
                n_autobuses=0,
            )
            for carril, t in self._ordenar(tiempos)
        ]
        ciclo_total = sum(f.duracion_total_s for f in fases)

        decision = DecisionSemaforica(
            timestamp=time.time(),
            fases=fases,
            tiempos_verde_s=dict(tiempos),
            fase_prioritaria="",
            nivel_congestion_global="desconocido",
            ciclo_total_s=round(ciclo_total, 1),
            origen="tiempo_fijo",
            motivo_prioridad="tiempo_fijo_fallback",
            n_frame_origen=n_frame,
        )

        self._registrar(decision, adaptativa=False)
        logger.warning(str(decision))
        return decision

    # ------------------------------------------------------------------
    # 3. Capa de priorización de autobús
    # ------------------------------------------------------------------

    def _autobuses_por_carril(self, detecciones_raw: List[dict]) -> Dict[str, int]:
        """
        Cuenta autobuses (clase 'autobús') por carril desde detecciones_raw.

        Extrae la información directamente de las detecciones individuales,
        sin necesidad de re-inferencia. Cada detección contiene el campo
        'carril' asignado por el detector.
        Protocolo, sección 2.3.7 — Mecanismos de priorización de flujo.
        """
        conteo: Dict[str, int] = defaultdict(int)
        for det in detecciones_raw:
            if det.get("clase") == "autobús":
                conteo[det.get("carril", "")] += 1
        return dict(conteo)

    def _aplicar_bonus_autobus(
        self,
        tiempos_base: Dict[str, float],
        buses_por_carril: Dict[str, int],
    ) -> Tuple[Dict[str, float], Dict[str, bool]]:
        """
        Suma el bonus de autobús a los carriles que tienen buses.

        El bonus es proporcional al número de autobuses detectados,
        acotado a max_autobuses_bonus para evitar monopolización del ciclo.

        Fórmula: t_bonus = min(n_buses, max_buses) × bonus_autobus_s

        Referencia: protocolo, sección 2.3.7; ILUNION (2024).

        Retorna
        -------
        tiempos_nuevos : dict[str, float]
        motivos_bus    : dict[str, bool]  — True si el carril recibió bonus
        """
        tiempos_nuevos: Dict[str, float] = {}
        motivos_bus:    Dict[str, bool]  = {}

        for carril, t_base in tiempos_base.items():
            n_buses = buses_por_carril.get(carril, 0)
            if n_buses > 0:
                bonus = min(n_buses, self.cfg.max_autobuses_bonus) * self.cfg.bonus_autobus_s
                tiempos_nuevos[carril] = t_base + bonus
                motivos_bus[carril]    = True
                logger.debug(
                    f"Bonus autobús → {carril}: "
                    f"{t_base:.0f}s + {bonus:.0f}s = {tiempos_nuevos[carril]:.0f}s "
                    f"({n_buses} bus(es))"
                )
            else:
                tiempos_nuevos[carril] = t_base
                motivos_bus[carril]    = False

        return tiempos_nuevos, motivos_bus

    # ------------------------------------------------------------------
    # 4. Capa de balanceo de ciclo
    # ------------------------------------------------------------------

    def _balancear_ciclo(
        self,
        tiempos_verde: Dict[str, float],
        conteo_por_carril: Dict[str, int],
    ) -> Dict[str, float]:
        """
        Escala los tiempos de verde para que el ciclo total esté dentro
        de [ciclo_min_s, ciclo_max_s], incluyendo amarillos y todo-rojos.

        Algoritmo
        ---------
        1. Calcular ciclo total con transiciones:
               ciclo = Σ(t_verde_i + t_amarillo + t_todo_rojo)
        2. Si ciclo < ciclo_min → escalar proporcionalmente al alza.
        3. Si ciclo > ciclo_max → recortar priorizando los carriles con
           menor congestión (se recortan primero los menos cargados),
           respetando siempre el tiempo mínimo de verde (t_mín = 12 s).

        Referencia: protocolo, sección 2.2.4 (método de Webster);
        sección 2.3.5 (ciclos adaptativos).
        """
        n_fases = len(tiempos_verde)
        t_trans = (self.cfg.tiempo_amarillo_s + self.cfg.tiempo_todo_rojo_s) * n_fases

        ciclo_solo_verdes_min = self.cfg.ciclo_min_s - t_trans
        ciclo_solo_verdes_max = self.cfg.ciclo_max_s - t_trans

        suma_verde = sum(tiempos_verde.values())

        # ── Caso 1: ciclo demasiado corto → escalar al alza ───────────────
        if suma_verde < ciclo_solo_verdes_min and suma_verde > 0:
            factor = ciclo_solo_verdes_min / suma_verde
            tiempos_verde = {c: round(t * factor, 1) for c, t in tiempos_verde.items()}
            logger.debug(f"Ciclo corto → escalado ×{factor:.2f}")

        # ── Caso 2: ciclo demasiado largo → recortar por prioridad ────────
        elif suma_verde > ciclo_solo_verdes_max:
            tiempos_verde = self._recortar_ciclo(
                tiempos_verde, ciclo_solo_verdes_max, conteo_por_carril
            )

        return tiempos_verde

    def _recortar_ciclo(
        self,
        tiempos: Dict[str, float],
        objetivo: float,
        conteo_por_carril: Dict[str, int],
    ) -> Dict[str, float]:
        """
        Recorta los tiempos de verde para alcanzar el objetivo,
        reduciendo primero los carriles con menor congestión.

        La política de recorte garantiza que ningún carril quede
        por debajo del tiempo mínimo de verde (protocolo, sec. 2.3.6).
        """
        from config.settings import TIEMPO_VERDE_MINIMO_S

        # Ordenar carriles de menor a mayor congestión (recortar primero los menos cargados)
        carriles_por_congestion = sorted(
            tiempos.keys(),
            key=lambda c: conteo_por_carril.get(c, 0),
        )

        tiempos_nuevos = dict(tiempos)
        exceso = sum(tiempos.values()) - objetivo

        for carril in carriles_por_congestion:
            if exceso <= 0:
                break
            disponible = tiempos_nuevos[carril] - TIEMPO_VERDE_MINIMO_S
            recorte    = min(disponible, exceso)
            if recorte > 0:
                tiempos_nuevos[carril] = round(tiempos_nuevos[carril] - recorte, 1)
                exceso -= recorte
                logger.debug(f"Recorte ciclo → {carril}: -{recorte:.0f}s")

        return tiempos_nuevos

    # ------------------------------------------------------------------
    # 5. Construcción de la salida
    # ------------------------------------------------------------------

    def _construir_fases(
        self,
        tiempos_verde: Dict[str, float],
        buses_por_carril: Dict[str, int],
        resultado: ResultadoDeteccion,
    ) -> List[FaseSemaforica]:
        """
        Empaqueta los tiempos calculados en objetos FaseSemaforica ordenados.
        El orden estándar es norte → centro → sur (ORDEN_FASES).
        """
        fases = []
        for carril in ORDEN_FASES:
            if carril not in tiempos_verde:
                continue
            fases.append(FaseSemaforica(
                carril=carril,
                tiempo_verde_s=tiempos_verde[carril],
                tiempo_amarillo_s=self.cfg.tiempo_amarillo_s,
                tiempo_todo_rojo_s=self.cfg.tiempo_todo_rojo_s,
                prioridad_autobus=buses_por_carril.get(carril, 0) > 0,
                n_vehiculos=resultado.conteo_por_carril.get(carril, 0),
                n_autobuses=buses_por_carril.get(carril, 0),
            ))
        return fases

    def _motivo_prioridad(
        self,
        fase_prioritaria: str,
        resultado: ResultadoDeteccion,
        buses_por_carril: Dict[str, int],
        motivos_bus: Dict[str, bool],
    ) -> str:
        """Genera una cadena descriptiva del motivo de priorización."""
        if motivos_bus.get(fase_prioritaria):
            return f"autobus_{fase_prioritaria}"
        if resultado.nivel_congestion == "congestionado":
            return f"congestion_{fase_prioritaria}"
        if resultado.nivel_congestion == "moderado":
            return f"moderado_{fase_prioritaria}"
        return f"distribucion_uniforme"

    # ------------------------------------------------------------------
    # 6. Historial y métricas
    # ------------------------------------------------------------------

    def _registrar(self, decision: DecisionSemaforica, adaptativa: bool) -> None:
        """Acumula la decisión en el historial y actualiza estadísticas."""
        self.historial.append(decision)
        s = self._stats
        s["decisiones_totales"]   += 1
        s["ciclo_acum_s"]         += decision.ciclo_total_s
        if adaptativa:
            s["decisiones_adaptativas"] += 1
        else:
            s["decisiones_fallback"]    += 1
        if decision.motivo_prioridad.startswith("autobus"):
            s["activaciones_bus"] += 1

    def resumen_sesion(self) -> str:
        """Retorna un resumen estadístico de la sesión como string."""
        s  = self._stats
        n  = max(s["decisiones_totales"], 1)
        return (
            f"\n{'='*60}\n"
            f"RESUMEN MOTOR — Actividad 10\n"
            f"  Decisiones totales     : {s['decisiones_totales']}\n"
            f"  Modo adaptativo        : {s['decisiones_adaptativas']}\n"
            f"  Modo fallback          : {s['decisiones_fallback']}\n"
            f"  Activaciones bus       : {s['activaciones_bus']}\n"
            f"  Ciclo promedio         : {s['ciclo_acum_s']/n:.1f} s\n"
            f"{'='*60}"
        )

    # ------------------------------------------------------------------
    # 7. Utilidades internas
    # ------------------------------------------------------------------

    @staticmethod
    def _ordenar(tiempos: Dict[str, float]) -> List[Tuple[str, float]]:
        """Retorna los carriles en ORDEN_FASES, descartando los ausentes."""
        return [(c, tiempos[c]) for c in ORDEN_FASES if c in tiempos]
