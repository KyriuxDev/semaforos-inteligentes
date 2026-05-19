"""
tests/simulacion.py — Actividad 11: Simulación y optimización de parámetros.

Implementa la Actividad 11 del plan de trabajo (06/05/26 – 13/05/26):
"Simular y optimizar los parámetros del sistema de semáforos inteligentes".

No requiere YOLOv5 ni cámaras: los ResultadoDeteccion se sintetizan
directamente con distribuciones vehiculares parametrizadas, permitiendo
evaluar cientos de configuraciones en segundos.

Flujo
-----
  1. Genera 6 escenarios de tráfico (libre → hora pico).
  2. Construye una grilla de 729 configuraciones de parámetros.
  3. Para cada configuración ejecuta MotorDecision sobre los 6 escenarios
     (50 frames c/u) y calcula 5 métricas de desempeño.
  4. Clasifica las configuraciones por score compuesto ponderado.
  5. Escribe el reporte en data/reporte_simulacion.txt e imprime un resumen.

Métricas
--------
  · eficiencia_verde  — Σ verde / ciclo_total  (mayor es mejor)
  · equidad_carriles  — 1 − CV(verdes)          (mayor es mejor)
  · tiempo_espera_est — (ciclo − verde) / 2     (menor es mejor)
  · activaciones_bus  — fases con bonus de autobús activado
  · score_global      — combinación lineal ponderada por escenario

Uso
---
  python -m tests.simulacion
  python main.py --simulacion
"""

from __future__ import annotations

import itertools
import random
import statistics
import time
from dataclasses import dataclass, field
from typing import Dict, List

from config.settings import ConfigMotor
from core.decision.motor import MotorDecision
from models.schemas import DecisionSemaforica, ResultadoDeteccion
from utils.logger import configurar_logging, get_logger

logger = get_logger("tests.simulacion")

# ── Reproducibilidad ────────────────────────────────────────────────────────
SEMILLA = 42
random.seed(SEMILLA)

# ── Parámetros de simulación ─────────────────────────────────────────────────
FRAMES_POR_ESCENARIO = 60   # frames sintéticos por escenario


# ===========================================================================
# SECCIÓN 1 — Escenarios de tráfico
# ===========================================================================

@dataclass
class Escenario:
    """
    Distribución vehicular para un escenario de tráfico.

    Atributos
    ----------
    nombre : str
    descripcion : str
    peso : float
        Importancia relativa en el score global (suma = 1.0 en ESCENARIOS).
    conteos : list[dict]
        Lista de distribuciones por frame:
        {"norte": int, "centro": int, "sur": int, "buses_norte": int}
    """
    nombre:      str
    descripcion: str
    peso:        float
    conteos:     List[Dict]


def _gen_conteos(
    norte_rng:    tuple,
    centro_rng:   tuple,
    sur_rng:      tuple,
    buses_norte:  int = 0,
    n:            int = FRAMES_POR_ESCENARIO,
) -> List[Dict]:
    """Genera N diccionarios de conteo vehicular con variación aleatoria."""
    return [
        {
            "norte":       random.randint(*norte_rng),
            "centro":      random.randint(*centro_rng),
            "sur":         random.randint(*sur_rng),
            "buses_norte": buses_norte,
        }
        for _ in range(n)
    ]


ESCENARIOS: List[Escenario] = [
    Escenario(
        nombre="libre",
        descripcion="Flujo libre — 0–2 veh/carril, sin buses",
        peso=0.10,
        conteos=_gen_conteos((0, 2), (0, 2), (0, 2)),
    ),
    Escenario(
        nombre="moderado",
        descripcion="Flujo moderado — 3–5 veh/carril",
        peso=0.15,
        conteos=_gen_conteos((3, 5), (3, 5), (3, 5)),
    ),
    Escenario(
        nombre="congestionado",
        descripcion="Flujo congestionado — 7–10 veh/carril",
        peso=0.20,
        conteos=_gen_conteos((7, 10), (7, 10), (7, 10)),
    ),
    Escenario(
        nombre="asimetrico",
        descripcion="Norte congestionado (8–12), sur libre (0–2)",
        peso=0.25,
        conteos=_gen_conteos((8, 12), (3, 5), (0, 2)),
    ),
    Escenario(
        nombre="con_autobus",
        descripcion="Bus prioritario en norte — 2–4 veh + 1 bus",
        peso=0.10,
        conteos=_gen_conteos((2, 4), (3, 5), (2, 4), buses_norte=1),
    ),
    Escenario(
        nombre="hora_pico",
        descripcion="Hora pico vespertina — alta demanda mixta + bus",
        peso=0.20,
        conteos=_gen_conteos((6, 10), (8, 12), (5, 9), buses_norte=1),
    ),
]


# ===========================================================================
# SECCIÓN 2 — Generador de ResultadoDeteccion sintético
# ===========================================================================

def _calcular_verde(n_veh: int, t_base: float, t_min: float, t_max: float) -> float:
    """Fórmula adaptativa: max(t_mín, min(t_máx, n_veh × t_base))."""
    return max(t_min, min(t_max, n_veh * t_base))


def _nivel_congestion(total: int) -> str:
    if total <= 2:  return "libre"
    if total <= 6:  return "moderado"
    return "congestionado"


def crear_resultado(
    frame_idx: int,
    conteo:    Dict,
    t_base:    float,
    t_min:     float,
    t_max:     float,
) -> ResultadoDeteccion:
    """
    Sintetiza un ResultadoDeteccion sin YOLOv5.

    Los autobuses se incluyen en detecciones_raw con clase='autobús' para
    que MotorDecision._autobuses_por_carril() los detecte correctamente.
    Los vehículos ligeros se representan con clase='automóvil'.
    """
    norte   = conteo["norte"]
    centro  = conteo["centro"]
    sur     = conteo["sur"]
    n_buses = conteo.get("buses_norte", 0)

    # El carril norte agrupa automóviles + autobuses
    conteo_carril = {
        "norte":  norte + n_buses,
        "centro": centro,
        "sur":    sur,
    }
    total = sum(conteo_carril.values())

    tiempos_verde = {
        c: _calcular_verde(conteo_carril[c], t_base, t_min, t_max)
        for c in ("norte", "centro", "sur")
    }

    raw: List[dict] = []
    for carril, n in [("norte", norte), ("centro", centro), ("sur", sur)]:
        for _ in range(n):
            raw.append({
                "clase":     "automóvil",
                "clase_id":  2,
                "carril":    carril,
                "confianza": 0.85,
                "bbox":      (0, 0, 0, 0),
                "centro":    (0, 0),
            })
    for _ in range(n_buses):
        raw.append({
            "clase":     "autobús",
            "clase_id":  5,
            "carril":    "norte",
            "confianza": 0.90,
            "bbox":      (0, 0, 0, 0),
            "centro":    (0, 0),
        })

    return ResultadoDeteccion(
        n_frame=frame_idx,
        timestamp=time.time(),
        total_vehiculos=total,
        conteo_por_tipo={
            "automóvil": norte + centro + sur,
            "autobús":   n_buses,
        },
        conteo_por_carril=conteo_carril,
        tiempo_verde_recomendado_s=tiempos_verde,
        detecciones_raw=raw,
        latencia_ms=0.0,
        nivel_congestion=_nivel_congestion(total),
    )


# ===========================================================================
# SECCIÓN 3 — Métricas de evaluación
# ===========================================================================

@dataclass
class MetricasSimulacion:
    """Métricas agregadas de una corrida de simulación sobre un escenario."""
    ciclo_promedio_s:    float = 0.0
    ciclo_std_s:         float = 0.0
    eficiencia_verde:    float = 0.0   # Σverde / ciclo_total  [0–1]
    equidad_carriles:    float = 0.0   # 1 − CV(verdes)        [0–1]
    tiempo_espera_est_s: float = 0.0   # Estimación media de espera/vehículo
    activaciones_bus:    int   = 0
    decisiones_fallback: int   = 0
    score:               float = 0.0


def calcular_metricas(decisiones: List[DecisionSemaforica]) -> MetricasSimulacion:
    """
    Calcula métricas de desempeño a partir de una lista de DecisionSemaforica.

    Tiempo de espera estimado
    --------------------------
    Para cada fase, los vehículos que llegan durante el rojo esperan en
    promedio (ciclo − verde) / 2 segundos.
    Se pondera por el número de vehículos en cada carril.

    Score parcial
    -------------
    Combinación lineal:
      0.30 · eficiencia_verde
    + 0.25 · equidad_carriles
    + 0.30 · (1 − espera_norm)   [normalizada con 90 s como máximo]
    + 0.15 · activacion_bus_norm
    """
    if not decisiones:
        return MetricasSimulacion()

    ciclos     = [d.ciclo_total_s for d in decisiones]
    ciclo_prom = statistics.mean(ciclos)
    ciclo_std  = statistics.stdev(ciclos) if len(ciclos) > 1 else 0.0

    # Eficiencia verde
    eficiencias = []
    for d in decisiones:
        if d.ciclo_total_s > 0:
            eficiencias.append(sum(d.tiempos_verde_s.values()) / d.ciclo_total_s)
    eficiencia_verde = statistics.mean(eficiencias) if eficiencias else 0.0

    # Equidad (1 − coeficiente de variación)
    equidades = []
    for d in decisiones:
        v = list(d.tiempos_verde_s.values())
        if len(v) > 1 and statistics.mean(v) > 0:
            cv = statistics.stdev(v) / statistics.mean(v)
            equidades.append(max(0.0, 1.0 - cv))
    equidad = statistics.mean(equidades) if equidades else 1.0

    # Tiempo de espera estimado ponderado por vehículos
    esperas: List[float] = []
    for d in decisiones:
        for fase in d.fases:
            n_veh = fase.n_vehiculos
            if n_veh > 0 and d.ciclo_total_s > 0:
                t_rojo  = d.ciclo_total_s - fase.tiempo_verde_s
                esperas.extend([t_rojo / 2] * n_veh)
    espera_est = statistics.mean(esperas) if esperas else ciclo_prom / 2

    # Contadores
    act_bus  = sum(1 for d in decisiones if d.motivo_prioridad.startswith("autobus"))
    fallback = sum(1 for d in decisiones if d.origen == "tiempo_fijo")

    # Score parcial
    score = (
        0.30 * eficiencia_verde
        + 0.25 * equidad
        + 0.30 * max(0.0, 1.0 - espera_est / 90.0)
        + 0.15 * min(1.0, act_bus / max(1, len(decisiones)) * 10)
    )

    return MetricasSimulacion(
        ciclo_promedio_s=round(ciclo_prom, 1),
        ciclo_std_s=round(ciclo_std, 1),
        eficiencia_verde=round(eficiencia_verde, 3),
        equidad_carriles=round(equidad, 3),
        tiempo_espera_est_s=round(espera_est, 1),
        activaciones_bus=act_bus,
        decisiones_fallback=fallback,
        score=round(score, 4),
    )


# ===========================================================================
# SECCIÓN 4 — Grilla de parámetros
# ===========================================================================

GRILLA: Dict[str, List[float]] = {
    "t_base":        [3.0, 4.0, 5.0],
    "t_min":         [10.0, 12.0, 15.0],
    "t_max":         [45.0, 60.0, 75.0],
    "bonus_autobus": [5.0,  8.0,  12.0],
    "ciclo_min":     [40.0, 45.0, 55.0],
    "ciclo_max":     [150.0, 180.0, 210.0],
}


@dataclass
class ConfigCandidada:
    """Una combinación de parámetros candidata."""
    t_base:        float
    t_min:         float
    t_max:         float
    bonus_autobus: float
    ciclo_min:     float
    ciclo_max:     float


def generar_grilla() -> List[ConfigCandidada]:
    """
    Genera el producto cartesiano de todos los valores de GRILLA,
    descartando combinaciones inválidas (t_min ≥ t_max o ciclo_min ≥ ciclo_max).
    """
    keys  = list(GRILLA.keys())
    vals  = [GRILLA[k] for k in keys]
    configs = []
    for combo in itertools.product(*vals):
        kw = dict(zip(keys, combo))
        if kw["t_min"] >= kw["t_max"]:
            continue
        if kw["ciclo_min"] >= kw["ciclo_max"]:
            continue
        configs.append(ConfigCandidada(**kw))
    return configs


# ===========================================================================
# SECCIÓN 5 — Motor de simulación
# ===========================================================================

@dataclass
class ResultadoConfig:
    """Métricas por escenario para una configuración de parámetros."""
    config:               ConfigCandidada
    metricas_escenario:   Dict[str, MetricasSimulacion] = field(default_factory=dict)
    score_global:         float = 0.0


def simular_config(
    config:    ConfigCandidada,
    escenarios: List[Escenario],
) -> ResultadoConfig:
    """
    Ejecuta MotorDecision sobre todos los escenarios con la configuración dada.

    Se instancia un MotorDecision fresco por escenario para evitar contaminación
    entre historiales.
    """
    cfg_motor = ConfigMotor(
        bonus_autobus_s=config.bonus_autobus,
        ciclo_min_s=config.ciclo_min,
        ciclo_max_s=config.ciclo_max,
        tiempo_amarillo_s=3.0,
        tiempo_todo_rojo_s=2.0,
        max_frames_sin_deteccion=10,
    )

    resultado = ResultadoConfig(config=config)

    for escenario in escenarios:
        motor      = MotorDecision(cfg_motor)
        decisiones: List[DecisionSemaforica] = []

        for i, conteo in enumerate(escenario.conteos):
            res_det  = crear_resultado(
                frame_idx=i,
                conteo=conteo,
                t_base=config.t_base,
                t_min=config.t_min,
                t_max=config.t_max,
            )
            decision = motor.decidir(res_det)
            decisiones.append(decision)

        resultado.metricas_escenario[escenario.nombre] = calcular_metricas(decisiones)

    # Score global: promedio ponderado por escenario
    resultado.score_global = round(
        sum(
            esc.peso * resultado.metricas_escenario[esc.nombre].score
            for esc in escenarios
        ),
        4,
    )
    return resultado


# ===========================================================================
# SECCIÓN 6 — Reporte
# ===========================================================================

_CFG_ACTUAL = ConfigCandidada(
    t_base=4.0, t_min=12.0, t_max=60.0,
    bonus_autobus=8.0, ciclo_min=45.0, ciclo_max=180.0,
)


def _fmt_cfg(c: ConfigCandidada) -> str:
    return (
        f"t_base={c.t_base:.1f}  t_min={c.t_min:.0f}  t_max={c.t_max:.0f}  "
        f"bonus={c.bonus_autobus:.0f}  c_min={c.ciclo_min:.0f}  c_max={c.ciclo_max:.0f}"
    )


def generar_reporte(
    resultados:   List[ResultadoConfig],
    ruta_salida:  str = "data/reporte_simulacion.txt",
) -> str:
    """
    Genera el reporte textual con ranking global, detalle de top-5 y
    comparativa contra la configuración actual del sistema.
    """
    top       = sorted(resultados, key=lambda r: r.score_global, reverse=True)
    n_configs = len(resultados)
    mejor     = top[0]

    # ── Configuración actual (simulada para comparativa) ───────────────────
    res_actual = simular_config(_CFG_ACTUAL, ESCENARIOS)

    SEP  = "═" * 72
    sep  = "─" * 72
    L    = []

    # Encabezado
    L += [
        SEP,
        "ACTIVIDAD 11 — SIMULACIÓN Y OPTIMIZACIÓN DE PARÁMETROS",
        "ITO — Sistema de Semáforos Inteligentes | Taller de Investigación II",
        SEP,
        f"  Configuraciones evaluadas : {n_configs}",
        f"  Escenarios por config.    : {len(ESCENARIOS)}",
        f"  Frames por escenario      : {FRAMES_POR_ESCENARIO}",
        f"  Total decisiones          : {n_configs * len(ESCENARIOS) * FRAMES_POR_ESCENARIO:,}",
        f"  Semilla aleatoria         : {SEMILLA}",
        "",
    ]

    # Ranking top-10
    L += [
        sep,
        "RANKING GLOBAL — TOP 10",
        sep,
        f"  {'#':>3}  {'t_base':>6}  {'t_min':>5}  {'t_max':>5}  "
        f"{'bonus':>5}  {'c_min':>5}  {'c_max':>5}  {'score':>7}",
        "  " + "─" * 56,
    ]
    for i, r in enumerate(top[:10], 1):
        c = r.config
        L.append(
            f"  {i:>3}  {c.t_base:>6.1f}  {c.t_min:>5.0f}  {c.t_max:>5.0f}  "
            f"{c.bonus_autobus:>5.0f}  {c.ciclo_min:>5.0f}  {c.ciclo_max:>5.0f}  "
            f"{r.score_global:>7.4f}"
        )
    # Posición de config actual
    pos_actual = next(
        (i + 1 for i, r in enumerate(top)
         if r.config == _CFG_ACTUAL or (
             r.config.t_base == _CFG_ACTUAL.t_base and
             r.config.t_min  == _CFG_ACTUAL.t_min  and
             r.config.t_max  == _CFG_ACTUAL.t_max  and
             r.config.bonus_autobus == _CFG_ACTUAL.bonus_autobus
         )),
        None
    )
    if pos_actual:
        L.append(f"\n  (Config. actual en posición #{pos_actual} de {n_configs})")
    L.append("")

    # Detalle top-5
    L += [sep, "DETALLE — TOP 5 CONFIGURACIONES", sep]
    for rank, r in enumerate(top[:5], 1):
        c = r.config
        L += [
            "",
            f"  [{rank}] score={r.score_global:.4f}  {_fmt_cfg(c)}",
            f"  {'Escenario':<16}  {'ciclo_avg':>9}  {'efic_verde':>10}  "
            f"{'equidad':>7}  {'espera_est':>10}  {'act_bus':>7}  {'score':>7}",
            "  " + "─" * 66,
        ]
        for nombre, m in r.metricas_escenario.items():
            L.append(
                f"  {nombre:<16}  {m.ciclo_promedio_s:>9.1f}  "
                f"{m.eficiencia_verde:>10.3f}  {m.equidad_carriles:>7.3f}  "
                f"{m.tiempo_espera_est_s:>10.1f}  {m.activaciones_bus:>7}  "
                f"{m.score:>7.4f}"
            )
    L.append("")

    # Configuración recomendada
    c = mejor.config
    L += [
        sep,
        "CONFIGURACIÓN RECOMENDADA",
        sep,
        f"  t_base (s/vehículo)     : {c.t_base}",
        f"  t_verde_mínimo (s)      : {c.t_min}",
        f"  t_verde_máximo (s)      : {c.t_max}",
        f"  bonus_autobús (s)       : {c.bonus_autobus}",
        f"  ciclo_mínimo (s)        : {c.ciclo_min}",
        f"  ciclo_máximo (s)        : {c.ciclo_max}",
        f"  score_global            : {mejor.score_global:.4f}",
        "",
        "  Para aplicar, actualiza config/settings.py:",
        f"    TIEMPO_BASE_POR_VEHICULO_S = {c.t_base}",
        f"    TIEMPO_VERDE_MINIMO_S      = {c.t_min}",
        f"    TIEMPO_VERDE_MAXIMO_S      = {c.t_max}",
        "",
        "  Y en ConfigMotor (config/settings.py, sección F):",
        f"    bonus_autobus_s = {c.bonus_autobus}",
        f"    ciclo_min_s     = {c.ciclo_min}",
        f"    ciclo_max_s     = {c.ciclo_max}",
        "",
    ]

    # Comparativa actual vs recomendada
    L += [
        sep,
        "COMPARATIVA — CONFIGURACIÓN ACTUAL vs RECOMENDADA",
        sep,
        f"  {'Métrica':<28}  {'Actual':>10}  {'Recomendada':>12}  {'Δ':>9}",
        "  " + "─" * 64,
        f"  {'Score global':<28}  {res_actual.score_global:>10.4f}  "
        f"{mejor.score_global:>12.4f}  "
        f"{(mejor.score_global - res_actual.score_global)*100:>+8.2f}%",
        "",
        f"  {'Escenario':<16}  {'espera actual':>13}  {'espera rec.':>11}  {'Δ (s)':>8}  {'mejora':>7}",
        "  " + "─" * 60,
    ]
    for esc in ESCENARIOS:
        m_a = res_actual.metricas_escenario.get(esc.nombre)
        m_r = mejor.metricas_escenario.get(esc.nombre)
        if m_a and m_r:
            delta  = m_r.tiempo_espera_est_s - m_a.tiempo_espera_est_s
            pct    = (delta / m_a.tiempo_espera_est_s * 100) if m_a.tiempo_espera_est_s > 0 else 0
            L.append(
                f"  {esc.nombre:<16}  {m_a.tiempo_espera_est_s:>13.1f}  "
                f"{m_r.tiempo_espera_est_s:>11.1f}  {delta:>+8.1f}  {pct:>+6.1f}%"
            )

    L += ["", SEP]
    reporte = "\n".join(L)

    import os
    os.makedirs("data", exist_ok=True)
    with open(ruta_salida, "w", encoding="utf-8") as f:
        f.write(reporte)
    logger.info(f"Reporte guardado en {ruta_salida}")

    return reporte


# ===========================================================================
# SECCIÓN 7 — Entry point
# ===========================================================================

def ejecutar_simulacion(ruta_reporte: str = "data/reporte_simulacion.txt") -> str:
    """
    Ejecuta la simulación completa y retorna el reporte como string.
    Llamable desde main.py con --simulacion o directamente como módulo.
    """
    configurar_logging()
    configs = generar_grilla()

    logger.info(
        f"Actividad 11 — Iniciando simulación: "
        f"{len(configs)} configuraciones × {len(ESCENARIOS)} escenarios "
        f"× {FRAMES_POR_ESCENARIO} frames"
    )

    resultados: List[ResultadoConfig] = []
    t0 = time.perf_counter()

    for i, config in enumerate(configs, 1):
        r = simular_config(config, ESCENARIOS)
        resultados.append(r)
        if i % 100 == 0 or i == len(configs):
            elapsed = time.perf_counter() - t0
            logger.info(f"  Progreso: {i}/{len(configs)} ({elapsed:.1f} s)")

    elapsed = time.perf_counter() - t0
    logger.info(f"Simulación completada en {elapsed:.2f} s")

    reporte = generar_reporte(resultados, ruta_salida=ruta_reporte)
    print(reporte)
    return reporte


if __name__ == "__main__":
    ejecutar_simulacion()