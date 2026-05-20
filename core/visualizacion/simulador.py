"""
core/visualizacion/simulador.py — Simulador visual comparativo en tiempo real.

Panel izquierdo  — LO QUE VE LA CÁMARA  (YOLOv5)
Panel derecho    — DECISION DEL SISTEMA  (Four Corners animado)

Fixes incluidos
---------------
  · Vehiculo._paso_linea(): vehículos que cruzaron la línea de pare
    siempre avanzan — no se quedan a media calle.
  · Enfoque._stop_efectivo(): car-following model — cada vehículo
    mantiene GAP px detrás del de adelante — no se enciman.
  · Enfoque._ordenar(): ordena la cola antes de actualizar.

Controles: q salir | p pausar/reanudar | s captura PNG
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import cv2
import numpy as np

from config.settings import ConfigDetector, ConfigMotor, ConfigPipeline
from core.decision.motor import MotorDecision
from core.detector.vehicular import DetectorVehicular
from models.schemas import DecisionSemaforica, ResultadoDeteccion
from utils.logger import get_logger

logger = get_logger("core.visualizacion.simulador")

# ===========================================================================
# Constantes
# ===========================================================================

PW = 640
PH = 480
H_HEAD = 38
H_FOOT = 80
H_ROAD = PH - H_HEAD - H_FOOT   # 362 px

IC_X = PW // 2               # 320
IC_Y = H_HEAD + H_ROAD // 2  # 219

LANE_W    = 22
DIR_LANES = 2
ROAD_HALF = DIR_LANES * 2 * LANE_W // 2   # 44 px
SEMAF_OFF = 14

INFERENCE_N  = 2
SPEED_FACTOR = 1.0
VEH_SPEED    = 75.0

C_BG      = (18,  20,  18)
C_ASPH    = (48,  50,  48)
C_INTER   = (58,  60,  58)
C_LANE    = (80,  80,  80)
C_CENTER  = (0,  170, 170)
C_STOP    = (210, 210, 210)
C_CONO    = (0,  110, 255)
C_CEBRA   = (200, 200, 200)
C_BUILD   = (35,  33,  42)
C_BUILD_E = (60,  58,  70)
C_CABLE   = (100, 100, 100)
C_VERDE   = (0,   210,  60)
C_AMARI   = (0,   210, 210)
C_ROJO    = (30,   30, 210)

CARRILES = ["norte", "centro", "sur"]

_VEH_COLS = [
    (30,144,255),(0,200,100),(0,165,255),
    (200,180,0),(160,80,220),(240,240,240),
    (50,50,180),(180,220,0),
]


# ===========================================================================
# Vehículo animado — car-following + no se queda a media calle
# ===========================================================================

@dataclass
class Vehiculo:
    x: float; y: float
    dx: float; dy: float
    color: tuple
    stop_val: float
    axis: str
    vw: int = 18; vh: int = 11

    def _paso_linea(self) -> bool:
        """True si el frente del vehículo ya cruzó la línea de pare."""
        if self.axis == 'y':
            f = self.y + self.vh/2 if self.dy > 0 else self.y - self.vh/2
            return f > self.stop_val if self.dy > 0 else f < self.stop_val
        else:
            f = self.x + self.vw/2 if self.dx > 0 else self.x - self.vw/2
            return f > self.stop_val if self.dx > 0 else f < self.stop_val

    def actualizar(self, dt: float, verde: bool, stop_ef: float) -> None:
        if self._paso_linea():
            self.x += self.dx * dt
            self.y += self.dy * dt
            return
        if self.axis == 'y':
            frente = self.y + self.vh/2 if self.dy > 0 else self.y - self.vh/2
            dist = (stop_ef - frente) if self.dy > 0 else (frente - stop_ef)
            if dist > 0.5:
                spd = VEH_SPEED if verde else min(VEH_SPEED, max(4.0, dist*2.5))
                self.y += (1 if self.dy > 0 else -1) * min(spd*dt, dist)
        else:
            frente = self.x + self.vw/2 if self.dx > 0 else self.x - self.vw/2
            dist = (stop_ef - frente) if self.dx > 0 else (frente - stop_ef)
            if dist > 0.5:
                spd = VEH_SPEED if verde else min(VEH_SPEED, max(4.0, dist*2.5))
                self.x += (1 if self.dx > 0 else -1) * min(spd*dt, dist)

    def fuera(self) -> bool:
        return (self.x < -60 or self.x > PW+60 or
                self.y < H_HEAD-60 or self.y > H_HEAD+H_ROAD+60)

    def draw(self, f: np.ndarray) -> None:
        x1 = int(self.x)-self.vw//2; y1 = int(self.y)-self.vh//2
        x2,y2 = x1+self.vw, y1+self.vh
        cv2.rectangle(f,(x1,y1),(x2,y2),self.color,-1)
        cv2.rectangle(f,(x1,y1),(x2,y2),(0,0,0),1)
        b = tuple(min(c+70,255) for c in self.color)
        if self.dy < 0 or (self.dy==0 and self.dx > 0):
            cv2.rectangle(f,(x2-5,y1+2),(x2-1,y2-2),b,-1)
        else:
            cv2.rectangle(f,(x1+1,y1+2),(x1+5,y2-2),b,-1)


# ===========================================================================
# Enfoque animado — car-following
# ===========================================================================

class Enfoque:
    GAP = 5

    def __init__(self, direction: str) -> None:
        self.direction = direction
        self.vehiculos: List[Vehiculo] = []
        self._timer = random.uniform(0, 2.5)
        rh = ROAD_HALF
        if direction == "norte":
            self._x0,self._x1 = IC_X-rh, IC_X
            self._sy = float(H_HEAD+4)
            self._dx,self._dy = 0.0, VEH_SPEED
            self._stop = float(IC_Y-rh-5)
            self._axis='y'; self._vw,self._vh=11,18
        elif direction == "sur":
            self._x0,self._x1 = IC_X, IC_X+rh
            self._sy = float(H_HEAD+H_ROAD-4)
            self._dx,self._dy = 0.0,-VEH_SPEED
            self._stop = float(IC_Y+rh+5)
            self._axis='y'; self._vw,self._vh=11,18
        elif direction == "este":
            self._y0,self._y1 = IC_Y-rh, IC_Y
            self._sx = float(PW-4)
            self._dx,self._dy = -VEH_SPEED, 0.0
            self._stop = float(IC_X+rh+5)
            self._axis='x'; self._vw,self._vh=18,11
        elif direction == "oeste":
            self._y0,self._y1 = IC_Y, IC_Y+rh
            self._sx = float(4)
            self._dx,self._dy = VEH_SPEED, 0.0
            self._stop = float(IC_X-rh-5)
            self._axis='x'; self._vw,self._vh=18,11

    def _ordenar(self) -> None:
        if self.direction == "norte":   self.vehiculos.sort(key=lambda v: -v.y)
        elif self.direction == "sur":   self.vehiculos.sort(key=lambda v:  v.y)
        elif self.direction == "este":  self.vehiculos.sort(key=lambda v:  v.x)
        elif self.direction == "oeste": self.vehiculos.sort(key=lambda v: -v.x)

    def _stop_efectivo(self, idx: int, verde: bool) -> float:
        base = (1e6 if (self.dy>0 or self.dx>0) else -1e6) if verde else self._stop
        if idx == 0:
            return base
        prev = self.vehiculos[idx-1]
        if self.direction == "norte":
            return min(base, prev.y - prev.vh/2 - self.GAP)
        elif self.direction == "sur":
            return max(base, prev.y + prev.vh/2 + self.GAP)
        elif self.direction == "este":
            return max(base, prev.x + prev.vw/2 + self.GAP)
        elif self.direction == "oeste":
            return min(base, prev.x - prev.vw/2 - self.GAP)
        return base

    def actualizar(self, dt: float, n_obj: int, verde: bool) -> None:
        self._ordenar()
        for i,v in enumerate(self.vehiculos):
            v.actualizar(dt, verde, self._stop_efectivo(i, verde))
        self.vehiculos = [v for v in self.vehiculos if not v.fuera()]
        self._timer += dt
        intervalo = max(0.4, 2.2 - n_obj*0.13)
        if len(self.vehiculos) < min(n_obj,7) and self._timer > intervalo:
            if self._libre():
                self._timer = 0.0; self._spawnear()

    def _libre(self) -> bool:
        sep=30
        if self.direction=="norte":  return all(v.y > H_HEAD+sep for v in self.vehiculos)
        elif self.direction=="sur":  return all(v.y < H_HEAD+H_ROAD-sep for v in self.vehiculos)
        elif self.direction=="este": return all(v.x < PW-sep for v in self.vehiculos)
        elif self.direction=="oeste":return all(v.x > sep for v in self.vehiculos)
        return True

    def _spawnear(self) -> None:
        color = random.choice(_VEH_COLS)
        if self.direction in ("norte","sur"):
            x = self._x0 + (random.randint(0,DIR_LANES-1)+0.5)*LANE_W
            y = self._sy
        else:
            x = self._sx
            y = self._y0 + (random.randint(0,DIR_LANES-1)+0.5)*LANE_W
        self.vehiculos.append(Vehiculo(
            x=x,y=y,dx=self._dx,dy=self._dy,color=color,
            stop_val=self._stop,axis=self._axis,vw=self._vw,vh=self._vh))

    def n_cola(self) -> int:
        if self.direction=="norte":  return sum(1 for v in self.vehiculos if v.y+v.vh/2 < IC_Y-ROAD_HALF)
        elif self.direction=="sur":  return sum(1 for v in self.vehiculos if v.y-v.vh/2 > IC_Y+ROAD_HALF)
        elif self.direction=="este": return sum(1 for v in self.vehiculos if v.x-v.vw/2 > IC_X+ROAD_HALF)
        elif self.direction=="oeste":return sum(1 for v in self.vehiculos if v.x+v.vw/2 < IC_X-ROAD_HALF)
        return 0

    def draw(self, f: np.ndarray) -> None:
        for v in self.vehiculos: v.draw(f)


# ===========================================================================
# Estado del ciclo semafórico
# ===========================================================================

class EstadoCiclo:
    def __init__(self, t_amarillo: float=3.0, t_todo_rojo: float=2.0) -> None:
        self._idx=0; self._sub="verde"; self._t=30.0
        self._tam=t_amarillo; self._ttr=t_todo_rojo
        self._dec: Optional[DecisionSemaforica]=None
        self._pend: Optional[DecisionSemaforica]=None

    def nueva_decision(self, d): self._pend=d

    def avanzar(self, dt):
        self._t -= dt
        if self._t > 0: return
        if self._sub=="verde":      self._sub="amarillo"; self._t=self._tam
        elif self._sub=="amarillo": self._sub="todo_rojo"; self._t=self._ttr
        elif self._sub=="todo_rojo":
            self._idx=(self._idx+1)%3; self._sub="verde"
            if self._pend: self._dec=self._pend; self._pend=None
            t=30.0
            if self._dec: t=self._dec.tiempos_verde_s.get(CARRILES[self._idx],30.0)
            self._t=t

    @property
    def activo(self): return CARRILES[self._idx]
    @property
    def subfase(self): return self._sub
    @property
    def t_rest(self): return max(0.0,self._t)
    def color(self,c):
        if self._sub=="todo_rojo": return C_ROJO
        if c==self.activo: return C_VERDE if self._sub=="verde" else C_AMARI
        return C_ROJO
    def verde(self,c): return c==self.activo and self._sub=="verde"


# ===========================================================================
# SimuladorVisual
# ===========================================================================

class SimuladorVisual:
    WINDOW = "ITO — Comparacion | Camara vs Decision del Sistema"

    def __init__(self, cfg_pipeline, cfg_detector, cfg_motor) -> None:
        self.cfg_p    = cfg_pipeline
        self.detector = DetectorVehicular(cfg_pipeline, cfg_detector)
        self.motor    = MotorDecision(cfg_motor)
        self.ciclo    = EstadoCiclo(cfg_motor.tiempo_amarillo_s, cfg_motor.tiempo_todo_rojo_s)
        self._enf: Dict[str,Enfoque] = {
            d: Enfoque(d) for d in ("norte","sur","este","oeste")}
        self._n_frame=0; self._last_t=0.0
        self._ult_res: Optional[ResultadoDeteccion]=None
        self._ult_dec: Optional[DecisionSemaforica]=None
        self._ult_frm: Optional[np.ndarray]=None

    def ejecutar(self) -> None:
        self.detector.iniciar()
        cap = cv2.VideoCapture(self.cfg_p.rtsp_url)
        if not cap.isOpened(): raise IOError(f"No se pudo abrir: {self.cfg_p.rtsp_url}")
        fps=cap.get(cv2.CAP_PROP_FPS) or 25.0
        fts=1.0/fps
        cv2.namedWindow(self.WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.WINDOW, PW*2+4, PH)
        self._last_t=time.perf_counter(); pausado=False
        logger.info(f"Simulador — {fps:.0f}fps | INFERENCE_N={INFERENCE_N}")
        try:
            while True:
                t0=time.perf_counter()
                dt=min(t0-self._last_t,0.1); self._last_t=t0
                if not pausado:
                    ret,frm=cap.read()
                    if not ret:
                        cap.set(cv2.CAP_PROP_POS_FRAMES,0); self._n_frame=0; continue
                    self._n_frame+=1; self._ult_frm=frm
                    if self._n_frame % INFERENCE_N == 0:
                        res=self.detector.inferir(frm,self._n_frame)
                        self.detector._actualizar_metricas(res)
                        dec=self.motor.decidir(res)
                        self._ult_res=res; self._ult_dec=dec
                        self.ciclo.nueva_decision(dec)
                        if self._n_frame%50==0:
                            logger.info(f"[{self._n_frame:05d}] veh={res.total_vehiculos} "
                                f"({res.nivel_congestion}) | ciclo={dec.ciclo_total_s:.0f}s "
                                f"prior={dec.fase_prioritaria} | lat={res.latencia_ms:.0f}ms")
                    else:
                        res,dec=self._ult_res,self._ult_dec
                else:
                    frm,res,dec=self._ult_frm,self._ult_res,self._ult_dec
                if res is None or frm is None: cv2.waitKey(1); continue
                self.ciclo.avanzar(dt*SPEED_FACTOR)
                n_n=res.conteo_por_carril.get("norte",0)
                n_c=res.conteo_por_carril.get("centro",0)
                n_s=res.conteo_por_carril.get("sur",0)
                self._enf["norte"].actualizar(dt,n_n,self.ciclo.verde("norte"))
                self._enf["sur"].actualizar(  dt,n_s,self.ciclo.verde("sur"))
                self._enf["este"].actualizar( dt,n_c//2,self.ciclo.verde("centro"))
                self._enf["oeste"].actualizar(dt,n_c//2,self.ciclo.verde("centro"))
                pizq=self._panel_camara(frm,res)
                pder=self._panel_aereo(res,dec)
                cv2.imshow(self.WINDOW,np.hstack([pizq,pder]))
                wms=max(1,int((fts-(time.perf_counter()-t0))*1000))
                k=cv2.waitKey(wms)&0xFF
                if k==ord("q"): break
                elif k==ord("p"): pausado=not pausado; logger.info("PAUSADO" if pausado else "REANUDADO")
                elif k==ord("s"):
                    nm=f"data/sim_{self._n_frame:05d}.png"
                    cv2.imwrite(nm,np.hstack([pizq,pder])); logger.info(f"Captura: {nm}")
        except KeyboardInterrupt: logger.info("Interrupcion.")
        finally:
            cap.release(); cv2.destroyAllWindows()
            self.detector._log_metricas(); print(self.motor.resumen_sesion())

    def _panel_camara(self,frm,res):
        p=cv2.resize(self.detector.anotar_frame(frm,res),(PW,PH))
        cv2.rectangle(p,(0,0),(260,24),(10,10,10),-1)
        cv2.putText(p,"LO QUE VE LA CAMARA — YOLOv5",(6,16),
                    cv2.FONT_HERSHEY_SIMPLEX,0.42,(160,160,160),1)
        return p

    def _panel_aereo(self,res,dec):
        f=np.full((PH,PW,3),C_BG,dtype=np.uint8)
        self._hdr(f); self._edificios(f); self._calles(f)
        self._marcas(f); self._cebras(f); self._conos(f)
        self._lineas_pare(f); self._cardinales(f)
        for e in self._enf.values(): e.draw(f)
        self._semaforos(f,res); self._footer(f,res,dec)
        return f

    def _hdr(self,f):
        cv2.rectangle(f,(0,0),(PW,H_HEAD),(12,12,12),-1)
        cv2.putText(f,"DECISION DEL SISTEMA — Four Corners Intersection",
                    (8,25),cv2.FONT_HERSHEY_SIMPLEX,0.44,(100,220,255),1)

    def _edificios(self,f):
        mg=6
        for (x1,y1),(x2,y2) in [
            ((0,H_HEAD),(IC_X-ROAD_HALF-mg,IC_Y-ROAD_HALF-mg)),
            ((IC_X+ROAD_HALF+mg,H_HEAD),(PW,IC_Y-ROAD_HALF-mg)),
            ((0,IC_Y+ROAD_HALF+mg),(IC_X-ROAD_HALF-mg,H_HEAD+H_ROAD)),
            ((IC_X+ROAD_HALF+mg,IC_Y+ROAD_HALF+mg),(PW,H_HEAD+H_ROAD))]:
            if x1>=x2 or y1>=y2: continue
            cv2.rectangle(f,(x1,y1),(x2,y2),C_BUILD,-1)
            cv2.rectangle(f,(x1,y1),(x2,y2),C_BUILD_E,1)
            for wx in range(x1+8,x2-6,18):
                for wy in range(y1+10,y2-6,16):
                    if wx+8<x2 and wy+8<y2:
                        cv2.rectangle(f,(wx,wy),(wx+8,wy+8),(45,40,55),-1)

    def _calles(self,f):
        cv2.rectangle(f,(IC_X-ROAD_HALF,H_HEAD),(IC_X+ROAD_HALF,H_HEAD+H_ROAD),C_ASPH,-1)
        cv2.rectangle(f,(0,IC_Y-ROAD_HALF),(PW,IC_Y+ROAD_HALF),C_ASPH,-1)
        cv2.rectangle(f,(IC_X-ROAD_HALF,IC_Y-ROAD_HALF),(IC_X+ROAD_HALF,IC_Y+ROAD_HALF),C_INTER,-1)

    def _marcas(self,f):
        for xo in [-LANE_W,LANE_W]:
            xc=IC_X+xo
            for y in range(H_HEAD,IC_Y-ROAD_HALF,14): cv2.line(f,(xc,y),(xc,min(y+7,IC_Y-ROAD_HALF)),C_LANE,1)
            for y in range(IC_Y+ROAD_HALF,H_HEAD+H_ROAD,14): cv2.line(f,(xc,y),(xc,min(y+7,H_HEAD+H_ROAD)),C_LANE,1)
        for yo in [-LANE_W,LANE_W]:
            yc=IC_Y+yo
            for x in range(0,IC_X-ROAD_HALF,14): cv2.line(f,(x,yc),(min(x+7,IC_X-ROAD_HALF),yc),C_LANE,1)
            for x in range(IC_X+ROAD_HALF,PW,14): cv2.line(f,(x,yc),(min(x+7,PW),yc),C_LANE,1)
        for y in range(H_HEAD,IC_Y-ROAD_HALF,12): cv2.line(f,(IC_X,y),(IC_X,min(y+6,IC_Y-ROAD_HALF)),C_CENTER,1)
        for y in range(IC_Y+ROAD_HALF,H_HEAD+H_ROAD,12): cv2.line(f,(IC_X,y),(IC_X,min(y+6,H_HEAD+H_ROAD)),C_CENTER,1)
        for x in range(0,IC_X-ROAD_HALF,12): cv2.line(f,(x,IC_Y),(min(x+6,IC_X-ROAD_HALF),IC_Y),C_CENTER,1)
        for x in range(IC_X+ROAD_HALF,PW,12): cv2.line(f,(x,IC_Y),(min(x+6,PW),IC_Y),C_CENTER,1)

    def _cebras(self,f):
        sw,sh,g=6,8,5
        for x in range(IC_X-ROAD_HALF,IC_X+ROAD_HALF,sw+g):
            cv2.rectangle(f,(x,IC_Y+ROAD_HALF+3),(min(x+sw,IC_X+ROAD_HALF),IC_Y+ROAD_HALF+3+sh),C_CEBRA,-1)
            cv2.rectangle(f,(x,IC_Y-ROAD_HALF-3-sh),(min(x+sw,IC_X+ROAD_HALF),IC_Y-ROAD_HALF-3),C_CEBRA,-1)
        for y in range(IC_Y-ROAD_HALF,IC_Y+ROAD_HALF,sw+g):
            cv2.rectangle(f,(IC_X+ROAD_HALF+3,y),(IC_X+ROAD_HALF+3+sh,min(y+sw,IC_Y+ROAD_HALF)),C_CEBRA,-1)
            cv2.rectangle(f,(IC_X-ROAD_HALF-3-sh,y),(IC_X-ROAD_HALF-3,min(y+sw,IC_Y+ROAD_HALF)),C_CEBRA,-1)

    def _conos(self,f):
        p=22
        for y in range(H_HEAD+8,IC_Y-ROAD_HALF,p): self._cono(f,IC_X-ROAD_HALF+2,y); self._cono(f,IC_X+ROAD_HALF-2,y)
        for y in range(IC_Y+ROAD_HALF+8,H_HEAD+H_ROAD,p): self._cono(f,IC_X-ROAD_HALF+2,y); self._cono(f,IC_X+ROAD_HALF-2,y)
        for x in range(8,IC_X-ROAD_HALF,p): self._cono(f,x,IC_Y-ROAD_HALF+2); self._cono(f,x,IC_Y+ROAD_HALF-2)
        for x in range(IC_X+ROAD_HALF+8,PW,p): self._cono(f,x,IC_Y-ROAD_HALF+2); self._cono(f,x,IC_Y+ROAD_HALF-2)

    @staticmethod
    def _cono(f,cx,cy):
        pts=np.array([[cx,cy-7],[cx-4,cy+5],[cx+4,cy+5]],np.int32)
        cv2.fillPoly(f,[pts],C_CONO)
        cv2.line(f,(cx-2,cy),(cx+2,cy),(200,200,200),1)

    def _lineas_pare(self,f):
        o=2
        cv2.line(f,(IC_X-ROAD_HALF,IC_Y+ROAD_HALF+o),(IC_X+ROAD_HALF,IC_Y+ROAD_HALF+o),C_STOP,2)
        cv2.line(f,(IC_X-ROAD_HALF,IC_Y-ROAD_HALF-o),(IC_X+ROAD_HALF,IC_Y-ROAD_HALF-o),C_STOP,2)
        cv2.line(f,(IC_X-ROAD_HALF-o,IC_Y-ROAD_HALF),(IC_X-ROAD_HALF-o,IC_Y+ROAD_HALF),C_STOP,2)
        cv2.line(f,(IC_X+ROAD_HALF+o,IC_Y-ROAD_HALF),(IC_X+ROAD_HALF+o,IC_Y+ROAD_HALF),C_STOP,2)

    def _cardinales(self,f):
        s=(cv2.FONT_HERSHEY_SIMPLEX,0.45,(70,70,70),1)
        cv2.putText(f,"N",(IC_X-6,H_HEAD+20),*s); cv2.putText(f,"S",(IC_X-5,H_HEAD+H_ROAD-6),*s)
        cv2.putText(f,"O",(5,IC_Y+6),*s);          cv2.putText(f,"E",(PW-17,IC_Y+6),*s)

    def _semaforos(self,f,res):
        rh=ROAD_HALF+SEMAF_OFF
        for px,py,c in [(IC_X-rh,IC_Y-rh,"norte"),(IC_X+rh,IC_Y-rh,"norte"),
                        (IC_X-rh,IC_Y+rh,"sur"),  (IC_X+rh,IC_Y+rh,"sur")]:
            cv2.line(f,(px,py),(IC_X,IC_Y),C_CABLE,1)
            hx=int(px+(IC_X-px)*0.55); hy=int(py+(IC_Y-py)*0.55)
            col=self.ciclo.color(c)
            cv2.rectangle(f,(hx-7,hy-9),(hx+7,hy+9),(20,20,20),-1)
            cv2.rectangle(f,(hx-7,hy-9),(hx+7,hy+9),(55,55,55),1)
            cv2.circle(f,(hx,hy),5,col,-1); cv2.circle(f,(hx,hy),5,(0,0,0),1)
        for px,py in [(IC_X-rh,IC_Y-rh),(IC_X+rh,IC_Y-rh)]:
            hx=int(px+(IC_X-px)*0.50); col=self.ciclo.color("centro")
            cv2.circle(f,(hx,IC_Y),5,col,-1); cv2.circle(f,(hx,IC_Y),5,(0,0,0),1)
        cxl,cyl=IC_X-rh,IC_Y-rh
        if self.ciclo.subfase=="verde":
            cv2.putText(f,f"{self.ciclo.t_rest:.0f}s",(cxl-8,cyl-12),cv2.FONT_HERSHEY_SIMPLEX,0.38,C_VERDE,1)
        elif self.ciclo.subfase=="amarillo":
            cv2.putText(f,"AMR",(cxl-8,cyl-12),cv2.FONT_HERSHEY_SIMPLEX,0.34,C_AMARI,1)
        cv2.putText(f,f"FASE: {self.ciclo.activo.upper()}",(IC_X-28,H_HEAD+16),
                    cv2.FONT_HERSHEY_SIMPLEX,0.33,(120,120,120),1)

    def _footer(self,f,res,dec):
        y0=H_HEAD+H_ROAD
        cv2.rectangle(f,(0,y0),(PW,PH),(12,12,12),-1)
        cv2.line(f,(0,y0),(PW,y0),(42,42,42),1)
        cnv={"libre":C_VERDE,"moderado":C_AMARI,"congestionado":C_ROJO}.get(res.nivel_congestion,(150,150,150))
        cv2.putText(f,f"Vehiculos: {res.total_vehiculos}   Congestion: {res.nivel_congestion.upper()}",
                    (8,y0+20),cv2.FONT_HERSHEY_SIMPLEX,0.45,cnv,1)
        cv2.putText(f,f"Ciclo: {dec.ciclo_total_s:.0f}s   Prior: {dec.fase_prioritaria}   [{dec.motivo_prioridad}]",
                    (8,y0+38),cv2.FONT_HERSHEY_SIMPLEX,0.37,(145,145,145),1)
        ts="   ".join(f"{c[0].upper()}:{t:.0f}s" for c,t in dec.tiempos_verde_s.items())
        cv2.putText(f,f"t_verde -> {ts}",(8,y0+56),cv2.FONT_HERSHEY_SIMPLEX,0.38,(100,200,255),1)
        cv2.putText(f,f"YOLOv5: {res.latencia_ms:.0f}ms   ITO Semaforos Inteligentes — Act.11",
                    (8,y0+72),cv2.FONT_HERSHEY_SIMPLEX,0.31,(52,52,52),1)