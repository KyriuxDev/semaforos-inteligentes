"""
core/visualizacion/video_comparativo.py
Genera dos videos animados de la intersección Four Corners + comparativa.

  1. data/semaforo_convencional.mp4  — tiempo fijo (30s/fase)
  2. data/semaforo_inteligente.mp4   — MotorDecision ITO adaptativo
  3. data/comparativa_final.mp4      — lado a lado

Mismo patrón de tráfico en ambos (semilla fija) → comparativa justa.

Uso:
    python main.py --comparativo
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import cv2
import numpy as np

from config.settings import ConfigMotor
from core.decision.motor import MotorDecision
from models.schemas import DecisionSemaforica, ResultadoDeteccion
from utils.logger import get_logger

logger = get_logger("core.visualizacion.video_comparativo")

# ===========================================================================
# Parámetros de video
# ===========================================================================

FPS        = 30
DURACION_S = 120
N_FRAMES   = FPS * DURACION_S

PW = 640; PH = 480
H_HEAD  = 50; H_FOOT = 90
H_SCENE = PH - H_HEAD - H_FOOT   # 340 px

IC_X = PW // 2
IC_Y = H_HEAD + H_SCENE // 2     # 220

LANE_W    = 22
DIR_LANES = 2
ROAD_HALF = DIR_LANES * 2 * LANE_W // 2   # 44 px

VEH_SPEED = 75.0
SEMILLA   = 42
CARRILES  = ["norte","centro","sur"]

C_BG=(18,20,18); C_ASPH=(48,50,48); C_INTER=(58,60,58)
C_LANE=(80,80,80); C_CENTER=(0,170,170); C_STOP=(210,210,210)
C_CONO=(0,110,255); C_CEBRA=(200,200,200)
C_BUILD=(35,33,42); C_BUILD_E=(60,58,70); C_CABLE=(100,100,100)
C_VERDE=(0,210,60); C_AMARI=(0,210,210); C_ROJO=(30,30,210)

_VEH_COLS=[
    (30,144,255),(0,200,100),(0,165,255),
    (200,180,0),(160,80,220),(240,240,240),
    (50,50,180),(180,220,0),
]


# ===========================================================================
# Patrón de tráfico
# ===========================================================================

@dataclass
class PatronTrafico:
    rng: random.Random = field(default_factory=lambda: random.Random(SEMILLA))

    def conteo_en(self, t: float) -> Dict[str,int]:
        if   t < 20: base={"norte":2,"centro":3,"sur":2}
        elif t < 50: base={"norte":8,"centro":6,"sur":4}
        elif t < 80: base={"norte":5,"centro":4,"sur":3}
        else:        base={"norte":2,"centro":2,"sur":1}
        return {k: max(0, v+self.rng.randint(-1,1)) for k,v in base.items()}

    def hay_autobus(self, t: float) -> bool:
        return 50 <= t <= 80 and int(t)%15==0


# ===========================================================================
# Vehículo animado — bug-fixed
# ===========================================================================

@dataclass
class Vehiculo:
    x:float; y:float; dx:float; dy:float
    color:tuple; stop_val:float; axis:str
    vw:int=18; vh:int=11

    def _paso_linea(self) -> bool:
        if self.axis=='y':
            f=self.y+self.vh/2 if self.dy>0 else self.y-self.vh/2
            return f>self.stop_val if self.dy>0 else f<self.stop_val
        else:
            f=self.x+self.vw/2 if self.dx>0 else self.x-self.vw/2
            return f>self.stop_val if self.dx>0 else f<self.stop_val

    def actualizar(self, dt:float, verde:bool, stop_ef:float) -> None:
        if self._paso_linea():
            self.x+=self.dx*dt; self.y+=self.dy*dt; return
        if self.axis=='y':
            frente=self.y+self.vh/2 if self.dy>0 else self.y-self.vh/2
            dist=(stop_ef-frente) if self.dy>0 else (frente-stop_ef)
            if dist>0.5:
                spd=VEH_SPEED if verde else min(VEH_SPEED,max(4.0,dist*2.5))
                self.y+=(1 if self.dy>0 else -1)*min(spd*dt,dist)
        else:
            frente=self.x+self.vw/2 if self.dx>0 else self.x-self.vw/2
            dist=(stop_ef-frente) if self.dx>0 else (frente-stop_ef)
            if dist>0.5:
                spd=VEH_SPEED if verde else min(VEH_SPEED,max(4.0,dist*2.5))
                self.x+=(1 if self.dx>0 else -1)*min(spd*dt,dist)

    def fuera(self) -> bool:
        return self.x<-60 or self.x>PW+60 or self.y<H_HEAD-60 or self.y>H_HEAD+H_SCENE+60

    def draw(self, f:np.ndarray) -> None:
        x1=int(self.x)-self.vw//2; y1=int(self.y)-self.vh//2
        x2,y2=x1+self.vw,y1+self.vh
        cv2.rectangle(f,(x1,y1),(x2,y2),self.color,-1)
        cv2.rectangle(f,(x1,y1),(x2,y2),(0,0,0),1)
        b=tuple(min(c+70,255) for c in self.color)
        if self.dy<0 or (self.dy==0 and self.dx>0): cv2.rectangle(f,(x2-5,y1+2),(x2-1,y2-2),b,-1)
        else: cv2.rectangle(f,(x1+1,y1+2),(x1+5,y2-2),b,-1)


# ===========================================================================
# Enfoque animado — car-following
# ===========================================================================

class Enfoque:
    GAP=5

    def __init__(self, direction:str, rng:random.Random) -> None:
        self.direction=direction; self.vehiculos:List[Vehiculo]=[]
        self._timer=rng.uniform(0,2.5); self._rng=rng
        rh=ROAD_HALF
        if direction=="norte":
            self._x0,self._x1=IC_X-rh,IC_X; self._sy=float(H_HEAD+4)
            self._dx,self._dy=0.0,VEH_SPEED; self._stop=float(IC_Y-rh-5)
            self._axis='y'; self._vw,self._vh=11,18
        elif direction=="sur":
            self._x0,self._x1=IC_X,IC_X+rh; self._sy=float(H_HEAD+H_SCENE-4)
            self._dx,self._dy=0.0,-VEH_SPEED; self._stop=float(IC_Y+rh+5)
            self._axis='y'; self._vw,self._vh=11,18
        elif direction=="este":
            self._y0,self._y1=IC_Y-rh,IC_Y; self._sx=float(PW-4)
            self._dx,self._dy=-VEH_SPEED,0.0; self._stop=float(IC_X+rh+5)
            self._axis='x'; self._vw,self._vh=18,11
        elif direction=="oeste":
            self._y0,self._y1=IC_Y,IC_Y+rh; self._sx=float(4)
            self._dx,self._dy=VEH_SPEED,0.0; self._stop=float(IC_X-rh-5)
            self._axis='x'; self._vw,self._vh=18,11

    def _ordenar(self):
        if self.direction=="norte":   self.vehiculos.sort(key=lambda v:-v.y)
        elif self.direction=="sur":   self.vehiculos.sort(key=lambda v: v.y)
        elif self.direction=="este":  self.vehiculos.sort(key=lambda v: v.x)
        elif self.direction=="oeste": self.vehiculos.sort(key=lambda v:-v.x)

    def _stop_ef(self, idx:int, verde:bool) -> float:
        base=(1e6 if (self._dy>0 or self._dx>0) else -1e6) if verde else self._stop
        if idx==0: return base
        prev=self.vehiculos[idx-1]
        if self.direction=="norte":  return min(base, prev.y-prev.vh/2-self.GAP)
        elif self.direction=="sur":  return max(base, prev.y+prev.vh/2+self.GAP)
        elif self.direction=="este": return max(base, prev.x+prev.vw/2+self.GAP)
        elif self.direction=="oeste":return min(base, prev.x-prev.vw/2-self.GAP)
        return base

    def actualizar(self, dt:float, n_obj:int, verde:bool) -> None:
        self._ordenar()
        for i,v in enumerate(self.vehiculos): v.actualizar(dt,verde,self._stop_ef(i,verde))
        self.vehiculos=[v for v in self.vehiculos if not v.fuera()]
        self._timer+=dt
        if len(self.vehiculos)<min(n_obj,7) and self._timer>max(0.4,2.2-n_obj*0.13):
            if self._libre(): self._timer=0.0; self._spawnear()

    def _libre(self):
        sep=30
        if self.direction=="norte":  return all(v.y>H_HEAD+sep for v in self.vehiculos)
        elif self.direction=="sur":  return all(v.y<H_HEAD+H_SCENE-sep for v in self.vehiculos)
        elif self.direction=="este": return all(v.x<PW-sep for v in self.vehiculos)
        elif self.direction=="oeste":return all(v.x>sep for v in self.vehiculos)
        return True

    def _spawnear(self):
        color=self._rng.choice(_VEH_COLS)
        if self.direction in ("norte","sur"):
            x=self._x0+(self._rng.randint(0,DIR_LANES-1)+0.5)*LANE_W; y=self._sy
        else:
            x=self._sx; y=self._y0+(self._rng.randint(0,DIR_LANES-1)+0.5)*LANE_W
        self.vehiculos.append(Vehiculo(x=x,y=y,dx=self._dx,dy=self._dy,color=color,
            stop_val=self._stop,axis=self._axis,vw=self._vw,vh=self._vh))

    def n_cola(self):
        if self.direction=="norte":  return sum(1 for v in self.vehiculos if v.y+v.vh/2<IC_Y-ROAD_HALF)
        elif self.direction=="sur":  return sum(1 for v in self.vehiculos if v.y-v.vh/2>IC_Y+ROAD_HALF)
        elif self.direction=="este": return sum(1 for v in self.vehiculos if v.x-v.vw/2>IC_X+ROAD_HALF)
        elif self.direction=="oeste":return sum(1 for v in self.vehiculos if v.x+v.vw/2<IC_X-ROAD_HALF)
        return 0

    def draw(self,f): 
        for v in self.vehiculos: v.draw(f)


# ===========================================================================
# Ciclos semafóricos
# ===========================================================================

class CicloFijo:
    T_VERDE=30.0; T_AM=3.0; T_TR=2.0
    def __init__(self): self._idx=0; self._sub="verde"; self._t=self.T_VERDE
    def avanzar(self,dt):
        self._t-=dt
        if self._t>0: return
        if self._sub=="verde":      self._sub="amarillo"; self._t=self.T_AM
        elif self._sub=="amarillo": self._sub="todo_rojo"; self._t=self.T_TR
        elif self._sub=="todo_rojo": self._idx=(self._idx+1)%3; self._sub="verde"; self._t=self.T_VERDE
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
    def ciclo_total(self): return (self.T_VERDE+self.T_AM+self.T_TR)*3


class CicloInteligente:
    def __init__(self):
        self._motor=MotorDecision(ConfigMotor(
            tiempo_amarillo_s=3.0,tiempo_todo_rojo_s=2.0,
            ciclo_min_s=40.0,ciclo_max_s=150.0,bonus_autobus_s=5.0))
        self._idx=0; self._sub="verde"; self._t=30.0
        self._pend:Optional[DecisionSemaforica]=None; self._ultimo=90.0
    def nueva_decision(self,res):
        dec=self._motor.decidir(res); self._pend=dec; self._ultimo=dec.ciclo_total_s
    def avanzar(self,dt):
        self._t-=dt
        if self._t>0: return
        if self._sub=="verde":      self._sub="amarillo"; self._t=3.0
        elif self._sub=="amarillo": self._sub="todo_rojo"; self._t=2.0
        elif self._sub=="todo_rojo":
            self._idx=(self._idx+1)%3; self._sub="verde"
            t=30.0
            if self._pend: t=self._pend.tiempos_verde_s.get(CARRILES[self._idx],30.0); self._pend=None
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
    def ciclo_total(self): return self._ultimo


# ===========================================================================
# Renderer compartido
# ===========================================================================

def _cono(f,cx,cy):
    pts=np.array([[cx,cy-7],[cx-4,cy+5],[cx+4,cy+5]],np.int32)
    cv2.fillPoly(f,[pts],C_CONO)
    cv2.line(f,(cx-2,cy),(cx+2,cy),(200,200,200),1)

def render_frame(f, ciclo, enfoques, metricas, titulo, color_t):
    f[:]=C_BG
    # Header
    cv2.rectangle(f,(0,0),(PW,H_HEAD),(12,12,12),-1)
    cv2.putText(f,titulo,(8,22),cv2.FONT_HERSHEY_SIMPLEX,0.50,color_t,1)
    cv2.putText(f,f"Fase: {ciclo.activo.upper()}  {ciclo.subfase.upper()}  Ciclo: {ciclo.ciclo_total():.0f}s",
                (8,40),cv2.FONT_HERSHEY_SIMPLEX,0.34,(110,110,110),1)
    # Edificios
    mg=6
    for (x1,y1),(x2,y2) in [((0,H_HEAD),(IC_X-ROAD_HALF-mg,IC_Y-ROAD_HALF-mg)),
        ((IC_X+ROAD_HALF+mg,H_HEAD),(PW,IC_Y-ROAD_HALF-mg)),
        ((0,IC_Y+ROAD_HALF+mg),(IC_X-ROAD_HALF-mg,H_HEAD+H_SCENE)),
        ((IC_X+ROAD_HALF+mg,IC_Y+ROAD_HALF+mg),(PW,H_HEAD+H_SCENE))]:
        if x1>=x2 or y1>=y2: continue
        cv2.rectangle(f,(x1,y1),(x2,y2),C_BUILD,-1)
        cv2.rectangle(f,(x1,y1),(x2,y2),C_BUILD_E,1)
        for wx in range(x1+8,x2-6,18):
            for wy in range(y1+10,y2-6,16):
                if wx+8<x2 and wy+8<y2: cv2.rectangle(f,(wx,wy),(wx+8,wy+8),(45,40,55),-1)
    # Calles
    cv2.rectangle(f,(IC_X-ROAD_HALF,H_HEAD),(IC_X+ROAD_HALF,H_HEAD+H_SCENE),C_ASPH,-1)
    cv2.rectangle(f,(0,IC_Y-ROAD_HALF),(PW,IC_Y+ROAD_HALF),C_ASPH,-1)
    cv2.rectangle(f,(IC_X-ROAD_HALF,IC_Y-ROAD_HALF),(IC_X+ROAD_HALF,IC_Y+ROAD_HALF),C_INTER,-1)
    # Marcas
    for xo in [-LANE_W,LANE_W]:
        xc=IC_X+xo
        for y in range(H_HEAD,IC_Y-ROAD_HALF,14): cv2.line(f,(xc,y),(xc,min(y+7,IC_Y-ROAD_HALF)),C_LANE,1)
        for y in range(IC_Y+ROAD_HALF,H_HEAD+H_SCENE,14): cv2.line(f,(xc,y),(xc,min(y+7,H_HEAD+H_SCENE)),C_LANE,1)
    for yo in [-LANE_W,LANE_W]:
        yc=IC_Y+yo
        for x in range(0,IC_X-ROAD_HALF,14): cv2.line(f,(x,yc),(min(x+7,IC_X-ROAD_HALF),yc),C_LANE,1)
        for x in range(IC_X+ROAD_HALF,PW,14): cv2.line(f,(x,yc),(min(x+7,PW),yc),C_LANE,1)
    for y in range(H_HEAD,IC_Y-ROAD_HALF,12): cv2.line(f,(IC_X,y),(IC_X,min(y+6,IC_Y-ROAD_HALF)),C_CENTER,1)
    for y in range(IC_Y+ROAD_HALF,H_HEAD+H_SCENE,12): cv2.line(f,(IC_X,y),(IC_X,min(y+6,H_HEAD+H_SCENE)),C_CENTER,1)
    for x in range(0,IC_X-ROAD_HALF,12): cv2.line(f,(x,IC_Y),(min(x+6,IC_X-ROAD_HALF),IC_Y),C_CENTER,1)
    for x in range(IC_X+ROAD_HALF,PW,12): cv2.line(f,(x,IC_Y),(min(x+6,PW),IC_Y),C_CENTER,1)
    # Cebras
    sw,sh,g=6,8,5
    for x in range(IC_X-ROAD_HALF,IC_X+ROAD_HALF,sw+g):
        cv2.rectangle(f,(x,IC_Y+ROAD_HALF+3),(min(x+sw,IC_X+ROAD_HALF),IC_Y+ROAD_HALF+3+sh),C_CEBRA,-1)
        cv2.rectangle(f,(x,IC_Y-ROAD_HALF-3-sh),(min(x+sw,IC_X+ROAD_HALF),IC_Y-ROAD_HALF-3),C_CEBRA,-1)
    for y in range(IC_Y-ROAD_HALF,IC_Y+ROAD_HALF,sw+g):
        cv2.rectangle(f,(IC_X+ROAD_HALF+3,y),(IC_X+ROAD_HALF+3+sh,min(y+sw,IC_Y+ROAD_HALF)),C_CEBRA,-1)
        cv2.rectangle(f,(IC_X-ROAD_HALF-3-sh,y),(IC_X-ROAD_HALF-3,min(y+sw,IC_Y+ROAD_HALF)),C_CEBRA,-1)
    # Conos
    p=22
    for y in range(H_HEAD+8,IC_Y-ROAD_HALF,p): _cono(f,IC_X-ROAD_HALF+2,y); _cono(f,IC_X+ROAD_HALF-2,y)
    for y in range(IC_Y+ROAD_HALF+8,H_HEAD+H_SCENE,p): _cono(f,IC_X-ROAD_HALF+2,y); _cono(f,IC_X+ROAD_HALF-2,y)
    for x in range(8,IC_X-ROAD_HALF,p): _cono(f,x,IC_Y-ROAD_HALF+2); _cono(f,x,IC_Y+ROAD_HALF-2)
    for x in range(IC_X+ROAD_HALF+8,PW,p): _cono(f,x,IC_Y-ROAD_HALF+2); _cono(f,x,IC_Y+ROAD_HALF-2)
    # Líneas de pare
    o=2
    cv2.line(f,(IC_X-ROAD_HALF,IC_Y+ROAD_HALF+o),(IC_X+ROAD_HALF,IC_Y+ROAD_HALF+o),C_STOP,2)
    cv2.line(f,(IC_X-ROAD_HALF,IC_Y-ROAD_HALF-o),(IC_X+ROAD_HALF,IC_Y-ROAD_HALF-o),C_STOP,2)
    cv2.line(f,(IC_X-ROAD_HALF-o,IC_Y-ROAD_HALF),(IC_X-ROAD_HALF-o,IC_Y+ROAD_HALF),C_STOP,2)
    cv2.line(f,(IC_X+ROAD_HALF+o,IC_Y-ROAD_HALF),(IC_X+ROAD_HALF+o,IC_Y+ROAD_HALF),C_STOP,2)
    # Vehículos
    for e in enfoques.values(): e.draw(f)
    # Semáforos cables
    rh=ROAD_HALF+14
    for px,py,c in [(IC_X-rh,IC_Y-rh,"norte"),(IC_X+rh,IC_Y-rh,"norte"),
                    (IC_X-rh,IC_Y+rh,"sur"),  (IC_X+rh,IC_Y+rh,"sur")]:
        cv2.line(f,(px,py),(IC_X,IC_Y),C_CABLE,1)
        hx=int(px+(IC_X-px)*0.55); hy=int(py+(IC_Y-py)*0.55)
        col=ciclo.color(c)
        cv2.rectangle(f,(hx-7,hy-9),(hx+7,hy+9),(20,20,20),-1)
        cv2.rectangle(f,(hx-7,hy-9),(hx+7,hy+9),(55,55,55),1)
        cv2.circle(f,(hx,hy),5,col,-1); cv2.circle(f,(hx,hy),5,(0,0,0),1)
    for px,py in [(IC_X-rh,IC_Y-rh),(IC_X+rh,IC_Y-rh)]:
        hx=int(px+(IC_X-px)*0.50); col=ciclo.color("centro")
        cv2.circle(f,(hx,IC_Y),5,col,-1); cv2.circle(f,(hx,IC_Y),5,(0,0,0),1)
    cxl,cyl=IC_X-rh,IC_Y-rh
    if ciclo.subfase=="verde": cv2.putText(f,f"{ciclo.t_rest:.0f}s",(cxl-8,cyl-12),cv2.FONT_HERSHEY_SIMPLEX,0.38,C_VERDE,1)
    elif ciclo.subfase=="amarillo": cv2.putText(f,"AMR",(cxl-8,cyl-12),cv2.FONT_HERSHEY_SIMPLEX,0.34,C_AMARI,1)
    # Puntos cardinales
    s=(cv2.FONT_HERSHEY_SIMPLEX,0.45,(70,70,70),1)
    cv2.putText(f,"N",(IC_X-6,H_HEAD+20),*s); cv2.putText(f,"S",(IC_X-5,H_HEAD+H_SCENE-6),*s)
    cv2.putText(f,"O",(5,IC_Y+6),*s);          cv2.putText(f,"E",(PW-17,IC_Y+6),*s)
    # Footer métricas
    y0=H_HEAD+H_SCENE
    cv2.rectangle(f,(0,y0),(PW,PH),(12,12,12),-1)
    cv2.line(f,(0,y0),(PW,y0),(42,42,42),1)
    cola=sum(e.n_cola() for e in enfoques.values())
    cv2.putText(f,f"Cola: {cola} veh   Liberados: {metricas.get('liberados',0)}",
                (8,y0+20),cv2.FONT_HERSHEY_SIMPLEX,0.42,color_t,1)
    cv2.putText(f,f"Espera acum: {metricas.get('espera_acum',0):.0f}s   Ciclos: {metricas.get('ciclos',0)}",
                (8,y0+38),cv2.FONT_HERSHEY_SIMPLEX,0.37,(140,140,140),1)
    cv2.putText(f,"ITO — Sistema Semaforos Inteligentes | Actividad 11",
                (8,y0+56),cv2.FONT_HERSHEY_SIMPLEX,0.32,(55,55,55),1)
    prog=int((metricas.get("t",0)/DURACION_S)*PW)
    cv2.rectangle(f,(0,y0+66),(prog,y0+73),(50,100,50),-1)
    cv2.rectangle(f,(0,y0+66),(PW,y0+73),(40,40,40),1)


# ===========================================================================
# Simulación y escritura de video
# ===========================================================================

def simular_y_grabar(modo:str, ruta:str, patron:PatronTrafico) -> Dict:
    rng=random.Random(SEMILLA+(0 if modo=="fijo" else 1))
    enf:Dict[str,Enfoque]={d:Enfoque(d,rng) for d in ("norte","sur","este","oeste")}
    ciclo=CicloFijo() if modo=="fijo" else CicloInteligente()
    writer=cv2.VideoWriter(ruta,cv2.VideoWriter_fourcc(*"mp4v"),FPS,(PW,PH))
    frame=np.zeros((PH,PW,3),dtype=np.uint8)
    met={"espera_acum":0.0,"liberados":0,"ciclos":0,"t":0.0}
    cola_prev={d:0 for d in enf}
    dt=1.0/FPS
    titulo=("SEMAFORO CONVENCIONAL — Tiempo Fijo" if modo=="fijo"
            else "SEMAFORO INTELIGENTE — ITO Adaptativo")
    color_t=((80,80,200) if modo=="fijo" else (0,200,80))
    logger.info(f"Generando [{modo}]: {ruta}")
    for fi in range(N_FRAMES):
        t=fi*dt; met["t"]=t
        conteo=patron.conteo_en(t)
        if modo=="inteligente":
            raw=[]
            for c,n in conteo.items():
                for _ in range(n): raw.append({"clase":"automóvil","clase_id":2,"carril":c,"confianza":0.85,"bbox":(0,0,0,0),"centro":(0,0)})
            if patron.hay_autobus(t):
                raw.append({"clase":"autobús","clase_id":5,"carril":"norte","confianza":0.90,"bbox":(0,0,0,0),"centro":(0,0)})
                conteo["norte"]+=1
            total=sum(conteo.values())
            nivel="libre" if total<=2 else("moderado" if total<=6 else "congestionado")
            from config.settings import TIEMPO_BASE_POR_VEHICULO_S as tb,TIEMPO_VERDE_MINIMO_S as tmn,TIEMPO_VERDE_MAXIMO_S as tmx
            tv={c:max(tmn,min(tmx,n*tb)) for c,n in conteo.items()}
            res=ResultadoDeteccion(n_frame=fi,timestamp=t,total_vehiculos=total,
                conteo_por_tipo={"automóvil":total},conteo_por_carril=conteo,
                tiempo_verde_recomendado_s=tv,detecciones_raw=raw,latencia_ms=0.0,nivel_congestion=nivel)
            ciclo.nueva_decision(res)
        prev_fase=ciclo.activo; ciclo.avanzar(dt)
        if ciclo.activo!=prev_fase: met["ciclos"]+=1
        n_n=conteo.get("norte",0); n_c=conteo.get("centro",0); n_s=conteo.get("sur",0)
        enf["norte"].actualizar(dt,n_n,ciclo.verde("norte"))
        enf["sur"].actualizar(  dt,n_s,ciclo.verde("sur"))
        enf["este"].actualizar( dt,n_c//2,ciclo.verde("centro"))
        enf["oeste"].actualizar(dt,n_c//2,ciclo.verde("centro"))
        for d,e in enf.items():
            ca=e.n_cola()
            if ca<cola_prev[d]: met["liberados"]+=cola_prev[d]-ca
            met["espera_acum"]+=ca*dt; cola_prev[d]=ca
        render_frame(frame,ciclo,enf,met,titulo,color_t)
        writer.write(frame)
        if fi%(FPS*10)==0:
            logger.info(f"  [{modo}] {t:.0f}s — cola={sum(e.n_cola() for e in enf.values())} lib={met['liberados']}")
    writer.release()
    logger.info(f"[{modo}] guardado: {ruta}")
    return met


def generar_comparativo(
    ruta_fijo:  str="data/semaforo_convencional.mp4",
    ruta_intel: str="data/semaforo_inteligente.mp4",
    ruta_final: str="data/comparativa_final.mp4",
) -> None:
    import os; os.makedirs("data",exist_ok=True)
    t0=time.perf_counter()
    mf=simular_y_grabar("fijo",       ruta_fijo,  PatronTrafico(rng=random.Random(SEMILLA)))
    mi=simular_y_grabar("inteligente",ruta_intel, PatronTrafico(rng=random.Random(SEMILLA)))
    logger.info("Combinando en comparativa_final.mp4...")
    c1=cv2.VideoCapture(ruta_fijo); c2=cv2.VideoCapture(ruta_intel)
    out=cv2.VideoWriter(ruta_final,cv2.VideoWriter_fourcc(*"mp4v"),FPS,(PW*2,PH))
    while True:
        r1,f1=c1.read(); r2,f2=c2.read()
        if not r1 or not r2: break
        combo=np.hstack([f1,f2]); cv2.line(combo,(PW,0),(PW,PH),(200,200,200),2)
        out.write(combo)
    c1.release(); c2.release(); out.release()
    me=mf["espera_acum"]; ii=mi["espera_acum"]
    print(f"\n{'='*60}")
    print("COMPARATIVA — Convencional vs Inteligente ITO")
    print(f"{'='*60}")
    print(f"  {'Métrica':<26} {'Convencional':>12} {'Inteligente':>12} {'Mejora':>8}")
    print(f"  {'-'*56}")
    print(f"  {'Espera acum. (s)':<26} {me:>12.0f} {ii:>12.0f} {(1-ii/max(me,1))*100:>+7.1f}%")
    print(f"  {'Vehículos liberados':<26} {mf['liberados']:>12} {mi['liberados']:>12} {(mi['liberados']/max(mf['liberados'],1)-1)*100:>+7.1f}%")
    print(f"  {'Ciclos completados':<26} {mf['ciclos']:>12} {mi['ciclos']:>12}")
    print(f"{'='*60}")
    print(f"\nArchivos en data/:")
    print(f"  {ruta_fijo}\n  {ruta_intel}\n  {ruta_final}  ← comparativa lado a lado")
    print(f"Tiempo total: {time.perf_counter()-t0:.1f}s")


if __name__=="__main__":
    from utils.logger import configurar_logging
    configurar_logging(); generar_comparativo()