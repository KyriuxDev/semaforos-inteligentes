# semaforos-inteligentes

**TECNOLÓGICO NACIONAL DE MÉXICO — INSTITUTO TECNOLÓGICO DE OAXACA**
Ingeniería en Sistemas Computacionales | Taller de Investigación II

> Sistema de semáforos inteligentes basado en detección vehicular para optimizar el flujo de tráfico en Oaxaca de Juárez, Oaxaca.

---

## Arquitectura del proyecto

```
semaforos-inteligentes/
│
├── main.py                        # Punto de entrada CLI
├── requirements.txt
│
├── config/
│   ├── __init__.py
│   └── settings.py                # ConfigPipeline, ConfigDetector, constantes
│
├── models/
│   ├── __init__.py
│   └── schemas.py                 # ResultadoDeteccion, DecisionSemaforica
│
├── core/
│   ├── pipeline/
│   │   └── preprocesamiento.py    # Actividad 8 — pipeline OpenCV (7 etapas)
│   ├── detector/
│   │   └── vehicular.py           # Actividad 9 — DetectorVehicular YOLOv5
│   └── decision/
│       └── motor.py               # Actividad 10 — MotorDecision (interfaz lista)
│
├── utils/
│   ├── logger.py                  # Logger centralizado
│   └── video.py                   # Generador de video sintético de prueba
│
├── tests/
│   └── (pruebas unitarias — Actividad 11)
│
└── data/                          # Videos de entrada y salida (no versionar)
    └── .gitkeep
```

### Flujo de datos entre capas

```
Cámara IP / video.mp4
        │
        ▼
core/pipeline/preprocesamiento.py   ← Actividad 8
        │  frame BGR → tensor RGB 640×640 normalizado
        ▼
core/detector/vehicular.py          ← Actividad 9  (actual)
        │  tensor → ResultadoDeteccion
        │  · conteo por tipo y carril
        │  · nivel de congestión
        │  · tiempo de verde recomendado/carril
        ▼
core/decision/motor.py              ← Actividad 10 (próxima)
        │  ResultadoDeteccion → DecisionSemaforica
        ▼
Controlador semafórico
```

---

## Instalación

```bash
# 1. Crear y activar entorno virtual
python3 -m venv venv
source venv/bin/activate          # Linux/macOS
venv\Scripts\activate             # Windows

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. PyTorch CPU (si no hay GPU NVIDIA)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 4. Verificar instalación
python -c "import torch, ultralytics, cv2; print('OK')"
```

---

## Uso

```bash
# Sobre un video existente
python main.py ruta/al/video.mp4

# Modo headless (sin ventana, solo log)
python main.py --noventana ruta/al/video.mp4
```

**Controles en ventana:** `q` salir · `p` pausar/reanudar · `s` captura PNG

---

## Parámetros operativos YOLOv5 (Actividad 9)

| Parámetro | Valor | Justificación |
|---|---|---|
| Modelo | `yolov5s` | Balance velocidad/precisión; 140 fps GPU (ScienceDirect 2024) |
| Confianza mínima | `0.45` | Minimiza falsos negativos en tráfico urbano |
| IoU NMS | `0.45` | Evita duplicados en vehículos agrupados |
| Tamaño imagen | `640 px` | Estándar YOLOv5 (Ultralytics 2024) |
| t_base | `4 s/veh` | Tiempo de verde base por vehículo (Blogs ETSII 2025) |
| t_mín | `12 s` | Tiempo mínimo garantizado (equidad) |
| t_máx | `60 s` | Tiempo máximo (evitar esperas largas) |

---

## Referencias

- San Miguel, S. (2024). *Sistema de coordinación de semáforos inteligentes con algoritmos de inteligencia artificial.* Revista ConCiencia Joven, 2, 32-38.
- Ultralytics (2024). *Comprehensive Guide to Ultralytics YOLOv5.* https://docs.ultralytics.com/yolov5/
- ScienceDirect (2024). *YOLOv5 — an overview.*
- Blogs ETSII URJC (2025). *Sistema de semáforos inteligente.*

---

## Equipo

| Nombre | Rol |
|---|---|
| Delgado Molina Karla Rocío | Desarrollo |
| Martínez Martínez Jesús Alexander | Desarrollo |
| Zarate Matus Ángel Adrián | Desarrollo |

**Catedrática:** Pérez López Otilia
