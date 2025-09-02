from typing import List, Tuple, Union
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Configuración del mapa
MAP_CENTER: Tuple[float, float] = (48.3794, 31.1656)
DEFAULT_ZOOM: int = 5
MAX_ENTRIES: int = 500

# Canales de Telegram a monitorear
CHANNELS: List[Union[str, int]] = [
    "Slavyangrad",
    "medmannews",
    "MiddleEastSpectator",
    "infodefENGLAND",
    "rnintel",
    "GeoPWatch",
    "intelslava",
    "wartranslated",
    "Suriyakmaps",
    "Eurekapress",
    "European_dissident"
]

# Grupo de monitorización
MONITOR_GROUP =-1003090741091

# Configuración de directorios
MEDIA_DIR = os.path.join(BASE_DIR, 'media')
DB_PATH = os.path.join(BASE_DIR, 'intel_data.db')
LOG_DIR = os.path.join(BASE_DIR, 'logs')

# Crear directorios necesarios
os.makedirs(MEDIA_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)