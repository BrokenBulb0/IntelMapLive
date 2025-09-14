from typing import List, Tuple, Union
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Configuración del mapa
MAP_CENTER: Tuple[float, float] = (0.00, 0.00)
DEFAULT_ZOOM: int = 1
MAX_ENTRIES: int = 500

# Canales de Telegram a monitorear
CHANNELS: List[Union[str, int]] = [
    "YOUR_FAVORITE_CHANNELS",
]

# Grupo de monitorización
MONITOR_GROUP =(number of your group)

# Configuración de directorios
MEDIA_DIR = os.path.join(BASE_DIR, 'media')
DB_PATH = os.path.join(BASE_DIR, 'intel_data.db')
LOG_DIR = os.path.join(BASE_DIR, 'logs')

# Crear directorios necesarios
os.makedirs(MEDIA_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
