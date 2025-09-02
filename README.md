# IntelLive Pro 🌍

by Br0kenBulb https://github.com/BrokenBulb0

**IntelLive Pro** es un sistema de **monitoreo geoestratégico en tiempo real** con integración de **Telegram** y visualización 3D interactiva con **PyDeck + Streamlit**.  
Permite recopilar, procesar y visualizar reportes multimedia geolocalizados desde múltiples canales de Telegram.

---

## Características Principales
- **Monitoreo en tiempo real** de fuentes de Telegram
- **Geolocalización automática** de reportes (NER + reglas + embeddings)
- **Visualización 3D interactiva** con PyDeck/Mapbox
- Gestión integrada de **imágenes, videos y audio**
- **Búsqueda avanzada** y filtrado temporal
- **Scroll inteligente** a reportes seleccionados
- IA local opcional (spaCy + sentence-transformers) para desambiguación

---

## Requisitos
- **Python 3.10+**
- **Cuenta de Telegram** con API ID / API HASH  
  [Crear app en my.telegram.org](https://my.telegram.org/apps)
- **Mapbox API Key** (opcional, para estilos de mapa mejorados)
- Dependencias Python (se instalan con `requirements.txt`)

---

## Instalación

```bash
# 1. Clonar el repositorio
git clone https://github.com/BrokenBulb0/IntelLive-Pro.git
cd IntelLive-Pro

# 2. Crear y activar entorno virtual
python3 -m venv venv
source venv/bin/activate   # en Linux/macOS
venv\Scripts\activate      # en Windows

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Descargar modelo de spaCy
python -m spacy download en_core_web_sm
# (opcional, mejor precisión)
python -m spacy download en_core_web_trf
