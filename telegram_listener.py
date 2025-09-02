#!/usr/bin/env python3
import asyncio
import hashlib
import logging
import os
import re
import sqlite3
from datetime import datetime
from typing import List, Optional, Tuple

from dotenv import load_dotenv
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
from geopy.geocoders import Nominatim
from telethon import TelegramClient, events
from telethon.errors import FloodWaitError, RPCError
from telethon.tl.types import Message

from config import CHANNELS, MONITOR_GROUP, MEDIA_DIR, DB_PATH, LOG_DIR

load_dotenv()
API_ID = int(os.getenv("API_ID"))
API_HASH = os.getenv("API_HASH")
SESSION_NAME = "intel_listener"

os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "listener.log")),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("TelegramListener")

try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
except Exception:
    nlp = None
    logger.warning("spaCy no disponible; NER deshabilitado")

geolocator = Nominatim(user_agent="intel_map_pro_v3")

conn = sqlite3.connect(DB_PATH, check_same_thread=False)
conn.execute("PRAGMA journal_mode=WAL")
c = conn.cursor()

c.execute(
    """
    CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY,
        text TEXT NOT NULL,
        media_paths TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        source_channel TEXT,
        telegram_msg_id INTEGER,
        content_hash TEXT UNIQUE
    )
"""
)
c.execute(
    """
    CREATE TABLE IF NOT EXISTS locations (
        id INTEGER PRIMARY KEY,
        message_id INTEGER,
        lat REAL,
        lon REAL,
        location_name TEXT,
        confidence REAL,
        FOREIGN KEY(message_id) REFERENCES messages(id)
    )
"""
)
conn.commit()

client = TelegramClient(SESSION_NAME, API_ID, API_HASH)

# -------------------- Geo helpers (anti-PERSON + country bias) --------------------
COUNTRY_HINTS = {
    "bulgaria": "bg", "españa": "es", "spain": "es", "france": "fr", "francia": "fr",
    "russia": "ru", "rusia": "ru", "ukraine": "ua", "ucrania": "ua", "poland": "pl",
    "polonia": "pl", "romania": "ro", "germany": "de", "alemania": "de", "italy": "it",
    "italia": "it", "turkey": "tr", "turquía": "tr", "georgia": "ge", "armenia": "am"
}
STOP_PERSON_NAMES = {
    "ursula", "maria", "maría", "juan", "john", "jose", "josé", "peter", "anna",
    "michael", "mike", "andrew", "andrés", "carlos", "pedro", "sofia", "sofía",
    "vladimir", "volodymyr", "yulia", "sergey", "margarita", "emmanuel", "macron",
    "von der", "leyen"
}
ONE_TOKEN_PLACE_ALLOWLIST = {
    "paris", "kyiv", "kiev", "gaza", "suez", "mosul", "aleppo", "homs", "lviv",
    "odessa", "odesa", "kharkiv", "donetsk", "luhansk", "sofia", "plovdiv", "sopot",
    "varna", "burgas", "sumy", "mariupol", "baghdad", "bagdad"
}
COORD_REGEX = r"(-?\d{1,2}\.\d+)[°,\s]+(-?\d{1,3}\.\d+)"

def _clean_token(t: str) -> str:
    return re.sub(r"[^A-Za-zÀ-ÖØ-öø-ÿ\- ]", "", t).strip().lower()

def _extract_country_hint(text: str) -> Optional[str]:
    t = _clean_token(text)
    for key, iso2 in COUNTRY_HINTS.items():
        if key in t.split() or f" {key} " in f" {t} ":
            return iso2
    return None

def _acceptable_place_token(tok: str) -> bool:
    tok_clean = _clean_token(tok)
    if not tok_clean or len(tok_clean) < 2:
        return False
    if any(n in tok_clean for n in STOP_PERSON_NAMES):
        return False
    if " " not in tok_clean and tok_clean not in ONE_TOKEN_PLACE_ALLOWLIST:
        return False
    return True

async def process_locations(msg: Message) -> List[Tuple[float, float, str, float]]:
    results: List[Tuple[float, float, str, float]] = []
    text = (msg.message or "").strip()
    if not text:
        return results

    # 1) Coordenadas explícitas
    try:
        for m in re.finditer(COORD_REGEX, text):
            lat, lon = map(float, m.groups())
            if -90 <= lat <= 90 and -180 <= lon <= 180:
                results.append((lat, lon, f"coords:{lat:.5f},{lon:.5f}", 1.0))
    except Exception:
        pass

    country_bias = _extract_country_hint(text)

    # 2) Candidatos por NER (solo GPE/LOC)
    candidates: List[str] = []
    try:
        if nlp:
            doc = nlp(text)
            for ent in doc.ents:
                if ent.label_ in ("GPE", "LOC") and _acceptable_place_token(ent.text):
                    candidates.append(ent.text)
        else:
            for tok in re.findall(r"\b[A-Z][A-Za-zÀ-ÖØ-öø-ÿ\-]{2,}(?:\s+[A-Z][A-Za-zÀ-ÖØ-öø-ÿ\-]{2,})*\b", text):
                if _acceptable_place_token(tok):
                    candidates.append(tok)
    except Exception:
        pass

    # dedup preservando orden
    seen = set()
    cand_unique = []
    for c in candidates:
        k = _clean_token(c)
        if k not in seen:
            seen.add(k)
            cand_unique.append(c)

    # 3) Geocodificar con sesgo de país
    for place in cand_unique:
        try:
            kwargs = dict(timeout=5)
            if country_bias:
                kwargs["country_codes"] = country_bias
            loc = geolocator.geocode(place, **kwargs)
            if not loc:
                continue
            lat, lon = float(loc.latitude), float(loc.longitude)
            if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                continue
            conf = 0.90 if country_bias else 0.85
            if country_bias and getattr(loc, "address", ""):
                if any(cc in loc.address.lower() for cc in (country_bias,)):
                    conf += 0.03
            results.append((lat, lon, place, min(conf, 0.97)))
        except GeocoderTimedOut:
            logger.warning(f"Timeout geocodificando {place}")
        except Exception as e:
            logger.debug(f"No se pudo geocodificar '{place}': {e}")

    return results

def _dedup_nearby(items: List[Tuple[float, float, str, float]], eps_m=100.0):
    out: List[Tuple[float, float, str, float]] = []
    for a in items:
        lat_a, lon_a = a[0], a[1]
        if any(((abs(lat_a - b[0]) * 111_000) ** 2 + (abs(lon_a - b[1]) * 85_000) ** 2) ** 0.5 < eps_m for b in out):
            continue
        out.append(a)
    return out

# -------------------- Media helpers (guardar con extensión) --------------------
from mimetypes import guess_extension

async def save_media_with_ext(msg: Message, media_dir: str) -> List[str]:
    os.makedirs(media_dir, exist_ok=True)
    paths: List[str] = []
    try:
        if msg.grouped_id:
            return paths  # álbum se maneja aparte
        data = await client.download_media(msg, file=bytes)
        if not data:
            return paths

        ext = None
        if getattr(msg, "document", None) and getattr(msg.document, "mime_type", None):
            ext = guess_extension(msg.document.mime_type) or None
        if not ext and getattr(msg, "file", None) and getattr(msg.file, "ext", None):
            ext = msg.file.ext
        if not ext and getattr(msg, "photo", None):
            ext = ".jpg"
        if not ext:
            ext = ".bin"

        fname = f"{msg.id}{ext}"
        path = os.path.join(media_dir, fname)
        with open(path, "wb") as f:
            f.write(data)
        paths.append(path)
    except FloodWaitError as e:
        logger.warning(f"FloodWait {e.seconds}s al guardar media")
        await asyncio.sleep(e.seconds + 1)
        return await save_media_with_ext(msg, media_dir)
    except Exception as e:
        logger.error(f"Error guardando media: {e}")
    return paths

async def save_album_with_ext(msgs: List[Message], media_dir: str) -> List[str]:
    os.makedirs(media_dir, exist_ok=True)
    out: List[str] = []
    for m in msgs:
        out += await save_media_with_ext(m, media_dir)
    return out

# -------------------- Forward / Copy --------------------
async def forward_message(msg: Message) -> None:
    await client.forward_messages(MONITOR_GROUP, msg)

async def forward_messages_bulk(msgs: List[Message]) -> None:
    await client.forward_messages(MONITOR_GROUP, msgs)

async def copy_text(msg: Message) -> bool:
    content = (msg.message or "").strip()
    if not content:
        return False
    await client.send_message(MONITOR_GROUP, content)
    return True

async def copy_media(msg: Message) -> bool:
    if not (msg.media or msg.photo or msg.document or msg.video or msg.audio or msg.voice):
        return False
    caption = (msg.message or "").strip() or None
    try:
        data = await client.download_media(msg, file=bytes)
        if not data:
            return False
        await client.send_file(MONITOR_GROUP, data, caption=caption)
        return True
    except FloodWaitError as e:
        logger.warning(f"FloodWait {e.seconds}s al copiar media")
        await asyncio.sleep(e.seconds + 1)
        return await copy_media(msg)
    except RPCError as e:
        logger.warning(f"RPCError copiando media: {e}")
        return False
    except Exception as e:
        logger.error(f"Error copiando media: {e}")
        return False

async def copy_album(msgs: List[Message]) -> bool:
    files: List[bytes] = []
    captions: List[str] = []
    try:
        for m in msgs:
            data = await client.download_media(m, file=bytes)
            if data:
                files.append(data)
            cap = (m.message or "").strip()
            if cap:
                captions.append(cap)
        if not files:
            return False
        merged_caption = "\n\n".join(captions) if captions else None
        await client.send_file(MONITOR_GROUP, files, caption=merged_caption, album=True)
        return True
    except FloodWaitError as e:
        logger.warning(f"FloodWait {e.seconds}s copiando álbum")
        await asyncio.sleep(e.seconds + 1)
        return await copy_album(msgs)
    except Exception as e:
        logger.error(f"Error copiando álbum: {e}")
        return False

async def forward_or_copy_single(msg: Message) -> None:
    try:
        await forward_message(msg)
        logger.info(f"Reenviado -> MONITOR_GROUP {MONITOR_GROUP}")
        return
    except RPCError as e:
        logger.warning(f"Forward bloqueado ({e.__class__.__name__}), copiando…")
    except Exception as e:
        logger.warning(f"Fallo forward: {e}; copiando…")

    if await copy_media(msg):
        logger.info("Media copiada a MONITOR_GROUP")
        return
    if await copy_text(msg):
        logger.info("Texto copiado a MONITOR_GROUP")
        return
    logger.info("Nada que copiar (sin texto ni media)")

async def forward_or_copy_album(msgs: List[Message]) -> None:
    try:
        await forward_messages_bulk(msgs)
        logger.info(f"Álbum reenviado -> MONITOR_GROUP {MONITOR_GROUP}")
        return
    except RPCError as e:
        logger.warning(f"Forward de álbum bloqueado ({e.__class__.__name__}), copiando…")
    except Exception as e:
        logger.warning(f"Fallo forward de álbum: {e}; copiando…")

    if await copy_album(msgs):
        logger.info("Álbum copiado a MONITOR_GROUP")
        return
    for m in msgs:
        await forward_or_copy_single(m)

# -------------------- Hash de contenido --------------------
def create_content_hash(text: str, media_paths: List[str]) -> str:
    h = hashlib.sha256()
    h.update((text or "").encode("utf-8"))
    for p in media_paths:
        h.update(os.path.basename(p).encode("utf-8"))
    return h.hexdigest()

# -------------------- Handlers --------------------
@client.on(events.NewMessage(chats=CHANNELS))
async def on_new_message(event: events.NewMessage.Event) -> None:
    try:
        start = datetime.now()
        msg = event.message

        media_paths = await save_media_with_ext(msg, MEDIA_DIR)
        content_hash = create_content_hash(msg.message or "", media_paths)

        c.execute("SELECT id FROM messages WHERE content_hash = ?", (content_hash,))
        if c.fetchone():
            logger.info("Duplicado: omitido")
            return

        c.execute(
            """
            INSERT INTO messages (text, media_paths, source_channel, telegram_msg_id, content_hash)
            VALUES (?, ?, ?, ?, ?)
            """,
            (msg.message, ",".join(media_paths), str(event.chat_id), msg.id, content_hash),
        )
        row_id = c.lastrowid

        locs = await process_locations(msg)
        locs = _dedup_nearby(locs)
        for lat, lon, name, conf in locs:
            c.execute(
                """
                INSERT INTO locations (message_id, lat, lon, location_name, confidence)
                VALUES (?, ?, ?, ?, ?)
                """,
                (row_id, lat, lon, name, conf),
            )

        conn.commit()
        await forward_or_copy_single(msg)

        elapsed = (datetime.now() - start).total_seconds()
        logger.info(f"Procesado OK | {elapsed:.2f}s")

    except FloodWaitError as e:
        logger.warning(f"FloodWait {e.seconds}s en on_new_message")
        await asyncio.sleep(e.seconds + 1)
    except Exception as e:
        logger.error(f"on_new_message error: {e}")
        conn.rollback()

@client.on(events.Album(chats=CHANNELS))
async def on_new_album(event: events.Album.Event) -> None:
    try:
        msgs = event.messages
        head = msgs[0]

        media_paths_all = await save_album_with_ext(msgs, MEDIA_DIR)
        content_hash = create_content_hash(
            " || ".join((m.message or "") for m in msgs),
            media_paths_all,
        )

        c.execute("SELECT id FROM messages WHERE content_hash = ?", (content_hash,))
        if c.fetchone():
            logger.info("Álbum duplicado: omitido")
            return

        c.execute(
            """
            INSERT INTO messages (text, media_paths, source_channel, telegram_msg_id, content_hash)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                " ⧉ ".join((m.message or "") for m in msgs if (m.message or "").strip()),
                ",".join(media_paths_all),
                str(event.chat_id),
                head.id,
                content_hash,
            ),
        )
        row_id = c.lastrowid

        all_locs: List[Tuple[float, float, str, float]] = []
        for m in msgs:
            all_locs += await process_locations(m)
        all_locs = _dedup_nearby(all_locs)

        for lat, lon, name, conf in all_locs:
            c.execute(
                """
                INSERT INTO locations (message_id, lat, lon, location_name, confidence)
                VALUES (?, ?, ?, ?, ?)
                """,
                (row_id, lat, lon, name, conf),
            )

        conn.commit()
        await forward_or_copy_album(msgs)
        logger.info("Álbum procesado OK")

    except FloodWaitError as e:
        logger.warning(f"FloodWait {e.seconds}s en on_new_album")
        await asyncio.sleep(e.seconds + 1)
    except Exception as e:
        logger.error(f"on_new_album error: {e}")
        conn.rollback()

# -------------------- Main --------------------
async def main() -> None:
    try:
        await client.start()
        logger.info("Conectado a Telegram")
        await client.run_until_disconnected()
    finally:
        await client.disconnect()
        conn.close()
        logger.info("Conexión cerrada")

if __name__ == "__main__":
    asyncio.run(main())
