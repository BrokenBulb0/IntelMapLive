#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, asyncio, hashlib, logging, math, os, re, sqlite3
from datetime import datetime, timedelta, timezone
from mimetypes import guess_extension
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
from geopy.geocoders import Nominatim
from telethon import TelegramClient, events
from telethon.errors import FloodWaitError, RPCError
from telethon.tl.custom.message import Message

from config import CHANNELS, MONITOR_GROUP, MEDIA_DIR, DB_PATH, LOG_DIR

# ===== ENV & LOG
load_dotenv()
API_ID = int(os.getenv("API_ID")); API_HASH = os.getenv("API_HASH")
SESSION_NAME = os.getenv("SESSION_NAME", "intel_listener")
os.makedirs(LOG_DIR, exist_ok=True); os.makedirs(MEDIA_DIR, exist_ok=True)
logging.basicConfig(level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.FileHandler(os.path.join(LOG_DIR, "listener.log")), logging.StreamHandler()])
logger = logging.getLogger("TelegramListener")

# ===== NLP opcional
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
except Exception:
    nlp = None
    logger.warning("spaCy no disponible; usando regex contextual")

# ===== Geocoder
geolocator = Nominatim(user_agent="intel_map_context_v3")

# ===== DB
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
conn.execute("PRAGMA journal_mode=WAL")
c = conn.cursor()

c.execute("""CREATE TABLE IF NOT EXISTS messages(
    id INTEGER PRIMARY KEY, text TEXT NOT NULL, media_paths TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    source_channel TEXT, telegram_msg_id INTEGER, content_hash TEXT)""")
c.execute("""CREATE UNIQUE INDEX IF NOT EXISTS idx_messages_src_msgid ON messages(source_channel, telegram_msg_id)""")
c.execute("""CREATE UNIQUE INDEX IF NOT EXISTS idx_messages_content_hash ON messages(content_hash)""")
c.execute("""CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp)""")

c.execute("""CREATE TABLE IF NOT EXISTS locations(
    id INTEGER PRIMARY KEY, message_id INTEGER, lat REAL, lon REAL,
    location_name TEXT, confidence REAL,
    is_area INTEGER DEFAULT 0, radius_km REAL,
    admin_kind TEXT, admin_code TEXT,  -- e.g. 'country', 'cn'
    FOREIGN KEY(message_id) REFERENCES messages(id))""")

c.execute("""CREATE TABLE IF NOT EXISTS geocache(
    q TEXT, country_codes TEXT, lat REAL, lon REAL, label TEXT,
    kind TEXT, bbox_area REAL, country_code TEXT,
    PRIMARY KEY (q, country_codes))""")

# idempotentes
for sql in (
    "ALTER TABLE locations ADD COLUMN is_area INTEGER DEFAULT 0",
    "ALTER TABLE locations ADD COLUMN radius_km REAL",
    "ALTER TABLE locations ADD COLUMN admin_kind TEXT",
    "ALTER TABLE locations ADD COLUMN admin_code TEXT",
    "ALTER TABLE geocache ADD COLUMN kind TEXT",
    "ALTER TABLE geocache ADD COLUMN bbox_area REAL",
    "ALTER TABLE geocache ADD COLUMN country_code TEXT",
    "CREATE TABLE IF NOT EXISTS channel_context(source_channel TEXT PRIMARY KEY,last_lat REAL,last_lon REAL,last_label TEXT,last_updated DATETIME DEFAULT CURRENT_TIMESTAMP)"
):
    try: c.execute(sql); conn.commit()
    except Exception: pass

# ===== Telegram
client = TelegramClient(SESSION_NAME, API_ID, API_HASH)

# ===== Reglas y helpers de contexto
ADMIN_HINTS = re.compile(
 r"\b(oblast|raion|governorate|province|prefecture|district|camp|front|checkpoint|border|bridge|airport|city of|"
 r"provincia|distrito|campamento|frente|puesto de control|frontera|puente|aeropuerto|ciudad de)\b", re.I)
PREP_NEAR = re.compile(
 r"\b(in|near|around|at|inside|outside|north of|south of|east of|west of|close to|"
 r"en|cerca de|al norte de|al sur de|al este de|al oeste de|junto a|próximo a)\b", re.I)
ACTION_HINTS = re.compile(
 r"\b(explosion|blast|strike|attack|airstrike|shelling|missile|drone|uav|raid|firefight|clashes|"
 r"protest|demonstration|riot|evacuate|land(ed|ing)|hit|destroyed|killed|wounded|"
 r"frente|ataque|bombardeo|misil|drone|dron|combate|enfrentamientos|protesta|manifestación|disturbios|evacuaci[oó]n|impact[oó]|"
 r"incursi[oó]n|ofensiva|defensiva|detonado|explosi[oó]n|fuego|tiroteo)\b", re.I)
TOPIC_HINTS = re.compile(
 r"\b(attention|focus|sanction(s)?|policy|talks|speech|address|support for|statement|"
 r"atenci[oó]n|sanciones?|pol[ií]tica|declaraci[oó]n|apoyo a|discurso|cumbre|summit|visit|visita)\b", re.I)

CONTEXT_RULES = [
    (re.compile(r"\bsahrawi refugee camps?\b", re.I), "Tindouf, Algeria", 0.18),
    (re.compile(r"\b(smara|auserd|awserd|dakhla|laayoune|boujdour)\s+camp(s)?\b", re.I), "Tindouf, Algeria", 0.16),
    (re.compile(r"\bgaza(\s+strip)?\b", re.I), "Gaza Strip, Palestine", 0.14),
    (re.compile(r"\bwest bank\b", re.I), "West Bank, Palestine", 0.12),
]
COORD_REGEX = r"(-?\d{1,2}\.\d+)[°,\s]+(-?\d{1,3}\.\d+)"
MIN_ACCEPT = 0.88
AREA_KINDS = {"country","state","region"}
BIG_AREA_KM2 = 4000.0

def _iso_utc(dt: datetime) -> str:
    if dt.tzinfo is None: dt = dt.replace(tzinfo=timezone.utc)
    else: dt = dt.astimezone(timezone.utc)
    return dt.strftime("%Y-%m-%d %H:%M:%S")

def _haversine_km(a_lat, a_lon, b_lat, b_lon) -> float:
    r = 6371.0; from math import radians, sin, cos, asin, sqrt
    phi1=radians(a_lat); phi2=radians(b_lat)
    dphi=radians(b_lat-a_lat); dl=radians(b_lon-a_lon)
    x=sin(dphi/2)**2+cos(phi1)*cos(phi2)*sin(dl/2)**2
    return 2*r*asin(min(1.0, sqrt(x)))

def get_channel_context(src: str):
    cur=conn.cursor(); cur.execute("SELECT last_lat,last_lon,last_label FROM channel_context WHERE source_channel=?", (str(src),))
    r=cur.fetchone(); 
    return (float(r[0]),float(r[1]),r[2] or "") if r and r[0] is not None and r[1] is not None else None

def update_channel_context(src: str, lat: float, lon: float, label: str):
    c.execute("""INSERT INTO channel_context(source_channel,last_lat,last_lon,last_label,last_updated)
                 VALUES(?,?,?,?,CURRENT_TIMESTAMP)
                 ON CONFLICT(source_channel) DO UPDATE SET last_lat=excluded.last_lat,last_lon=excluded.last_lon,
                 last_label=excluded.last_label,last_updated=CURRENT_TIMESTAMP""",
              (str(src), float(lat), float(lon), str(label or ""))); conn.commit()

def _bbox_area_km2(b): 
    try:
        s,n,w,e = map(float, b); lat_km=(n-s)*111.0; mid=(n+s)/2.0
        lon_km=(e-w)*111.0*max(0.1, math.cos(math.radians(mid))); return abs(lat_km*lon_km)
    except Exception: return None

async def geocode_cached(q: str, country_codes: Optional[str]):
    key_cc = country_codes or ""; cur = conn.cursor()
    cur.execute("SELECT lat,lon,label,kind,bbox_area,country_code FROM geocache WHERE q=? AND country_codes=?", (q,key_cc))
    r = cur.fetchone()
    if r: 
        return float(r[0]), float(r[1]), r[2], (r[3] or None), (float(r[4]) if r[4] is not None else None), (r[5] or None)

    def _do_geocode():
        kw=dict(timeout=7, addressdetails=True)
        if country_codes: kw["country_codes"]=country_codes
        return geolocator.geocode(q, **kw)

    try: loc = await asyncio.to_thread(_do_geocode)
    except (GeocoderTimedOut, GeocoderUnavailable):
        logger.warning(f"Geocoder issue: {q}"); return None
    except Exception as e:
        logger.debug(f"Geocode fail {q}: {e}"); return None
    if not loc: return None

    lat,lon = float(loc.latitude), float(loc.longitude)
    label = getattr(loc,"address",q); kind=None; area=None; ccode=None
    try:
        raw = getattr(loc,"raw",{}) or {}
        kind = raw.get("type") or None
        bbox = raw.get("boundingbox") or None
        if bbox: area = _bbox_area_km2(bbox)
        addr = raw.get("address") or {}
        ccode = (addr.get("country_code") or "").lower() or None
    except Exception: pass

    try:
        c.execute("INSERT OR REPLACE INTO geocache(q,country_codes,lat,lon,label,kind,bbox_area,country_code) VALUES(?,?,?,?,?,?,?,?)",
                  (q,key_cc,lat,lon,label,kind,area,ccode)); conn.commit()
    except Exception: pass
    return lat,lon,label,kind,area,ccode

def _extract_candidates(text: str) -> List[str]:
    out=[]
    if nlp:
        try:
            for ent in nlp(text).ents:
                if ent.label_ in ("GPE","LOC","FAC"):
                    t=ent.text.strip()
                    if t and len(t)>=3: out.append(t)
        except Exception: pass
    if not out:
        try:
            for m in re.finditer(
                r"\b(in|near|around|at|inside|outside|north of|south of|east of|west of|city of|"
                r"en|cerca de|al norte de|al sur de|al este de|al oeste de|junto a|próximo a)\s+"
                r"([A-ZÁÉÍÓÚÑ][\w\-ÁÉÍÓÚÑ]+(?:\s+[A-ZÁÉÍÓÚÑ][\w\-ÁÉÍÓÚÑ]+){0,3})", text):
                out.append(m.group(2).strip())
        except Exception: pass
    seen=set(); uniq=[]
    for s in out:
        k=s.lower()
        if k not in seen: seen.add(k); uniq.append(s)
    return uniq

def _preposition_near(text: str, place: str) -> bool:
    try:
        pat = re.compile(
            rf"\b(in|near|around|at|inside|outside|north of|south of|east of|west of|city of|"
            rf"en|cerca de|al norte de|al sur de|al este de|al oeste de|junto a|próximo a)\s+{re.escape(place)}\b", re.I)
        return bool(pat.search(text))
    except Exception: return False

def _action_near_place(text: str, place: str) -> bool:
    try:
        for m in re.finditer(re.escape(place), text, flags=re.I):
            w = text[max(0,m.start()-60):min(len(text), m.end()+60)]
            if ACTION_HINTS.search(w): return True
    except Exception: pass
    return False

def _context_score_base(text: str, place_text: str, kind: Optional[str], area_km2: Optional[float]) -> float:
    base = 0.86 if " " not in place_text else 0.88
    has_prep = _preposition_near(text, place_text)
    if has_prep: base += 0.04
    if ADMIN_HINTS.search(text or ""): base += 0.03
    act_near = _action_near_place(text or "", place_text)
    topic_near = TOPIC_HINTS.search(text or "") is not None

    if kind:
        k = kind.lower()
        if k in {"city","town","village","hamlet","suburb","neighbourhood","neighborhood","locality","island","airport","bridge","checkpoint","port"}:
            base += 0.03
        if k in AREA_KINDS:
            if not act_near and not has_prep: base -= 0.10
            elif topic_near and not act_near: base -= 0.06
            elif act_near: base += 0.02

    if area_km2 is not None:
        if area_km2 < 200.0: base += 0.02
        elif area_km2 < 4000.0: base += 0.01

    return max(0.0, min(base, 0.96))

def _channel_boost(src: str, lat: float, lon: float) -> float:
    ctx=get_channel_context(src)
    if not ctx: return 0.0
    d=_haversine_km(ctx[0],ctx[1],lat,lon)
    return 0.05 if d<50 else (0.03 if d<150 else (0.02 if d<300 else 0.0))

async def _resolve_contextual(text: str, source_channel: str, cand_strings: List[str]):
    if not cand_strings: return None
    hits=[]
    for s in cand_strings:
        g=await geocode_cached(s, None)
        if not g: continue
        lat,lon,label,kind,area,ccode=g
        if not (-90<=lat<=90 and -180<=lon<=180): continue
        hits.append((lat,lon,label or s, s, (kind or "").lower(), area, (ccode or None)))

    if not hits: return None

    neighbor_bonus=0.0
    if len(hits)>=2:
        for i in range(len(hits)):
            for j in range(i+1,len(hits)):
                if _haversine_km(hits[i][0],hits[i][1],hits[j][0],hits[j][1])<40:
                    neighbor_bonus=0.02; break

    best=None
    for lat,lon,label,orig,kind,area,ccode in hits:
        score=_context_score_base(text, orig, kind, area)
        score+=_channel_boost(source_channel, lat, lon)
        score+=neighbor_bonus
        score=min(score,0.97)
        if score>=MIN_ACCEPT:
            is_area = 1 if (kind in AREA_KINDS or (area and area>BIG_AREA_KM2)) else 0
            radius_km = None  # ya no lo usa la app
            cand=(lat,lon,label,score,is_area,radius_km,kind,ccode)
            if best is None: best=cand
            else:
                if cand[3] > best[3] + 0.03: best=cand
                elif abs(cand[3]-best[3])<=0.03 and best[4]==1 and cand[4]==0:
                    best=cand
    return best

def _dedup_nearby(items, eps_m=120.0):
    out=[]
    for a in items:
        lat_a,lon_a=a[0],a[1]
        if any(((abs(lat_a-b[0])*111000)**2+(abs(lon_a-b[1])*85000)**2)**0.5<eps_m for b in out): continue
        out.append(a)
    return out

# ===== Media helpers
def _detect_ext_from_msg(msg: Message) -> str:
    ext=None
    if getattr(msg,"document",None) and getattr(msg.document,"mime_type",None):
        ext=guess_extension(msg.document.mime_type) or None
        if ext==".jpe": ext=".jpg"
    if not ext and getattr(msg,"file",None) and getattr(msg.file,"ext",None): ext=msg.file.ext
    if not ext and getattr(msg,"photo",None): ext=".jpg"
    return ext or ".bin"

async def save_media_with_ext(msg: Message, media_dir: str) -> List[str]:
    paths=[]
    try:
        if msg.grouped_id: return paths
        data=await client.download_media(msg, file=bytes)
        if not data: return paths
        path=os.path.join(media_dir, f"{msg.id}{_detect_ext_from_msg(msg)}")
        with open(path,"wb") as f: f.write(data)
        paths.append(path)
    except FloodWaitError as e:
        await asyncio.sleep(e.seconds+1); return await save_media_with_ext(msg, media_dir)
    except Exception as e:
        logger.error(f"save_media error: {e}")
    return paths

async def save_album_with_ext(msgs: List[Message], media_dir: str) -> List[str]:
    out=[]
    for m in msgs:
        try:
            data=await client.download_media(m, file=bytes)
            if not data: continue
            path=os.path.join(media_dir, f"{m.id}{_detect_ext_from_msg(m)}")
            with open(path,"wb") as f: f.write(data)
            out.append(path)
        except FloodWaitError as e:
            await asyncio.sleep(e.seconds+1); return await save_album_with_ext(msgs, media_dir)
        except Exception as e:
            logger.error(f"save_album item error: {e}")
    return out

# ===== Persistencia
def create_content_hash(text: str, media_paths: List[str]) -> str:
    h=hashlib.sha256(); h.update((text or "").encode("utf-8"))
    for p in media_paths: h.update(os.path.basename(p).encode("utf-8"))
    return h.hexdigest()

def message_exists(src: str, mid: int) -> bool:
    cur=conn.cursor(); cur.execute("SELECT 1 FROM messages WHERE source_channel=? AND telegram_msg_id=? LIMIT 1",(str(src),int(mid)))
    return cur.fetchone() is not None

def try_insert_message(text: str, media_paths: List[str], src: str, mid: int, msg_dt: Optional[datetime]=None) -> Optional[int]:
    content_hash=create_content_hash(text or "", media_paths)
    try:
        if msg_dt is not None:
            c.execute("""INSERT INTO messages(text,media_paths,timestamp,source_channel,telegram_msg_id,content_hash)
                         VALUES(?,?,?,?,?,?)""",
                      (text, ",".join(media_paths), _iso_utc(msg_dt), str(src), int(mid), content_hash))
        else:
            c.execute("""INSERT INTO messages(text,media_paths,source_channel,telegram_msg_id,content_hash)
                         VALUES(?,?,?,?,?)""",
                      (text, ",".join(media_paths), str(src), int(mid), content_hash))
        conn.commit(); return c.lastrowid
    except sqlite3.IntegrityError:
        return None

def insert_locations(message_id: int, locs):
    if not locs or message_id is None: return
    for lat,lon,name,conf,is_area,radius_km,admin_kind,admin_code in locs:
        c.execute("""INSERT INTO locations(message_id,lat,lon,location_name,confidence,is_area,radius_km,admin_kind,admin_code)
                     VALUES(?,?,?,?,?,?,?,?,?)""",
                  (message_id,float(lat),float(lon),str(name or ""),float(conf),int(is_area or 0),
                   float(radius_km) if radius_km else None, admin_kind, admin_code))
    conn.commit()

# ===== Forward/Copy
async def forward_message(msg: Message): await client.forward_messages(MONITOR_GROUP, msg)
async def forward_messages_bulk(msgs: List[Message]): await client.forward_messages(MONITOR_GROUP, msgs)

async def copy_text(msg: Message)->bool:
    t=(msg.message or "").strip(); 
    if not t: return False
    await client.send_message(MONITOR_GROUP, t); return True

async def copy_media(msg: Message)->bool:
    if not (msg.media or msg.photo or msg.document or msg.video or msg.audio or msg.voice): return False
    cap=(msg.message or "").strip() or None
    try:
        data=await client.download_media(msg, file=bytes)
        if not data: return False
        await client.send_file(MONITOR_GROUP, data, caption=cap); return True
    except FloodWaitError as e:
        await asyncio.sleep(e.seconds+1); return await copy_media(msg)
    except Exception as e:
        logger.error(f"copy_media error: {e}"); return False

async def copy_album(msgs: List[Message])->bool:
    files=[]; caps=[]
    try:
        for m in msgs:
            d=await client.download_media(m, file=bytes)
            if d: files.append(d)
            cap=(m.message or "").strip()
            if cap: caps.append(cap)
        if not files: return False
        await client.send_file(MONITOR_GROUP, files, caption=("\n\n".join(caps) if caps else None), album=True)
        return True
    except FloodWaitError as e:
        await asyncio.sleep(e.seconds+1); return await copy_album(msgs)
    except Exception as e:
        logger.error(f"copy_album error: {e}"); return False

async def forward_or_copy_single(msg: Message, do_forward=True):
    if not do_forward: return
    try: await forward_message(msg); return
    except Exception: pass
    if await copy_media(msg): return
    if await copy_text(msg): return

async def forward_or_copy_album(msgs: List[Message], do_forward=True):
    if not do_forward: return
    try: await forward_messages_bulk(msgs); return
    except Exception: pass
    if await copy_album(msgs): return
    for m in msgs: await forward_or_copy_single(m, True)

# ===== Core
async def process_locations(msg: Message):
    text=(msg.message or "").strip()
    out=[]
    # reglas fuertes
    for rx,label,boost in CONTEXT_RULES:
        if rx.search(text or ""):
            hit=await geocode_cached(label, None)
            if hit:
                lat,lon,_,_,_,cc=hit
                out.append((lat,lon,label,min(0.92+boost,0.99),0,None,"",cc))
    # coords explícitas
    try:
        for m in re.finditer(COORD_REGEX, text or ""):
            lat,lon=map(float,m.groups())
            if -90<=lat<=90 and -180<=lon<=180:
                out.append((lat,lon,f"coords:{lat:.5f},{lon:.5f}",1.0,0,None,"",""))
    except Exception: pass
    if out: return _dedup_nearby(out)
    # contextual
    cand=_extract_candidates(text or "")
    best=await _resolve_contextual(text or "", str(msg.chat_id), cand)
    if best:
        lat,lon,label,conf,is_area,radius_km,kind,ccode = best
        return [(lat,lon,label,conf,is_area,radius_km,(kind or ""), (ccode or ""))]
    return []

async def handle_single_message(msg: Message, do_forward=True):
    try:
        if message_exists(str(msg.chat_id), msg.id): return
        media_paths=await save_media_with_ext(msg, MEDIA_DIR)
        row_id=try_insert_message(msg.message or "", media_paths, str(msg.chat_id), msg.id,
                                  msg.date if isinstance(msg.date, datetime) else None)
        if row_id is not None:
            locs=await process_locations(msg); insert_locations(row_id, locs)
            try:
                cur=conn.cursor()
                cur.execute("SELECT lat,lon,location_name FROM locations WHERE message_id=? ORDER BY confidence DESC LIMIT 1",(row_id,))
                r=cur.fetchone()
                if r and r[0] is not None and r[1] is not None:
                    update_channel_context(str(msg.chat_id), float(r[0]), float(r[1]), r[2] or "")
            except Exception: pass
            if do_forward: await forward_or_copy_single(msg)
    except FloodWaitError as e:
        await asyncio.sleep(e.seconds+1)
    except Exception as e:
        logger.error(f"handle_single_message error: {e}")

async def handle_album_messages(msgs: List[Message], chat_id: int, do_forward=True):
    try:
        head=msgs[0]
        if message_exists(str(chat_id), head.id): return
        media_paths_all=await save_album_with_ext(msgs, MEDIA_DIR)
        flat_text=" ⧉ ".join((m.message or "").strip() for m in msgs if (m.message or "").strip())
        row_id=try_insert_message(flat_text, media_paths_all, str(chat_id), head.id,
                                  head.date if isinstance(head.date, datetime) else None)
        if row_id is not None:
            all_locs=[]
            for m in msgs: all_locs+=await process_locations(m)
            insert_locations(row_id, _dedup_nearby(all_locs))
            try:
                cur=conn.cursor()
                cur.execute("SELECT lat,lon,location_name FROM locations WHERE message_id=? ORDER BY confidence DESC LIMIT 1",(row_id,))
                r=cur.fetchone()
                if r and r[0] is not None and r[1] is not None:
                    update_channel_context(str(chat_id), float(r[0]), float(r[1]), r[2] or "")
            except Exception: pass
            if do_forward: await forward_or_copy_album(msgs)
    except FloodWaitError as e:
        await asyncio.sleep(e.seconds+1); await handle_album_messages(msgs, chat_id, do_forward)
    except Exception as e:
        logger.error(f"handle_album_messages error: {e}")

@client.on(events.NewMessage(chats=CHANNELS))
async def on_new_message(event): await handle_single_message(event.message, True)

@client.on(events.Album(chats=CHANNELS))
async def on_new_album(event): await handle_album_messages(event.messages, event.chat_id, True)

async def backfill_channels(days: int, do_forward: bool):
    since_dt = datetime.now(timezone.utc) - timedelta(days=days)
    for ch in CHANNELS:
        current_group: Dict[int, List[Message]] = {}; last_gid=None
        async for msg in client.iter_messages(ch, reverse=True, offset_date=since_dt):
            gid=getattr(msg,"grouped_id",None)
            if gid:
                if last_gid is None or gid==last_gid:
                    current_group.setdefault(gid,[]).append(msg); last_gid=gid; continue
                else:
                    group=current_group.pop(last_gid,[])
                    if group: await handle_album_messages(group, group[0].chat_id, do_forward)
                    current_group.setdefault(gid,[]).append(msg); last_gid=gid; continue
            if last_gid is not None:
                group=current_group.pop(last_gid,[])
                if group: await handle_album_messages(group, group[0].chat_id, do_forward)
                last_gid=None
            await handle_single_message(msg, do_forward)
        if last_gid is not None:
            group=current_group.pop(last_gid,[])
            if group: await handle_album_messages(group, group[0].chat_id, do_forward)

async def run(args):
    await client.start(); logger.info("Conectado a Telegram")
    if args.backfill_days is not None:
        await backfill_channels(args.backfill_days, do_forward=(not args.no_forward))
    if args.only_backfill:
        await client.disconnect(); conn.close(); return
    logger.info("Escuchando en vivo…"); await client.run_until_disconnected()

def parse_args():
    p=argparse.ArgumentParser(description="IntelMap listener (context + areas + admin_code)")
    p.add_argument("--backfill-days", type=int, default=None)
    p.add_argument("--no-forward", action="store_true")
    p.add_argument("--only-backfill", action="store_true")
    return p.parse_args()

if __name__=="__main__":
    args=parse_args()
    try: asyncio.run(run(args))
    finally:
        try: conn.close()
        except Exception: pass
