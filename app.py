#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, sqlite3, unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import humanize
import pandas as pd
import pydeck as pdk
import streamlit as st
from bs4 import BeautifulSoup

# ====== Config (no tocar) ======
from config import (
    MAP_CENTER, DEFAULT_ZOOM, DB_PATH, MEDIA_DIR,
    COUNTRY_CENTROIDS, COUNTRY_ALIASES
)

# ====== Mapbox token (acepta dos nombres) + fallback ======
try:
    _mapbox_token = os.getenv("MAPBOX_API_KEY") or os.getenv("MAPBOX_ACCESS_TOKEN") or ""
    pdk.settings.mapbox_api_key = _mapbox_token
except Exception:
    _mapbox_token = ""

# ====== UI Colors ======
PAGE_TITLE = "IntelLive Pro"
CARD_BG = "#161B22"      # Fondo tarjeta
CARD_BORDER = "#30363D"  # Borde tarjeta
PRIMARY = "#58A6FF"      # Azul elegante
BG = "#0D1117"           # Fondo global
TEXT_MAIN = "#E6EDF3"
TEXT_SUBTLE = "#8B949E"

IMG_EXT={".png",".jpg",".jpeg",".webp",".bmp",".gif"}
VID_EXT={".mp4",".webm",".mov",".m4v"}
AUD_EXT={".mp3",".wav",".ogg",".m4a",".aac"}

try:
    from streamlit_autorefresh import st_autorefresh
except Exception:
    st_autorefresh = None

# ====== Setup ======
def setup_app():
    st.set_page_config(page_title=PAGE_TITLE, page_icon="🌍", layout="wide")
    Path(MEDIA_DIR).mkdir(parents=True, exist_ok=True)
    Path(os.path.dirname(DB_PATH) or ".").mkdir(parents=True, exist_ok=True)

    st.markdown(f"""
    <style>
    :root {{
        --bg:{BG}; --card:{CARD_BG}; --border:{CARD_BORDER}; --primary:{PRIMARY};
        --text-main:{TEXT_MAIN}; --text-subtle:{TEXT_SUBTLE};
    }}
    .stApp {{
        background:var(--bg);
        color:var(--text-main);
        font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Arial;
    }}
    .headline {{
        display:flex; align-items:center; gap:12px; margin-bottom:8px;
    }}
    .headline h1 {{
        margin:0; color:var(--text-main); font-size:1.4rem;
    }}
    .subtle {{
        color:var(--text-subtle); font-size:0.9rem;
    }}
    .card {{
        background:var(--card);
        border:1px solid var(--border);
        border-radius:12px;
        padding:14px;
        display:flex; flex-direction:column; gap:10px; height:100%;
        box-shadow:0 2px 6px rgba(0,0,0,0.4);
    }}
    .card .title {{
        color: var(--primary);
        font-weight: 600;
        font-size: 1rem;
    }}
    .card .meta {{
        color:var(--text-subtle);
        font-size:0.8rem;
    }}
    .card .text {{
        color:var(--text-main);
        font-size:0.92rem;
        line-height:1.4;
        max-height:7.5em;
        overflow:hidden;
    }}
    div[data-testid="stMetricValue"] {{
        color: var(--primary);
    }}
    </style>
    """, unsafe_allow_html=True)

    st.session_state.setdefault("selected_report", None)
    st.session_state.setdefault("page", 1)
    st.session_state.setdefault("last_query", "")
    st.session_state.setdefault("last_range", "30 días")
    st.session_state.setdefault("show_all", False)
    st.session_state.setdefault("auto_refresh_on", False)
    st.session_state.setdefault("only_coords", False)
    st.session_state.setdefault("center_nonce", 0)
    st.session_state.setdefault("refresh_nonce", 0)

# ====== Utils ======
def _norm(s: str) -> str:
    s = (s or "").strip().lower()
    s = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in s if not unicodedata.combining(ch))

def as_int(x, default: int = 0) -> int:
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return default
        return int(float(x))
    except Exception:
        return default

def as_float(x, default: Optional[float] = None) -> Optional[float]:
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return default
        return float(x)
    except Exception:
        return default

def clean_text(text: Optional[str]) -> str:
    if not text: return ""
    try: clean = BeautifulSoup(text, "lxml").get_text()
    except Exception: clean = BeautifulSoup(text, "html.parser").get_text()
    clean = re.sub(r"@\w+\s?|https?://\S+|[^\w\s.,!?¿¡áéíóúÁÉÍÓÚñÑ\-']", " ", clean)
    return " ".join(clean.strip().split())

def _valid_coords(lat, lon) -> bool:
    try: lat=float(lat); lon=float(lon); return -90<=lat<=90 and -180<=lon<=180
    except Exception: return False

def time_ago(ts: Optional[pd.Timestamp]) -> str:
    if ts is None or pd.isna(ts): return ""
    now = datetime.now(timezone.utc)
    if ts.tzinfo is None: ts = ts.tz_localize("UTC")
    return humanize.naturaltime(now - ts).replace(" ago", "")

def split_media_paths(media_paths: Optional[str]) -> List[str]:
    if not media_paths:
        return []
    parts = [p.strip() for p in str(media_paths).split(",") if p.strip()]
    resolved = []
    for p in parts:
        if p.lower().startswith(("http://", "https://")):
            resolved.append(p); continue
        base = os.path.basename(p)
        cand = []
        if os.path.isabs(p):
            cand.append(p); cand.append(os.path.join(MEDIA_DIR, base)); cand.append(os.path.join("media", base))
        else:
            cand.append(os.path.join(MEDIA_DIR, p)); cand.append(os.path.join(MEDIA_DIR, base)); cand.append(os.path.join("media", base))
        chosen = None
        for c in cand:
            try:
                if os.path.exists(c): chosen = c; break
            except Exception: continue
        if chosen: resolved.append(chosen)
    out, seen = [], set()
    for r in resolved:
        if r not in seen:
            out.append(r); seen.add(r)
    return out

def pick_preview(paths: List[str]) -> Tuple[Optional[str], Optional[str]]:
    if not paths: return None, None
    def ext(p): return os.path.splitext(p)[1].lower()
    imgs=[p for p in paths if ext(p) in IMG_EXT or p.lower().startswith(("http://","https://")) and any(x in p.lower() for x in (".png",".jpg",".jpeg",".webp",".gif",".bmp"))]
    if imgs: return "image", imgs[0]
    vids=[p for p in paths if ext(p) in VID_EXT or p.lower().startswith(("http://","https://")) and any(x in p.lower() for x in (".mp4",".webm",".mov",".m4v"))]
    if vids: return "video", vids[0]
    auds=[p for p in paths if ext(p) in AUD_EXT or p.lower().startswith(("http://","https://")) and any(x in p.lower() for x in (".mp3",".wav",".ogg",".m4a",".aac"))]
    if auds: return "audio", auds[0]
    return "other", paths[0]

# ====== Overrides ======
def ensure_override_table():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS location_overrides(
                message_id INTEGER PRIMARY KEY,
                lat REAL, lon REAL, location_name TEXT,
                is_area INTEGER DEFAULT 0, radius_km REAL,
                admin_kind TEXT, admin_code TEXT,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )""")
        conn.commit()

def save_override(message_id:int, lat:float, lon:float, name:str, is_area:int=0,
                  radius_km:Optional[float]=None, admin_kind:Optional[str]=None, admin_code:Optional[str]=None):
    ensure_override_table()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""INSERT INTO location_overrides(message_id,lat,lon,location_name,is_area,radius_km,admin_kind,admin_code,updated_at)
                        VALUES(?,?,?,?,?,?,?,?,CURRENT_TIMESTAMP)
                        ON CONFLICT(message_id) DO UPDATE SET
                          lat=excluded.lat, lon=excluded.lon, location_name=excluded.location_name,
                          is_area=excluded.is_area, radius_km=excluded.radius_km,
                          admin_kind=excluded.admin_kind, admin_code=excluded.admin_code,
                          updated_at=CURRENT_TIMESTAMP""",
                     (int(message_id), float(lat), float(lon), str(name or ""),
                      int(is_area or 0), (None if radius_km is None else float(radius_km)),
                      (admin_kind or None), (admin_code or None)))
        conn.commit()

def delete_override(message_id:int):
    ensure_override_table()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM location_overrides WHERE message_id=?", (int(message_id),))
        conn.commit()

# ====== Borrar reportes (hard delete) ======
def delete_report(message_id: int):
    """Elimina un reporte de la DB y sus relaciones."""
    with _db_connect() as conn:
        conn.execute("DELETE FROM messages WHERE id=?", (int(message_id),))
        conn.execute("DELETE FROM locations WHERE message_id=?", (int(message_id),))
        conn.execute("DELETE FROM location_overrides WHERE message_id=?", (int(message_id),))
        conn.commit()

# ====== DB helpers ======
def _parse_timestamp_series(s: pd.Series) -> pd.Series:
    s0 = pd.to_datetime(s, utc=True, errors="coerce")
    if s0.notna().any(): return s0
    s1 = pd.to_numeric(s, errors="coerce")
    if s1.notna().any():
        ts = pd.to_datetime(s1, unit="s", utc=True, errors="coerce")
        if ts.notna().any(): return ts
        ts = pd.to_datetime(s1, unit="ms", utc=True, errors="coerce")
        return ts
    return s0

def _db_connect():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    try:
        conn.execute("PRAGMA busy_timeout=30000;")
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
    except Exception: pass
    return conn

def _db_stats() -> Dict[str, Optional[str]]:
    stats = {"path": DB_PATH, "exists": os.path.exists(DB_PATH), "size": None, "count": None, "last_id": None, "last_ts": None}
    if not stats["exists"]: return stats
    stats["size"] = os.path.getsize(DB_PATH)
    try:
        with _db_connect() as conn:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*), MAX(id), MAX(timestamp) FROM messages")
            row = cur.fetchone() or (0, None, None)
            stats["count"], stats["last_id"], stats["last_ts"] = row[0], row[1], row[2]
    except Exception: pass
    return stats

def _db_mtime() -> float:
    try: return os.path.getmtime(DB_PATH)
    except Exception: return 0.0

@st.cache_data(ttl=0)
def load_data(db_mtime: float, nonce: int) -> pd.DataFrame:
    base_cols = ["id","text","timestamp","media_paths","latitude","longitude","ubicacion","confidence",
                 "is_area","radius_km","admin_kind","admin_code","media_files","media_count","time_ago",
                 "area_str","overridden"]
    query = """
    WITH best_loc AS (
      SELECT l.message_id, l.lat AS latitude, l.lon AS longitude, l.location_name AS ubicacion,
             l.confidence AS confidence, l.is_area, l.radius_km, l.admin_kind, l.admin_code
      FROM locations l
      JOIN (SELECT message_id, MAX(confidence) AS maxc FROM locations GROUP BY message_id) mx
        ON l.message_id=mx.message_id AND l.confidence=mx.maxc
    ),
    eff AS (
      SELECT
        m.id, m.text, m.timestamp, m.media_paths,
        COALESCE(o.lat, b.latitude) AS latitude,
        COALESCE(o.lon, b.longitude) AS longitude,
        COALESCE(o.location_name, b.ubicacion) AS ubicacion,
        b.confidence,
        COALESCE(o.is_area, b.is_area) AS is_area,
        COALESCE(o.radius_km, b.radius_km) AS radius_km,
        COALESCE(o.admin_kind, b.admin_kind) AS admin_kind,
        COALESCE(o.admin_code, b.admin_code) AS admin_code,
        CASE WHEN o.message_id IS NOT NULL THEN 1 ELSE 0 END AS overridden
      FROM messages m
      LEFT JOIN best_loc b ON m.id=b.message_id
      LEFT JOIN location_overrides o ON o.message_id=m.id
    )
    SELECT * FROM eff ORDER BY timestamp DESC
    """
    ensure_override_table()
    with _db_connect() as conn:
        df = pd.read_sql(query, conn)
    if df.empty: return pd.DataFrame(columns=base_cols)
    df["timestamp"] = _parse_timestamp_series(df["timestamp"])
    df = df.sort_values("timestamp", ascending=False, kind="mergesort").reset_index(drop=True)
    df["text"] = df["text"].fillna("").map(clean_text)
    df["media_files"] = df["media_paths"].fillna("").map(split_media_paths)
    df["media_count"] = df["media_files"].map(len)
    df["time_ago"] = df["timestamp"].map(time_ago)
    df["latitude"]  = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["confidence"] = pd.to_numeric(df.get("confidence"), errors="coerce")
    df["radius_km"] = pd.to_numeric(df.get("radius_km"), errors="coerce")
    df["is_area"] = pd.to_numeric(df.get("is_area"), errors="coerce").fillna(0).astype(int)
    def _area_str(r):
        try: r=float(r); return f" · 🟦 área ~{r:.0f} km" if r>0 else ""
        except Exception: return ""
    df["area_str"] = df["radius_km"].map(_area_str)
    df["overridden"] = pd.to_numeric(df["overridden"], errors="coerce").fillna(0).astype(int)
    for col in base_cols:
        if col not in df.columns: df[col]=None
    return df

# ====== Inferencia país por texto ======
def _build_alias_regex() -> Optional[re.Pattern]:
    if not COUNTRY_ALIASES: return None
    keys = set(COUNTRY_ALIASES.keys())
    for iso2, (_, _, label) in COUNTRY_CENTROIDS.items():
        if label: keys.add(_norm(label))
    ordered = sorted(keys, key=lambda s: (-len(s), s))
    try: return re.compile(r"\b(" + "|".join(re.escape(k) for k in ordered) + r")\b", re.IGNORECASE)
    except Exception: return None

_ALIAS_RE = _build_alias_regex()

def augment_with_text_country_inference(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or _ALIAS_RE is None: return df
    mask = df["latitude"].isna() | df["longitude"].isna()
    if not mask.any(): return df
    def _infer_row(row):
        text = (row.get("text") or "")
        m = _ALIAS_RE.search(_norm(text))
        if not m: return row
        key = m.group(1).lower()
        iso = COUNTRY_ALIASES.get(key)
        if not iso:
            for k_iso, (_, _, label) in COUNTRY_CENTROIDS.items():
                if label and _norm(label) == key:
                    iso = k_iso; break
        if not iso or iso not in COUNTRY_CENTROIDS: return row
        lat, lon, label = COUNTRY_CENTROIDS[iso]
        if pd.isna(row.get("latitude")): row["latitude"] = lat
        if pd.isna(row.get("longitude")): row["longitude"] = lon
        current = as_int(row.get("is_area"), 0)
        row["is_area"] = 1 if current == 0 else current
        row["admin_code"] = row.get("admin_code") or iso
        row["admin_kind"] = row.get("admin_kind") or "country"
        row["ubicacion"] = row.get("ubicacion") or label
        if pd.isna(row.get("confidence")): row["confidence"] = 0.35
        return row
    df.loc[mask] = df.loc[mask].apply(_infer_row, axis=1)
    return df

# ====== Mapa ======
def render_map(points_df: pd.DataFrame, selected: Optional[dict]):
    pts = points_df.copy()
    pts = pts.dropna(subset=["latitude","longitude"])
    if not pts.empty: pts = pts[(pts["latitude"].between(-90,90)) & (pts["longitude"].between(-180,180))]
    if selected and _valid_coords(selected.get("latitude"), selected.get("longitude")):
        zoom = 6 if as_int(selected.get("is_area"), 0)==1 else 13
        lat0 = float(selected["latitude"]); lon0 = float(selected["longitude"])
        deck_key_base = f"deck_{selected.get('id')}"
    elif not pts.empty:
        lat0 = float(pts["latitude"].mean()); lon0 = float(pts["longitude"].mean()); zoom = 3.5
        deck_key_base = "deck_default"
    else:
        lat0, lon0 = MAP_CENTER; zoom = DEFAULT_ZOOM
        deck_key_base = "deck_empty"
    deck_key = f"{deck_key_base}_{st.session_state.get('center_nonce',0)}"
    view_state = pdk.ViewState(latitude=lat0, longitude=lon0, zoom=zoom, pitch=45)
    plot_df = pts.copy()
    plot_df["hover_name"] = plot_df["ubicacion"].fillna("")
    layer_points = pdk.Layer(
        "ScatterplotLayer", data=plot_df, get_position=["longitude","latitude"],
        get_radius=9, radius_units="pixels", pickable=True,
        stroked=True, filled=True,
        get_fill_color=[88, 166, 255, 225],
        get_line_color=[120, 180, 255, 255],
        line_width_min_pixels=1,
    )
    has_mapbox = bool(_mapbox_token)
    layers = [layer_points]; map_style = "mapbox://styles/mapbox/satellite-streets-v11"
    if not has_mapbox:
        tile_layer = pdk.Layer(
            "TileLayer", data="https://tile.openstreetmap.org/{z}/{x}/{y}.png",
            min_zoom=0, max_zoom=19, tile_size=256, opacity=1.0,
        )
        layers = [tile_layer, layer_points]; map_style = None
    deck = pdk.Deck(
        layers=layers, initial_view_state=view_state, map_style=map_style,
        tooltip={
            "html": """
                <div style="background:#0D1117dd;border:2px solid #58A6FF;border-radius:8px;padding:12px;max-width:420px">
                    <div style="color:#58A6FF;font-weight:700;font-size:1.05rem">{hover_name}</div>
                    <div style="color:#E6EDF3;margin:8px 0;font-size:0.9rem">{text}</div>
                    <div style="color:#8B949E;font-size:0.8rem">
                        🕒 {time_ago}{area_str} · 📎 {media_count} archivos
                    </div>
                </div>
            """
        },
    )
    st.pydeck_chart(deck, use_container_width=True, key=deck_key)

# ====== Corrección ======
# (se mantiene igual que en tu código previo, con delete_override)

# ====== Listado ======
def render_list(df: pd.DataFrame):
    PAGE_SIZE=12
    total=len(df); total_pages=max(1, (total+PAGE_SIZE-1)//PAGE_SIZE)
    c1,c2,c3=st.columns([1,2,1])
    with c2:
        st.write("")
        st.markdown(f'<div class="headline"><h1>📰 Reportes</h1><span class="subtle">{total} items</span></div>', unsafe_allow_html=True)
    with c3:
        st.selectbox("Página", options=list(range(1,total_pages+1)), key="page", index=min(st.session_state.page-1, total_pages-1))
    start=(st.session_state.page-1)*PAGE_SIZE; end=start+PAGE_SIZE
    page_df=df.iloc[start:end].reset_index(drop=True)
    cols=st.columns(3, gap="small")
    for i,row in page_df.iterrows():
        with cols[i%3]:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            title=row.get("ubicacion") or "Ubicación"
            badge=" ✅ Corregida" if as_int(row.get("overridden"),0)==1 else ""
            st.markdown(f"<div class='title'>{title}{badge}</div>", unsafe_allow_html=True)
            meta=row.get("time_ago","")
            if pd.notna(row.get("confidence")): meta=f"{meta} · conf {row['confidence']:.2f}"
            is_area_flag = (as_int(row.get("is_area"),0)==1)
            if is_area_flag and pd.notna(row.get("radius_km")):
                meta=f"{meta} · área ~{float(row['radius_km']):.0f} km"
            st.markdown(f"<div class='meta'>{meta}</div>", unsafe_allow_html=True)
            txt=(row.get("text") or "")
            if len(txt)>500: txt=txt[:500].rstrip()+"…"
            st.markdown(f"<div class='text'>{txt}</div>", unsafe_allow_html=True)
            media_files=row.get("media_files") or []
            kind,path=pick_preview(media_files)
            if kind=="image" and path: st.image(path, use_container_width=True)
            elif kind=="video" and path: st.video(path)
            elif kind=="audio" and path: st.audio(path)
            elif path: st.caption(f"📎 {os.path.basename(path)}")
            colb1,colb2,colb3 = st.columns([1,1,1])
            if colb1.button("📍 Centrar", key=f"center_{row['id']}"):
                st.session_state.selected_report={
                    "id": int(row["id"]),
                    "latitude": as_float(row.get("latitude")),
                    "longitude": as_float(row.get("longitude")),
                    "ubicacion": row.get("ubicacion") or "",
                    "text": row.get("text") or "",
                    "time_ago": row.get("time_ago") or "",
                    "media_count": int(row.get("media_count") or 0),
                    "is_area": as_int(row.get("is_area"),0),
                    "admin_code": row.get("admin_code") or "",
                    "admin_kind": row.get("admin_kind") or ""
                }
                st.session_state.center_nonce += 1; st.rerun()
            if colb2.button("✏️ Corregir", key=f"fixbtn_{row['id']}"):
                st.session_state[f"show_fix_{row['id']}"]=True
            if colb3.button("🗑️ Borrar", key=f"delete_{row['id']}"):
                delete_report(int(row["id"]))
                st.success("Reporte eliminado ✓")
                st.rerun()
            if st.session_state.get(f"show_fix_{row['id']}", False):
                _correction_form(row)
            st.markdown("</div>", unsafe_allow_html=True)

# ====== Main ======
def main():
    setup_app()
    with st.sidebar:
        st.markdown("Controles")
        show_all=st.checkbox("Mostrar todo el historial", value=st.session_state.show_all); st.session_state.show_all=show_all
        time_range = st.selectbox("Rango temporal", ["24h","3 días","30 días"] if not show_all else ["todo"], index=2 if not show_all else 0)
        q = st.text_input("Buscar texto…", value=st.session_state.last_query).strip()
        only_coords = st.checkbox("Solo con ubicación", value=st.session_state.only_coords); st.session_state.only_coords=only_coords
        auto_refresh = st.checkbox("Auto-actualizar cada 15s", value=st.session_state.auto_refresh_on); st.session_state.auto_refresh_on=auto_refresh
        colr1, colr2 = st.columns([1,1])
        if colr1.button("Recargar ahora"):
            st.session_state.refresh_nonce += 1; st.rerun()
        with st.expander("Diagnóstico DB"):
            stats = _db_stats()
            st.write(f"Ruta DB: `{stats['path']}`")
            st.write(f"Existe: {stats['exists']} · Tamaño: {stats['size']} bytes")
            st.write(f"Mensajes: {stats['count']} · Último ID: {stats['last_id']} · Último timestamp: {stats['last_ts']}")
        with st.expander("Basemap"):
            provider = "Mapbox" if (_mapbox_token) else "OpenStreetMap (TileLayer)"
            st.write(f"Proveedor: **{provider}**")
            st.write(f"Tiene token: `{bool(_mapbox_token)}`")
    if st.session_state.auto_refresh_on and st_autorefresh is not None:
        st_autorefresh(interval=15000, key="data_refresh")
    if q != st.session_state.last_query or time_range != st.session_state.last_range:
        st.session_state.page = 1; st.session_state.last_query = q; st.session_state.last_range = time_range
    db_mtime = _db_mtime()
    df = load_data(db_mtime, st.session_state.refresh_nonce)
    if not df.empty and time_range != "todo":
        days = 1 if time_range=="24h" else (3 if time_range=="3 días" else 30)
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)
        df = df[df["timestamp"].notna() & (df["timestamp"] >= cutoff)]
    if not df.empty and q:
        df = df[df["text"].str.contains(re.escape(q), case=False, na=False)]
    df = augment_with_text_country_inference(df)
    df_for_list = df.copy(); df_for_map = df.copy()
    if not df_for_map.empty:
        df_for_map = df_for_map.dropna(subset=["latitude","longitude"])
        df_for_map = df_for_map[(df_for_map["latitude"].between(-90,90)) & (df_for_map["longitude"].between(-180,180))]
    if st.session_state.only_coords: df_for_list = df_for_map
    st.markdown('<div class="headline"><h1>Mapa</h1></div>', unsafe_allow_html=True)
    cA,cB,cC,cD,cE = st.columns(5)
    total=len(df_for_list)
    with_media = int((df_for_list["media_count"]>0).sum()) if not df_for_list.empty else 0
    total_files = int(df_for_list["media_count"].sum()) if not df_for_list.empty else 0
    with_areas  = int((df_for_list["is_area"]==1).sum()) if not df_for_list.empty else 0
    with_points_map = len(df_for_map) if not df_for_map.empty else 0
    cA.metric("Reportes", total); cB.metric("Con media", with_media)
    cC.metric("Archivos totales", total_files); cD.metric("ÁREAS", with_areas); cE.metric("PUNTOS (en mapa)", with_points_map)
    render_map(df_for_map, st.session_state.get("selected_report"))
    st.markdown("---"); render_list(df_for_list)

if __name__=="__main__":
    main()
