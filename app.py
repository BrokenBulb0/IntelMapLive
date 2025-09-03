#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json, os, re, sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import humanize
import pandas as pd
import pydeck as pdk
import streamlit as st
from bs4 import BeautifulSoup

# ========= Config obligatoria (tu archivo) =========
from config import (
    MAP_CENTER, DEFAULT_ZOOM, DB_PATH, MEDIA_DIR, DATA_DIR,
    COUNTRY_CENTROIDS, COUNTRY_ALIASES
)

# Ruta opcional al geojson (si no existe, la app sigue funcionando)
COUNTRIES_GEOJSON = getattr(__import__("config"), "COUNTRIES_GEOJSON",
                            os.path.join(DATA_DIR, "countries.geojson"))

# Mapbox token opcional (tiles)
try:
    pdk.settings.mapbox_api_key = os.getenv("MAPBOX_API_KEY", "")
except Exception:
    pass

# ========= UI =========
PAGE_TITLE="IntelLive Pro"
CARD_BG="#0F1426"; CARD_BORDER="#243145"; PRIMARY="#FF4B4B"
IMG_EXT={".png",".jpg",".jpeg",".webp",".bmp",".gif"}
VID_EXT={".mp4",".webm",".mov",".m4v"}
AUD_EXT={".mp3",".wav",".ogg",".m4a",".aac"}

try:
    from streamlit_autorefresh import st_autorefresh
except Exception:
    st_autorefresh = None

# ========= Setup =========
def setup_app():
    st.set_page_config(page_title=PAGE_TITLE, page_icon="🌍", layout="wide")
    Path(MEDIA_DIR).mkdir(parents=True, exist_ok=True)
    Path(DATA_DIR).mkdir(parents=True, exist_ok=True)

    st.markdown(f"""
    <style>
    :root {{ --bg:#0A0F24; --card:{CARD_BG}; --border:{CARD_BORDER}; --primary:{PRIMARY}; }}
    .stApp {{ background:var(--bg); color:#e5e7eb; font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Arial; }}
    .headline {{ display:flex; align-items:center; gap:12px; margin-bottom:8px; }}
    .headline h1 {{ margin:0; color:#fff; font-size:1.4rem; }}
    .subtle {{ color:#94a3b8; font-size:0.9rem }}
    .card {{ background:var(--card); border:1px solid var(--border); border-radius:10px; padding:12px; display:flex; flex-direction:column; gap:8px; height:100%; }}
    .card .title {{ color: var(--primary); font-weight: 600; font-size: 0.95rem; }}
    .card .meta {{ color:#9ca3af; font-size:0.8rem }}
    .card .text {{ color:#e5e7eb; font-size:0.9rem; line-height:1.35; max-height:7.5em; overflow:hidden; }}
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

# ========= Utils =========
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
    if not media_paths: return []
    parts = [p.strip() for p in str(media_paths).split(",") if p.strip()]
    out = []
    for p in parts:
        base = os.path.basename(p)
        full = p if os.path.isabs(p) else os.path.join(MEDIA_DIR, base)
        out.append(full)
    seen=set(); uniq=[]
    for p in out:
        if p not in seen: uniq.append(p); seen.add(p)
    return [p for p in uniq if os.path.exists(p)]

def pick_preview(paths: List[str]) -> Tuple[Optional[str], Optional[str]]:
    if not paths: return None, None
    imgs=[p for p in paths if os.path.splitext(p)[1].lower() in IMG_EXT]
    if imgs: return "image", imgs[0]
    vids=[p for p in paths if os.path.splitext(p)[1].lower() in VID_EXT]
    if vids: return "video", vids[0]
    auds=[p for p in paths if os.path.splitext(p)[1].lower() in AUD_EXT]
    if auds: return "audio", auds[0]
    return "other", paths[0]

# ========= Overrides =========
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

# ========= Países: GeoJSON (opcional) y lookup =========
def _feature_centroid(geom: dict) -> Optional[Tuple[float,float]]:
    if not geom: return None
    t = geom.get("type"); coords = geom.get("coordinates")
    if not coords: return None
    lat_sum=lon_sum=count=0
    if t=="Polygon":
        for ring in coords:
            for lon,lat in ring:
                lat_sum+=lat; lon_sum+=lon; count+=1
    elif t=="MultiPolygon":
        for poly in coords:
            for ring in poly:
                for lon,lat in ring:
                    lat_sum+=lat; lon_sum+=lon; count+=1
    return (lat_sum/count, lon_sum/count) if count else None

@st.cache_data(ttl=3600)
def load_countries_geojson(path:str):
    if not os.path.exists(path): return None
    with open(path,"r",encoding="utf-8") as f:
        return json.load(f)

@st.cache_data(ttl=3600)
def build_country_lookup(geojson: Optional[dict]):
    """
    Retorna:
      lookup: dict name_lower -> {name, code(ISO2/ADM0_A3), centroid}
      pattern: regex para detectar nombres en texto
    Si no hay geojson, arma un lookup mínimo con COUNTRY_CENTROIDS/ALIASES.
    """
    if not geojson:
        # construir lookup mínimo desde config
        inv = {}  # nombre_legible_lower -> ISO2
        for iso2, (lat, lon, label) in COUNTRY_CENTROIDS.items():
            inv[label.strip().lower()] = iso2
        for alias, iso2 in COUNTRY_ALIASES.items():
            inv[alias.strip().lower()] = iso2

        lookup = {}
        for name, iso2 in inv.items():
            lat, lon, label = COUNTRY_CENTROIDS.get(iso2, (None, None, None))
            if lat is None: continue
            lookup[name] = {"name": label or name.title(), "code": iso2, "centroid": (lat, lon)}
        pattern = re.compile(r"\b(" + "|".join(re.escape(n) for n in lookup.keys()) + r")\b", re.IGNORECASE) if lookup else re.compile(r"$a")
        return lookup, pattern

    lookup={}
    names=set()
    for f in geojson.get("features", []):
        p=f.get("properties", {})
        canon_name = (p.get("NAME") or p.get("NAME_EN") or p.get("ADMIN") or p.get("SOVEREIGNT") or "").strip()
        if not canon_name: continue
        # preferimos ISO_A2 si existe, si no ADM0_A3
        code = (p.get("ISO_A2") or p.get("ADM0_A3") or p.get("WB_A2") or "").strip().upper()
        centroid = _feature_centroid(f.get("geometry"))

        # variantes para matchear
        alts = {
            canon_name,
            p.get("NAME_LONG"), p.get("ADMIN"), p.get("ABBREV"),
            p.get("NAME_EN"), p.get("SOVEREIGNT")
        }
        # también incorpora alias de config que mapeen a este code
        for alias, iso2 in COUNTRY_ALIASES.items():
            if iso2 and code and iso2.upper()==code.upper():
                alts.add(alias)

        altnames = {re.sub(r"\s+", " ", (x or "")).strip().lower() for x in alts if x}
        altnames = {n for n in altnames if len(n) >= 3}
        for n in altnames:
            lookup[n] = {"name": canon_name, "code": code, "centroid": centroid}
            names.add(n)

    sorted_names = sorted(names, key=lambda s: (-len(s), s))
    pattern = re.compile(r"\b(" + "|".join(re.escape(n) for n in sorted_names) + r")\b", re.IGNORECASE) if sorted_names else re.compile(r"$a")
    return lookup, pattern

def infer_country_from_text(text: str, lookup: dict, pattern: re.Pattern):
    if not text: return None
    m = pattern.search(text.lower())
    if m:
        rec = lookup.get(m.group(1).lower())
        if rec and rec.get("centroid"): return rec
    # intento por alias directos si no hubo match (fallback rápido)
    for alias, iso2 in COUNTRY_ALIASES.items():
        if re.search(rf"\b{re.escape(alias)}\b", text, flags=re.IGNORECASE):
            cent = COUNTRY_CENTROIDS.get(iso2)
            if cent:
                lat, lon, label = cent
                return {"name": label, "code": iso2, "centroid": (lat,lon)}
    return None

# ========= Datos =========
def _parse_timestamp_series(s: pd.Series) -> pd.Series:
    s0 = pd.to_datetime(s, utc=True, errors="coerce")
    if s0.notna().any(): 
        return s0
    s1 = pd.to_numeric(s, errors="coerce")
    if s1.notna().any():
        ts = pd.to_datetime(s1, unit="s", utc=True, errors="coerce")
        if ts.notna().any(): return ts
        ts = pd.to_datetime(s1, unit="ms", utc=True, errors="coerce")
        return ts
    return s0

@st.cache_data(ttl=15)
def load_data() -> pd.DataFrame:
    """
    Una fila por mensaje, priorizando override si existe.
    Sin filtros de fecha/coords aquí; se aplican fuera.
    """
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
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql(query, conn)

    if df.empty:
        return pd.DataFrame(columns=base_cols)

    df["timestamp"] = _parse_timestamp_series(df["timestamp"])
    df = df.sort_values("timestamp", ascending=False, kind="mergesort").reset_index(drop=True)

    df["text"] = df["text"].fillna("").map(clean_text)
    df["media_files"] = df["media_paths"].fillna("").map(split_media_paths)
    df["media_count"] = df["media_files"].map(len)
    df["time_ago"] = df["timestamp"].map(time_ago)

    df["latitude"]  = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["confidence"] = pd.to_numeric(df.get("confidence"), errors="coerce")

    def _area_str(r):
        try:
            r=float(r); return f" · 🟦 área ~{r:.0f} km" if r>0 else ""
        except Exception:
            return ""
    df["area_str"] = df["radius_km"].map(_area_str)
    df["overridden"] = df["overridden"].fillna(0).astype(int)

    for col in base_cols:
        if col not in df.columns: df[col]=None
    return df

# ========= Fallback: inferir país por texto cuando no hay coords =========
def augment_with_text_country_inference(df: pd.DataFrame, lookup: dict, pattern: re.Pattern) -> pd.DataFrame:
    if df.empty: return df
    mask = df["latitude"].isna() | df["longitude"].isna()
    if not mask.any(): return df

    def _infer_row(row):
        rec = infer_country_from_text(row.get("text",""), lookup, pattern)
        if not rec or not rec.get("centroid"): 
            return row
        lat, lon = rec["centroid"]
        if pd.isna(row.get("latitude")): row["latitude"] = lat
        if pd.isna(row.get("longitude")): row["longitude"] = lon
        row["is_area"] = 1 if pd.isna(row.get("is_area")) or int(row.get("is_area") or 0)==0 else int(row["is_area"])
        row["admin_code"] = row.get("admin_code") or rec.get("code")
        row["admin_kind"] = row.get("admin_kind") or "country"
        row["ubicacion"] = row.get("ubicacion") or rec.get("name")
        if pd.isna(row.get("confidence")): row["confidence"] = 0.35  # marcada como inferida
        return row

    df.loc[mask] = df.loc[mask].apply(_infer_row, axis=1)
    return df

# ========= Map helpers =========
def _find_country_feature(geojson, admin_code: Optional[str], label: Optional[str]):
    if not geojson: return None
    feats = geojson.get("features", [])
    code = (admin_code or "").upper()
    name = (label or "").lower()

    def _props_ok(p):
        vals = [p.get("ISO_A2"), p.get("ADM0_A3"), p.get("WB_A2")]
        vals = [str(v).upper() for v in vals if v]
        return code and (code in vals)

    for f in feats:
        p=f.get("properties",{})
        if code and _props_ok(p):
            return f
    if name:
        for f in feats:
            p=f.get("properties",{})
            n=(p.get("NAME") or p.get("NAME_EN") or p.get("ADMIN") or "").lower()
            if n and (n==name or name in n):
                return f
    return None

def render_map(points_df: pd.DataFrame, selected: Optional[dict], countries):
    pts = points_df.copy()
    pts = pts.dropna(subset=["latitude","longitude"])
    if not pts.empty:
        pts = pts[(pts["latitude"].between(-90,90)) & (pts["longitude"].between(-180,180))]

    # View
    if selected and _valid_coords(selected.get("latitude"), selected.get("longitude")):
        zoom = 6 if int(selected.get("is_area",0))==1 else 13
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

    layers=[]
    # capa países (debajo) solo si hay geojson
    if countries:
        countries_layer = pdk.Layer(
            "GeoJsonLayer",
            data=countries,
            pickable=True,
            stroked=True,
            filled=True,
            opacity=0.05,
            get_fill_color=[0, 150, 255, 15],
            get_line_color=[80, 160, 255, 120],
            line_width_min_pixels=1,
            auto_highlight=True,
            highlight_color=[0, 255, 255, 140],
        )
        layers.append(countries_layer)

    # puntos (encima)
    layer_points = pdk.Layer(
        "ScatterplotLayer",
        data=pts,
        get_position=["longitude","latitude"],
        get_radius=9,
        radius_units="pixels",
        pickable=True,
        stroked=True,
        filled=True,
        get_fill_color=[255, 0, 0, 225],
        get_line_color=[255, 120, 120, 255],
        line_width_min_pixels=1,
    )
    layers.append(layer_points)

    deck = pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        map_style="mapbox://styles/mapbox/satellite-streets-v11",
        tooltip={
            "html": """
                <div style="background:#0A0F24dd;border:2px solid #FF4B4B;border-radius:8px;padding:12px;max-width:420px">
                    <div style="color:#FF4B4B;font-weight:700;font-size:1.05rem">{ubicacion}{properties.ADMIN}</div>
                    <div style="color:#E0F2FE;margin:8px 0;font-size:0.9rem">{text}</div>
                    <div style="color:#94A3B8;font-size:0.8rem">
                        🕒 {time_ago}{area_str} · 📎 {media_count} archivos
                    </div>
                </div>
            """
        },
    )
    st.pydeck_chart(deck, use_container_width=True, key=deck_key)

# ========= Corrección UI =========
def _resolve_place_by_name(name: str, countries: Optional[dict]) -> Optional[Dict]:
    """Devuelve dict con lat, lon, label, code (si país)."""
    if not name: return None
    q = name.strip().lower()

    # 1) GeoJSON (si existe)
    if countries:
        # construir lookup simple en memoria
        feats = countries.get("features", [])
        best = None
        for f in feats:
            p = f.get("properties", {})
            labels = [
                p.get("NAME"), p.get("NAME_EN"), p.get("ADMIN"),
                p.get("SOVEREIGNT"), p.get("NAME_LONG"), p.get("ABBREV")
            ]
            labels = [str(x).strip().lower() for x in labels if x]
            if any(q == lab or q in lab for lab in labels):
                cen = _feature_centroid(f.get("geometry"))
                if cen:
                    code = (p.get("ISO_A2") or p.get("ADM0_A3") or "").strip().upper() or None
                    best = {"lat": cen[0], "lon": cen[1], "label": p.get("ADMIN") or p.get("NAME") or name, "code": code}
                    break
        if best: return best

    # 2) Alias + centroides de config (país)
    iso2 = COUNTRY_ALIASES.get(q)
    if not iso2:
        # intenta con nombres legibles del dict de centroides
        for k_iso2, (lat, lon, label) in COUNTRY_CENTROIDS.items():
            if q == (label or "").strip().lower():
                iso2 = k_iso2
                break
    if iso2 and iso2 in COUNTRY_CENTROIDS:
        lat, lon, label = COUNTRY_CENTROIDS[iso2]
        return {"lat": lat, "lon": lon, "label": label, "code": iso2}

    # 3) (Opcional) Geocoder online si geopy está instalado
    try:
        from geopy.geocoders import Nominatim
        geocoder = Nominatim(user_agent="intellive_pro")
        loc = geocoder.geocode(name, timeout=10)
        if loc:
            return {"lat": float(loc.latitude), "lon": float(loc.longitude), "label": name, "code": None}
    except Exception:
        pass

    return None

def _correction_form(row, countries):
    with st.form(f"fix_{row['id']}"):
        st.markdown("**✏️ Corregir ubicación**")
        tabs = st.tabs(["Por nombre", "Por lat/lon"])

        with tabs[0]:
            place_name = st.text_input("Lugar (país/ciudad/área)", value=row.get("ubicacion") or "")
            coln1, coln2 = st.columns(2)
            with coln1:
                save_as_area = st.checkbox("Guardar como área (país)", value=True)
            with coln2:
                radius_km = st.number_input("Radio km (opcional para área)", value=float(row.get("radius_km") or 0.0), min_value=0.0, step=10.0)
        with tabs[1]:
            lat=st.number_input("Latitud", value=float(row["latitude"]) if pd.notna(row["latitude"]) else 0.0, format="%.6f")
            lon=st.number_input("Longitud", value=float(row["longitude"]) if pd.notna(row["longitude"]) else 0.0, format="%.6f")
            label=st.text_input("Etiqueta (opcional)", value=row.get("ubicacion") or "")

        colf1, colf2 = st.columns([1,1])
        submit = colf1.form_submit_button("Guardar")
        if submit:
            # Prioridad: si puso nombre, usamos nombre; si no, lat/lon.
            if place_name.strip():
                hit = _resolve_place_by_name(place_name, countries)
                if not hit:
                    st.error("No pude resolver ese lugar. Agrega un alias en config.COUNTRY_ALIASES o provee lat/lon.")
                    return
                save_override(
                    int(row["id"]), float(hit["lat"]), float(hit["lon"]),
                    hit["label"], int(save_as_area),
                    (radius_km if save_as_area and radius_km>0 else None),
                    "country" if (save_as_area and hit.get("code")) else None,
                    hit.get("code")
                )
                st.success(f"Ubicación corregida: {hit['label']} ✓"); st.rerun()
            else:
                if not _valid_coords(lat,lon):
                    st.error("Coordenadas inválidas"); return
                save_override(
                    int(row["id"]), float(lat), float(lon),
                    label, 0, None, None, None
                )
                st.success("Ubicación corregida ✓"); st.rerun()

    if int(row.get("overridden") or 0)==1:
        if st.button("🗑️ Borrar corrección", key=f"del_{row['id']}"):
            delete_override(int(row["id"])); st.success("Corrección eliminada"); st.rerun()

# ========= Listado =========
def render_list(df: pd.DataFrame, countries):
    PAGE_SIZE=12
    total=len(df); total_pages=max(1, (total+PAGE_SIZE-1)//PAGE_SIZE)

    c1,c2,c3=st.columns([1,2,1])
    with c2:
        st.write("")
        st.markdown(f'<div class="headline"><h1>📰 Reportes</h1><span class="subtle">{total} items</span></div>', unsafe_allow_html=True)
    with c3:
        st.selectbox("Página", options=list(range(1,total_pages+1)),
                     key="page", index=min(st.session_state.page-1, total_pages-1))

    start=(st.session_state.page-1)*PAGE_SIZE; end=start+PAGE_SIZE
    page_df=df.iloc[start:end].reset_index(drop=True)

    cols=st.columns(3, gap="small")
    for i,row in page_df.iterrows():
        with cols[i%3]:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            title=row.get("ubicacion") or "Ubicación"
            badge=" ✅ Corregida" if int(row.get("overridden") or 0)==1 else ""
            st.markdown(f"<div class='title'>{title}{badge}</div>", unsafe_allow_html=True)

            meta=row.get("time_ago","")
            if pd.notna(row.get("confidence")): meta=f"{meta} · conf {row['confidence']:.2f}"
            if int(row.get("is_area") or 0)==1 and pd.notna(row.get("radius_km")):
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

            colb1,colb2 = st.columns([1,1])
            if colb1.button("📍 Centrar", key=f"center_{row['id']}"):
                st.session_state.selected_report={
                    "id": int(row["id"]),
                    "latitude": float(row["latitude"]) if pd.notna(row["latitude"]) else None,
                    "longitude": float(row["longitude"]) if pd.notna(row["longitude"]) else None,
                    "ubicacion": row.get("ubicacion") or "",
                    "text": row.get("text") or "",
                    "time_ago": row.get("time_ago") or "",
                    "media_count": int(row.get("media_count") or 0),
                    "is_area": int(row.get("is_area") or 0),
                    "admin_code": row.get("admin_code") or "",
                    "admin_kind": row.get("admin_kind") or ""
                }
                st.session_state.center_nonce += 1; st.rerun()
            if colb2.button("✏️ Corregir", key=f"fixbtn_{row['id']}"):
                st.session_state[f"show_fix_{row['id']}"]=True

            if st.session_state.get(f"show_fix_{row['id']}", False):
                _correction_form(row, countries)

            st.markdown("</div>", unsafe_allow_html=True)

# ========= Main =========
def main():
    setup_app()

    with st.sidebar:
        st.markdown("### ⚙️ Controles")
        show_all=st.checkbox("Mostrar todo el historial", value=st.session_state.show_all); st.session_state.show_all=show_all
        time_range = st.selectbox("Rango temporal", ["24h","3 días","30 días"] if not show_all else ["todo"], index=2 if not show_all else 0)
        q = st.text_input("Buscar texto…", value=st.session_state.last_query).strip()
        only_coords = st.checkbox("Solo con ubicación", value=st.session_state.only_coords); st.session_state.only_coords=only_coords
        auto_refresh = st.checkbox("Auto-actualizar cada 15s", value=st.session_state.auto_refresh_on); st.session_state.auto_refresh_on=auto_refresh

    if st.session_state.auto_refresh_on and st_autorefresh is not None:
        st_autorefresh(interval=15000, key="data_refresh")

    if q != st.session_state.last_query or time_range != st.session_state.last_range:
        st.session_state.page = 1
        st.session_state.last_query = q
        st.session_state.last_range = time_range

    df = load_data()

    # Filtro temporal robusto
    if not df.empty and time_range != "todo":
        days = 1 if time_range=="24h" else (3 if time_range=="3 días" else 30)
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)
        df = df[df["timestamp"].notna() & (df["timestamp"] >= cutoff)]

    # Texto
    if not df.empty and q:
        df = df[df["text"].str.contains(re.escape(q), case=False, na=False)]

    # Países (para hover y mejor inferencia)
    countries = load_countries_geojson(COUNTRIES_GEOJSON)
    lookup, pattern = build_country_lookup(countries)

    # Fallback de país por texto si faltan coords
    df = augment_with_text_country_inference(df, lookup, pattern)

    # Vistas
    df_for_list = df.copy()
    df_for_map = df.copy()
    if not df_for_map.empty:
        df_for_map = df_for_map.dropna(subset=["latitude","longitude"])
        df_for_map = df_for_map[(df_for_map["latitude"].between(-90,90)) & (df_for_map["longitude"].between(-180,180))]
    if st.session_state.only_coords:
        df_for_list = df_for_map

    st.markdown('<div class="headline"><h1>🌍 Mapa</h1></div>', unsafe_allow_html=True)
    cA,cB,cC,cD,cE = st.columns(5)

    total=len(df_for_list)
    with_media = int((df_for_list["media_count"]>0).sum()) if not df_for_list.empty else 0
    total_files = int(df_for_list["media_count"].sum()) if not df_for_list.empty else 0
    with_areas  = int((df_for_list["is_area"]==1).sum()) if not df_for_list.empty else 0
    with_points_map = len(df_for_map) if not df_for_map.empty else 0

    cA.metric("Reportes", total)
    cB.metric("Con media", with_media)
    cC.metric("Archivos totales", total_files)
    cD.metric("ÁREAS", with_areas)
    cE.metric("PUNTOS (en mapa)", with_points_map)

    render_map(df_for_map, st.session_state.get("selected_report"), countries)
    st.markdown("---")
    render_list(df_for_list, countries)

if __name__=="__main__":
    main()
