#!/usr/bin/env python3
import os
import re
import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Tuple, Optional

import humanize
import pandas as pd
import pydeck as pdk
import streamlit as st
from bs4 import BeautifulSoup

from config import MAP_CENTER, DEFAULT_ZOOM, DB_PATH, MEDIA_DIR

# ===================== Constantes UI =====================
PAGE_TITLE = "IntelLive Pro"
CARD_BG = "#0F1426"
CARD_BORDER = "#243145"
PRIMARY = "#FF4B4B"

IMG_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
VID_EXT = {".mp4", ".webm", ".mov", ".m4v"}
AUD_EXT = {".mp3", ".wav", ".ogg", ".m4a", ".aac"}

# ===================== Setup =====================
def setup_app() -> None:
    st.set_page_config(page_title=PAGE_TITLE, page_icon="🌍", layout="wide")

    Path(MEDIA_DIR).mkdir(parents=True, exist_ok=True)

    st.markdown(
        f"""
        <style>
        :root {{
          --bg: #0A0F24;
          --card: {CARD_BG};
          --border: {CARD_BORDER};
          --primary: {PRIMARY};
        }}
        .stApp {{ background: var(--bg); color: #e5e7eb; font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Arial; }}

        .headline {{
          display:flex; align-items:center; gap:12px; margin-bottom:8px;
        }}
        .headline h1 {{ margin:0; color:#fff; font-size:1.4rem; }}
        .subtle {{ color:#94a3b8; font-size:0.9rem }}

        .card {{
          background: var(--card);
          border: 1px solid var(--border);
          border-radius: 10px;
          padding: 12px;
          display:flex;
          flex-direction:column;
          gap:8px;
          height: 100%;
        }}
        .card .title {{ color: var(--primary); font-weight: 600; font-size: 0.95rem; }}
        .card .meta {{ color:#9ca3af; font-size:0.8rem }}
        .card .text {{ color:#e5e7eb; font-size:0.9rem; line-height:1.35; max-height:7.5em; overflow:hidden; }}

        .btnrow {{ display:flex; gap:8px; }}
        .thumb {{ width:100%; border-radius:8px; max-height:180px; object-fit:cover; border:1px solid var(--border); }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    if "selected_report" not in st.session_state:
        st.session_state.selected_report = None

    if "page" not in st.session_state:
        st.session_state.page = 1

# ===================== Utils =====================
def clean_text(text: Optional[str]) -> str:
    if not text:
        return ""
    try:
        clean = BeautifulSoup(text, "lxml").get_text()
    except Exception:
        clean = BeautifulSoup(text, "html.parser").get_text()
    clean = re.sub(r"@\w+\s?|https?://\S+|[^\w\s.,!?¿¡áéíóúÁÉÍÓÚñÑ\-]", " ", clean)
    return " ".join(clean.strip().split())

def _valid_coords(lat, lon) -> bool:
    try:
        lat = float(lat); lon = float(lon)
        return -90 <= lat <= 90 and -180 <= lon <= 180
    except Exception:
        return False

def time_ago(ts: pd.Timestamp) -> str:
    now = datetime.now(timezone.utc)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return humanize.naturaltime(now - ts).replace(" ago", "")

def split_media_paths(media_paths: Optional[str]) -> List[str]:
    if not media_paths:
        return []
    parts = [p.strip() for p in str(media_paths).split(",") if p.strip()]
    out = []
    for p in parts:
        base = os.path.basename(p)
        full = p if os.path.isabs(p) else os.path.join(MEDIA_DIR, base)
        out.append(full)
    seen, uniq = set(), []
    for p in out:
        if p not in seen:
            uniq.append(p); seen.add(p)
    return [p for p in uniq if os.path.exists(p)]

def pick_preview(paths: List[str]) -> Tuple[Optional[str], Optional[str]]:
    if not paths: return None, None
    imgs = [p for p in paths if os.path.splitext(p)[1].lower() in IMG_EXT]
    if imgs: return "image", imgs[0]
    vids = [p for p in paths if os.path.splitext(p)[1].lower() in VID_EXT]
    if vids: return "video", vids[0]
    auds = [p for p in paths if os.path.splitext(p)[1].lower() in AUD_EXT]
    if auds: return "audio", auds[0]
    return "other", paths[0]

# ===================== Carga de datos =====================
@st.cache_data(ttl=15)
def load_data(days: int = 30) -> pd.DataFrame:
    base_cols = [
        "id","text","timestamp","media_paths",
        "latitude","longitude","ubicacion","confidence"
    ]
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql(
            f"""
            SELECT
                m.id,
                m.text,
                m.timestamp,
                m.media_paths,
                l.lat AS latitude,
                l.lon AS longitude,
                l.location_name AS ubicacion,
                l.confidence AS confidence
            FROM messages m
            LEFT JOIN locations l ON m.id = l.message_id
            WHERE m.timestamp > datetime('now', '-{days} days')
            ORDER BY m.id ASC, l.confidence DESC
            """,
            conn,
        )
    if df.empty:
        return pd.DataFrame(columns=base_cols)

    df = (
        df.sort_values(["id", "confidence"], ascending=[True, False])
          .groupby("id", as_index=False)
          .first()
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df = df[df["timestamp"].notna()]
    df["text"] = df["text"].fillna("").map(clean_text)
    df["time_ago"] = df["timestamp"].map(time_ago)
    df["media_files"] = df["media_paths"].map(split_media_paths)
    df["media_count"] = df["media_files"].map(len)
    return df

# ===================== Geolocalización bajo demanda =====================
PHRASE_GEO_HINTS = [
    (r"\bsahrawi refugee camps?\b", (27.70, -8.15, "Sahrawi Refugee Camps (Tindouf, Algeria)")),
    (r"\btindouf\b", (27.71, -8.15, "Tindouf, Algeria")),
    (r"\bgaza\b", (31.40, 34.30, "Gaza")),
    (r"\bkhan\s*younis\b", (31.34, 34.30, "Khan Younis")),
    (r"\brafah\b", (31.29, 34.25, "Rafah")),
    (r"\bsopot\b", (42.65, 24.75, "Sopot, Bulgaria")),
    (r"\bkyiv|kiev\b", (50.45, 30.52, "Kyiv, Ukraine")),
]

def infer_location_from_text(text: str) -> Optional[dict]:
    t = clean_text(text).lower()
    for pattern, (lat, lon, name) in PHRASE_GEO_HINTS:
        if re.search(pattern, t):
            return {"latitude": lat, "longitude": lon, "ubicacion": name}
    return None

# ===================== Mapa =====================
def render_map(points_df: pd.DataFrame, selected: Optional[dict]) -> None:
    pts = points_df.copy()
    if not pts.empty:
        pts["latitude"] = pd.to_numeric(pts["latitude"], errors="coerce")
        pts["longitude"] = pd.to_numeric(pts["longitude"], errors="coerce")
        pts = pts.dropna(subset=["latitude", "longitude"])

    if selected and _valid_coords(selected.get("latitude"), selected.get("longitude")):
        lat0 = float(selected["latitude"]); lon0 = float(selected["longitude"]); zoom = 13
    elif not pts.empty:
        lat0 = float(pts["latitude"].mean()); lon0 = float(pts["longitude"].mean()); zoom = 3.5
    else:
        lat0, lon0 = MAP_CENTER; zoom = DEFAULT_ZOOM

    view_state = pdk.ViewState(latitude=lat0, longitude=lon0, zoom=zoom, pitch=45)

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=pts if not pts.empty else pd.DataFrame(columns=["longitude","latitude"]),
        get_position=["longitude", "latitude"],
        get_radius=250,
        radius_scale=1,
        radius_min_pixels=6,
        radius_max_pixels=80,
        pickable=True,
        stroked=True,
    )

    st.pydeck_chart(
        pdk.Deck(
            layers=[layer],
            initial_view_state=view_state,
            map_style="mapbox://styles/mapbox/satellite-streets-v11",
            tooltip={
                "html": """
                    <div style="background:#0A0F24dd;border:2px solid #FF4B4B;border-radius:8px;padding:12px;max-width:420px">
                        <div style="color:#FF4B4B;font-weight:700;font-size:1.05rem">{ubicacion}</div>
                        <div style="color:#E0F2FE;margin:8px 0;font-size:0.9rem">{text}</div>
                        <div style="color:#94A3B8;font-size:0.8rem">🕒 {time_ago} · 📎 {media_count} archivos</div>
                    </div>
                """
            },
        )
    )

# ===================== Listado =====================
def render_list(df: pd.DataFrame) -> None:
    PAGE_SIZE = 12
    total = len(df)
    total_pages = max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE)

    c1, c2, c3 = st.columns([1,2,1])
    with c2:
        st.write("")
        st.markdown(
            f'<div class="headline"><h1>📰 Reportes</h1><span class="subtle">{total} items</span></div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.selectbox(
            "Página",
            options=list(range(1, total_pages + 1)),
            key="page",
            index=min(st.session_state.page - 1, total_pages - 1),
        )

    start = (st.session_state.page - 1) * PAGE_SIZE
    end = start + PAGE_SIZE
    page_df = df.iloc[start:end].reset_index(drop=True)

    cols = st.columns(3, gap="small")
    for i, row in page_df.iterrows():
        with cols[i % 3]:
            with st.container():
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown(f"<div class='title'>{row.get('ubicacion') or 'Sin ubicación'}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='meta'>{row.get('time_ago', '')}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='text'>{(row.get('text') or '')[:500]}</div>", unsafe_allow_html=True)

                lat, lon = row.get("latitude"), row.get("longitude")
                has_coords = _valid_coords(lat, lon)
                colb1, colb2, colb3 = st.columns([1, 1, 1])
                if colb1.button("📍 Centrar", key=f"center_{row['id']}", disabled=not has_coords, use_container_width=True):
                    st.session_state.selected_report = {
                        "id": int(row["id"]),
                        "latitude": float(lat),
                        "longitude": float(lon),
                        "ubicacion": row.get("ubicacion") or "",
                        "text": row.get("text") or "",
                        "time_ago": row.get("time_ago") or "",
                        "media_count": int(row.get("media_count") or 0),
                    }

                if colb2.button("🧭 Inferir", key=f"infer_{row['id']}", use_container_width=True):
                    hint = infer_location_from_text(row.get("text") or "")
                    if hint:
                        st.session_state.selected_report = {
                            "id": int(row["id"]),
                            "latitude": hint["latitude"],
                            "longitude": hint["longitude"],
                            "ubicacion": hint["ubicacion"],
                            "text": row.get("text") or "",
                            "time_ago": row.get("time_ago") or "",
                            "media_count": int(row.get("media_count") or 0),
                        }
                        st.toast(f"Ubicación sugerida: {hint['ubicacion']}", icon="🧭")
                    else:
                        st.toast("Sin sugerencia confiable para este texto.", icon="⚠️")

                colb3.write("")
                media_files = row.get("media_files") or []
                kind, path = pick_preview(media_files)
                if kind == "image" and path:
                    st.image(path, use_container_width=True)
                elif kind == "video" and path:
                    st.video(path)
                elif kind == "audio" and path:
                    st.audio(path)
                elif path:
                    st.caption(f"📎 {os.path.basename(path)}")

                st.markdown("</div>", unsafe_allow_html=True)

# ===================== Main =====================
def main() -> None:
    setup_app()

    with st.sidebar:
        st.markdown("### ⚙️ Controles")
        time_range = st.selectbox("Rango temporal", ["24h", "3 días", "30 días", "todo"], index=2)
        q = st.text_input("Buscar texto…").strip().lower()

    days = 30
    if time_range == "24h": days = 1
    elif time_range == "3 días": days = 3
    elif time_range == "todo": days = 3650

    df = load_data(days=days)

    if not df.empty and q:
        df = df[df["text"].str.lower().str.contains(q, na=False)]

    # 🔧 Ensure media_count is always available
    if "media_count" not in df.columns:
        df["media_files"] = df["media_paths"].map(split_media_paths)
        df["media_count"] = df["media_files"].map(len)

    st.markdown('<div class="headline"><h1>🌍 Mapa</h1></div>', unsafe_allow_html=True)
    cA, cB, cC, cD = st.columns(4)
    cA.metric("Reportes", len(df))
    cB.metric("Con media", int(df["media_count"].sum()))
    cC.metric("Con ubicación", int(df[["latitude","longitude"]].dropna().shape[0]))
    cD.metric("Rango días", days if days < 3650 else "Todos")

    render_map(df, st.session_state.get("selected_report"))

    st.markdown("---")
    render_list(df)

if __name__ == "__main__":
    main()
