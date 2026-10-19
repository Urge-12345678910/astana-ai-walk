"""
Прогулки по Астане: интерфейс на Streamlit.
Запуск:  streamlit run app.py
"""
import os

import folium
import streamlit as st
from streamlit_folium import st_folium

from src.rag_2gis_engine import (ASTANA_CENTER, DEFAULT_MODEL, NEAR_RADIUS_M,
                                 DGisError, get_recommendations)

st.set_page_config(page_title="Прогулки по Астане", layout="wide", initial_sidebar_state="expanded")

# --------------------------------------------------------------------------- #
# Оформление
# Палитра: холодный свет и стекло Астаны, солнечный акцент только в фокусе поля.
#   бумага #EDF2F4, панель #E0E8EB, чернила #13232E, сталь #586A75,
#   линия #C4D1D7, небо #007C99, солнце #E0A100
# Шрифты: Unbounded (заголовки, широкая геометрия) + Golos Text (текст).
# --------------------------------------------------------------------------- #
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Golos+Text:wght@400;500;600&family=Unbounded:wght@500;600&display=swap');

:root{
  --paper:#EDF2F4; --panel:#E0E8EB; --ink:#13232E; --steel:#586A75;
  --line:#C4D1D7; --sky:#007C99; --sky-deep:#055F75; --sun:#E0A100;
}
html, body, .stApp, [data-testid="stAppViewContainer"]{
  background:var(--paper); color:var(--ink);
  font-family:'Golos Text', system-ui, sans-serif;
}
[data-testid="stDecoration"], footer, #MainMenu{display:none !important;}
.block-container{max-width:1320px; padding:2.4rem 2.6rem 4rem;}

/* Боковая панель */
[data-testid="stSidebar"]{background:var(--panel); border-right:1px solid var(--line);}
.side-title{font-family:'Unbounded',sans-serif; font-weight:500; font-size:.95rem; margin:.2rem 0 1.2rem;}
.side-h{font-weight:600; font-size:.92rem; margin:1.6rem 0 .5rem; padding-top:1.1rem; border-top:1px solid var(--line);}
[data-testid="stSidebar"] label p{font-size:.84rem; color:var(--steel); font-weight:500;}
[data-testid="stSidebar"] .stCaption, [data-testid="stSidebar"] [data-testid="stCaptionContainer"]{color:var(--steel);}

/* Шапка с силуэтом города: единственный выразительный элемент страницы */
.mast{position:relative; min-height:15.5rem; padding-bottom:9rem; margin:0 0 2rem; border-bottom:2px solid var(--ink);}
.mast h1{
  position:relative; z-index:1;
  font-family:'Unbounded',sans-serif; font-weight:600; letter-spacing:-0.02em;
  font-size:clamp(1.8rem, 3.4vw, 3rem); line-height:1.06; margin:0 0 .8rem; padding:0;
}
.mast p{position:relative; z-index:1; color:var(--steel); max-width:50ch; margin:0; font-size:1.02rem; line-height:1.55;}
.sk{position:absolute; right:0; bottom:0; width:min(100%,1200px); height:auto; display:block;}
.sk .a{fill:#C9D6DB;} .sk .b{fill:#B3C5CD;} .sk .c{fill:#007C99; opacity:.85;} .sk .g{fill:#E0A100;} .sk .p{fill:#EDF2F4;}
.sk .l{fill:none; stroke:#EDF2F4; stroke-width:1.5; opacity:.8;}

/* Поля ввода */
[data-baseweb="input"], [data-baseweb="base-input"]{background:#fff !important; border-radius:2px !important;}
[data-baseweb="input"]{border:1px solid var(--line) !important; box-shadow:none !important; outline:none !important;}
[data-baseweb="input"]:focus-within, [data-baseweb="base-input"]:focus-within{
  border-color:var(--sun) !important; box-shadow:0 0 0 2px rgba(224,161,0,.3) !important;}
[data-testid="stSlider"] [role="slider"]{background:var(--sky) !important; box-shadow:none !important;}
input{color:var(--ink) !important; font-family:'Golos Text',sans-serif !important;}
[data-testid="stSidebar"] input{font-size:.95rem !important; padding:.55rem .75rem !important;}
[data-testid="stMain"] .stTextInput input, section.main .stTextInput input{font-size:1.1rem !important; padding:.85rem 1rem !important;}
[data-testid="stForm"]{border:0 !important; padding:0 !important;}
.stTextInput label p{font-size:.9rem; color:var(--steel); font-weight:500;}

/* Кнопки */
[data-testid="stFormSubmitButton"] button{
  background:var(--sky); color:#fff; border:0; border-radius:2px;
  padding:.7rem 1.7rem; font-weight:500;
}
[data-testid="stFormSubmitButton"] button p{color:#fff;}
[data-testid="stFormSubmitButton"] button:hover{background:var(--sky-deep); color:#fff;}
.stButton button{
  width:100%; background:transparent; border:1px solid var(--line); border-radius:2px;
  color:var(--ink); font-weight:400; text-align:left; justify-content:flex-start; padding:.55rem .8rem;
}
.stButton button:hover{border-color:var(--sky); color:var(--sky);}
.stButton button p{font-size:.9rem;}

/* Заголовки колонок */
.colhead{font-family:'Unbounded',sans-serif; font-weight:500; font-size:.9rem; margin:1.6rem 0 .8rem;}
.found{color:var(--steel); font-size:.92rem; margin:0 0 .3rem;}
.hint{color:var(--steel); max-width:52ch; line-height:1.6;}

/* Текст рекомендаций (разметка приходит от Gemini) */
.stMarkdown h2{
  font-family:'Unbounded',sans-serif; font-weight:500; font-size:.95rem !important; color:var(--steel);
  margin:2.2rem 0 .4rem; padding:0 0 .5rem; border-bottom:1px solid var(--line);
}
.stMarkdown h3{
  font-family:'Golos Text',sans-serif; font-weight:600; font-size:1.2rem !important;
  border-left:4px solid var(--sky); padding:.1rem 0 .1rem .8rem; margin:1.6rem 0 .5rem;
}
.stMarkdown p, .stMarkdown li{line-height:1.6; font-size:1rem; max-width:64ch;}
.stMarkdown strong{font-weight:600;}

/* Карта */
iframe{border:1px solid var(--line) !important; border-top:3px solid var(--sky) !important; border-radius:2px;}
.legend{font-size:.85rem; color:var(--steel); margin-top:.7rem; display:flex; gap:1.4rem; flex-wrap:wrap;}
.dot{display:inline-block; width:.7rem; height:.7rem; border-radius:50%; margin-right:.4rem; vertical-align:-1px;}
.dot.red{background:#D62839;} .dot.green{background:#2E9E5B;}
.dash{display:inline-block; width:1.1rem; border-top:2px dashed var(--sky); margin-right:.4rem; vertical-align:3px;}

@media (max-width: 900px){ .block-container{padding:1.4rem 1rem 3rem;} }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
# Состояние
# --------------------------------------------------------------------------- #
# Координаты живут в ключах виджетов. Клик по карте кладётся в pending_coords
# и применяется в начале следующего прогона (менять ключ уже созданного
# виджета Streamlit не разрешает).
st.session_state.setdefault("lat_input", ASTANA_CENTER[0])
st.session_state.setdefault("lon_input", ASTANA_CENTER[1])
st.session_state.setdefault("last_click", None)
st.session_state.setdefault("result", None)
st.session_state.setdefault("query", "")

if "pending_coords" in st.session_state:
    st.session_state.lat_input, st.session_state.lon_input = st.session_state.pop("pending_coords")


def use_example(text: str) -> None:
    """Подставляет пример в поле и сразу запускает поиск."""
    st.session_state.query = text
    st.session_state.run_search = True


# --------------------------------------------------------------------------- #
# Боковая панель
# --------------------------------------------------------------------------- #
sb = st.sidebar
sb.markdown('<div class="side-title">Параметры</div>', unsafe_allow_html=True)
gemini_key = sb.text_input("Gemini API Key", type="password",
                           value=os.getenv("GEMINI_API_KEY", ""))
dgis_key = sb.text_input("2GIS API Key", type="password",
                         value=os.getenv("DGIS_API_KEY", ""),
                         help="Ключ из кабинета platform.2gis.ru")
model_name = sb.text_input("Модель Gemini", value=DEFAULT_MODEL)

sb.markdown('<div class="side-h">Точка поиска</div>', unsafe_allow_html=True)
user_lat = sb.number_input("Широта (Lat)", key="lat_input", min_value=50.9,
                           max_value=51.4, format="%.6f", step=0.001)
user_lon = sb.number_input("Долгота (Lon)", key="lon_input", min_value=71.2,
                           max_value=71.7, format="%.6f", step=0.001)
sb.caption("Координаты можно не вводить: нажмите на карту, и точка переместится.")

sb.markdown('<div class="side-h">Время</div>', unsafe_allow_html=True)
time_available = sb.slider("Сколько у вас времени, минут", 15, 180, 60, step=15)

# --------------------------------------------------------------------------- #
# Шапка и поиск
# --------------------------------------------------------------------------- #
st.markdown(
    '<div class="mast"><h1>Прогулки по Астане</h1>'
    '<p>Напишите, чего хотите и сколько у вас времени. Подберём места рядом '
    'с выбранной точкой по данным 2ГИС и объясним, почему они подходят.</p>'
    '<svg class="sk" viewBox="0 0 1200 140" preserveAspectRatio="xMaxYMax meet" aria-hidden="true"><rect class="b" x="70" y="112" width="120" height="28"/><rect class="b" x="100" y="100" width="60" height="12"/><path class="b" d="M105 100 Q130 66 155 100 Z"/><rect class="b" x="129" y="46" width="2" height="26"/><rect class="a" x="312" y="128" width="36" height="12"/><path class="b" d="M318 140 L326 76 L334 76 L342 140 Z"/><rect class="b" x="329" y="26" width="2" height="12"/><circle class="g" cx="330" cy="58" r="22"/><circle class="p" cx="330" cy="58" r="9"/><path class="a" d="M480 140 L560 82 Q600 55 622 6 L626 6 Q648 55 688 82 L768 140 Z"/><path class="l" d="M624 6 L596 140 M624 6 L652 140 M624 6 L560 140 M624 6 L690 140"/><circle class="c" cx="900" cy="88" r="52"/><ellipse class="l" cx="900" cy="88" rx="52" ry="14"/><ellipse class="l" cx="900" cy="88" rx="22" ry="52"/><rect class="b" x="1020" y="118" width="80" height="22"/><path class="b" d="M1035 118 Q1060 82 1085 118 Z"/><rect class="b" x="1008" y="62" width="5" height="78"/><rect class="b" x="1016" y="80" width="5" height="60"/><rect class="b" x="1099" y="80" width="5" height="60"/><rect class="b" x="1107" y="62" width="5" height="78"/><rect class="b" x="1135" y="70" width="16" height="70"/><rect class="b" x="1155" y="50" width="18" height="90"/><rect class="b" x="1163" y="28" width="2" height="24"/><rect class="b" x="1177" y="84" width="20" height="56"/></svg></div>',
    unsafe_allow_html=True)

with st.form("search"):
    st.text_input("Что вы ищете и какие у вас предпочтения?", key="query",
                  placeholder="Например: тихое место с кофе и розеткой")
    submitted = st.form_submit_button("Подобрать места")

examples = ["Хочу погулять с друзьями", "Тихое место с кофе и розеткой",
            "Компьютерный клуб", "Еду с ребёнком, что посмотреть рядом"]
for col, text in zip(st.columns(len(examples)), examples):
    col.button(text, key=f"ex_{text}", on_click=use_example, args=(text,))

run = submitted or st.session_state.pop("run_search", False)
query = st.session_state.query.strip()

if run:
    if not gemini_key:
        st.warning("Введите Gemini API Key в боковой панели.")
    elif not query:
        st.warning("Опишите, что вы ищете.")
    else:
        with st.spinner("Ищу места и подбираю варианты"):
            try:
                res = get_recommendations(
                    query=query, lat=user_lat, lon=user_lon,
                    minutes=time_available, gemini_key=gemini_key,
                    dgis_key=dgis_key or None, model=model_name)
                res["origin"] = (user_lat, user_lon)
                st.session_state.result = res
            except DGisError as e:
                st.session_state.result = None
                st.error(str(e))
            except Exception as e:  # noqa: BLE001
                st.session_state.result = None
                st.error(f"Не удалось выполнить поиск: {type(e).__name__}: {e}")

result = st.session_state.result
col_left, col_right = st.columns([5, 6], gap="large")

# --------------------------------------------------------------------------- #
# Левая колонка: рекомендации
# --------------------------------------------------------------------------- #
with col_left:
    st.markdown('<div class="colhead">Рекомендации</div>', unsafe_allow_html=True)
    if result:
        st.markdown(
            f'<p class="found">Искали: {", ".join(result["terms"])}. '
            f'Рядом найдено {len(result["near"])}, в других районах {len(result["far"])}.</p>',
            unsafe_allow_html=True)
        st.markdown(result["text"])
    else:
        st.markdown(
            '<p class="hint">Здесь появятся места с объяснением, что там делать и сколько '
            'времени это займёт. Выберите пример выше или напишите свой запрос.</p>',
            unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
# Правая колонка: карта
# --------------------------------------------------------------------------- #
with col_right:
    st.markdown('<div class="colhead">Карта</div>', unsafe_allow_html=True)

    m = folium.Map(location=[user_lat, user_lon], zoom_start=12 if result else 13,
                   tiles="OpenStreetMap", control_scale=True)
    # Приглушаем подложку, чтобы цветные точки читались лучше
    m.get_root().header.add_child(folium.Element(
        "<style>.leaflet-tile-pane{filter:grayscale(.85) contrast(.92) brightness(1.06);}</style>"))

    # Граница «близкой» зоны
    folium.Circle([user_lat, user_lon], radius=NEAR_RADIUS_M, color="#007C99",
                  weight=1.5, dash_array="6 6", fill=False,
                  tooltip="Граница ближней зоны, 4,5 км",
                  bubbling_mouse_events=False).add_to(m)

    # Красная точка — место поиска
    folium.CircleMarker([user_lat, user_lon], radius=9, color="#FFFFFF", weight=3,
                        fill=True, fill_color="#D62839", fill_opacity=1,
                        tooltip="Точка поиска",
                        bubbling_mouse_events=False).add_to(m)

    # Зелёные точки — найденные места
    if result:
        for group, label in ((result["near"], "Рядом"), (result["far"], "В другом районе")):
            for p in group:
                rating = f"<br>Рейтинг: {p.rating}" if p.rating else ""
                popup = folium.Popup(
                    f"<b>{p.name}</b><br>{p.address}{rating}"
                    f"<br>{label}: {p.distance_m/1000:.1f} км, пешком {p.walk_min} мин",
                    max_width=260)
                folium.CircleMarker([p.lat, p.lon], radius=7, color="#FFFFFF", weight=2,
                                    fill=True, fill_color="#2E9E5B", fill_opacity=1,
                                    tooltip=p.name, popup=popup,
                                    bubbling_mouse_events=False).add_to(m)

    map_data = st_folium(m, height=580, use_container_width=True,
                         returned_objects=["last_clicked"], key="main_map")

    st.markdown(
        '<div class="legend">'
        '<span><span class="dot red"></span>Точка поиска</span>'
        '<span><span class="dot green"></span>Найденные места</span>'
        '<span><span class="dash"></span>Граница 4,5 км</span></div>',
        unsafe_allow_html=True)

    # Клик по карте: обновляем координаты и перезапускаем скрипт
    click = (map_data or {}).get("last_clicked")
    if click:
        new_point = (round(click["lat"], 6), round(click["lng"], 6))
        if new_point != st.session_state.last_click:
            st.session_state.last_click = new_point
            st.session_state.pending_coords = new_point
            st.rerun()