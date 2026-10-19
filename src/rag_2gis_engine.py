"""
Бэкенд ИИ-помощника по прогулкам в Астане.

Пайплайн (RAG):
  1. Разбор разговорного запроса -> поисковые термины для 2GIS.
  2. Retrieval: 2GIS Places API (близкие места + альтернативы по городу).
  3. Фильтрация мусора, подсчёт расстояний, разделение на near / far.
  4. Generation: Gemini выбирает лучшее ИЗ ПОЛУЧЕННЫХ данных и пишет ответ.
"""
from __future__ import annotations

import json
import math
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import TimeoutError as FuturesTimeout
from dataclasses import dataclass, field, asdict
from typing import Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import google.generativeai as genai  # старый пакет google-generativeai

# --------------------------------------------------------------------------- #
# Константы
# --------------------------------------------------------------------------- #
DGIS_URL = "https://catalog.api.2gis.com/3.0/items"
ASTANA_CENTER = (51.1282, 71.4304)      # (lat, lon) — Байтерек
NEAR_RADIUS_M = 4500                    # "близко" — до 4.5 км
FAR_RADIUS_M = 25000                    # поиск альтернатив по всему городу
DEFAULT_MODEL = "gemini-3.8-flash"      # можно сменить в сайдбаре

# Разговорные фразы -> поисковые рубрики 2GIS.
# Правила ключей: обычный ключ совпадает с НАЧАЛОМ слова ("дет" -> "детей"),
# ключ с "=" — только слово целиком ("=бар" не сработает на "барбекю").
# Более специфичные категории стоят выше.
QUERY_MAP: list[tuple[tuple[str, ...], list[str]]] = [
    (("компьютерн", "киберспорт", "кибер", "геймерск", "игровой клуб", "=пк"),
     ["компьютерный клуб", "киберклуб", "игровой клуб"]),
    (("настольн", "настолк"),
     ["настольные игры", "антикафе", "игровой клуб"]),
    (("боулинг", "бильярд", "квест"),
     ["боулинг", "бильярд", "квест"]),
    (("мини-футбол", "футбол"),
     ["футбольное поле", "спортивный комплекс", "стадион"]),
    (("баскетбол",), ["баскетбольная площадка", "спортивный комплекс"]),
    (("плаван", "бассейн"), ["бассейн"]),
    (("спорт", "тренировк", "потренироваться", "бегать", "пробежк"),
     ["спортивный комплекс", "парк", "фитнес-клуб"]),
    (("погулять", "прогулк", "гулять", "прогуляться", "свежий воздух"),
     ["парк", "сквер", "набережная"]),
    (("учиться", "учёб", "учеб", "работать", "поработать", "розетк", "ноутбук"),
     ["коворкинг", "библиотека", "кофейня"]),
    (("кофе", "тихое место"), ["кофейня"]),
    (("ребён", "ребен", "дет", "с детьми"),
     ["детская площадка", "детский центр", "музей"]),
    (("музей", "выставк", "искусств"), ["музей", "выставочный зал", "галерея"]),
    (("поесть", "перекус", "голод", "обед", "ужин", "ресторан"),
     ["ресторан", "кафе"]),
    (("ночь", "ночн", "поздно", "=бар", "=бары", "дискотек", "караоке"),
     ["ночной клуб", "бар", "караоке"]),
    (("смотров", "вид на город", "фотосесс"),
     ["смотровая площадка", "парк"]),
    (("турист", "достопримечательност", "посмотреть"),
     ["достопримечательность", "музей", "памятник"]),
    (("кино", "фильм"), ["кинотеатр"]),
]

# Мусорные объекты: магазины стройматериалов, электротоваров, автозапчастей и т.п.
JUNK_KEYWORDS = (
    "строительн", "стройматериал", "стройтовар", "электротовар", "электроинструмент",
    "электроника", "бытовая техника", "автозапчаст", "запчаст", "автосервис",
    "шиномонтаж", "автомойк", "сантехник", "хозтовар", "хозяйственные товары",
    "ломбард", "инструмент", "крепёж", "крепеж", "лакокрасочн", "отделочные",
    "мебельный магазин", "оптовая", "склад", "ритуал", "автоаксессуар",
)


# --------------------------------------------------------------------------- #
# Модель данных
# --------------------------------------------------------------------------- #
@dataclass
class Place:
    id: str
    name: str
    lat: float
    lon: float
    address: str = ""
    rubrics: list[str] = field(default_factory=list)
    rating: Optional[float] = None
    reviews: Optional[int] = None
    schedule: str = ""
    distance_m: float = 0.0

    @property
    def walk_min(self) -> int:
        return max(1, round(self.distance_m / 1000 / 5 * 60))   # 5 км/ч

    @property
    def car_min(self) -> int:
        return max(1, round(self.distance_m / 1000 / 25 * 60))  # ~25 км/ч по городу


# --------------------------------------------------------------------------- #
# Вспомогательные функции
# --------------------------------------------------------------------------- #
def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Расстояние между двумя точками по сфере, в метрах."""
    r = 6371000
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _kw_match(kw: str, text: str) -> bool:
    """Совпадение ключа: по началу слова, либо по слову целиком (префикс '=')."""
    if kw.startswith("="):
        return re.search(r"(?<!\w)" + re.escape(kw[1:]) + r"(?!\w)", text) is not None
    return re.search(r"(?<!\w)" + re.escape(kw), text) is not None


def interpret_query(query: str) -> tuple[list[str], bool]:
    """Переводит разговорный запрос в поисковые термины 2GIS.
    Возвращает (термины, распознан_ли_запрос)."""
    q = query.lower()
    terms: list[str] = []
    for keywords, mapped in QUERY_MAP:
        if any(_kw_match(k, q) for k in keywords):
            for t in mapped:
                if t not in terms:
                    terms.append(t)
    if terms:
        return terms[:5], True
    return [query.strip()], False      # не распознали — ищем как есть


# --- Проверка релевантности: подходит ли найденное место под поисковый термин ---
_GENERIC_STEMS = {"клуб", "центр", "компл", "зал", "площа", "пол"}


def _stem(word: str) -> str:
    n = len(word)
    if n <= 3:
        return word
    if n <= 5:
        return word[:-1] if word[-1] in "аяеоыиьй" else word
    return word[:max(4, n - 3)]


def is_relevant(place: "Place", term: str) -> bool:
    """Место подходит, если его название/рубрики содержат основу значимого слова термина.
    Общие слова (клуб, центр, зал...) не считаются, если есть более specific слово."""
    stems = [_stem(w) for w in re.findall(r"\w+", term.lower()) if len(w) >= 3]
    if not stems:
        return True
    key = [x for x in stems if x not in _GENERIC_STEMS] or stems
    hay = re.sub(r"\W", "", (place.name + " " + " ".join(place.rubrics)).lower())
    return any(k in hay for k in key)


def is_junk(name: str, rubrics: list[str]) -> bool:
    text = (name + " " + " ".join(rubrics)).lower()
    return any(k in text for k in JUNK_KEYWORDS)


def _format_schedule(schedule: dict | None) -> str:
    """Сжимает расписание 2GIS в короткую строку. Пусто, если данных нет."""
    if not schedule:
        return ""
    if schedule.get("is_24x7"):
        return "круглосуточно"
    days = [("Mon", "Пн"), ("Tue", "Вт"), ("Wed", "Ср"), ("Thu", "Чт"),
            ("Fri", "Пт"), ("Sat", "Сб"), ("Sun", "Вс")]
    parts = []
    for key, label in days:
        wh = (schedule.get(key) or {}).get("working_hours") or []
        if wh:
            parts.append(f"{label} {wh[0].get('from')}–{wh[0].get('to')}")
    return ", ".join(parts)


def _parse_item(item: dict, user_lat: float, user_lon: float) -> Optional[Place]:
    pt = item.get("point")
    if not pt:
        return None
    rubrics = [r.get("name", "") for r in item.get("rubrics", [])]
    name = item.get("name") or item.get("full_name") or ""
    if not name or is_junk(name, rubrics):
        return None
    reviews = item.get("reviews") or {}
    return Place(
        id=str(item.get("id")),
        name=name,
        lat=pt["lat"],
        lon=pt["lon"],
        address=item.get("address_name", ""),
        rubrics=rubrics,
        rating=reviews.get("general_rating"),
        reviews=reviews.get("general_review_count"),
        schedule=_format_schedule(item.get("schedule")),
        distance_m=haversine_m(user_lat, user_lon, pt["lat"], pt["lon"]),
    )


# --------------------------------------------------------------------------- #
# Retrieval: 2GIS
# --------------------------------------------------------------------------- #
class DGisError(RuntimeError):
    pass


def _make_session() -> requests.Session:
    """Сессия с авто-повторами: переживает кратковременные обрывы соединения."""
    retry = Retry(total=2, connect=2, read=1, backoff_factor=0.5,
                  status_forcelist=(429, 500, 502, 503, 504),
                  allowed_methods=("GET",))
    sess = requests.Session()
    sess.mount("https://", HTTPAdapter(max_retries=retry))
    return sess


_SESSION = _make_session()


def _dgis_request(params: dict) -> list[dict]:
    """Один запрос к 2GIS. При ошибке расширенных полей повторяет с базовыми."""
    full_fields = ("items.point,items.address_name,items.rubrics,"
                   "items.reviews,items.schedule")
    for fields in (full_fields, "items.point,items.address_name,items.rubrics"):
        try:
            resp = _SESSION.get(DGIS_URL, params={**params, "fields": fields},
                                timeout=(6, 15))
        except requests.RequestException as e:
            # Не показываем исходное сообщение: в нём URL с API-ключом
            raise DGisError(f"Нет связи с 2GIS ({type(e).__name__}). "
                            f"Проверьте интернет/VPN и попробуйте ещё раз.") from None
        try:
            data = resp.json()
        except ValueError:
            raise DGisError(f"2GIS вернул не-JSON (HTTP {resp.status_code})")
        meta = data.get("meta", {})
        code = meta.get("code", resp.status_code)
        if code == 200:
            return data.get("result", {}).get("items", [])
        msg = (meta.get("error") or {}).get("message", "неизвестная ошибка")
        # 2GIS отдаёт 404 при "ничего не найдено" — это не ошибка
        if code == 404:
            return []
        # нет прав на расширенные поля — пробуем базовые
        if code in (400, 403) and fields == full_fields:
            continue
        raise DGisError(f"2GIS API: {code} — {msg}")
    return []


def search_2gis(term: str, lat: float, lon: float, radius: int, api_key: str,
                by_distance: bool, pages: int = 2) -> list[Place]:
    """Поиск одного термина вокруг точки (lat, lon) в заданном радиусе."""
    places: list[Place] = []
    for page in range(1, pages + 1):
        params = {
            "q": term,
            "point": f"{lon},{lat}",       # 2GIS: СНАЧАЛА долгота!
            "radius": radius,
            "type": "branch",
            "page": page,
            "page_size": 10,
            "locale": "ru_KZ",
            "key": api_key,
        }
        if by_distance:
            params["sort_point"] = f"{lon},{lat}"
            params["sort"] = "distance"
        try:
            items = _dgis_request(params)
        except DGisError:
            if page == 1:
                raise          # первая страница обязана прийти
            break              # дальше — берём то, что уже есть
        for it in items:
            p = _parse_item(it, lat, lon)
            if p:
                places.append(p)
        if len(items) < 10:
            break
    return places


SEARCH_DEADLINE_S = 40   # общий лимит на весь поиск в 2GIS


def retrieve_places(query: str, lat: float, lon: float, dgis_key: str
                    ) -> tuple[list[Place], list[Place], list[str]]:
    """Возвращает (близкие, альтернативы, использованные термины).
    Все запросы идут параллельно, на весь поиск есть общий лимит времени."""
    terms, mapped = interpret_query(query)
    near: dict[str, Place] = {}
    far: dict[str, Place] = {}
    errors: list[DGisError] = []

    def job(kind: str, term: str) -> tuple[str, list[Place]]:
        if kind == "near":   # вокруг пользователя, сортировка по расстоянию
            found = search_2gis(term, lat, lon, NEAR_RADIUS_M, dgis_key,
                                by_distance=True)
        else:                # вокруг центра города, большой радиус
            found = search_2gis(term, ASTANA_CENTER[0], ASTANA_CENTER[1],
                                FAR_RADIUS_M, dgis_key, by_distance=False)
        # Отсеиваем нечёткие совпадения 2GIS (только для распознанных запросов)
        if mapped:
            found = [p for p in found if is_relevant(p, term)]
        return kind, found

    ex = ThreadPoolExecutor(max_workers=6)
    futures = [ex.submit(job, kind, t) for t in terms for kind in ("near", "far")]
    try:
        for fut in as_completed(futures, timeout=SEARCH_DEADLINE_S):
            try:
                kind, places = fut.result()
            except DGisError as e:
                errors.append(e)
                continue
            for p in places:
                p.distance_m = haversine_m(lat, lon, p.lat, p.lon)
                if kind == "near" and p.distance_m <= NEAR_RADIUS_M:
                    near[p.id] = p
                elif kind == "far" and p.distance_m > NEAR_RADIUS_M:
                    far[p.id] = p
    except FuturesTimeout:
        errors.append(DGisError("2GIS отвечает слишком медленно (превышен лимит времени)."))
    finally:
        ex.shutdown(wait=False, cancel_futures=True)

    # близкие имеют приоритет над дальними
    for pid in near:
        far.pop(pid, None)

    if errors and not near and not far:
        raise errors[0]

    near_list = sorted(near.values(), key=lambda p: p.distance_m)[:12]
    far_list = sorted(far.values(),
                      key=lambda p: (-(p.rating or 0), p.distance_m))[:8]
    return near_list, far_list, terms


# --------------------------------------------------------------------------- #
# Generation: Gemini
# --------------------------------------------------------------------------- #
def _place_for_prompt(p: Place) -> dict:
    d = asdict(p)
    d["distance_km"] = round(p.distance_m / 1000, 2)
    d["walk_min"], d["car_min"] = p.walk_min, p.car_min
    for k in ("id", "lat", "lon", "distance_m"):
        d.pop(k, None)
    return d


SYSTEM_PROMPT = """Ты городской гид по Астане. Тебе передают запрос пользователя, его запас времени \
и данные 2ГИС о местах. Пиши по-русски, спокойно и по делу, как знающий местный житель.

Правила:
1. Рекомендуй только места из переданных списков. Не придумывай названия, адреса и часы работы.
2. Выбери 1-3 места из блока «БЛИЗКО» и 1-2 из блока «АЛЬТЕРНАТИВЫ В ДРУГИХ РАЙОНАХ». \
Если блок пуст, пропусти его.
3. Учитывай запас времени: дорога туда и обратно плюс время на месте не должны его превышать. \
Не подходящие по времени места в основной блок не включай. В блоке альтернатив такое место \
допустимо, если прямо сказать, что понадобится больше времени.
4. Если часов работы в данных нет, напиши «часов работы в данных нет, лучше проверить». \
Никогда не пиши «открыто сейчас» без данных.
5. Различай источник. В поле «Почему подходит» опирайся на данные 2ГИС (рубрика, рейтинг, часы, \
расстояние) и пиши об этом прямо. Поле «Совет» это твоя идея, а не данные 2ГИС.
6. Никаких эмодзи, иконок, декоративных символов и восклицательных знаков. Без вступления и заключения.

Формат ответа (Markdown), строго:
## Рядом
### Название места
Расстояние и время в пути одной строкой, например: 600 м, 8 минут пешком.

**Почему подходит.** 1-2 предложения.

**Что делать.** Конкретный план на месте, примерно N минут.

**Совет.** Одна необязательная мысль от тебя.

## В других районах
(те же поля)

## Маршрут
Одна-две строки с порядком посещения, только если уместно сочетать несколько мест.
"""


def generate_recommendations(query: str, minutes: int, near: list[Place],
                             far: list[Place], gemini_key: str, model: str) -> str:
    genai.configure(api_key=gemini_key)
    gemini = genai.GenerativeModel(
        model_name=model,
        system_instruction=SYSTEM_PROMPT,
        generation_config=genai.GenerationConfig(temperature=0.4),
    )
    payload = {
        "запрос_пользователя": query,
        "оставшееся_время_мин": minutes,
        "БЛИЗКО (до 4.5 км)": [_place_for_prompt(p) for p in near],
        "АЛЬТЕРНАТИВЫ В ДРУГИХ РАЙОНАХ": [_place_for_prompt(p) for p in far],
    }
    resp = gemini.generate_content(
        json.dumps(payload, ensure_ascii=False, indent=1),
        request_options={"timeout": 60})
    return resp.text or "Модель вернула пустой ответ."


def fallback_text(near: list[Place], far: list[Place]) -> str:
    """Простой вывод без ИИ, на случай ошибки Gemini."""
    def fmt(title: str, items: list[Place]) -> str:
        if not items:
            return ""
        rows = []
        for p in items[:5]:
            extra = f", рейтинг {p.rating}" if p.rating else ""
            rows.append(f"- **{p.name}**, {p.distance_m/1000:.1f} км, "
                        f"пешком {p.walk_min} мин{extra}")
        return f"## {title}\n" + "\n".join(rows) + "\n\n"
    return fmt("Рядом", near) + fmt("В других районах", far)


# --------------------------------------------------------------------------- #
# Публичная точка входа
# --------------------------------------------------------------------------- #
def get_recommendations(query: str, lat: float, lon: float, minutes: int,
                        gemini_key: str, dgis_key: Optional[str] = None,
                        model: str = DEFAULT_MODEL) -> dict:
    dgis_key = dgis_key or os.getenv("DGIS_API_KEY", "")
    if not dgis_key:
        raise DGisError("Не задан ключ 2GIS (поле в сайдбаре или переменная DGIS_API_KEY).")

    near, far, terms = retrieve_places(query, lat, lon, dgis_key)

    if not near and not far:
        text = "Подходящих мест не нашлось. Попробуйте переформулировать запрос или выбрать другую точку на карте."
    else:
        try:
            text = generate_recommendations(query, minutes, near, far, gemini_key, model)
        except Exception as e:  # noqa: BLE001
            text = (f"ИИ-анализ недоступен ({type(e).__name__}). Проверьте название модели "
                    f"в боковой панели. Ниже список найденных мест без пояснений.\n\n"
                    + fallback_text(near, far))
    return {"text": text, "near": near, "far": far, "terms": terms}