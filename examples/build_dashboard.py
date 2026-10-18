"""Собирает единый dashboard.html из готовых результатов исследования
Кировского административного округа Омска.

Скрипт НИЧЕГО не скачивает из интернета и НЕ повторяет расчёты BlocksNet —
он только читает GeoJSON/CSV, ранее сохранённые ноутбуком
``examples/omsk_kirovsky_pipeline.ipynb``, и собирает из них HTML-страницу
с интерактивной картой, графиками, таблицами и выводами.

Запуск (из корня проекта):
    .venv\\Scripts\\python.exe examples\\build_dashboard.py

Результат:
    examples/data/omsk_kirovsky/dashboard.html
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import folium
import geopandas as gpd
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import colors as mcolors

BASE = Path(__file__).resolve().parent / "data" / "omsk_kirovsky"
OUTPUT = BASE / "dashboard.html"

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["figure.dpi"] = 110
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.25


# --------------------------------------------------------------------------- #
# Палитры и русские подписи (согласованы с ноутбуком)
# --------------------------------------------------------------------------- #
LAND_USE_LABELS = {
    "residential": "Жилая застройка",
    "business": "Бизнес",
    "recreation": "Рекреация",
    "industrial": "Промышленность",
    "transport": "Транспорт",
    "special": "Специальное",
    "agriculture": "Сельское хозяйство",
}
LAND_USE_COLORS = {
    "Жилая застройка": "#f7e7a1",
    "Бизнес": "#f5a25d",
    "Рекреация": "#7ecb8b",
    "Промышленность": "#b07cc6",
    "Транспорт": "#e8604c",
    "Специальное": "#9aa7b8",
    "Сельское хозяйство": "#c9b458",
}
LAND_USE_NA_COLOR = "#eef0f2"

STAGE_FRAC = {"low-rise": 0.0, "mid-rise": 0.25, "high-mid-rise": 0.5, "high-rise": 0.75, "tower": 1.0}
STAGE_RU = {
    "low-rise": "Малоэтажная",
    "mid-rise": "Среднеэтажная",
    "high-mid-rise": "Многоэтажная (высокая/средняя)",
    "high-rise": "Высотная",
    "tower": "Башенная",
}
TYPE_RU = {"residential": "жилая", "mixed-use": "смешанная", "non-residential": "нежилая"}
DENSITY_RU = {"low-density": "низкой плотности", "high-density": "высокой плотности"}

STRELKA_RU = {
    "low-rise non-residential": "Малоэтажная нежилая",
    "individual residential": "Индивидуальная жилая застройка",
    "low-rise model": "Малоэтажная модель",
    "mid-rise": "Среднеэтажная базовая",
    "mid-rise non-residential": "Среднеэтажная нежилая",
    "mid-rise microdistrict": "Среднеэтажный микрорайон",
    "mid-rise block": "Среднеэтажный квартал",
    "central model": "Центральная модель",
    "high-rise non-residential": "Высотная нежилая",
    "high-rise soviet microdistrict": "Высотный советский микрорайон",
    "high-rise modern microdistrict": "Высотный современный микрорайон",
}
STRELKA_COLORS = {
    "low-rise non-residential": "#9e9e9e",
    "individual residential": "#a5d6a7",
    "low-rise model": "#66bb6a",
    "mid-rise": "#4db6ac",
    "mid-rise non-residential": "#90a4ae",
    "mid-rise microdistrict": "#64b5f6",
    "mid-rise block": "#42a5f5",
    "central model": "#ffb74d",
    "high-rise non-residential": "#7986cb",
    "high-rise soviet microdistrict": "#7e57c2",
    "high-rise modern microdistrict": "#ab47bc",
}
MORPH_NA_COLOR = "#e6e6e6"


def spacematrix_label(value) -> str:
    if pd.isna(value):
        return "Нет данных"
    tokens = str(value).split()
    stage = next((t for t in tokens if t in STAGE_FRAC), None)
    mtype = next((t for t in tokens if t in TYPE_RU), None)
    dens = next((t for t in tokens if t in DENSITY_RU), None)
    parts = [STAGE_RU[stage]] if stage else []
    if mtype:
        parts.append(TYPE_RU[mtype])
    label = " ".join(parts) if parts else str(value)
    if dens:
        label += f" ({DENSITY_RU[dens]})"
    return label


def spacematrix_color(value) -> str:
    if pd.isna(value):
        return MORPH_NA_COLOR
    tokens = str(value).split()
    stage = next((t for t in tokens if t in STAGE_FRAC), "low-rise")
    mtype = next((t for t in tokens if t in TYPE_RU), "residential")
    dens = next((t for t in tokens if t in DENSITY_RU), None)
    frac = STAGE_FRAC[stage]
    if mtype == "residential":
        hue = 0.33 + (0.78 - 0.33) * frac
    elif mtype == "mixed-use":
        hue = 0.12
    else:
        hue = 0.55
    lightness = 0.72 - 0.34 * frac
    lightness += 0.05 if dens == "low-density" else (-0.05 if dens == "high-density" else 0)
    lightness = min(0.85, max(0.35, lightness))
    r, g, b = __import__("colorsys").hls_to_rgb(hue, lightness, 0.55)
    return "#%02x%02x%02x" % (int(r * 255), int(g * 255), int(b * 255))


def strelka_label(value) -> str:
    if pd.isna(value):
        return "Нет данных"
    orig = str(value)
    return f"{STRELKA_RU.get(orig, orig)} ({orig})"


def strelka_color(value) -> str:
    if pd.isna(value):
        return MORPH_NA_COLOR
    return STRELKA_COLORS.get(str(value), "#bdbdbd")


# --------------------------------------------------------------------------- #
# Загрузка готовых результатов (только чтение локальных файлов)
# --------------------------------------------------------------------------- #
def read_table(name: str) -> pd.DataFrame:
    path = BASE / name
    df = pd.read_csv(path)
    if df.columns[0].startswith("Unnamed"):
        df = df.set_index(df.columns[0])
    return df


GEOJSON_PRECISION = 5


def _round_floats(obj):
    if isinstance(obj, float):
        return round(obj, GEOJSON_PRECISION)
    if isinstance(obj, list):
        return [_round_floats(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _round_floats(v) for k, v in obj.items()}
    return obj


def minimize_geojson(data):
    """Округляет координаты и числа, чтобы заметно уменьшить размер HTML."""
    return _round_floats(data)


def load_data() -> gpd.GeoDataFrame:
    blocks = gpd.read_file(BASE / "blocks.geojson")
    master = blocks.reset_index(drop=True).copy()
    master.index = blocks.index

    density = read_table("density.csv")
    prov = read_table("provision_school.csv")
    space = read_table("spacematrix.csv")
    strelka = read_table("strelka.csv")
    area = read_table("area_accessibility.csv")

    master = master.join(
        density[["fsi", "gsi", "mxi", "l", "osr", "share_living", "share_non_living", "site_area",
                 "footprint_area", "build_floor_area", "living_area", "non_living_area"]]
    )
    master = master.join(
        prov[["demand", "capacity", "demand_left", "demand_within", "demand_without",
              "capacity_left", "capacity_within", "capacity_without",
              "provision_strong", "provision_weak"]].rename(columns={"capacity": "capacity_schools"})
    )
    master = master.join(area[["area_accessibility"]])
    master["morphotype_spacematrix"] = space["morphotype"]
    master["spacematrix_cluster"] = space["cluster"]
    master["morphotype_strelka"] = strelka["morphotype"]

    try:
        bb = gpd.read_file(BASE / "buildings_blocks.geojson")[["population"]]
        master = master.join(bb)
    except Exception:
        master["population"] = 0.0

    try:
        sb = gpd.read_file(BASE / "schools_blocks.geojson")[["capacity", "count"]]
        master = master.join(sb.rename(columns={"capacity": "school_capacity", "count": "school_count"}))
    except Exception:
        master["school_capacity"] = 0.0
        master["school_count"] = 0.0

    try:
        landuse = gpd.read_file(BASE / "blocks_landuse.geojson")
        master["land_use"] = landuse["land_use"]
        master["land_use_share"] = landuse["share"]
    except Exception:
        master["land_use"] = None
        master["land_use_share"] = None

    master["land_use_ru"] = master["land_use"].apply(
        lambda v: "Нет данных" if pd.isna(v) else LAND_USE_LABELS.get(str(v).lower(), "Нет данных")
    )
    master["land_use_color"] = master["land_use_ru"].map(LAND_USE_COLORS).fillna(LAND_USE_NA_COLOR)
    master["sm_label"] = master["morphotype_spacematrix"].apply(spacematrix_label)
    master["sm_color"] = master["morphotype_spacematrix"].apply(spacematrix_color)
    master["st_label"] = master["morphotype_strelka"].apply(strelka_label)
    master["st_color"] = master["morphotype_strelka"].apply(strelka_color)

    for col in ["fsi", "gsi", "mxi", "l", "osr", "share_living", "share_non_living", "area_accessibility",
                "spacematrix_cluster", "land_use_share"]:
        if col in master.columns:
            master[col] = pd.to_numeric(master[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return master


# --------------------------------------------------------------------------- #
# Интерактивная карта
# --------------------------------------------------------------------------- #
def _style_choropleth(col, cmap_obj, norm):
    def style(feature):
        value = feature["properties"].get(col)
        if value is None:
            return {"fillColor": MORPH_NA_COLOR, "color": "#888888", "weight": 0.3, "fillOpacity": 0.2}
        return {
            "fillColor": mcolors.to_hex(cmap_obj(norm(value))),
            "color": "#ffffff",
            "weight": 0.4,
            "fillOpacity": 0.78,
        }

    return style


def _style_by_color(color_col):
    def style(feature):
        return {
            "fillColor": feature["properties"].get(color_col, MORPH_NA_COLOR),
            "color": "black",
            "weight": 0.5,
            "fillOpacity": 0.8,
        }

    return style


def add_geo_layer(m, gdf, cols, name, style_fn, aliases, show=False):
    data = minimize_geojson(json.loads(gdf[cols + ["geometry"]].to_json()))
    tip_cols = [c for c in cols if aliases.get(c, c)]
    folium.GeoJson(
        data,
        name=name,
        show=show,
        style_function=style_fn,
        highlight_function=lambda _f: {"weight": 2.5, "color": "#000000"},
        tooltip=folium.GeoJsonTooltip(
            fields=tip_cols,
            aliases=[aliases.get(c, c) for c in tip_cols],
            localize=True,
            sticky=False,
        ),
    ).add_to(m)


def build_map(master: gpd.GeoDataFrame) -> folium.Map:
    boundaries = gpd.read_file(BASE / "boundaries.geojson")
    c = boundaries.geometry.centroid
    center = [float(c.y.mean()), float(c.x.mean())]

    m = folium.Map(location=center, zoom_start=12, tiles="CartoDB positron", control_scale=True)
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(m)

    numeric_layers = [
        ("fsi", "FSI — коэффициент плотности застройки", "YlOrRd"),
        ("gsi", "GSI — занятость территории пятном", "Blues"),
        ("mxi", "MXI — доля жилой функции", "PuBuGn"),
        ("l", "L — средняя этажность", "magma"),
        ("osr", "OSR — свободная площадь на человека", "GnBu"),
        ("population", "Население квартала, чел", "YlOrBr"),
        ("provision_strong", "Обеспеченность школами (сильная, 0–1)", "RdYlGn"),
        ("provision_weak", "Обеспеченность школами (слабая, 0–1)", "RdYlGn"),
        ("area_accessibility", "Доступность по площади, мин", "RdYlGn_r"),
    ]
    numeric_norms = {}
    for i, (col, title, cmap_name) in enumerate(numeric_layers):
        series = pd.to_numeric(master[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if series.empty:
            continue
        vmin, vmax = float(series.min()), float(series.max())
        cmap_obj = plt.get_cmap(cmap_name)
        norm = mcolors.Normalize(vmin, vmax if vmax > vmin else vmin + 1.0)
        numeric_norms[col] = (cmap_name, vmin, vmax)
        add_geo_layer(
            m, master, [col], title, _style_choropleth(col, cmap_obj, norm), {col: title}, show=(i == 0)
        )

    add_geo_layer(
        m, master, ["land_use", "land_use_color"], "Землепользование",
        _style_by_color("land_use_color"),
        {"land_use": "Тип землепользования", "land_use_color": ""},
    )
    add_geo_layer(
        m, master, ["sm_label", "sm_color"], "Морфотипы Spacematrix",
        _style_by_color("sm_color"),
        {"sm_label": "Морфотип", "sm_color": ""},
    )
    add_geo_layer(
        m, master, ["st_label", "st_color"], "Морфотипы Strelka",
        _style_by_color("st_color"),
        {"st_label": "Морфотип", "st_color": ""},
    )

    add_geo_layer(
        m, master, ["population", "l", "fsi", "demand", "provision_strong", "area_accessibility",
                    "spacematrix_cluster", "land_use_share"],
        "Показатели квартала (подсказка)",
        lambda _f: {"fillColor": "#ffffff", "color": "#111111", "weight": 0.6, "fillOpacity": 0.0},
        {
            "population": "Население, чел",
            "l": "Этажность",
            "fsi": "FSI",
            "demand": "Спрос на школы, мест",
            "provision_strong": "Обеспеченность (сильная)",
            "area_accessibility": "Доступность, мин",
            "spacematrix_cluster": "Кластер Spacematrix",
            "land_use_share": "Доля преобладающей зоны",
        },
    )

    try:
        schools = gpd.read_file(BASE / "schools.geojson")
        folium.GeoJson(
            minimize_geojson(json.loads(schools.to_json())),
            name="Школы",
            show=True,
            style_function=lambda _f: {"color": "#c62828", "weight": 1.0, "fillColor": "#e53935", "fillOpacity": 0.85},
            tooltip=folium.GeoJsonTooltip(fields=["amenity"], aliases=["Объект"]),
        ).add_to(m)
    except Exception:
        pass

    folium.GeoJson(
        minimize_geojson(json.loads(boundaries[["geometry"]].to_json())),
        name="Граница округа",
        style_function=lambda _f: {"color": "#222222", "weight": 2.5, "fillOpacity": 0.0},
    ).add_to(m)

    source_layers = [
        ("roads.geojson", "Дороги", {"color": "gray", "weight": 1}, {"LineString", "MultiLineString"}),
        ("railways.geojson", "Железные дороги", {"color": "#455a64", "weight": 1.2, "dashArray": "6,4"},
         {"LineString", "MultiLineString"}),
        ("water.geojson", "Вода", {"color": "#42a5f5", "weight": 0.8, "fillColor": "#90caf9", "fillOpacity": 0.6},
         None),
    ]
    for fname, layer_name, st, geom_types in source_layers:
        try:
            g = gpd.read_file(BASE / fname)
            if geom_types:
                g = g[g.geometry.geom_type.isin(geom_types)]
            folium.GeoJson(
                minimize_geojson(json.loads(g[["geometry"]].to_json())),
                name=f"Источник: {layer_name}",
                show=False,
                style_function=lambda _f, style=st: style,
            ).add_to(m)
        except Exception:
            pass

    folium.LayerControl(collapsed=False, position="topright").add_to(m)
    return m, numeric_norms


# --------------------------------------------------------------------------- #
# Графики (matplotlib -> base64 PNG)
# --------------------------------------------------------------------------- #
def fig_uri(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def bar_chart(series, colors, title, ylabel, rotation=40):
    series = series[series > 0]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(range(len(series)), series.values,
           color=[colors.get(k, "#90a4ae") for k in series.index], edgecolor="black", linewidth=0.4)
    ax.set_xticks(range(len(series)))
    ax.set_xticklabels(series.index, rotation=rotation, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    return fig_uri(fig)


def hist_chart(values, title, xlabel, color="#42a5f5", bins=30, vmax=None):
    values = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if vmax is not None:
        values = values[values <= vmax]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.hist(values, bins=bins, color=color, edgecolor="white", linewidth=0.4)
    ax.axvline(values.mean(), color="#c62828", linestyle="--", linewidth=1.2,
               label=f"среднее = {values.mean():.2f}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Число кварталов")
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8)
    return fig_uri(fig)


def build_charts(master: gpd.GeoDataFrame) -> dict:
    developed = master[(master["fsi"].fillna(0) > 0)]
    demand_blocks = master[master["demand"].fillna(0) > 0]

    return {
        "land_use": bar_chart(master["land_use_ru"].value_counts(), LAND_USE_COLORS,
                              "Землепользование: число кварталов по типу", "Кварталов"),
        "sm": bar_chart(master["sm_label"].value_counts(),
                        {l: spacematrix_color(master.loc[master["sm_label"] == l, "morphotype_spacematrix"].iloc[0])
                         for l in master["sm_label"].unique()},
                        "Морфотипы Spacematrix: число кварталов", "Кварталов"),
        "st": bar_chart(master["st_label"].value_counts(),
                        {l: strelka_color(master.loc[master["st_label"] == l, "morphotype_strelka"].iloc[0])
                         for l in master["st_label"].unique()},
                        "Морфотипы Strelka: число кварталов", "Кварталов"),
        "population": hist_chart(master["population"], "Распределение населения по кварталам",
                                 "Население, чел", color="#ffb74d", bins=40),
        "fsi": hist_chart(developed["fsi"], "Плотность застройки (FSI) в застроенных кварталах",
                          "FSI, м²/м²", color="#7e57c2"),
        "l": hist_chart(developed["l"], "Этажность (L) в застроенных кварталах",
                        "Средняя этажность", color="#26a69a", bins=25),
        "provision": hist_chart(demand_blocks["provision_strong"],
                                "Обеспеченность школами (сильная) среди кварталов со спросом",
                                "Доля спроса, покрытая (0–1)", color="#66bb6a", bins=20, vmax=1.0),
        "access": hist_chart(master["area_accessibility"],
                             "Доступность по площади (среднее время до объектов)",
                             "Минут", color="#ef5350", bins=40),
    }


# --------------------------------------------------------------------------- #
# Таблицы
# --------------------------------------------------------------------------- #
STAT_FIELDS = {
    "fsi": "FSI, м²/м²",
    "gsi": "GSI",
    "mxi": "MXI",
    "l": "Этажность L",
    "osr": "OSR — свободная площадь",
    "share_living": "Доля жилой площади",
    "share_non_living": "Доля нежилой площади",
    "footprint_area": "Пятно застройки, м²",
    "build_floor_area": "Площадь этажей, м²",
    "living_area": "Жилая площадь, м²",
    "non_living_area": "Нежилая площадь, м²",
    "population": "Население, чел",
    "demand": "Спрос на школы, мест",
    "capacity_schools": "Ёмкость школ, мест",
    "demand_left": "Не покрытый спрос, мест",
    "demand_within": "Спрос в пределах норматива",
    "demand_without": "Спрос за пределами норматива",
    "capacity_left": "Свободная ёмкость школ",
    "capacity_within": "Ёмкость в пределах норматива",
    "capacity_without": "Ёмкость за пределами норматива",
    "provision_strong": "Обеспеченность (сильная)",
    "provision_weak": "Обеспеченность (слабая)",
    "area_accessibility": "Доступность, мин",
    "land_use_share": "Доля преобладающей зоны",
    "spacematrix_cluster": "Кластер Spacematrix",
}


def build_tables(master: gpd.GeoDataFrame) -> dict:
    summary = read_table("results_summary.csv")
    summary_html = summary.T.rename(columns={summary.T.columns[0]: "Значение"}).to_html(
        classes="table", border=0
    )

    stats_src = master[[c for c in STAT_FIELDS if c in master.columns]].replace([np.inf, -np.inf], np.nan)
    stats = stats_src.describe().T
    stats.index = [STAT_FIELDS[i] for i in stats.index]
    stats = stats[["count", "mean", "std", "min", "50%", "max"]].round(3)
    stats.columns = ["Кварталов", "Среднее", "Ст. откл.", "Мин", "Медиана", "Макс"]
    stats_html = stats.to_html(classes="table", border=0)

    try:
        clusters = read_table("spacematrix_clusters.csv")
        clusters = clusters.rename(columns={
            "cluster": "Кластер", "l": "Этажность", "fsi": "FSI", "mxi": "MXI",
            "l_interpretation": "Этажность (EN)", "fsi_interpretation": "Плотность (EN)",
            "mxi_interpretation": "Смешанность (EN)", "morphotype": "Морфотип (EN)",
        })
        clusters["Морфотип"] = clusters["Морфотип (EN)"].apply(spacematrix_label)
        cols = ["Кластер", "Этажность", "FSI", "MXI", "Морфотип", "Этажность (EN)",
                "Плотность (EN)", "Смешанность (EN)", "Морфотип (EN)"]
        view = clusters[cols].copy()
        num_cols = view.select_dtypes("number").columns
        view[num_cols] = view[num_cols].round(3)
        clusters_html = view.to_html(classes="table", border=0, index=False)
    except Exception:
        clusters_html = "<p>Таблица кластеров недоступна.</p>"

    try:
        links = read_table("provision_links_school.csv")
        links["value"] = pd.to_numeric(links["value"], errors="coerce")
        links = links.sort_values("value", ascending=False).head(25)
        links = links.rename(columns={"source": "Квартал-житель", "target": "Квартал-школа",
                                      "value": "Учеников"})
        links_html = links[["Квартал-житель", "Квартал-школа", "Учеников"]].to_html(
            classes="table", border=0, index=False
        )
    except Exception:
        links_html = "<p>Таблица связей квартал–школа недоступна.</p>"

    top = master.sort_values("population", ascending=False).head(15)
    top = top[["population", "l", "fsi", "demand", "provision_strong", "sm_label"]].copy()
    top.index.name = "Квартал"
    num_cols = top.select_dtypes("number").columns
    top[num_cols] = top[num_cols].round(2)
    top = top.rename(columns={
        "population": "Население, чел", "l": "Этажность", "fsi": "FSI",
        "demand": "Спрос, мест", "provision_strong": "Обеспеченность",
        "sm_label": "Морфотип Spacematrix",
    })
    top_html = top.to_html(classes="table", border=0)

    return {
        "summary": summary_html,
        "stats": stats_html,
        "clusters": clusters_html,
        "links": links_html,
        "top": top_html,
    }


# --------------------------------------------------------------------------- #
# Выводы
# --------------------------------------------------------------------------- #
def ru_int(value) -> str:
    return f"{int(round(value)):,}".replace(",", "\u00a0")


def build_conclusions(master: gpd.GeoDataFrame) -> str:
    n = len(master)
    population = master["population"].fillna(0).sum()
    schools = master["school_count"].fillna(0).sum()
    capacity = master["school_capacity"].fillna(0).sum()

    demand_blocks = master[master["demand"].fillna(0) > 0]
    mean_prov = demand_blocks["provision_strong"].mean()
    share_good = (demand_blocks["provision_strong"] >= 0.5).mean() * 100 if len(demand_blocks) else float("nan")
    n_with_school = int((master["school_capacity"].fillna(0) > 0).sum())

    acc = pd.to_numeric(master["area_accessibility"], errors="coerce")
    acc = acc[acc < 60]
    mean_acc = acc.mean()

    developed = master[master["fsi"].fillna(0) > 0]
    densest = developed.loc[developed["fsi"].idxmax()] if len(developed) else None
    tallest = developed.loc[developed["l"].idxmax()] if len(developed) else None

    lu_mode = master["land_use_ru"].value_counts()
    lu_top = lu_mode.index[0] if len(lu_mode) else "—"
    lu_top_share = (lu_mode.iloc[0] / n * 100) if len(lu_mode) else 0.0

    sm_mode = developed["sm_label"].value_counts()
    sm_top = sm_mode.index[0] if len(sm_mode) else "—"
    st_mode = master["st_label"].value_counts()
    st_top = st_mode.index[0] if len(st_mode) else "—"

    residential_share = (master["land_use_ru"] == "Жилая застройка").mean() * 100

    items = [
        f"В округе <b>{n}</b> кварталов, суммарное население — <b>{ru_int(population)}</b> чел.",
        f"Учтено <b>{int(schools)}</b> школ(-ы) общей ёмкостью <b>{ru_int(capacity)}</b> мест; "
        f"школы есть в {n_with_school} кварталах.",
        f"Спрос на школьные места есть в <b>{len(demand_blocks)}</b> кварталах. Средняя сильная обеспеченность — "
        f"<b>{mean_prov:.2f}</b>, доля кварталов с обеспеченностью от 0.5 — <b>{share_good:.1f}%</b>.",
        f"Средняя доступность по площади (без выбросов ≥ 60 мин) — <b>{mean_acc:.1f} мин</b> пешком.",
        f"Преобладающее землепользование — «<b>{lu_top}</b>» ({lu_top_share:.0f}% кварталов); "
        f"на жилую застройку приходится {residential_share:.0f}% кварталов.",
        f"Среди застроенных кварталов доминирует морфотип Spacematrix «<b>{sm_top}</b>»; "
        f"по Strelka чаще всего встречается «<b>{st_top}</b>».",
    ]
    if densest is not None:
        items.append(f"Самый плотный квартал — №{densest.name} (FSI = {densest['fsi']:.2f}, этажность ≈ {densest['l']:.1f}).")
    if tallest is not None:
        items.append(f"Самая высокая застройка — квартал №{tallest.name} (этажность ≈ {tallest['l']:.1f}).")

    lis = "".join(f"<li>{t}</li>" for t in items)
    return f"<ul class='conclusions'>{lis}</ul>"


# --------------------------------------------------------------------------- #
# Легенды для карты
# --------------------------------------------------------------------------- #
def gradient_css(cmap_name: str, steps: int = 8) -> str:
    cmap_obj = plt.get_cmap(cmap_name)
    cols = [mcolors.to_hex(cmap_obj(i / (steps - 1))) for i in range(steps)]
    return "linear-gradient(to right, " + ", ".join(cols) + ")"


def build_legends(master: gpd.GeoDataFrame, numeric_norms: dict) -> str:
    blocks = []

    for col, (cmap_name, vmin, vmax) in numeric_norms.items():
        title = next(t for c, t, _ in [
            ("fsi", "FSI — коэффициент плотности застройки", ""),
            ("gsi", "GSI — занятость территории пятном", ""),
            ("mxi", "MXI — доля жилой функции", ""),
            ("l", "L — средняя этажность", ""),
            ("osr", "OSR — свободная площадь на человека", ""),
            ("population", "Население квартала, чел", ""),
            ("provision_strong", "Обеспеченность школами (сильная, 0–1)", ""),
            ("provision_weak", "Обеспеченность школами (слабая, 0–1)", ""),
            ("area_accessibility", "Доступность по площади, мин", ""),
        ] if c == col)
        blocks.append(
            f"<div class='legend-item'><div class='legend-title'>{title}</div>"
            f"<div class='gradient' style='background:{gradient_css(cmap_name)}'></div>"
            f"<div class='gradient-labels'><span>{vmin:.2f}</span><span>{vmax:.2f}</span></div></div>"
        )

    cat_layers = [
        ("Землепользование", master["land_use_ru"].value_counts().index, master["land_use_color"], False),
        ("Морфотипы Spacematrix", master["sm_label"].value_counts().index, master["sm_color"], True),
        ("Морфотипы Strelka", master["st_label"].value_counts().index, master["st_color"], True),
    ]
    for title, labels, color_series, is_morph in cat_layers:
        lookup = dict(zip(master["land_use_ru" if title == "Землепользование" else
                          ("sm_label" if "Spacematrix" in title else "st_label")], color_series))
        rows = "".join(
            f"<div class='swatch-row'><span class='swatch' style='background:{lookup.get(l, MORPH_NA_COLOR)}'></span>"
            f"<span>{l}</span></div>"
            for l in labels
        )
        blocks.append(f"<div class='legend-item'><div class='legend-title'>{title}</div>{rows}</div>")

    blocks.append(
        "<div class='legend-item'><div class='legend-title'>Прочее</div>"
        "<div class='swatch-row'><span class='swatch' style='background:#e53935'></span><span>Школа</span></div>"
        "<div class='swatch-row'><span class='swatch' style='background:transparent;border:2px solid #222'></span><span>Граница округа</span></div>"
        "<div class='swatch-row'><span class='swatch' style='background:#808080'></span><span>Источник: дороги</span></div>"
        "<div class='swatch-row'><span class='swatch' style='background:#455a64'></span><span>Источник: ж/д</span></div>"
        "<div class='swatch-row'><span class='swatch' style='background:#90caf9'></span><span>Источник: вода</span></div>"
        "</div>"
    )
    return "".join(blocks)


# --------------------------------------------------------------------------- #
# Сборка HTML
# --------------------------------------------------------------------------- #
CSS = """
:root { --bg:#f4f6f8; --card:#ffffff; --ink:#1f2933; --muted:#5c6b7a; --accent:#1e88e5; }
* { box-sizing: border-box; }
body { margin:0; font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background:var(--bg); color:var(--ink); }
header { background:linear-gradient(135deg,#1e3c72,#2a5298); color:#fff; padding:28px 32px; }
header h1 { margin:0 0 6px; font-size:26px; }
header p { margin:0; opacity:.9; }
.container { max-width:1400px; margin:0 auto; padding:24px 32px 60px; }
.cards { display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:16px; margin:24px 0; }
.card { background:var(--card); border-radius:12px; padding:18px 20px; box-shadow:0 1px 4px rgba(0,0,0,.08); }
.card .value { font-size:26px; font-weight:700; color:var(--accent); }
.card .label { font-size:13px; color:var(--muted); margin-top:4px; }
h2 { font-size:20px; margin:32px 0 14px; border-left:4px solid var(--accent); padding-left:10px; }
.map-wrap { display:grid; grid-template-columns:1fr 320px; gap:16px; }
.map-frame { background:var(--card); border-radius:12px; overflow:hidden; box-shadow:0 1px 4px rgba(0,0,0,.08); }
.map-frame iframe { width:100%; height:720px; border:0; display:block; }
.legend-panel { background:var(--card); border-radius:12px; padding:16px; box-shadow:0 1px 4px rgba(0,0,0,.08); max-height:720px; overflow-y:auto; }
.legend-item { margin-bottom:16px; }
.legend-title { font-weight:600; font-size:13px; margin-bottom:6px; }
.gradient { height:12px; border-radius:6px; }
.gradient-labels { display:flex; justify-content:space-between; font-size:11px; color:var(--muted); }
.swatch-row { display:flex; align-items:center; gap:8px; font-size:12px; margin:3px 0; }
.swatch { width:16px; height:16px; border-radius:4px; display:inline-block; flex:none; }
.charts { display:grid; grid-template-columns:repeat(auto-fit,minmax(420px,1fr)); gap:18px; }
.charts .card img { width:100%; height:auto; }
.table { border-collapse:collapse; width:100%; font-size:13px; background:var(--card); }
.table th, .table td { padding:6px 10px; border-bottom:1px solid #e3e8ee; text-align:right; white-space:nowrap; }
.table th:first-child, .table td:first-child { text-align:left; }
.table thead th { background:#eef3f8; position:sticky; top:0; }
.table-wrap { overflow-x:auto; background:var(--card); border-radius:12px; box-shadow:0 1px 4px rgba(0,0,0,.08); padding:6px; }
.conclusions { background:var(--card); border-radius:12px; padding:18px 22px 18px 40px; box-shadow:0 1px 4px rgba(0,0,0,.08); line-height:1.7; }
ul.conclusions li { margin-bottom:8px; }
.two-col { display:grid; grid-template-columns:1fr 1fr; gap:18px; }
footer { text-align:center; color:var(--muted); font-size:12px; padding:24px; }
@media (max-width:1000px){ .map-wrap,.two-col{grid-template-columns:1fr;} }
"""


def main() -> None:
    if not BASE.exists():
        raise SystemExit(f"Не найдена папка с результатами: {BASE}")

    master = load_data()
    m, numeric_norms = build_map(master)
    charts = build_charts(master)
    tables = build_tables(master)
    conclusions = build_conclusions(master)
    legends = build_legends(master, numeric_norms)

    population = master["population"].fillna(0).sum()
    capacity = master["school_capacity"].fillna(0).sum()
    schools = int(round(master["school_count"].fillna(0).sum()))

    # srcdoc кладём в атрибут с одинарными кавычками: так не нужно экранировать
    # все кавычки JSON (их сотни тысяч), файл получается заметно меньше.
    map_srcdoc = m.get_root().render().replace("&", "&amp;").replace("'", "&#39;")

    cards = f"""
    <div class="cards">
      <div class="card"><div class="value">{len(master)}</div><div class="label">Кварталов</div></div>
      <div class="card"><div class="value">{ru_int(population)}</div><div class="label">Население, чел</div></div>
      <div class="card"><div class="value">{schools}</div><div class="label">Школ</div></div>
      <div class="card"><div class="value">{ru_int(capacity)}</div><div class="label">Мест в школах</div></div>
    </div>
    """

    def chart_block(key, title):
        return f"<div class='card'><h3>{title}</h3><img src='{charts[key]}' alt='{title}'></div>"

    html_doc = f"""<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Дашборд: Кировский округ Омска</title>
<style>{CSS}</style>
</head>
<body>
<header>
  <h1>Кировский административный округ, Омск</h1>
  <p>Интерактивный дашборд по результатам расчётов BlocksNet: землепользование, плотность, морфотипы, обеспеченность школами и доступность.</p>
</header>
<div class="container">
  {cards}

  <h2>Интерактивная карта</h2>
  <div class="map-wrap">
    <div class="map-frame"><iframe title="Карта" srcdoc='{map_srcdoc}'></iframe></div>
    <div class="legend-panel">{legends}</div>
  </div>

  <h2>Выводы</h2>
  {conclusions}

  <h2>Графики</h2>
  <div class="charts">
    {chart_block('land_use', 'Землепользование')}
    {chart_block('sm', 'Морфотипы Spacematrix')}
    {chart_block('st', 'Морфотипы Strelka')}
    {chart_block('population', 'Население')}
    {chart_block('fsi', 'Плотность FSI')}
    {chart_block('l', 'Этажность L')}
    {chart_block('provision', 'Обеспеченность школами')}
    {chart_block('access', 'Доступность по площади')}
  </div>

  <h2>Сводка</h2>
  <div class="two-col">
    <div class="table-wrap">{tables['summary']}</div>
    <div class="table-wrap">{tables['stats']}</div>
  </div>

  <h2>Кластеры Spacematrix</h2>
  <div class="table-wrap">{tables['clusters']}</div>

  <h2>Связи кварталов и школ (топ-25 по числу учеников)</h2>
  <div class="table-wrap">{tables['links']}</div>

  <h2>Топ-15 кварталов по населению</h2>
  <div class="table-wrap">{tables['top']}</div>

  <footer>Собрано из локальных результатов ноутбука на BlocksNet 1.0.0a9. Данные — OpenStreetMap.</footer>
</div>
</body>
</html>
"""

    OUTPUT.write_text(html_doc, encoding="utf-8")
    print(f"Готово: {OUTPUT}")


if __name__ == "__main__":
    main()
