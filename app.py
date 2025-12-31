# app.py
import io
import json
import zipfile

import streamlit as st
import pandas as pd

from lead_miner import run_pipeline, COUNTRY_PROFILES

st.set_page_config(page_title="Evergreen Lead Miner", layout="wide")

st.title("Evergreen Lead Miner")
st.caption("OSM (clean-ish) + CommonCrawl (volume) → merge → checker → kept/rejected CSV")

DEFAULT_BLACKLIST = {
    "yelp.com","angi.com","homeadvisor.com","bbb.org","yellowpages.com",
    "facebook.com","instagram.com","linkedin.com","youtube.com","tiktok.com",
    "pinterest.com","wikipedia.org"
}

PRESETS = {
    "HVAC (USA)": {
        "location_query": "United States",
        "country_profile": "USA",
        "osm_tags": [{"craft":"hvac"}],
        "cc_tokens": ["hvac","heating","air-conditioning","ac-repair","heat-pump","furnace","services"],
        "cc_tlds": ["com","net","org","us"],
        "pos": [r"\bhvac\b", r"air\s*conditioning|a/c\b|ac\s*repair", r"heating|furnace|boiler", r"heat\s*pump", r"duct"],
        "neg": [r"\bbest\b|\btop\s*\d+\b|\bdirectory\b|\breviews?\b", r"near\s*me", r"affiliate|lead"],
    },
    "Plumber (USA)": {
        "location_query": "United States",
        "country_profile": "USA",
        "osm_tags": [{"craft":"plumber"}],
        "cc_tokens": ["plumber","plumbing","drain","leak","water-heater","services"],
        "cc_tlds": ["com","net","org","us"],
        "pos": [r"\bplumb", r"drain|pipe|leak|water\s*heater"],
        "neg": [r"\bbest\b|\btop\s*\d+\b|\bdirectory\b|\breviews?\b"],
    },
    "Electrician (USA)": {
        "location_query": "United States",
        "country_profile": "USA",
        "osm_tags": [{"craft":"electrician"}],
        "cc_tokens": ["electrician","electrical","panel","breaker","wiring","services"],
        "cc_tlds": ["com","net","org","us"],
        "pos": [r"\belectrician\b|\belectrical\b", r"panel|breaker|wiring|rewire"],
        "neg": [r"\bbest\b|\btop\s*\d+\b|\bdirectory\b|\breviews?\b"],
    },
    "Dentist (USA)": {
        "location_query": "United States",
        "country_profile": "USA",
        "osm_tags": [{"amenity":"dentist"}],
        "cc_tokens": ["dentist","dental","implants","orthodontics","invisalign","services"],
        "cc_tlds": ["com","net","org","us"],
        "pos": [r"\bdentist\b|\bdental\b", r"implant|invisalign|orthodont"],
        "neg": [r"\bbest\b|\btop\s*\d+\b|\bdirectory\b|\breviews?\b"],
    },
}

with st.sidebar:
    st.header("Job configuration")

    preset_name = st.selectbox("Preset", list(PRESETS.keys()) + ["Custom"], index=0)
    preset = PRESETS.get(preset_name, None)

    if preset_name != "Custom":
        location_query = preset["location_query"]
        country_profile = preset["country_profile"]
        osm_tags = preset["osm_tags"]
        cc_tokens = preset["cc_tokens"]
        cc_tlds = preset["cc_tlds"]
        pos = preset["pos"]
        neg = preset["neg"]
    else:
        location_query = st.text_input("Location query (Nominatim)", "United States")
        country_profile = st.selectbox("Country profile (for phone/address hints)", list(COUNTRY_PROFILES.keys()), index=0)

        osm_tags_raw = st.text_area("OSM tag filters (JSON list)", '[{"craft":"hvac"}]')
        try:
            osm_tags = json.loads(osm_tags_raw)
            if not isinstance(osm_tags, list):
                raise ValueError()
        except Exception:
            st.error('Invalid JSON for OSM tags. Example: [{"craft":"hvac"}]')
            st.stop()

        cc_tokens = [x.strip() for x in st.text_input("CommonCrawl tokens (comma)", "hvac,heating,air-conditioning,services").split(",") if x.strip()]
        cc_tlds = [x.strip() for x in st.text_input("TLDs (comma)", "com,net,org,us").split(",") if x.strip()]
        pos = [x for x in st.text_area("Positive regex (one per line)", r"\bhvac\b\nheating\nair\s*conditioning").splitlines() if x.strip()]
        neg = [x for x in st.text_area("Negative regex (one per line)", r"\bbest\b\n\bdirectory\b\nreviews?").splitlines() if x.strip()]

    sources = st.multiselect("Sources", ["OSM", "CommonCrawl"], default=["OSM", "CommonCrawl"])
    strictness = st.slider("Strictness", 0, 2, 1, help="0 = tolérant, 2 = strict (plus clean, moins de résultats)")

    st.subheader("Limits (safety)")
    max_merged = st.number_input("Max merged domains (cap)", min_value=200, max_value=200000, value=15000, step=1000)
    osm_tile_step = st.number_input("OSM tile step (deg)", min_value=0.8, max_value=10.0, value=4.0, step=0.2, help="Plus petit = plus de requêtes Overpass.")
    osm_max = st.number_input("OSM max domains", min_value=200, max_value=200000, value=25000, step=1000)
    cc_max = st.number_input("CommonCrawl max domains", min_value=200, max_value=300000, value=60000, step=2000)
    cc_latest = st.number_input("CommonCrawl latest crawls", min_value=1, max_value=3, value=2, step=1)

    run_btn = st.button("Run", type="primary")

if run_btn:
    with st.spinner("Running collection + checker…"):
        df_osm, df_cc, df_merged, kept, rejected = run_pipeline(
            location_query=location_query,
            country_profile_key=country_profile,
            bbox=None,
            use_nominatim=True,
            sources=sources,
            osm_tag_filters=osm_tags,
            osm_tile_step_deg=float(osm_tile_step),
            osm_max_domains=int(osm_max),
            cc_tokens=cc_tokens,
            cc_tlds=cc_tlds,
            cc_latest_crawls=int(cc_latest),
            cc_max_domains=int(cc_max),
            max_merged_domains=int(max_merged),
            positive_keywords=pos,
            negative_keywords=neg,
            strictness=int(strictness),
            blacklist_domains=DEFAULT_BLACKLIST,
        )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("OSM raw", 0 if df_osm is None else len(df_osm))
    c2.metric("CommonCrawl raw", 0 if df_cc is None else len(df_cc))
    c3.metric("Merged domains", 0 if df_merged is None else len(df_merged))
    c4.metric("Kept", 0 if kept is None else len(kept))

    st.subheader("Kept (clean list)")
    st.dataframe(kept.head(300), use_container_width=True)

    st.subheader("Rejected (why dropped)")
    st.dataframe(rejected.head(300), use_container_width=True)

    # ZIP export
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("raw_osm.csv", (df_osm.to_csv(index=False) if df_osm is not None else ""))
        z.writestr("raw_commoncrawl.csv", (df_cc.to_csv(index=False) if df_cc is not None else ""))
        z.writestr("merged_domains.csv", (df_merged.to_csv(index=False) if df_merged is not None else ""))
        z.writestr("kept.csv", kept.to_csv(index=False) if kept is not None else "")
        z.writestr("rejected.csv", rejected.to_csv(index=False) if rejected is not None else "")

    st.download_button(
        "Download ZIP (CSVs)",
        data=buf.getvalue(),
        file_name="lead_miner_results.zip",
        mime="application/zip",
    )
else:
    st.info("Configure the job in the sidebar, then click **Run**.")
