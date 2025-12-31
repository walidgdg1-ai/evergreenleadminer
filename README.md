# Evergreen Lead Miner (OSM + CommonCrawl → Checker)

This is a small Streamlit web app that:
1) Collects candidate business websites from:
   - OpenStreetMap (Overpass) — cleaner but incomplete
   - CommonCrawl (CDX index) — huge volume, noisier
2) Deduplicates by domain
3) Fetches each site and scores relevance with regex rules
4) Exports: raw_osm.csv, raw_commoncrawl.csv, merged_domains.csv, kept.csv, rejected.csv

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy (Streamlit Community Cloud)
- Push this repo to GitHub
- Deploy app.py

## Notes / Safety
- Overpass is shared infra. Keep `OSM tile step` reasonable (e.g., 3–6 degrees) and don't spam it.
- CommonCrawl patterns can explode; use caps and tokens.

## Customize (evergreen)
- Change OSM tags (example):
  - plumber: [{"craft":"plumber"}]
  - electrician: [{"craft":"electrician"}]
  - dentist: [{"amenity":"dentist"}]
- Change regex lists in the UI:
  - Positive regex = signals the niche
  - Negative regex = directories/listicles/aggregators
