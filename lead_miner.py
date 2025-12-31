# evergreen_lead_miner/lead_miner.py
"""
Evergreen Lead Miner
- Sources: OpenStreetMap (Overpass) + CommonCrawl (CDX)
- Then: merge domains + HTTP/HTML checker (regex scoring) -> kept/rejected

Notes:
- Overpass is community infrastructure. Keep tile_step reasonable and don't hammer it.
- CommonCrawl is huge; use caps and tokens.
"""

from __future__ import annotations

import io
import json
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

import pandas as pd
import requests
import tldextract
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
from tenacity import retry, stop_after_attempt, wait_exponential


HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; EvergreenLeadMiner/1.0; +https://example.com)"
}

# --------------------------
# Utils
# --------------------------

def normalize_url(u: str) -> Optional[str]:
    if not u:
        return None
    u = str(u).strip()
    if not u:
        return None
    if not u.startswith(("http://", "https://")):
        u = "https://" + u
    return u

def root_domain(url: str) -> Optional[str]:
    try:
        ext = tldextract.extract(url)
        if not ext.domain or not ext.suffix:
            return None
        return f"{ext.domain}.{ext.suffix}".lower()
    except Exception:
        return None

def safe_get_json(url: str, params: Optional[dict] = None, timeout: int = 30) -> dict:
    r = requests.get(url, params=params, headers=HEADERS, timeout=timeout)
    r.raise_for_status()
    return r.json()

# --------------------------
# Geo: Nominatim bbox
# --------------------------

def get_bbox_from_nominatim(query: str) -> Optional[Tuple[float, float, float, float]]:
    """
    Returns bbox (latS, lonW, latN, lonE) from Nominatim.
    """
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": query, "format": "json", "limit": 1}
    r = requests.get(url, params=params, headers=HEADERS, timeout=30)
    r.raise_for_status()
    data = r.json()
    if not data:
        return None
    bb = data[0].get("boundingbox")  # [south, north, west, east]
    if not bb or len(bb) != 4:
        return None
    s, n, w, e = map(float, bb)
    return (s, w, n, e)

def bbox_tiles(bbox: Tuple[float, float, float, float], step_deg: float) -> List[Tuple[float, float, float, float]]:
    latS, lonW, latN, lonE = bbox
    tiles: List[Tuple[float, float, float, float]] = []
    lat = latS
    while lat < latN:
        lon = lonW
        while lon < lonE:
            tiles.append((lat, lon, min(lat + step_deg, latN), min(lon + step_deg, lonE)))
            lon += step_deg
        lat += step_deg
    return tiles

# --------------------------
# OSM Overpass collector
# --------------------------

class OverpassError(Exception):
    pass

DEFAULT_OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=25))
def overpass_query(endpoint: str, query: str, timeout_s: int) -> dict:
    r = requests.post(endpoint, data={"data": query}, headers=HEADERS, timeout=timeout_s)
    if r.status_code != 200:
        raise OverpassError(f"HTTP {r.status_code}: {r.text[:200]}")
    return r.json()

def build_overpass_union(tag_filters: Sequence[Dict[str, str]], s: float, w: float, n: float, e: float, timeout_s: int) -> str:
    parts: List[str] = []
    for tf in tag_filters:
        for k, v in tf.items():
            parts.append(f'node["{k}"="{v}"]({s},{w},{n},{e});')
            parts.append(f'way["{k}"="{v}"]({s},{w},{n},{e});')
            parts.append(f'relation["{k}"="{v}"]({s},{w},{n},{e});')
    return f"""
[out:json][timeout:{timeout_s}];
(
{''.join(parts)}
);
out center tags;
""".strip()

def extract_osm_website(tags: dict) -> Optional[str]:
    tags = tags or {}
    for k in ["website", "contact:website", "url", "contact:url"]:
        v = tags.get(k)
        if isinstance(v, str) and v.strip():
            v = v.strip()
            if v.startswith(("http://", "https://")):
                return v
            if re.match(r"^[\w\.-]+\.[a-z]{2,}", v, re.I):
                return "https://" + v
    return None

def collect_from_osm(
    bbox: Tuple[float, float, float, float],
    tag_filters: Sequence[Dict[str, str]],
    tile_step_deg: float = 4.0,
    sleep_s: float = 0.8,
    timeout_s: int = 180,
    endpoints: Optional[Sequence[str]] = None,
    max_domains: int = 50000,
) -> pd.DataFrame:
    eps = list(endpoints) if endpoints else list(DEFAULT_OVERPASS_ENDPOINTS)
    tiles = bbox_tiles(bbox, tile_step_deg)

    rows: List[dict] = []
    seen: Set[str] = set()

    for (s, w, n, e) in tiles:
        q = build_overpass_union(tag_filters, s, w, n, e, timeout_s)

        data = None
        for ep in eps:
            try:
                data = overpass_query(ep, q, timeout_s)
                break
            except Exception:
                data = None

        if not data:
            time.sleep(sleep_s)
            continue

        for el in data.get("elements", []):
            tags = el.get("tags", {}) or {}
            site = extract_osm_website(tags)
            if not site:
                continue
            site = normalize_url(site)
            if not site:
                continue
            dom = root_domain(site)
            if not dom or dom in seen:
                continue
            seen.add(dom)

            rows.append(
                {
                    "source": "osm",
                    "domain": dom,
                    "url": site,
                    "name": tags.get("name"),
                    "city": tags.get("addr:city"),
                    "state": tags.get("addr:state"),
                    "postcode": tags.get("addr:postcode"),
                }
            )

            if len(seen) >= max_domains:
                return pd.DataFrame(rows)

        time.sleep(sleep_s)

    return pd.DataFrame(rows)

# --------------------------
# CommonCrawl collector
# --------------------------

def get_latest_cdx_apis(n: int = 2) -> List[str]:
    info = safe_get_json("https://index.commoncrawl.org/collinfo.json", timeout=30)
    info = sorted(info, key=lambda x: x.get("id", ""), reverse=True)
    return [x["cdx-api"] for x in info[:n]]

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=25))
def cdx_query(cdx_api: str, url_pattern: str, page: int = 0) -> List[dict]:
    params = {
        "url": url_pattern,
        "output": "json",
        "fl": "url,status,mime",
        "filter": "status:200",
        "collapse": "urlkey",
        "page": page,
    }
    r = requests.get(cdx_api, params=params, headers=HEADERS, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"CDX HTTP {r.status_code}: {r.text[:200]}")
    # JSON Lines: one object per line
    out: List[dict] = []
    for line in r.text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            # If a weird line appears, skip it.
            continue
    return out

def collect_from_commoncrawl(
    tokens: Sequence[str],
    tlds: Sequence[str],
    max_latest_crawls: int = 2,
    max_pages_per_pattern: int = 30,
    max_urls_per_pattern: int = 15000,
    sleep_s: float = 0.2,
    max_domains: int = 80000,
) -> pd.DataFrame:
    cdx_apis = get_latest_cdx_apis(max_latest_crawls)

    # Build patterns like *.com/*token*
    clean_tokens = []
    for t in tokens:
        t = (t or "").strip().lower()
        if not t:
            continue
        # keep token safe-ish
        t = re.sub(r"[^a-z0-9\-\_]+", "", t)
        if len(t) >= 2:
            clean_tokens.append(t)

    patterns: List[str] = []
    for tld in tlds:
        tld = (tld or "").strip().lower()
        if not tld:
            continue
        for tok in clean_tokens:
            patterns.append(f"*.{tld}/*{tok}*")

    rows: List[dict] = []
    seen: Set[str] = set()

    for api in cdx_apis:
        for pat in patterns:
            got = 0
            for page in range(max_pages_per_pattern):
                data = cdx_query(api, pat, page=page)
                if not data:
                    break
                for item in data:
                    u = item.get("url")
                    mime = (item.get("mime") or "").lower()
                    if not u or "text/html" not in mime:
                        continue
                    u = normalize_url(u)
                    if not u:
                        continue
                    dom = root_domain(u)
                    if not dom or dom in seen:
                        continue
                    seen.add(dom)
                    rows.append({"source": "commoncrawl", "domain": dom, "url": u, "pattern": pat, "cdx_api": api})
                    got += 1
                    if got >= max_urls_per_pattern or len(seen) >= max_domains:
                        break
                if got >= max_urls_per_pattern or len(seen) >= max_domains:
                    break
                time.sleep(sleep_s)
            if len(seen) >= max_domains:
                return pd.DataFrame(rows)

    return pd.DataFrame(rows)

# --------------------------
# Checker
# --------------------------

@dataclass
class CountryProfile:
    name: str
    phone_regex: str
    address_hint_regex: str

COUNTRY_PROFILES: Dict[str, CountryProfile] = {
    "USA": CountryProfile("USA", r"(\+1[\s\-\.]?)?(\(?\d{3}\)?[\s\-\.]?)\d{3}[\s\-\.]?\d{4}", r"\b[A-Z]{2}\s*\d{5}(?:-\d{4})?\b|\bUSA\b|\bUnited States\b"),
    "UK":  CountryProfile("UK",  r"(\+44\s?7\d{3}|\(?07\d{3}\)?)\s?\d{3}\s?\d{3}", r"\b[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}\b|\bUnited Kingdom\b|\bUK\b"),
    "FR":  CountryProfile("FR",  r"(\+33|0)\s*[1-9](?:[\s\.-]?\d{2}){4}", r"\b\d{5}\b|\bFrance\b"),
    "BE":  CountryProfile("BE",  r"(\+32|0)\s*\d(?:[\s\.-]?\d{2}){3,4}", r"\b\d{4}\b|\bBelgium\b|\bBruxelles\b|\bBrussels\b"),
}

SCHEMA_LOCALBUSINESS_RE = re.compile(r"LocalBusiness|Organization|HVACBusiness|Dentist|Plumber|Electrician|Attorney|Lawyer", re.I)

def fetch_html(url: str, timeout_s: int = 15) -> Tuple[Optional[str], str, Optional[str]]:
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout_s, allow_redirects=True)
        if r.status_code >= 400:
            return None, f"http_{r.status_code}", None
        ctype = (r.headers.get("content-type", "") or "").lower()
        if "text/html" not in ctype:
            return None, "non_html", None
        return r.text, "ok", str(r.url)
    except requests.RequestException:
        return None, "error", None

def visible_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for t in soup(["script", "style", "noscript"]):
        t.decompose()
    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    h = " ".join(x.get_text(" ", strip=True) for x in soup.find_all(["h1", "h2", "h3"])[:20])
    body = soup.get_text(" ", strip=True)[:20000]
    return " ".join([title, h, body])

def score_patterns(text: str, patterns: Sequence[str], weight: int) -> Tuple[int, int]:
    s = 0
    hits = 0
    for rx in patterns:
        if not rx:
            continue
        if re.search(rx, text, flags=re.I):
            s += weight
            hits += 1
    return s, hits

def decide(pos_score: int, neg_score: int, strictness: int, has_phone: bool, has_address_hint: bool, has_schema: bool) -> bool:
    # 0 tolérant | 1 normal | 2 strict
    if strictness == 0:
        return (pos_score >= 4 and neg_score <= 10) or (pos_score >= 8)
    if strictness == 1:
        return (pos_score >= 6 and neg_score <= 8 and (has_phone or has_address_hint or has_schema)) or (pos_score >= 10 and neg_score <= 10)
    return (pos_score >= 8 and neg_score <= 7 and has_phone and (has_address_hint or has_schema))

def analyze_domain(
    domain: str,
    url: str,
    positive: Sequence[str],
    negative: Sequence[str],
    strictness: int,
    profile: CountryProfile,
    timeout_s: int = 15,
    fetch_internal_pages: bool = True,
    internal_paths: Sequence[str] = ("/services", "/service", "/about", "/contact"),
    max_internal_pages: int = 2,
) -> dict:
    html, st, final = fetch_html(url, timeout_s)
    if st != "ok" and url.startswith("https://"):
        html, st, final = fetch_html("http://" + url[len("https://"):], timeout_s)

    if st != "ok" or not html:
        return {"domain": domain, "url": url, "final_url": final, "keep": False, "reason": st, "pos_score": 0, "neg_score": 0}

    txt = visible_text(html)

    pos_s, pos_h = score_patterns(txt, positive, weight=4)
    neg_s, neg_h = score_patterns(txt, negative, weight=3)

    has_phone = bool(re.search(profile.phone_regex, txt))
    has_address = bool(re.search(profile.address_hint_regex, txt, flags=re.I))
    has_schema = bool(SCHEMA_LOCALBUSINESS_RE.search(html))

    # Borderline enrichment: try 1-2 internal pages
    if fetch_internal_pages and 4 <= pos_s < 8:
        fetched = 0
        base = (final or url).rstrip("/")
        for path in internal_paths:
            if fetched >= max_internal_pages:
                break
            h2, st2, _ = fetch_html(base + path, timeout_s)
            if st2 == "ok" and h2:
                t2 = visible_text(h2)
                ps2, _ = score_patterns(t2, positive, weight=4)
                ns2, _ = score_patterns(t2, negative, weight=3)
                pos_s += max(0, ps2 // 2)
                neg_s += max(0, ns2 // 2)
                fetched += 1
            time.sleep(0.05)

    keep = decide(pos_s, neg_s, strictness, has_phone, has_address, has_schema)

    return {
        "domain": domain,
        "url": url,
        "final_url": final,
        "keep": keep,
        "reason": "ok" if keep else "not_relevant",
        "pos_score": pos_s,
        "neg_score": neg_s,
        "has_phone": has_phone,
        "has_address_hint": has_address,
        "has_schema": has_schema,
    }

# --------------------------
# Orchestrator
# --------------------------

def run_pipeline(
    location_query: str,
    country_profile_key: str,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    use_nominatim: bool = True,
    sources: Sequence[str] = ("OSM", "CommonCrawl"),
    osm_tag_filters: Optional[Sequence[Dict[str, str]]] = None,
    osm_tile_step_deg: float = 4.0,
    osm_sleep_s: float = 0.8,
    osm_max_domains: int = 25000,
    cc_tokens: Optional[Sequence[str]] = None,
    cc_tlds: Sequence[str] = ("com", "net", "org", "us"),
    cc_latest_crawls: int = 2,
    cc_max_domains: int = 60000,
    cc_max_pages_per_pattern: int = 25,
    cc_max_urls_per_pattern: int = 12000,
    cc_sleep_s: float = 0.2,
    positive_keywords: Optional[Sequence[str]] = None,
    negative_keywords: Optional[Sequence[str]] = None,
    strictness: int = 1,
    blacklist_domains: Optional[Set[str]] = None,
    max_merged_domains: int = 15000,
    checker_workers: int = 24,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns: df_osm_raw, df_cc_raw, df_merged_domains, df_kept, df_rejected
    """

    profile = COUNTRY_PROFILES.get(country_profile_key, COUNTRY_PROFILES["USA"])

    if bbox is None and use_nominatim:
        bbox = get_bbox_from_nominatim(location_query)
    if bbox is None:
        raise ValueError("No bbox available. Provide bbox or enable Nominatim with a valid location_query.")

    osm_tag_filters = list(osm_tag_filters) if osm_tag_filters else [{"craft": "hvac"}]
    cc_tokens = list(cc_tokens) if cc_tokens else ["hvac", "heating", "air-conditioning", "services"]
    positive = list(positive_keywords) if positive_keywords else [r"\bhvac\b", r"heating", r"air\s*conditioning"]
    negative = list(negative_keywords) if negative_keywords else [r"\bbest\b|\btop\s*\d+\b|\bdirectory\b|\breviews?\b"]

    bl = set(blacklist_domains) if blacklist_domains else set()

    # Collect
    df_osm = pd.DataFrame()
    df_cc = pd.DataFrame()

    if "OSM" in sources:
        df_osm = collect_from_osm(
            bbox=bbox,
            tag_filters=osm_tag_filters,
            tile_step_deg=osm_tile_step_deg,
            sleep_s=osm_sleep_s,
            max_domains=osm_max_domains,
        )

    if "CommonCrawl" in sources:
        df_cc = collect_from_commoncrawl(
            tokens=cc_tokens,
            tlds=cc_tlds,
            max_latest_crawls=cc_latest_crawls,
            max_pages_per_pattern=cc_max_pages_per_pattern,
            max_urls_per_pattern=cc_max_urls_per_pattern,
            sleep_s=cc_sleep_s,
            max_domains=cc_max_domains,
        )

    frames = [d for d in [df_osm, df_cc] if d is not None and not d.empty]
    if not frames:
        return df_osm, df_cc, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    df_all = pd.concat(frames, ignore_index=True)
    df_all["domain"] = df_all["domain"].astype(str).str.lower()

    # Blacklist + dedupe
    df_all = df_all[~df_all["domain"].isin(bl)].copy()
    df_merged = df_all.drop_duplicates(subset=["domain"]).reset_index(drop=True)

    if max_merged_domains and len(df_merged) > max_merged_domains:
        df_merged = df_merged.head(max_merged_domains).copy()

    # Checker
    kept, rejected = [], []
    with ThreadPoolExecutor(max_workers=checker_workers) as ex:
        futs = [
            ex.submit(
                analyze_domain,
                r["domain"],
                r["url"],
                positive,
                negative,
                strictness,
                profile,
            )
            for _, r in df_merged.iterrows()
        ]
        for fut in as_completed(futs):
            res = fut.result()
            (kept if res.get("keep") else rejected).append(res)

    df_kept = pd.DataFrame(kept).sort_values(["pos_score", "neg_score"], ascending=[False, True])
    df_rej = pd.DataFrame(rejected)

    return df_osm, df_cc, df_merged, df_kept, df_rej
