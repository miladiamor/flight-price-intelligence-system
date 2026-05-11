#!/usr/bin/env python3
"""
Google Flights scraper with three stabilizers:

A) --first-screen-only      : do not click "More flights"; capture what's initially rendered.
B) --stale-safe / --no-stale-safe:
   default True; safely re-query elements to avoid StaleElementReferenceException.
C) --js-snapshot            : collect card data entirely via JavaScript (very stable).

Other features:
- If URL has a 'tfs' param, we swap the embedded date for each target day.
- If URL has no 'tfs', we open the calendar and pick the date via the UI.
- Handles Google consent page where it appears.

Outputs either a console table or CSV (append mode).
"""

from __future__ import annotations
import os, argparse, base64, binascii, csv, re, sys, time, unicodedata, urllib.parse
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple

from selenium import webdriver
from selenium.common.exceptions import (
    NoSuchElementException,
    StaleElementReferenceException,
    TimeoutException,
    ElementNotInteractableException,
)
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager

DEFAULT_URL = (
    "https://www.google.com/travel/flights/search?"
    "tfs=CBwQAhooEgoyMDI1LTEwLTA0agwIAhIIL20vMDQ0cnZyDAgCEggvbS8wN2Rma0AB"
    "SAFwAYIBCwj___________8BmAED&tfu=EgYIABABGAA&tcfs=ChMKCC9tLzA0NHJ2GgdKYWthcnRhUgRgAXgB"
)
DEFAULT_DATE_LABEL = "Wed, Oct 8"

# ----------------------------- CLI -----------------------------

def parse_arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Scrape Google Flights results.")
    p.add_argument("--url", default=DEFAULT_URL, help="Google Flights URL (share/search link).")
    p.add_argument("--target-date", default=DEFAULT_DATE_LABEL,
                   help="Used only when no --dates/--range; e.g. 'Wed, Oct 8'.")
    p.add_argument("--max-results", type=int, default=0, help="Max cards per date (0 = all).")
    p.add_argument("--csv-output", help="Append results to this CSV.")
    p.add_argument("--dates", help="Comma-separated list YYYY-MM-DD.")
    p.add_argument("--range-start", help="YYYY-MM-DD (inclusive).")
    p.add_argument("--range-end", help="YYYY-MM-DD (inclusive).")
    p.add_argument("--year-offsets", default="0,1",
                   help="Used only with --target-date; e.g. '0,1' for this year and next.")
    p.add_argument("--no-table", action="store_true", help="Don’t print the table.")
    p.add_argument("--origin", type=str, default="", help="Origin IATA/city (for output column).")
    p.add_argument("--destination", type=str, default="", help="Destination IATA/city (for output column).")

    # Stabilizers
    p.add_argument("--first-screen-only", action="store_true",
                   help="Capture only the initially visible list; do not click 'More flights'.")
    p.add_argument("--stale-safe", action=argparse.BooleanOptionalAction, default=True,
                   help="Re-query elements to avoid stale refs (default on).")
    p.add_argument("--js-snapshot", action="store_true",
                   help="Collect cards with one JavaScript query (stable & fast).")

    # Headless control
    p.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True,
                   help="Run Chrome headless (default on). Use --no-headless to show the browser.")

    return p.parse_args()

def configure_driver(headless: bool = True) -> webdriver.Chrome:
    options = Options()
    if headless:
        options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1400,1080")
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    driver.set_page_load_timeout(180)
    return driver

def normalise(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).replace("\xa0", " ").strip()
    # Ensure "PM+1" → "PM +1"
    return re.sub(r"([AP]M)(\+\d)", r"\1 \2", text)

# ---------------------- Consent + Calendar ----------------------

def handle_consent_if_needed(driver: webdriver.Chrome, wait: WebDriverWait) -> None:
    try:
        wait.until(lambda d: True)  # give WebDriver a moment
        if "consent.google" not in driver.current_url:
            return
        for xp in (
            "//button[.//div[normalize-space()='Reject all']]",
            "//button[.//span[normalize-space()='Reject all']]",
            "//button[normalize-space()='Reject all']",
            "//button[.//div[normalize-space()='Accept all']]",
            "//button[normalize-space()='Accept all']",
        ):
            try:
                btn = WebDriverWait(driver, 5).until(EC.element_to_be_clickable((By.XPATH, xp)))
                driver.execute_script("arguments[0].click();", btn)
                time.sleep(0.8)
                return
            except TimeoutException:
                continue
    except Exception:
        pass

def set_departure_date_via_ui(driver: webdriver.Chrome, depart_date_iso: str) -> None:
    wait = WebDriverWait(driver, 40)
    # open date picker
    date_btn = wait.until(EC.element_to_be_clickable(
        (By.CSS_SELECTOR, "button[aria-label*='Departure'], button[aria-label*='Depart']")))
    driver.execute_script("arguments[0].click();", date_btn)

    # direct data-iso if possible
    try:
        cell = wait.until(EC.element_to_be_clickable(
            (By.CSS_SELECTOR, f"div[role='gridcell'][data-iso='{depart_date_iso}']")))
        driver.execute_script("arguments[0].click();", cell)
    except TimeoutException:
        # fallback by aria-label
        y, m, d = depart_date_iso.split("-")
        xpath = (f"//div[@role='gridcell' and contains(@aria-label,'{int(d)}') and "
                 f"(contains(@aria-label,'-{m}-') or contains(@aria-label,' {int(m)} ') or contains(@aria-label,' {m}-')) "
                 f"and contains(@aria-label,'{y}')]")
        cell = wait.until(EC.element_to_be_clickable((By.XPATH, xpath)))
        driver.execute_script("arguments[0].click();", cell)

    done_btn = wait.until(EC.element_to_be_clickable((By.XPATH, "//button[.//span[normalize-space()='Done']]")))
    driver.execute_script("arguments[0].click();", done_btn)

    wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "div[jscontroller='yGdjUc']")))
    time.sleep(0.5)

# --------------------------- URL helpers -------------------------

def build_url_for_date(base_url: str, target_date: date) -> str:
    parsed = urllib.parse.urlparse(base_url)
    query = urllib.parse.parse_qs(parsed.query)
    tfs_value = query.get("tfs", [None])[0]
    if not tfs_value:
        return base_url

    pad = "=" * (-len(tfs_value) % 4)
    try:
        decoded = base64.urlsafe_b64decode(tfs_value + pad)
    except (ValueError, binascii.Error):
        return base_url

    date_match = re.search(rb"\d{4}-\d{2}-\d{2}", decoded)
    if not date_match:
        return base_url

    new_date_bytes = target_date.strftime("%Y-%m-%d").encode("ascii")
    updated = decoded[: date_match.start()] + new_date_bytes + decoded[date_match.end():]
    new_tfs = base64.urlsafe_b64encode(updated).decode("ascii").rstrip("=")

    query["tfs"] = [new_tfs]
    new_query = urllib.parse.urlencode(query, doseq=True)
    return urllib.parse.urlunparse(parsed._replace(query=new_query))

def extract_year_from_url(base_url: str) -> Optional[int]:
    parsed = urllib.parse.urlparse(base_url)
    query = urllib.parse.parse_qs(parsed.query)
    tfs_value = query.get("tfs", [None])[0]
    if not tfs_value:
        return None
    match = re.search(r"Egoy[A-Za-z0-9_-]{4}", tfs_value)
    if not match:
        return None
    chunk = match.group(0)
    pad = "=" * ((4 - len(chunk) % 4) % 4)
    try:
        decoded = base64.urlsafe_b64decode(chunk + pad)
    except (binascii.Error, ValueError):
        return None
    year_match = re.search(rb"(20\d{2})", decoded)
    return int(year_match.group(1).decode()) if year_match else None

def label_to_date(label: str, year: int) -> date:
    _dow, month_day = label.split(", ")
    month_name, day_str = month_day.split(" ")
    try:
        month = datetime.strptime(month_name, "%b").month
    except ValueError:
        month = datetime.strptime(month_name, "%B").month
    return date(year, month, int(day_str))

def shift_year_safe(target: date, offset: int) -> date:
    year = target.year + offset
    try:
        return target.replace(year=year)
    except ValueError:
        return target.replace(year=year, day=target.day - 1)

# --------------------- Parsing helpers (shared) ---------------------

def parse_dates_from_label(label: str, travel_year: int) -> Tuple[str, str]:
    """Parse 'Departs on Fri, Nov 8 and arrives on Fri, Nov 8' into ISO dates."""
    label = normalise(label).replace("\u202f", " ")
    def _extract(frag: str) -> str:
        m = re.search(r"on ([A-Za-z]+, [A-Za-z]+ \d{1,2})", frag)
        if not m:
            return ""
        txt = m.group(1).strip()
        for fmt in ("%A, %B %d", "%A, %b %d", "%a, %B %d", "%a, %b %d"):
            try:
                return datetime.strptime(f"{txt} {travel_year}", fmt + " %Y").date().isoformat()
            except ValueError:
                continue
        return ""
    if " and arrives " in label:
        left, right = label.split(" and arrives ", 1)
    else:
        left, right = label, ""
    dep_iso = _extract(left)
    arr_iso = _extract(right)
    if dep_iso and arr_iso:
        dep_dt = datetime.fromisoformat(dep_iso)
        arr_dt = datetime.fromisoformat(arr_iso)
        while arr_dt < dep_dt:
            arr_dt += timedelta(days=1)
        arr_iso = arr_dt.isoformat()
    return dep_iso, arr_iso

# --------------------- Collectors: JS & Selenium ---------------------

def gather_cards_js_snapshot(driver, travel_year: int, max_results: int, first_screen_only: bool) -> List[Dict]:
    """
    Collect card data fully in JS to avoid staleness. Optionally only first screen.
    """
    # Optionally scroll to the bottom to load more (if not first_screen_only)
    if not first_screen_only:
        last = -1
        for _ in range(30):
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(0.6)
            h = driver.execute_script("return document.body.scrollHeight;")
            if h == last:
                break
            last = h

    script = r"""
    const out = [];
    const cards = Array.from(document.querySelectorAll("div[jscontroller='yGdjUc']"));
    for (const c of cards) {
      const get = (sel) => {
        const el = c.querySelector(sel);
        return el ? el.textContent.trim() : "";
      };
      const getAttr = (sel, attr) => {
        const el = c.querySelector(sel);
        return el ? el.getAttribute(attr) || "" : "";
      };

      const price = get(".YMlIz.FpEdX span");
      // airlines can be split across spans; join uniques
      const an = [];
      const aspans = c.querySelectorAll(".Ir0Voe > .sSHqwe.tPgKwe.ogfYpf span");
      aspans.forEach(s => { const t=s.textContent.trim(); if(t && !an.includes(t)) an.push(t); });
      let airlines = an.length ? an.join(" + ") : get(".Ir0Voe > .sSHqwe.tPgKwe.ogfYpf");

      const dep = get(".zxVSec span[role='text']:nth-of-type(1)");
      const arr = get(".zxVSec span[role='text']:nth-of-type(2)");

      const labelEl = c.querySelector(".zxVSec.YMlIz.tPgKwe.ogfYpf .mv1WYe");
      const label = labelEl ? (labelEl.getAttribute("aria-label")||"") : "";

      const duration = get(".gvkrdb.AdWm1c.tPgKwe.ogfYpf");
      const stopsText = get(".EfT7Ae.AdWm1c.tPgKwe span.ogfYpf");
      let stopsCount = null;
      if (stopsText.toLowerCase().includes("nonstop")) { stopsCount = 0; }
      else { const m = stopsText.match(/(\d+)/); if (m) stopsCount = parseInt(m[1],10); }

      out.push({airlines, price, dep, arr, label, duration, stopsText, stopsCount});
    }
    return out;
    """
    rows = driver.execute_script(script)

    # Map to Python rows and parse dates
    out: List[Dict] = []
    for r in rows[: (None if max_results == 0 else max_results)]:
        dep_iso, arr_iso = ("", "")
        if r.get("label"):
            dep_iso, arr_iso = parse_dates_from_label(r["label"], travel_year)
        out.append({
            "airlines": normalise(r.get("airlines","")),
            "price": normalise(r.get("price","")),
            "departure": normalise(r.get("dep","")),
            "arrival": normalise(r.get("arr","")),
            "departure_date": dep_iso,
            "arrival_date": arr_iso,
            "duration": normalise(r.get("duration","")),
            "stops_text": normalise(r.get("stopsText","")),
            "stops_count": r.get("stopsCount", None),
        })
    return out

def gather_cards_selenium(driver, travel_year: int, max_results: int,
                          first_screen_only: bool, stale_safe: bool) -> List[Dict]:
    """
    Selenium collector. If stale_safe=True, we re-query cards & fields each time.
    """
    wait = WebDriverWait(driver, 45)
    try:
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "div[jscontroller='yGdjUc']")))
    except TimeoutException:
        return []

    def visible_cards_count() -> int:
        return len(driver.find_elements(By.CSS_SELECTOR, "div[jscontroller='yGdjUc']"))

    # Scroll if not first screen only (also try to click More flights)
    if not first_screen_only:
        more_xpaths = [
            "//button[.//span[contains(text(),'More flights')]]",
            "//button[.//span[contains(text(),'View more flights')]]",
            "//button[.//span[contains(text(),'Show more flights')]]",
            "//div[@role='button'][.//span[contains(text(),'More flights')]]",
        ]
        seen = 0
        for _ in range(40):
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(0.7)
            # click a visible "more" if present
            clicked = False
            for xp in more_xpaths:
                elems = driver.find_elements(By.XPATH, xp)
                for el in elems:
                    if el.is_displayed():
                        try:
                            driver.execute_script("arguments[0].click();", el)
                            time.sleep(1.0)
                            clicked = True
                            break
                        except Exception:
                            continue
                if clicked:
                    break
            count = visible_cards_count()
            if count == seen:
                break
            seen = count

    # Now read cards. If stale_safe, read by index & re-query each field by CSS each time.
    rows: List[Dict] = []
    total = visible_cards_count()
    limit = total if max_results == 0 else min(total, max_results)

    for i in range(limit):
        try:
            if stale_safe:
                # re-query the i-th card by index every time
                cards = driver.find_elements(By.CSS_SELECTOR, "div[jscontroller='yGdjUc']")
                card = cards[i]
                def q(css): 
                    try:
                        return card.find_element(By.CSS_SELECTOR, css)
                    except NoSuchElementException:
                        return None
                def qt(css):
                    el = q(css)
                    return normalise(el.text) if el else ""
                def ga(el, attr):
                    try:
                        return el.get_attribute(attr) if el else ""
                    except StaleElementReferenceException:
                        return ""
                price = qt(".YMlIz.FpEdX span")
                # airlines
                names = []
                for sp in card.find_elements(By.CSS_SELECTOR, ".Ir0Voe > .sSHqwe.tPgKwe.ogfYpf span"):
                    t = normalise(sp.text)
                    if t and t not in names:
                        names.append(t)
                airlines = " + ".join(names) if names else qt(".Ir0Voe > .sSHqwe.tPgKwe.ogfYpf")
                # times
                times = card.find_elements(By.CSS_SELECTOR, ".zxVSec span[role='text']")
                dep = normalise(times[0].text) if len(times) > 0 else ""
                arr = normalise(times[1].text) if len(times) > 1 else ""
                label_el = q(".zxVSec.YMlIz.tPgKwe.ogfYpf .mv1WYe")
                label = ga(label_el, "aria-label") or ""
                dep_iso, arr_iso = parse_dates_from_label(label, travel_year) if label else ("","")
                duration = qt(".gvkrdb.AdWm1c.tPgKwe.ogfYpf")
                stops_text = ""
                scount = None
                elst = q(".EfT7Ae.AdWm1c.tPgKwe span.ogfYpf")
                if elst:
                    stops_text = normalise(elst.text)
                    if "nonstop" in stops_text.lower():
                        scount = 0
                    else:
                        m = re.search(r"(\d+)", stops_text)
                        scount = int(m.group(1)) if m else None
            else:
                # classic direct read (may throw stale)
                card = driver.find_elements(By.CSS_SELECTOR, "div[jscontroller='yGdjUc']")[i]
                price = normalise(card.find_element(By.CSS_SELECTOR, ".YMlIz.FpEdX span").text)
                names = []
                for sp in card.find_elements(By.CSS_SELECTOR, ".Ir0Voe > .sSHqwe.tPgKwe.ogfYpf span"):
                    t = normalise(sp.text)
                    if t and t not in names:
                        names.append(t)
                airlines = " + ".join(names) if names else normalise(
                    card.find_element(By.CSS_SELECTOR, ".Ir0Voe > .sSHqwe.tPgKwe.ogfYpf").text)
                times = card.find_elements(By.CSS_SELECTOR, ".zxVSec span[role='text']")
                dep = normalise(times[0].text) if len(times) > 0 else ""
                arr = normalise(times[1].text) if len(times) > 1 else ""
                label = card.find_element(By.CSS_SELECTOR, ".zxVSec.YMlIz.tPgKwe.ogfYpf .mv1WYe").get_attribute("aria-label")
                dep_iso, arr_iso = parse_dates_from_label(label, travel_year) if label else ("","")
                duration = normalise(card.find_element(By.CSS_SELECTOR, ".gvkrdb.AdWm1c.tPgKwe.ogfYpf").text)
                stxt = normalise(card.find_element(By.CSS_SELECTOR, ".EfT7Ae.AdWm1c.tPgKwe span.ogfYpf").text)
                stops_text = stxt
                if "nonstop" in stxt.lower():
                    scount = 0
                else:
                    m = re.search(r"(\d+)", stxt)
                    scount = int(m.group(1)) if m else None

            rows.append({
                "airlines": airlines, "price": price,
                "departure": dep, "arrival": arr,
                "departure_date": dep_iso, "arrival_date": arr_iso,
                "duration": duration, "stops_text": stops_text, "stops_count": scount,
            })
        except StaleElementReferenceException:
            # Skip this one if it still went stale
            continue
        except Exception:
            continue

    return rows

# ------------------------- Output shaping -------------------------

def normalise_output_rows(raw_rows: List[Dict[str, Optional[str]]],
                          target_date: date, origin: str, destination: str) -> List[Dict[str, str]]:
    final: List[Dict[str, str]] = []
    for r in raw_rows:
        dep_date = r.get("departure_date") or target_date.isoformat()
        arr_label = r.get("arrival") or ""
        arr_date = r.get("arrival_date") or ""
        arrival_display = f"{arr_label} ({arr_date})" if arr_date and arr_label else (arr_date or arr_label)
        stops_text = r.get("stops_text") or ("Nonstop" if r.get("stops_count") == 0 else "Unknown")
        final.append({
            "Airline": r.get("airlines", ""), "Date": dep_date,
            "Departure Time": r.get("departure", ""), "Arrival Time": arrival_display,
            "Duration": r.get("duration", ""), "Stops": stops_text, "Price": r.get("price", ""),
            "Origin": origin, "Destination": destination, "Search Date": datetime.today().date().isoformat(),
        })
    return final

def format_output(rows: List[Dict[str, str]]) -> None:
    if not rows:
        print("No flight results captured.")
        return
    headers = ["Airline","Date","Departure Time","Arrival Time","Duration","Stops","Price","Origin","Destination","Search Date"]
    widths = [len(h) for h in headers]
    for row in rows:
        for i, key in enumerate(headers):
            widths[i] = max(widths[i], len(row.get(key, "")))
    fmt = "  ".join([f"{{:<{w}}}" for w in widths])
    print(fmt.format(*headers))
    print("-" * (sum(widths) + 2*(len(widths)-1)))
    for row in rows:
        print(fmt.format(*(row.get(k, "") for k in headers)))

def write_csv(rows: List[Dict[str, str]], path: str) -> None:
    fields = ["Airline","Date","Departure Time","Arrival Time","Duration","Stops","Price","Origin","Destination", "Search Date"]
    need_header = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if need_header:
            w.writeheader()
        w.writerows(rows)

# ------------------------------ Main ------------------------------

def main() -> int:
    args = parse_arguments()

    # Build date list
    if args.dates:
        dates: List[date] = []
        for chunk in args.dates.split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            try:
                dates.append(datetime.fromisoformat(chunk).date())
            except ValueError as exc:
                raise SystemExit(f"Invalid date in --dates: {chunk}") from exc
    elif args.range_start and args.range_end:
        try:
            start = datetime.fromisoformat(args.range_start).date()
            end = datetime.fromisoformat(args.range_end).date()
        except ValueError as exc:
            raise SystemExit("Invalid --range-start/--range-end; use YYYY-MM-DD") from exc
        if end < start:
            raise SystemExit("--range-end must be on or after --range-start")
        dates = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    else:
        base_year = extract_year_from_url(args.url) or datetime.now().year
        base_date = label_to_date(args.target_date, base_year)
        offs = [int(x) for x in args.year_offsets.split(",") if x.strip()]
        if not offs:
            offs = [0]
        dates = [shift_year_safe(base_date, o) for o in offs]

    # de-dup dates
    seen = set()
    unique_dates = [d for d in dates if not (d in seen or seen.add(d))]

    driver = configure_driver(headless=args.headless)
    wait = WebDriverWait(driver, 45)

    all_rows: List[Dict[str, str]] = []
    try:
        # Open URL + consent
        driver.get(args.url)
        handle_consent_if_needed(driver, wait)

        # Let Google rewrite to canonical with tfs=
        base_search_url = args.url
        for _ in range(14):  # ~7s
            cur = driver.current_url
            if "tfs=" in cur:
                base_search_url = cur
                break
            time.sleep(0.5)

        # Iterate the dates
        for dt in unique_dates:
            url_for_date = build_url_for_date(base_search_url, dt)
            driver.get(url_for_date)

            if "tfs=" not in url_for_date:
                try:
                    set_departure_date_via_ui(driver, dt.isoformat())
                except Exception:
                    pass

            # Choose collection mode
            if args.js_snapshot:
                raw = gather_cards_js_snapshot(
                    driver,
                    travel_year=dt.year,
                    max_results=args.max_results,
                    first_screen_only=args.first_screen_only
                )
            else:
                raw = gather_cards_selenium(
                    driver,
                    travel_year=dt.year,
                    max_results=args.max_results,
                    first_screen_only=args.first_screen_only,
                    stale_safe=args.stale_safe,
                )

            normalised = normalise_output_rows(raw, dt, args.origin, args.destination)
            print(f"Collected {len(normalised)} flights for {dt.isoformat()} ({args.origin} → {args.destination})")
            all_rows.extend(normalised)

    finally:
        driver.quit()

    if not args.no_table:
        format_output(all_rows)
    else:
        print(f"Total flights collected: {len(all_rows)} across {len(unique_dates)} dates.")

    if args.csv_output:
        write_csv(all_rows, args.csv_output)
        print(f"\nSaved {len(all_rows)} rows to {args.csv_output}.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
