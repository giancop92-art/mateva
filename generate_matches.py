import json
import os
import re
import time
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from io import StringIO

import pandas as pd
import requests
from google import genai
from google.genai import types

# Configurazione stagione 2026/27
SEASON_YEAR = "2026"
FD_CSV_URL = "https://www.football-data.co.uk/mmz4281/2627/I1.csv"
UNDERSTAT_URL = f"https://understat.com/league/Serie_A/{SEASON_YEAR}"
ESPN_SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/soccer/ita.1/scoreboard"
DAYS_AHEAD = 5

# Lista modelli con fallback automatico in caso di 503 (high demand) o 429
MODELS_FALLBACK = [
    "gemini-3.8-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
]
SLEEP_SECONDS = 15  # Rispetta il limite di 5 RPM sul piano gratuito

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

# Mappa normalizzazione nomi ESPN/Understat -> football-data.co.uk
TEAM_ALIASES = {
    "as roma": "roma",
    "ac milan": "milan",
    "inter milan": "inter",
    "internazionale": "inter",
    "hellas verona": "verona",
    "ssc napoli": "napoli",
    "ss lazio": "lazio",
    "ac monza": "monza",
    "us lecce": "lecce",
    "udinese calcio": "udinese",
    "bologna fc": "bologna",
    "acf fiorentina": "fiorentina",
    "atalanta bc": "atalanta",
    "torino fc": "torino",
    "genoa cfc": "genoa",
    "cagliari calcio": "cagliari",
    "parma calcio": "parma",
    "como 1907": "como",
    "us sassuolo": "sassuolo",
    "sassuolo calcio": "sassuolo",
    "pisa sc": "pisa",
    "us cremonese": "cremonese",
    "frosinone calcio": "frosinone",
    "venezia fc": "venezia",
    "empoli fc": "empoli",
}


def normalize_team_name(name):
    """Normalizza il nome squadra per fare match affidabile con football-data.co.uk."""
    if not name:
        return ""
    low = name.strip().lower()
    if low in TEAM_ALIASES:
        return TEAM_ALIASES[low]
    for prefix in ("ac ", "as ", "fc ", "us ", "ss ", "ssc ", "acf ", "hellas "):
        if low.startswith(prefix):
            low = low[len(prefix):].strip()
    return low


def fetch_football_data():
    """Scarica il CSV stagionale della Serie A da football-data.co.uk."""
    try:
        r = requests.get(FD_CSV_URL, headers=HEADERS, timeout=20)
        r.raise_for_status()
        df = pd.read_csv(StringIO(r.text))
        df = df.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"])
        return df
    except Exception as e:
        print(f"[WARN] Impossibile scaricare football-data.co.uk: {e}")
        return pd.DataFrame()


def compute_league_table_and_stats(fd_df):
    """Calcola classifica reale, forma e split Casa/Trasferta per tutte le squadre da I1.csv."""
    if fd_df.empty:
        return {}

    teams = {}

    def ensure_team(t_norm, display_name):
        if t_norm not in teams:
            teams[t_norm] = {
                "display_name": display_name,
                "played": 0,
                "points": 0,
                "gf": 0,
                "ga": 0,
                "form": [],
                "home": {"played": 0, "pts": 0, "gf": 0, "ga": 0, "corners": [], "cards": [], "shots": [], "sot": []},
                "away": {"played": 0, "pts": 0, "gf": 0, "ga": 0, "corners": [], "cards": [], "shots": [], "sot": []},
            }

    for _, row in fd_df.iterrows():
        h_raw = str(row["HomeTeam"]).strip()
        a_raw = str(row["AwayTeam"]).strip()
        h = normalize_team_name(h_raw)
        a = normalize_team_name(a_raw)
        ensure_team(h, h_raw)
        ensure_team(a, a_raw)

        hg = int(row["FTHG"])
        ag = int(row["FTAG"])

        if hg > ag:
            hp, ap, hf, af = 3, 0, "W", "L"
        elif hg < ag:
            hp, ap, hf, af = 0, 3, "L", "W"
        else:
            hp, ap, hf, af = 1, 1, "D", "D"

        # Aggiorna totali
        teams[h]["played"] += 1
        teams[h]["points"] += hp
        teams[h]["gf"] += hg
        teams[h]["ga"] += ag
        teams[h]["form"].append(hf)

        teams[a]["played"] += 1
        teams[a]["points"] += ap
        teams[a]["gf"] += ag
        teams[a]["ga"] += hg
        teams[a]["form"].append(af)

        # Aggiorna split Casa
        teams[h]["home"]["played"] += 1
        teams[h]["home"]["pts"] += hp
        teams[h]["home"]["gf"] += hg
        teams[h]["home"]["ga"] += ag
        if pd.notna(row.get("HC")):
            teams[h]["home"]["corners"].append(float(row["HC"]))
        if pd.notna(row.get("HY")):
            teams[h]["home"]["cards"].append(float(row["HY"]) + (float(row["HR"]) if pd.notna(row.get("HR")) else 0.0))
        if pd.notna(row.get("HS")):
            teams[h]["home"]["shots"].append(float(row["HS"]))
        if pd.notna(row.get("HST")):
            teams[h]["home"]["sot"].append(float(row["HST"]))

        # Aggiorna split Trasferta
        teams[a]["away"]["played"] += 1
        teams[a]["away"]["pts"] += ap
        teams[a]["away"]["gf"] += ag
        teams[a]["away"]["ga"] += hg
        if pd.notna(row.get("AC")):
            teams[a]["away"]["corners"].append(float(row["AC"]))
        if pd.notna(row.get("AY")):
            teams[a]["away"]["cards"].append(float(row["AY"]) + (float(row["AR"]) if pd.notna(row.get("AR")) else 0.0))
        if pd.notna(row.get("AS")):
            teams[a]["away"]["shots"].append(float(row["AS"]))
        if pd.notna(row.get("AST")):
            teams[a]["away"]["sot"].append(float(row["AST"]))

    # Ordina classifica (punti, diff reti, gol fatti)
    ranked = sorted(
        teams.items(),
        key=lambda kv: (kv[1]["points"], kv[1]["gf"] - kv[1]["ga"], kv[1]["gf"]),
        reverse=True,
    )

    def avg(lst):
        return round(sum(lst) / len(lst), 2) if lst else None

    result = {}
    for pos, (t_norm, d) in enumerate(ranked, start=1):
        hp = d["home"]["played"]
        ap = d["away"]["played"]
        result[t_norm] = {
            "position": pos,
            "played": d["played"],
            "points": d["points"],
            "goals_for": d["gf"],
            "goals_against": d["ga"],
            "recent_form": d["form"][-5:],
            "home_split": {
                "label": "Casa",
                "played": hp,
                "points_per_game": round(d["home"]["pts"] / hp, 2) if hp else None,
                "gf_per_game": round(d["home"]["gf"] / hp, 2) if hp else None,
                "ga_per_game": round(d["home"]["ga"] / hp, 2) if hp else None,
                "corners_for_avg": avg(d["home"]["corners"]),
                "cards_avg": avg(d["home"]["cards"]),
                "shots_avg": avg(d["home"]["shots"]),
                "shots_on_target_avg": avg(d["home"]["sot"]),
            },
            "away_split": {
                "label": "Trasferta",
                "played": ap,
                "points_per_game": round(d["away"]["pts"] / ap, 2) if ap else None,
                "gf_per_game": round(d["away"]["gf"] / ap, 2) if ap else None,
                "ga_per_game": round(d["away"]["ga"] / ap, 2) if ap else None,
                "corners_for_avg": avg(d["away"]["corners"]),
                "cards_avg": avg(d["away"]["cards"]),
                "shots_avg": avg(d["away"]["shots"]),
                "shots_on_target_avg": avg(d["away"]["sot"]),
            },
        }
    return result


def extract_understat_json(html, var_name):
    """Estrae un blocco JSON.parse('...') dalla pagina HTML di Understat."""
    pattern = rf"var\s+{var_name}\s*=\s*JSON\.parse\('([^']+)'\)"
    match = re.search(pattern, html)
    if not match:
        return None
    raw = match.group(1).encode("utf-8").decode("unicode_escape")
    return json.loads(raw)


def fetch_understat_data():
    """Recupera calendario e xG/xGA da Understat (se accessibile)."""
    try:
        r = requests.get(UNDERSTAT_URL, headers=HEADERS, timeout=25)
        r.raise_for_status()
        html = r.text
        dates_data = extract_understat_json(html, "datesData") or []
        teams_data = extract_understat_json(html, "teamsData") or {}
        return dates_data, teams_data
    except Exception as e:
        print(f"[WARN] Impossibile scaricare Understat: {e}")
        return [], {}


def get_understat_xg(team_name, teams_data):
    """Restituisce xG/gara e xGA/gara da Understat se presenti, altrimenti None."""
    target = normalize_team_name(team_name)
    for _, tdata in teams_data.items():
        if normalize_team_name(tdata.get("title", "")) == target:
            hist = tdata.get("history", [])
            n = len(hist)
            if n > 0:
                xg = sum(float(x.get("xG", 0)) for x in hist)
                xga = sum(float(x.get("xGA", 0)) for x in hist)
                return round(xg / n, 2), round(xga / n, 2)
    return None, None


def get_upcoming_fixtures(dates_data):
    """Filtra le partite di Serie A in programma nei prossimi DAYS_AHEAD giorni."""
    now = datetime.now(timezone.utc)
    limit = now + timedelta(days=DAYS_AHEAD)
    fixtures = []

    # 1. Prova da Understat
    for m in dates_data:
        if m.get("isResult"):
            continue
        dt_str = m.get("datetime")
        if not dt_str:
            continue
        try:
            dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        if now - timedelta(hours=2) <= dt <= limit:
            fixtures.append({
                "home": m["h"]["title"],
                "away": m["a"]["title"],
                "kickoff_iso": dt.astimezone(timezone(timedelta(hours=1))).isoformat(),
                "date_slug": dt.strftime("%Y-%m-%d"),
            })

    if fixtures:
        return fixtures

    # 2. Fallback pubblico da ESPN
    try:
        r = requests.get(ESPN_SCOREBOARD_URL, headers=HEADERS, timeout=20)
        r.raise_for_status()
        data = r.json()
        for ev in data.get("events", []):
            status = ev.get("status", {}).get("type", {}).get("completed", False)
            if status:
                continue
            comp = ev["competitions"][0]
            home = next(c["team"]["displayName"] for c in comp["competitors"] if c["homeAway"] == "home")
            away = next(c["team"]["displayName"] for c in comp["competitors"] if c["homeAway"] == "away")
            dt_iso = ev.get("date", "")
            fixtures.append({
                "home": home,
                "away": away,
                "kickoff_iso": dt_iso,
                "date_slug": dt_iso[:10] if dt_iso else now.strftime("%Y-%m-%d"),
            })
    except Exception as e:
        print(f"[WARN] Fallback ESPN fallito: {e}")

    return fixtures


def fetch_google_news_context(home, away, max_items=10):
    """Recupera gratuitamente notizie recenti (probabili formazioni, infortuni, arbitro) via Google News RSS."""
    query = f"{home} {away} probabili formazioni infortuni arbitro Serie A"
    url = f"https://news.google.com/rss/search?q={urllib.parse.quote(query)}&hl=it&gl=IT&ceid=IT:it"
    headlines = []
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        r.raise_for_status()
        root = ET.fromstring(r.content)
        for item in root.findall(".//item")[:max_items]:
            title = item.findtext("title", "").strip()
            pub_date = item.findtext("pubDate", "").strip()
            if title:
                headlines.append(f"- {title} ({pub_date})")
    except Exception as e:
        print(f"   [WARN] Google News RSS non disponibile per {home}-{away}: {e}")

    return "\n".join(headlines) if headlines else "Nessuna notizia recente estratta via RSS."


def build_team_payload(team_name, side, league_table, teams_data):
    """Costruisce il blocco dati reali verificati per una squadra."""
    norm = normalize_team_name(team_name)
    stats = league_table.get(norm)
    xg, xga = get_understat_xg(team_name, teams_data)

    if not stats:
        return {
            "name": team_name,
            "position": None,
            "points": None,
            "goals_for": None,
            "goals_against": None,
            "xg_per_game": xg,
            "xga_per_game": xga,
            "recent_form": [],
            "split_stats": {
                "label": "Casa" if side == "home" else "Trasferta",
                "played": None,
                "points_per_game": None,
                "gf_per_game": None,
                "ga_per_game": None,
                "corners_for_avg": None,
                "cards_avg": None,
                "shots_avg": None,
                "shots_on_target_avg": None,
            },
        }

    split_key = "home_split" if side == "home" else "away_split"
    return {
        "name": team_name,
        "position": stats["position"],
        "points": stats["points"],
        "goals_for": stats["goals_for"],
        "goals_against": stats["goals_against"],
        "xg_per_game": xg,
        "xga_per_game": xga,
        "recent_form": stats["recent_form"],
        "split_stats": stats[split_key],
    }


def build_prompt(fixture, home_data, away_data, news_context):
    """Costruisce il prompt senza numeri di esempio finti e con regole anti-allucinazione."""
    slug = re.sub(r"[^a-z0-9]+", "-", f"{fixture['date_slug']}-{fixture['home']}-{fixture['away']}".lower()).strip("-")

    return f"""
Sei un analista calcistico che applica rigorosamente la metodologia 'football-match-evaluator-v3' per la Serie A 2026/27.

Analizza la partita: {fixture['home']} vs {fixture['away']} (Kickoff: {fixture['kickoff_iso']}).

DATI REALI VERIFICATI DA FOOTBALL-DATA.CO.UK (USALI OBBLIGATORIAMENTE, NON MODIFICARLI MAI):
- CASA: {json.dumps(home_data, ensure_ascii=False)}
- TRASFERTA: {json.dumps(away_data, ensure_ascii=False)}

NOTIZIE RECENTI (Google News RSS - probabili formazioni, assenze, arbitro):
{news_context}

REGOLE TASSATIVE:
1. Usa ESATTAMENTE i numeri reali forniti sopra per classifica, punti, gol, forma, split, tiri, corner e cartellini. Se un campo è null (es. xg_per_game), lascialo null.
2. Esito 1X2: basa le probabilità sul divario reale di classifica, punti/gara, gol fatti/subiti, tiri in porta e forma. La somma dei 'mid' di home_win + draw + away_win DEVE fare 100.
3. Gol: over_2_5.mid + under_2_5.mid = 100; btts_yes.mid + btts_no.mid = 100.
4. Corner e Cartellini: usa le medie reali di corner e cartellini fornite negli split Casa/Trasferta.
5. GIOCATORI: VIETATO scrivere nomi generici o segnaposto come "Attaccante Torino", "Como Midfielder", "Cagliari Striker", "Difensore X". Inserisci SOLO nomi e cognomi reali di calciatori effettivamente presenti nelle notizie RSS o nella rosa attuale; se non hai nomi certi, lascia l'array vuoto [].
6. Scrivi tutti i testi (drivers, anomalies, coverage_gaps, pre_kickoff_checks) in ITALIANO.

RISPONDI ESCLUSIVAMENTE CON UN SINGOLO OGGETTO JSON VALIDO con questa struttura esatta:
{{
  "id": "{slug}",
  "competition": "Serie A",
  "context_type": "domestic_league",
  "matchday": "<stringa es. 12ª Giornata>",
  "kickoff_iso": "{fixture['kickoff_iso']}",
  "venue": "<nome stadio>",
  "referee": {{
    "name": "<nome arbitro se presente nelle notizie, altrimenti Da verificare>",
    "cards_per_game": null,
    "penalties_rate_pct": null,
    "strictness": null
  }},
  "teams": {{
    "home": {{
      "name": "{fixture['home']}",
      "position": "<intero dai dati reali>",
      "points": "<intero dai dati reali>",
      "goals_for": "<intero dai dati reali>",
      "goals_against": "<intero dai dati reali>",
      "xg_per_game": "<float dai dati reali o null>",
      "xga_per_game": "<float dai dati reali o null>",
      "recent_form": ["<dai dati reali>"],
      "split_stats": {{
        "label": "Casa",
        "played": "<intero>",
        "points_per_game": "<float>",
        "gf_per_game": "<float>",
        "ga_per_game": "<float>",
        "corners_for_avg": "<float>",
        "cards_avg": "<float>"
      }},
      "absences": {{
        "confirmed_out": ["<solo se citati nelle notizie>"],
        "doubtful": ["<solo se citati nelle notizie>"]
      }},
      "rest_days": null,
      "upcoming_cup_fixture": null,
      "probable_formation": "<modulo o null>"
    }},
    "away": {{
      "name": "{fixture['away']}",
      "position": "<intero dai dati reali>",
      "points": "<intero dai dati reali>",
      "goals_for": "<intero dai dati reali>",
      "goals_against": "<intero dai dati reali>",
      "xg_per_game": "<float dai dati reali o null>",
      "xga_per_game": "<float dai dati reali o null>",
      "recent_form": ["<dai dati reali>"],
      "split_stats": {{
        "label": "Trasferta",
        "played": "<intero>",
        "points_per_game": "<float>",
        "gf_per_game": "<float>",
        "ga_per_game": "<float>",
        "corners_for_avg": "<float>",
        "cards_avg": "<float>"
      }},
      "absences": {{
        "confirmed_out": ["<solo se citati nelle notizie>"],
        "doubtful": ["<solo se citati nelle notizie>"]
      }},
      "rest_days": null,
      "upcoming_cup_fixture": null,
      "probable_formation": "<modulo o null>"
    }}
  }},
  "analysis": {{
    "1_outcome_1x2": {{
      "home_win": {{ "min": 0, "max": 0, "mid": 0.0 }},
      "draw": {{ "min": 0, "max": 0, "mid": 0.0 }},
      "away_win": {{ "min": 0, "max": 0, "mid": 0.0 }},
      "drivers": ["<driver in italiano basato sui numeri reali>"],
      "anomalies": ["<eventuale anomalia in italiano>"]
    }},
    "2_goals": {{
      "over_2_5": {{ "min": 0, "max": 0, "mid": 0.0 }},
      "under_2_5": {{ "min": 0, "max": 0, "mid": 0.0 }},
      "btts_yes": {{ "min": 0, "max": 0, "mid": 0.0 }},
      "btts_no": {{ "min": 0, "max": 0, "mid": 0.0 }},
      "drivers": ["<driver gol in italiano>"]
    }},
    "3_corners": {{
      "line": 9.5,
      "over_line": {{ "min": 0, "max": 0, "mid": 0.0 }},
      "under_line": {{ "min": 0, "max": 0, "mid": 0.0 }},
      "most_corners_team": "home|away",
      "most_corners_prob": {{ "min": 0, "max": 0, "mid": 0.0 }},
      "drivers": ["<driver corner in italiano>"]
    }},
    "4_team_cards": {{
      "line": 3.5,
      "over_line": {{ "min": 0, "max": 0, "mid": 0.0 }},
      "under_line": {{ "min": 0, "max": 0, "mid": 0.0 }},
      "most_cards_team": "home|away",
      "most_cards_prob": {{ "min": 0, "max": 0, "mid": 0.0 }},
      "drivers": ["<driver cartellini in italiano>"]
    }},
    "5_penalties": {{
      "penalty_yes": {{ "min": 0, "max": 0, "mid": 0.0 }},
      "favored_team": "home|away",
      "drivers": ["<driver rigori in italiano>"]
    }},
    "6_player_goals": [
      {{
        "player": "<Nome Cognome reale>",
        "team": "home|away",
        "role": "ATT|CC",
        "expected_starter": true,
        "set_piece_duty": "none|penalties|freekicks",
        "prob": {{ "min": 0, "max": 0, "mid": 0.0 }},
        "key_metric": "<motivazione in italiano>"
      }}
    ],
    "7_player_cards": [
      {{
        "player": "<Nome Cognome reale>",
        "team": "home|away",
        "role": "DC|CC|TD|TS",
        "cautioned_diffida": false,
        "prob": {{ "min": 0, "max": 0, "mid": 0.0 }},
        "key_metric": "<motivazione in italiano>"
      }}
    ]
  }},
  "data_quality": {{
    "confidence": "medium",
    "sources_used": ["football-data.co.uk", "Google News RSS"],
    "coverage_gaps": ["<limiti reali in italiano>"],
    "pre_kickoff_checks": ["<verifiche pre-gara in italiano>"]
  }}
}}
"""


def enforce_ground_truth(match_obj, home_data, away_data):
    """Sovrascrive i dati oggettivi delle squadre con quelli calcolati in Python per impedire allucinazioni."""
    for side, real_data in (("home", home_data), ("away", away_data)):
        if side in match_obj.get("teams", {}):
            t = match_obj["teams"][side]
            t["name"] = real_data["name"]
            t["position"] = real_data["position"]
            t["points"] = real_data["points"]
            t["goals_for"] = real_data["goals_for"]
            t["goals_against"] = real_data["goals_against"]
            t["xg_per_game"] = real_data["xg_per_game"]
            t["xga_per_game"] = real_data["xga_per_game"]
            t["recent_form"] = real_data["recent_form"]
            t["split_stats"] = {
                "label": real_data["split_stats"]["label"],
                "played": real_data["split_stats"]["played"],
                "points_per_game": real_data["split_stats"]["points_per_game"],
                "gf_per_game": real_data["split_stats"]["gf_per_game"],
                "ga_per_game": real_data["split_stats"]["ga_per_game"],
                "corners_for_avg": real_data["split_stats"]["corners_for_avg"],
                "cards_avg": real_data["split_stats"]["cards_avg"],
            }
    return match_obj


def parse_json_response(text):
    """Pulisce eventuali fence markdown e converte la risposta in dizionario Python."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("Nessun oggetto JSON trovato nella risposta.")
    return json.loads(cleaned[start : end + 1])


def generate_with_fallback(client, prompt):
    """Prova i modelli in MODELS_FALLBACK in sequenza se uno è sovraccarico (503) o esaurito (429)."""
    last_error = None
    for model_name in MODELS_FALLBACK:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.2,
                ),
            )
            return parse_json_response(response.text), model_name
        except Exception as e:
            last_error = e
            print(f"   [WARN] {model_name} non disponibile ({str(e)[:90]}...), provo fallback...")
            time.sleep(3)
    raise last_error


def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Variabile d'ambiente GEMINI_API_KEY mancante.")

    client = genai.Client(api_key=api_key)

    print("1. Scaricamento dati Serie A da football-data.co.uk e Understat...")
    fd_df = fetch_football_data()
    league_table = compute_league_table_and_stats(fd_df)
    print(f"   -> Calcolata classifica reale per {len(league_table)} squadre da I1.csv.")

    dates_data, teams_data = fetch_understat_data()
    fixtures = get_upcoming_fixtures(dates_data)
    print(f"2. Trovate {len(fixtures)} partite nei prossimi {DAYS_AHEAD} giorni.")

    analyzed_matches = []
    for idx, fix in enumerate(fixtures, start=1):
        print(f"[{idx}/{len(fixtures)}] Analisi {fix['home']} vs {fix['away']}...")
        h_data = build_team_payload(fix["home"], "home", league_table, teams_data)
        a_data = build_team_payload(fix["away"], "away", league_table, teams_data)
        news_context = fetch_google_news_context(fix["home"], fix["away"])
        prompt = build_prompt(fix, h_data, a_data, news_context)

        try:
            match_obj, used_model = generate_with_fallback(client, prompt)
            match_obj = enforce_ground_truth(match_obj, h_data, a_data)
            analyzed_matches.append(match_obj)
            print(f"   -> OK ({used_model}): {fix['home']} vs {fix['away']}")
        except Exception as e:
            print(f"   -> [ERRORE] {fix['home']} vs {fix['away']}: {e}")

        if idx < len(fixtures):
            time.sleep(SLEEP_SECONDS)

    output = {
        "schema_version": "3.0",
        "generated_at": datetime.now(timezone(timedelta(hours=1))).isoformat(),
        "matches": analyzed_matches,
    }

    with open("matches.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"Completato! Salvate {len(analyzed_matches)} partite in matches.json.")


if __name__ == "__main__":
    main()
