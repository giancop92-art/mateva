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


def fetch_football_data():
    """Scarica il CSV stagionale della Serie A da football-data.co.uk."""
    try:
        r = requests.get(FD_CSV_URL, headers=HEADERS, timeout=20)
        r.raise_for_status()
        return pd.read_csv(StringIO(r.text))
    except Exception as e:
        print(f"[WARN] Impossibile scaricare football-data.co.uk: {e}")
        return pd.DataFrame()


def extract_understat_json(html, var_name):
    """Estrae un blocco JSON.parse('...') dalla pagina HTML di Understat."""
    pattern = rf"var\s+{var_name}\s*=\s*JSON\.parse\('([^']+)'\)"
    match = re.search(pattern, html)
    if not match:
        return None
    raw = match.group(1).encode("utf-8").decode("unicode_escape")
    return json.loads(raw)


def fetch_understat_data():
    """Recupera calendario, metriche squadra (xG, xGA, PPDA) e giocatori da Understat."""
    try:
        r = requests.get(UNDERSTAT_URL, headers=HEADERS, timeout=25)
        r.raise_for_status()
        html = r.text
        dates_data = extract_understat_json(html, "datesData") or []
        teams_data = extract_understat_json(html, "teamsData") or {}
        players_data = extract_understat_json(html, "playersData") or []
        return dates_data, teams_data, players_data
    except Exception as e:
        print(f"[WARN] Impossibile scaricare Understat: {e}")
        return [], {}, []


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

    # 2. Fallback pubblico da ESPN se Understat non restituisce gare
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


def fetch_google_news_context(home, away, max_items=8):
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


def summarize_team_stats(team_name, fd_df, teams_data):
    """Calcola un riepilogo numerico pulito per la squadra da passare a Gemini."""
    summary = {"team": team_name, "understat": {}, "football_data": {}}

    # Statistiche Understat (xG, xGA, npxG, Deep)
    for _, tdata in teams_data.items():
        if tdata.get("title", "").lower() == team_name.lower():
            hist = tdata.get("history", [])
            n = len(hist)
            if n > 0:
                xg = sum(float(x.get("xG", 0)) for x in hist)
                xga = sum(float(x.get("xGA", 0)) for x in hist)
                npxg = sum(float(x.get("npxG", 0)) for x in hist)
                scored = sum(int(x.get("scored", 0)) for x in hist)
                missed = sum(int(x.get("missed", 0)) for x in hist)
                pts = sum(int(x.get("pts", 0)) for x in hist)
                deep = sum(int(x.get("deep", 0)) for x in hist)
                form = [x.get("result", "").upper() for x in hist[-5:]]
                summary["understat"] = {
                    "played": n,
                    "points": pts,
                    "goals_for": scored,
                    "goals_against": missed,
                    "xg_per_game": round(xg / n, 2),
                    "xga_per_game": round(xga / n, 2),
                    "npxg_per_game": round(npxg / n, 2),
                    "deep_per_game": round(deep / n, 1),
                    "recent_form": form,
                }
            break

    # Statistiche football-data.co.uk (corner, falli, gialli)
    if not fd_df.empty and "HomeTeam" in fd_df.columns:
        mask_h = fd_df["HomeTeam"].str.lower().str.contains(team_name.lower()[:5], na=False)
        mask_a = fd_df["AwayTeam"].str.lower().str.contains(team_name.lower()[:5], na=False)
        home_df = fd_df[mask_h]
        away_df = fd_df[mask_a]
        if len(home_df) > 0:
            summary["football_data"]["home_corners_avg"] = round(home_df["HC"].mean(), 2) if "HC" in home_df else None
            summary["football_data"]["home_yellows_avg"] = round(home_df["HY"].mean(), 2) if "HY" in home_df else None
        if len(away_df) > 0:
            summary["football_data"]["away_corners_avg"] = round(away_df["AC"].mean(), 2) if "AC" in away_df else None
            summary["football_data"]["away_yellows_avg"] = round(away_df["AY"].mean(), 2) if "AY" in away_df else None

    return summary


def build_prompt(fixture, home_stats, away_stats, news_context):
    """Costruisce il prompt con le regole di football-match-evaluator-v3 e lo schema JSON esatto."""
    slug = re.sub(r"[^a-z0-9]+", "-", f"{fixture['date_slug']}-{fixture['home']}-{fixture['away']}".lower()).strip("-")

    return f"""
Sei un analista calcistico che applica rigorosamente la metodologia 'football-match-evaluator-v3' per la Serie A 2026/27.

Analizza la partita: {fixture['home']} vs {fixture['away']} (Kickoff: {fixture['kickoff_iso']}).

DATI NUMERICI GIA' ESTRATTI (usali come base oggettiva):
- Casa: {json.dumps(home_stats, ensure_ascii=False)}
- Trasferta: {json.dumps(away_stats, ensure_ascii=False)}

NOTIZIE RECENTI (Google News RSS - probabili formazioni, assenze, arbitro):
{news_context}

REGOLE DI VALUTAZIONE:
- Esito 1X2: incrocia differenziale xG/xGA, over/under-performance, ingressi in area (deep), forma, assenze e calendario. La somma dei valori 'mid' di home_win + draw + away_win DEVE fare 100.
- Gol: valuta npxG, palle inattive, transizioni e solidità difensiva. over_2_5.mid + under_2_5.mid = 100; btts_yes.mid + btts_no.mid = 100.
- Corner: valuta ampiezza, blocco basso vs dominio territoriale e medie corner.
- Cartellini: valuta falli, pressing, transizioni e severità arbitro.
- Rigori: valuta ingressi in area e storico rigori.
- Giocatori: indica 2-3 possibili marcatori e 2-3 possibili ammoniti con probabilità realistiche.
- Non inventare dati mancanti: se un numero (es. media arbitro o posizione esatta) non è ricavabile dai dati forniti, usa null e segnalalo in coverage_gaps.

RISPONDI ESCLUSIVAMENTE CON UN SINGOLO OGGETTO JSON VALIDO che rispetti ESATTAMENTE questa struttura:
{{
  "id": "{slug}",
  "competition": "Serie A",
  "context_type": "domestic_league",
  "matchday": "es. 12ª Giornata",
  "kickoff_iso": "{fixture['kickoff_iso']}",
  "venue": "Nome Stadio",
  "referee": {{
    "name": "Nome Arbitro o Da designare",
    "cards_per_game": 4.2,
    "penalties_rate_pct": 25,
    "strictness": "low|medium|high"
  }},
  "teams": {{
    "home": {{
      "name": "{fixture['home']}",
      "position": 10,
      "points": 15,
      "goals_for": 12,
      "goals_against": 14,
      "xg_per_game": 1.25,
      "xga_per_game": 1.40,
      "recent_form": ["W", "D", "L", "W", "L"],
      "split_stats": {{
        "label": "Casa",
        "played": 5,
        "points_per_game": 1.4,
        "gf_per_game": 1.2,
        "ga_per_game": 1.0,
        "corners_for_avg": 4.8,
        "cards_avg": 2.1
      }},
      "absences": {{
        "confirmed_out": ["Giocatore A"],
        "doubtful": ["Giocatore B"]
      }},
      "rest_days": 7,
      "upcoming_cup_fixture": null,
      "probable_formation": "3-5-2"
    }},
    "away": {{
      "name": "{fixture['away']}",
      "position": 4,
      "points": 22,
      "goals_for": 21,
      "goals_against": 10,
      "xg_per_game": 1.95,
      "xga_per_game": 0.95,
      "recent_form": ["W", "W", "D", "W", "L"],
      "split_stats": {{
        "label": "Trasferta",
        "played": 5,
        "points_per_game": 2.0,
        "gf_per_game": 1.8,
        "ga_per_game": 0.8,
        "corners_for_avg": 5.5,
        "cards_avg": 1.8
      }},
      "absences": {{
        "confirmed_out": [],
        "doubtful": []
      }},
      "rest_days": 7,
      "upcoming_cup_fixture": null,
      "probable_formation": "4-3-3"
    }}
  }},
  "analysis": {{
    "1_outcome_1x2": {{
      "home_win": {{ "min": 25, "max": 29, "mid": 27.0 }},
      "draw": {{ "min": 26, "max": 30, "mid": 28.0 }},
      "away_win": {{ "min": 43, "max": 47, "mid": 45.0 }},
      "drivers": ["Driver 1", "Driver 2"],
      "anomalies": ["Eventuale anomalia statistica"]
    }},
    "2_goals": {{
      "over_2_5": {{ "min": 50, "max": 55, "mid": 52.5 }},
      "under_2_5": {{ "min": 45, "max": 50, "mid": 47.5 }},
      "btts_yes": {{ "min": 48, "max": 53, "mid": 50.5 }},
      "btts_no": {{ "min": 47, "max": 52, "mid": 49.5 }},
      "drivers": ["Driver gol 1", "Driver gol 2"]
    }},
    "3_corners": {{
      "line": 9.5,
      "over_line": {{ "min": 48, "max": 53, "mid": 50.5 }},
      "under_line": {{ "min": 47, "max": 52, "mid": 49.5 }},
      "most_corners_team": "away",
      "most_corners_prob": {{ "min": 58, "max": 63, "mid": 60.5 }},
      "drivers": ["Driver corner 1"]
    }},
    "4_team_cards": {{
      "line": 3.5,
      "over_line": {{ "min": 55, "max": 60, "mid": 57.5 }},
      "under_line": {{ "min": 40, "max": 45, "mid": 42.5 }},
      "most_cards_team": "home",
      "most_cards_prob": {{ "min": 54, "max": 59, "mid": 56.5 }},
      "drivers": ["Driver cartellini 1"]
    }},
    "5_penalties": {{
      "penalty_yes": {{ "min": 22, "max": 27, "mid": 24.5 }},
      "favored_team": "away",
      "drivers": ["Driver rigori 1"]
    }},
    "6_player_goals": [
      {{
        "player": "Nome Giocatore",
        "team": "away",
        "role": "ATT",
        "expected_starter": true,
        "set_piece_duty": "penalties",
        "prob": {{ "min": 35, "max": 40, "mid": 37.5 }},
        "key_metric": "Sintesi xG/ruolo"
      }}
    ],
    "7_player_cards": [
      {{
        "player": "Nome Giocatore",
        "team": "home",
        "role": "CC",
        "cautioned_diffida": false,
        "prob": {{ "min": 28, "max": 33, "mid": 30.5 }},
        "key_metric": "Sintesi falli/matchup"
      }}
    ]
  }},
  "data_quality": {{
    "confidence": "medium",
    "sources_used": ["Understat", "football-data.co.uk", "Google News RSS"],
    "coverage_gaps": ["Eventuali limiti dati"],
    "pre_kickoff_checks": ["Verifiche 1h prima del match"]
  }}
}}
"""


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
    dates_data, teams_data, _ = fetch_understat_data()

    fixtures = get_upcoming_fixtures(dates_data)
    print(f"2. Trovate {len(fixtures)} partite nei prossimi {DAYS_AHEAD} giorni.")

    analyzed_matches = []
    for idx, fix in enumerate(fixtures, start=1):
        print(f"[{idx}/{len(fixtures)}] Analisi {fix['home']} vs {fix['away']}...")
        h_stats = summarize_team_stats(fix["home"], fd_df, teams_data)
        a_stats = summarize_team_stats(fix["away"], fd_df, teams_data)
        news_context = fetch_google_news_context(fix["home"], fix["away"])
        prompt = build_prompt(fix, h_stats, a_stats, news_context)

        try:
            match_obj, used_model = generate_with_fallback(client, prompt)
            analyzed_matches.append(match_obj)
            print(f"   -> OK ({used_model}): {fix['home']} vs {fix['away']}")
        except Exception as e:
            print(f"   -> [ERRORE] {fix['home']} vs {fix['away']}: {e}")

        # Pausa per rispettare i limiti RPM del piano gratuito
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
