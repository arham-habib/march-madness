"""
DEPRACATED
Use the Async version in the same folder
"""



import os
import json
import time
import logging
import requests
import pandas as pd
import argparse
from tqdm import tqdm
from pathlib import Path
from typing import Optional, Dict, Any, List

# Constants
BASE_URL = "https://ncaa-api.henrygd.me/game/{}/boxscore"
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent.parent / "data"
LOG_DIR = SCRIPT_DIR.parent.parent / "logs"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

def fetch_box_score(sport: str, division: str, game_id: int) -> Optional[Dict[str, Any]]:
    """Fetch player box score data for a given game."""
    url = BASE_URL.format(game_id)
    retries = 3
    backoff = 1

    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=3)
            if response.status_code == 404:
                logging.warning(f"No game data available for {game_id}.")
                return None
            response.raise_for_status()
            data = response.json()
            if not data or "teams" not in data:
                logging.warning(f"No box score available for game {game_id}.")
                return None
            return data
        except requests.exceptions.RequestException as e:
            logging.warning(f"Attempt {attempt + 1}: Failed to fetch {game_id}: {e}")
            time.sleep(backoff)
            backoff *= 2

    logging.error(f"Failed to fetch box score for game {game_id} after {retries} attempts.")
    return None

def parse_box_scores(data: Dict[str, Any], game_id: int, date: str) -> pd.DataFrame:
    """Extract relevant player stats from the box score data."""
    if not data or "teams" not in data:
        return pd.DataFrame()

    player_list = []
    for team in data["teams"]:
        team_id = team.get("teamId", "")
        team_name = next((t["shortName"] for t in data.get("meta", {}).get("teams", []) if str(t.get("id", "")) == str(team_id)), "Unknown")

        for player in team.get("playerStats", []):
            player_list.append({
                "gameID": game_id,
                "date": date,
                "team": team_name,
                "player": f"{player.get('firstName', '')} {player.get('lastName', '')}".strip(),
                "position": player.get("position", "Unknown"),
                "minutesPlayed": player.get("minutesPlayed", "-1"),
                "fieldGoalsMade": player.get("fieldGoalsMade", "-1"),
                "threePointsMade": player.get("threePointsMade", "-1"),
                "freeThrowsMade": player.get("freeThrowsMade", "-1"),
                "totalRebounds": player.get("totalRebounds", "-1"),
                "offensiveRebounds": player.get("offensiveRebounds", "-1"),
                "assists": player.get("assists", "-1"),
                "personalFouls": player.get("personalFouls", "-1"),
                "steals": player.get("steals", "-1"),
                "turnovers": player.get("turnovers", "-1"),
                "blockedShots": player.get("blockedShots", "-1"),
                "points": player.get("points", "-1"),
            })

    return pd.DataFrame(player_list)

def scrape_box_scores(sport: str, division: str, year: int, game_id: Optional[int] = None):
    """Scrape player box scores for a specific game or all games in a given year."""
    game_file = DATA_DIR / f"ncaab_{year}_{sport}_{division}.parquet"
    if not game_file.exists():
        logging.error(f"Game data file not found: {game_file}")
        raise FileNotFoundError(f"Missing game data file: {game_file}")

    # Load existing game data
    game_data = pd.read_parquet(game_file)
    
    if game_id:
        game_ids = game_data[game_data["url"] == str(game_id)]
    else:
        game_ids = game_data  # Use all games

    all_data = []
    for _, row in tqdm(game_ids.iterrows(), total=len(game_ids), desc="Scraping box scores"):
        game_id = row["url"]
        date = row["date"]
        data = fetch_box_score(sport, division, game_id)
        df = parse_box_scores(data, game_id, date)

        if not df.empty:
            all_data.append(df)
        else:
            logging.warning(f"No data for game {game_id}.")

    # Save full dataset
    if all_data:
        full_df = pd.concat(all_data, ignore_index=True)
        full_df.to_parquet(DATA_DIR / f"box_score_{sport}_{year}_{division}.parquet", index=False)
        logging.info(f"Saved all box scores data ({len(full_df)} rows) to box_scores_{sport}_{year}_{division}.parquet.")

def main():
    parser = argparse.ArgumentParser(description="Scrape NCAA player box scores.")
    parser.add_argument("--sport", type=str, choices=["men", "women"], default="men", help="Sport category: men or women (default: men)")
    parser.add_argument("--division", type=str, choices=["d1", "d2", "d3"], default="d1", help="NCAA division (default: d1)")
    parser.add_argument("--year", type=int, required=True, help="Year to scrape data for")
    parser.add_argument("--game_id", type=int, help="Optional game ID for testing a single game")

    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        filename=LOG_DIR / f"box_score_{args.year}_{args.sport}_{args.division}.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    scrape_box_scores(args.sport, args.division, args.year, args.game_id)

if __name__ == "__main__":
    main()
