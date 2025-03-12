import os
import json
import time
import logging
import aiohttp
import asyncio
import pandas as pd
import argparse
from tqdm.asyncio import tqdm
from pathlib import Path
from typing import Optional, Dict, Any, List
from aiolimiter import AsyncLimiter

# Constants
BASE_URL = "https://ncaa-api.henrygd.me/game/{}/boxscore"
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent.parent / "data"
LOG_DIR = SCRIPT_DIR.parent.parent / "logs"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Rate limiter (4 requests per second)
RATE_LIMITER = AsyncLimiter(4, 1)

async def fetch_box_score(session: aiohttp.ClientSession, game_id: int) -> Optional[Dict[str, Any]]:
    """Fetch player box score data for a given game asynchronously."""
    url = BASE_URL.format(game_id)
    retries = 3
    backoff = 1

    for attempt in range(retries):
        async with RATE_LIMITER:
            try:
                async with session.get(url, timeout=5) as response:
                    if response.status == 404:
                        logging.warning(f"No game data available for {game_id}.")
                        return None
                    response.raise_for_status()
                    data = await response.json()
                    if not data or "teams" not in data:
                        logging.warning(f"No box score available for game {game_id}.")
                        return None
                    return data
            except asyncio.TimeoutError:
                logging.warning(f"Timeout on attempt {attempt + 1} for game {game_id}. Retrying...")
                await asyncio.sleep(backoff)
                backoff *= 2
                continue
            except aiohttp.ClientError as e:
                logging.warning(f"Attempt {attempt + 1}: Failed to fetch {game_id}: {e}")
                await asyncio.sleep(backoff)
                backoff *= 2

    logging.error(f"Failed to fetch box score for game {game_id} after {retries} attempts.")
    return None

async def parse_box_scores(data: Dict[str, Any], game_id: int, date: str) -> pd.DataFrame:
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

async def scrape_box_scores(sport: str, division: str, year: int, game_id: Optional[int] = None):
    """Scrape player box scores asynchronously."""
    game_file = DATA_DIR / f"ncaab_{year}_{sport}_{division}.parquet"
    if not game_file.exists():
        logging.error(f"Game data file not found: {game_file}")
        raise FileNotFoundError(f"Missing game data file: {game_file}")

    game_data = pd.read_parquet(game_file)
    game_ids = game_data[game_data["url"] == str(game_id)] if game_id else game_data

    async with aiohttp.ClientSession() as session:
        tasks = [fetch_box_score(session, row["url"]) for _, row in game_ids.iterrows()]
        responses = await tqdm.gather(*tasks)

    all_data = []
    for response, (_, row) in zip(responses, game_ids.iterrows()):
        if response:
            df = await parse_box_scores(response, row["url"], row["date"])
            if not df.empty:
                all_data.append(df)
        else:
            logging.warning(f"No data for game {row['url']}.")

    if all_data:
        full_df = pd.concat(all_data, ignore_index=True)
        full_df.to_parquet(DATA_DIR / f"box_score_{sport}_{year}_{division}.parquet", index=False)
        logging.info(f"Saved all box scores data ({len(full_df)} rows) to box_scores_{sport}_{year}_{division}.parquet.")

async def main():
    parser = argparse.ArgumentParser(description="Scrape NCAA player box scores asynchronously.")
    parser.add_argument("--sport", type=str, choices=["men", "women"], default="men", help="Sport category: men or women (default: men)")
    parser.add_argument("--division", type=str, choices=["d1", "d2", "d3"], default="d1", help="NCAA division (default: d1)")
    parser.add_argument("--year", type=int, required=True, help="Year to scrape data for")
    parser.add_argument("--game_id", type=int, help="Optional game ID for testing a single game")
    
    args = parser.parse_args()
    
    logging.basicConfig(
        filename=LOG_DIR / f"box_score_{args.year}_{args.sport}_{args.division}.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    
    await scrape_box_scores(args.sport, args.division, args.year, args.game_id)

if __name__ == "__main__":
    asyncio.run(main())
