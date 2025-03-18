import polars as pl

def load_player_stats(filepath: str) -> pl.LazyFrame:
    """
    Load and preprocess player statistics from a parquet file.
    
    Args:
        filepath: Path to the parquet file containing player statistics
        
    Returns:
        LazyFrame with preprocessed player statistics
    """
    return (
        pl.scan_parquet(filepath)
        .unique()
        .sort("gameID", "date", "player")
        .group_by(["gameID", "date", "player"])
        .first()
        .with_columns([
            pl.col("date").str.strptime(pl.Date, format="%Y-%m-%d"),
            pl.col("minutesPlayed").cast(pl.Int32, strict=False),
            pl.col("fieldGoalsMade").str.split("-").list.first().cast(pl.Int32, strict=False).alias("fieldGoalsMade"),
            pl.col("fieldGoalsMade").str.split("-").list.last().cast(pl.Int32, strict=False).alias("fieldGoalsAttempted"), 
            pl.col("threePointsMade").str.split("-").list.first().cast(pl.Int32, strict=False).alias("threePointsMade"),
            pl.col("threePointsMade").str.split("-").list.last().cast(pl.Int32, strict=False).alias("threePointsAttempted"),
            pl.col("freeThrowsMade").str.split("-").list.first().cast(pl.Int32, strict=False).alias("freeThrowsMade"),
            pl.col("freeThrowsMade").str.split("-").list.last().cast(pl.Int32, strict=False).alias("freeThrowsAttempted"),
            pl.col("totalRebounds").cast(pl.Int32, strict=False),
            pl.col("offensiveRebounds").cast(pl.Int32, strict=False),
            pl.col("assists").cast(pl.Int32, strict=False),
            pl.col("personalFouls").cast(pl.Int32, strict=False),
            pl.col("steals").cast(pl.Int32, strict=False),
            pl.col("turnovers").cast(pl.Int32, strict=False),
            pl.col("blockedShots").cast(pl.Int32, strict=False),
            pl.col("points").cast(pl.Int32, strict=False)
        ])
    )

def load_schedule(filepath: str) -> pl.LazyFrame:
    """
    Load and preprocess game schedule data from a parquet file.
    
    Args:
        filepath: Path to the parquet file containing game schedule data
        
    Returns:
        LazyFrame with preprocessed game schedule data
    """
    return (
        pl.scan_parquet(filepath)
        .filter(
            (pl.col("home_score").str.strip_chars() != "") & 
            (pl.col("away_score").str.strip_chars() != "")
        )
        .with_columns([
            pl.col("date").str.strptime(pl.Date, format="%Y-%m-%d"),
            pl.col("home_score").cast(pl.Int32),
            pl.col("away_score").cast(pl.Int32), 
            pl.col("url").alias("gameID")
        ])
        .unique()
    )