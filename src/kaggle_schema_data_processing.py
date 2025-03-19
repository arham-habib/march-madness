import polars as pl
import numpy as np

"""
Feature engineering pipeline for March Madness data.
- Shuffles data randomly between home and away to remove location bias for model training.
"""

class MarchMadnessKaggleFeaturePipeline:
    def __init__(self, game_data_df: pl.DataFrame, 
                 initial_elo: float = 1500, k_factor: float = 20, ema_alpha: float = 0.1):
        """
        Initialize with game-level data containing team statistics.

        Params:
            game_data_df: Polars DataFrame with game matchups and team stats (new schema).
            initial_elo: Default starting ELO for new teams.
            k_factor: ELO adjustment factor.
            ema_alpha: Smoothing factor for EMA.
        """
        self.game_data_df = game_data_df.with_row_count("gameID")  # Add unique gameID
        self.initial_elo = initial_elo
        self.k_factor = k_factor
        self.ema_alpha = ema_alpha
        self.team_elos = {}

        self._create_games_df()  # Derive games_df from game_data_df

    def _create_games_df(self) -> None:
        """Create games_df by mapping winning/losing teams to home/away based on WLoc."""
        self.games_df = self.game_data_df.with_columns([
            pl.when(pl.col("WLoc") == "H")
              .then(pl.col("WTeamID"))
              .when(pl.col("WLoc") == "A")
              .then(pl.col("LTeamID"))
              .otherwise(pl.col("WTeamID"))
              .alias("home_team"),
            
            pl.when(pl.col("WLoc") == "H")
              .then(pl.col("LTeamID"))
              .when(pl.col("WLoc") == "A")
              .then(pl.col("WTeamID"))
              .otherwise(pl.col("LTeamID"))
              .alias("away_team"),
            
            pl.when(pl.col("WLoc") == "H")
              .then(pl.col("WScore"))
              .when(pl.col("WLoc") == "A")
              .then(pl.col("LScore"))
              .otherwise(pl.col("WScore"))
              .alias("home_score"),
            
            pl.when(pl.col("WLoc") == "H")
              .then(pl.col("LScore"))
              .when(pl.col("WLoc") == "A")
              .then(pl.col("WScore"))
              .otherwise(pl.col("LScore"))
              .alias("away_score"),
        ]).select([
            "gameID", "Season", "DayNum", "home_team", "away_team", "home_score", "away_score"
        ])

    def _expected_result(self, rating_a: float, rating_b: float) -> float:
        """Calculate expected score for team A based on ELO ratings."""
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

    def _update_elo(self, elo_a: float, elo_b: float, score_a: float, score_b: float) -> tuple:
        """Update ELO ratings based on game result using 538's formula."""
        mov = score_a - score_b if score_a > score_b else score_b - score_a
        expected_a = self._expected_result(elo_a, elo_b)
        result_a = 1 if score_a > score_b else 0
        mov_multiplier = ((mov + 3) ** 0.8) / (7.5 + 0.006 * (elo_a - elo_b))
        k_mov = self.k_factor * mov_multiplier
        new_elo_a = elo_a + k_mov * (result_a - expected_a)
        new_elo_b = elo_b + k_mov * ((1 - result_a) - (1 - expected_a))
        return new_elo_a, new_elo_b

    def generate_elo_features(self) -> None:
        """Compute ELO ratings for teams, including lagged values and changes."""
        game_rows = []
        self.games_df = self.games_df.sort(["Season", "DayNum"], descending=[False, False])  # Sort chronologically descending

        for row in self.games_df.iter_rows(named=True):
            home, away = row["home_team"], row["away_team"]
            home_score, away_score = row["home_score"], row["away_score"]

            home_elo = self.team_elos.get(home, self.initial_elo)
            away_elo = self.team_elos.get(away, self.initial_elo)

            home_lagged_elo = home_elo
            away_lagged_elo = away_elo

            new_home_elo, new_away_elo = self._update_elo(home_elo, away_elo, home_score, away_score)

            home_elo_change = new_home_elo - home_elo
            away_elo_change = new_away_elo - away_elo

            self.team_elos[home] = new_home_elo
            self.team_elos[away] = new_away_elo

            game_rows.append({
                **row,
                "elo_home": new_home_elo,
                "elo_away": new_away_elo,
                "lagged_elo_home": home_lagged_elo,
                "lagged_elo_away": away_lagged_elo,
                "elo_change_home": home_elo_change,
                "elo_change_away": away_elo_change
            })
        elo_df = pl.DataFrame(game_rows)
        elo_df = elo_df.with_columns([
            pl.col("elo_change_home").shift(1).alias("lagged_elo_change_home"),
            pl.col("elo_change_away").shift(1).alias("lagged_elo_change_away"),
            pl.col("elo_change_home").shift(1).ewm_mean(alpha=self.ema_alpha).alias("elo_change_ema_home"),
            pl.col("elo_change_away").shift(1).ewm_mean(alpha=self.ema_alpha).alias("elo_change_ema_away")
        ])
        elo_df = elo_df.with_columns(pl.col('gameID').cast(pl.UInt32))
        self.games_df = elo_df

    def aggregate_team_stats(self) -> pl.DataFrame:
        """Extract team-level stats from game_data_df for winning and losing teams."""
        winning_stats = self.game_data_df.select([
            "gameID", "Season", "DayNum",
            pl.col("WTeamID").alias("team"),
            pl.col("WFGM").alias("fg_made"),
            pl.col("WFGA").alias("fg_attempted"),
            (pl.col("WFGM") - pl.col("WFGM3")).alias("2p_made"),
            (pl.col("WFGA") - pl.col("WFGA3")).alias("2p_attempted"),
            pl.col("WFGM3").alias("3p_made"),
            pl.col("WFGA3").alias("3p_attempted"),
            pl.col("WFTM").alias("ft_made"),
            pl.col("WFTA").alias("ft_attempted"),
            pl.col("WScore").alias("team_points"),
            pl.col("WOR").alias("off_rebounds"),
            pl.col("WDR").alias("def_rebounds"),
            (pl.col("WOR") + pl.col("WDR")).alias("total_rebounds"),
            pl.col("WAst").alias("assists"),
            pl.col("WTO").alias("turnovers"),
            pl.col("WStl").alias("steals"),
            pl.col("WBlk").alias("blocks"),
            pl.col("WPF").alias("fouls")
        ])
        
        losing_stats = self.game_data_df.select([
            "gameID", "Season", "DayNum",
            pl.col("LTeamID").alias("team"),
            pl.col("LFGM").alias("fg_made"),
            pl.col("LFGA").alias("fg_attempted"),
            (pl.col("LFGM") - pl.col("LFGM3")).alias("2p_made"),
            (pl.col("LFGA") - pl.col("LFGA3")).alias("2p_attempted"),
            pl.col("LFGM3").alias("3p_made"),
            pl.col("LFGA3").alias("3p_attempted"),
            pl.col("LFTM").alias("ft_made"),
            pl.col("LFTA").alias("ft_attempted"),
            pl.col("LScore").alias("team_points"),
            pl.col("LOR").alias("off_rebounds"),
            pl.col("LDR").alias("def_rebounds"),
            (pl.col("LOR") + pl.col("LDR")).alias("total_rebounds"),
            pl.col("LAst").alias("assists"),
            pl.col("LTO").alias("turnovers"),
            pl.col("LStl").alias("steals"),
            pl.col("LBlk").alias("blocks"),
            pl.col("LPF").alias("fouls")
        ])
        
        return pl.concat([winning_stats, losing_stats]).sort(["Season", "DayNum"], descending=[False, False])

    def merge_opponent_stats(self, df: pl.DataFrame) -> pl.DataFrame:
        """Merge opponent stats to compute rebound rates, block rate, and ratings."""
        df = df.join(
            df.select([
                "gameID", "team", "off_rebounds", "def_rebounds", "fg_attempted",
                "turnovers", "ft_attempted", "team_points"
            ]).rename({
                "team": "opponent",
                "off_rebounds": "opponent_off_rebounds",
                "def_rebounds": "opponent_def_rebounds",
                "fg_attempted": "opponent_fg_attempted",
                "turnovers": "opponent_turnovers",
                "ft_attempted": "opponent_ft_attempted",
                "team_points": "opponent_points"
            }),
            left_on="gameID", right_on="gameID", how="left"
        ).filter(pl.col("team") != pl.col("opponent"))

        df = df.with_columns([
            (pl.col("fg_attempted") - pl.col("off_rebounds") + pl.col("turnovers") + (0.44 * pl.col("ft_attempted"))).alias("possessions"),
            (pl.col("opponent_fg_attempted") - pl.col("opponent_off_rebounds") + pl.col("opponent_turnovers") + (0.44 * pl.col("opponent_ft_attempted"))).alias("opponent_possessions"),
        ])

        df = df.with_columns([
            (pl.col("team_points") / pl.col("possessions") * 100).alias("offensive_rating"),
            (pl.col("opponent_points") / pl.col("opponent_possessions") * 100).alias("defensive_rating"),
            (pl.col("off_rebounds") / (pl.col("off_rebounds") + pl.col("opponent_def_rebounds"))).alias("off_rebound_rate"),
            (pl.col("def_rebounds") / (pl.col("def_rebounds") + pl.col("opponent_off_rebounds"))).alias("def_rebound_rate"),
            (pl.col("blocks") / (pl.col("opponent_fg_attempted") + 1)).alias("block_rate")
        ])
        return df.sort(["Season", "DayNum"], descending=[False, False])

    def compute_efficiency_metrics(self, df: pl.DataFrame) -> pl.DataFrame:
        """Compute field goal percentages, turnover rate, and assist/turnover ratio."""
        return df.with_columns([
            (pl.col("2p_made") / pl.col("2p_attempted")).alias("2p_pct"),
            (pl.col("3p_made") / pl.col("3p_attempted")).alias("3p_pct"),
            (pl.col("ft_made") / pl.col("ft_attempted")).alias("ft_pct"),
            (pl.col("turnovers") / (pl.col("2p_attempted") + pl.col("3p_attempted") + 0.44 * pl.col("ft_attempted") + pl.col("turnovers"))).alias("turnover_rate"),
            (pl.col("assists") / (pl.col("turnovers") + 1)).alias("assist_turnover")
        ]).sort(["Season", "DayNum"], descending=[False, False])

    def compute_ema(self, df: pl.DataFrame) -> pl.DataFrame:
        """Compute exponential moving averages (EMA) over past games."""
        return (
            df.sort(["team", "Season", "DayNum"], descending=[False, False, False])
            .with_columns([
                pl.col("team_points").shift(1).ewm_mean(self.ema_alpha, min_periods=10).over("team").alias("ema_points"),
                pl.col("2p_pct").shift(1).ewm_mean(self.ema_alpha, min_periods=10).over("team").alias("ema_2p_pct"),
                pl.col("3p_pct").shift(1).ewm_mean(self.ema_alpha, min_periods=10).over("team").alias("ema_3p_pct"),
                pl.col("ft_pct").shift(1).ewm_mean(self.ema_alpha, min_periods=10).over("team").alias("ema_ft_pct"),
                pl.col("off_rebound_rate").shift(1).ewm_mean(self.ema_alpha, min_periods=10).over("team").alias("ema_off_rebound_rate"),
                pl.col("def_rebound_rate").shift(1).ewm_mean(self.ema_alpha, min_periods=10).over("team").alias("ema_def_rebound_rate"),
                pl.col("turnover_rate").shift(1).ewm_mean(self.ema_alpha, min_periods=10).over("team").alias("ema_turnover_rate"),
                pl.col("assist_turnover").shift(1).ewm_mean(self.ema_alpha, min_periods=10).over("team").alias("ema_assist_turnover"),
                pl.col("block_rate").shift(1).ewm_mean(self.ema_alpha, min_periods=10).over("team").alias("ema_block_rate"),
                pl.col("offensive_rating").shift(1).ewm_mean(self.ema_alpha, min_periods=10).over("team").alias("ema_offensive_rating"),
                pl.col("defensive_rating").shift(1).ewm_mean(self.ema_alpha, min_periods=10).over("team").alias("ema_defensive_rating")
            ])
        )

    def merge_with_matchups(self, df: pl.DataFrame) -> pl.DataFrame:
        """Merge EMA features with game matchups."""
        df_home = df.rename({col: col + "_home" for col in df.columns if col not in ["gameID", "team"]})
        df_away = df.rename({col: col + "_away" for col in df.columns if col not in ["gameID", "team"]})
        return (
            self.games_df
            .join(df_home, left_on=["gameID", "home_team"], right_on=["gameID", "team"])
            .join(df_away, left_on=["gameID", "away_team"], right_on=["gameID", "team"])
            .select([
                "gameID", "Season", "DayNum", "home_team", "away_team", "home_score", "away_score",
                # ELO features
                "elo_home", "elo_change_home", "lagged_elo_home", "lagged_elo_change_home", "elo_change_ema_home",
                "elo_away", "elo_change_away", "lagged_elo_away", "lagged_elo_change_away", "elo_change_ema_away",
                # EMA Features for Home Team
                "ema_points_home", "ema_2p_pct_home", "ema_3p_pct_home", "ema_ft_pct_home", "ema_offensive_rating_home", "ema_defensive_rating_home",
                "ema_off_rebound_rate_home", "ema_def_rebound_rate_home", "ema_turnover_rate_home", "ema_assist_turnover_home", "ema_block_rate_home",
                # EMA Features for Away Team
                "ema_points_away", "ema_2p_pct_away", "ema_3p_pct_away", "ema_ft_pct_away", "ema_offensive_rating_away", "ema_defensive_rating_away",
                "ema_off_rebound_rate_away", "ema_def_rebound_rate_away", "ema_turnover_rate_away", "ema_assist_turnover_away", "ema_block_rate_away",
                # Result
                (pl.col("home_score") > pl.col("away_score")).alias("home_win")
            ]).sort(["Season", "DayNum"], descending=[False, False])
        )

    def unpack_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Unpack home/away features into per-team statistics."""
        home_records = df.select([
            "gameID", "Season", "DayNum",
            pl.col("home_team").alias("team"),
            pl.col("elo_home").alias("elo"),
            pl.col("elo_change_home").alias("elo_change"),
            pl.col("lagged_elo_home").alias("lagged_elo"),
            pl.col("lagged_elo_change_home").alias("lagged_elo_change"),
            pl.col("elo_change_ema_home").alias("elo_change_ema"),
            pl.col("ema_points_home").alias("ema_points"),
            pl.col("ema_2p_pct_home").alias("ema_2p_pct"),
            pl.col("ema_3p_pct_home").alias("ema_3p_pct"),
            pl.col("ema_ft_pct_home").alias("ema_ft_pct"),
            pl.col("ema_off_rebound_rate_home").alias("ema_off_rebound_rate"),
            pl.col("ema_def_rebound_rate_home").alias("ema_def_rebound_rate"),
            pl.col("ema_turnover_rate_home").alias("ema_turnover_rate"),
            pl.col("ema_assist_turnover_home").alias("ema_assist_turnover"),
            pl.col("ema_block_rate_home").alias("ema_block_rate"),
            pl.col("ema_offensive_rating_home").alias("ema_offensive_rating"),
            pl.col("ema_defensive_rating_home").alias("ema_defensive_rating")
        ])

        away_records = df.select([
            "gameID", "Season", "DayNum",
            pl.col("away_team").alias("team"),
            pl.col("elo_away").alias("elo"),
            pl.col("elo_change_away").alias("elo_change"),
            pl.col("lagged_elo_away").alias("lagged_elo"),
            pl.col("lagged_elo_change_away").alias("lagged_elo_change"),
            pl.col("elo_change_ema_away").alias("elo_change_ema"),
            pl.col("ema_points_away").alias("ema_points"),
            pl.col("ema_2p_pct_away").alias("ema_2p_pct"),
            pl.col("ema_3p_pct_away").alias("ema_3p_pct"),
            pl.col("ema_ft_pct_away").alias("ema_ft_pct"),
            pl.col("ema_off_rebound_rate_away").alias("ema_off_rebound_rate"),
            pl.col("ema_def_rebound_rate_away").alias("ema_def_rebound_rate"),
            pl.col("ema_turnover_rate_away").alias("ema_turnover_rate"),
            pl.col("ema_assist_turnover_away").alias("ema_assist_turnover"),
            pl.col("ema_block_rate_away").alias("ema_block_rate"),
            pl.col("ema_offensive_rating_away").alias("ema_offensive_rating"),
            pl.col("ema_defensive_rating_away").alias("ema_defensive_rating")
        ])

        return pl.concat([home_records, away_records]).sort(["Season", "DayNum", "gameID", "team"], descending=[False, False, False, False])

    def randomize_home_away(self, df: pl.DataFrame) -> pl.DataFrame:
        """Randomly shuffle home/away teams to remove location bias."""
        df = df.clone()
        np.random.seed(42)  # Reproducibility
        n_rows = df.height
        n_swaps = n_rows // 2
        swap_mask = np.zeros(n_rows, dtype=bool)
        swap_mask[:n_swaps] = True
        np.random.shuffle(swap_mask)
        df = df.with_columns(pl.lit(swap_mask).alias("swap"))

        df_swapped = df.with_columns([
            # Swap teams and scores
            pl.when(pl.col("swap")).then(pl.col("away_team")).otherwise(pl.col("home_team")).alias("home_team"),
            pl.when(pl.col("swap")).then(pl.col("home_team")).otherwise(pl.col("away_team")).alias("away_team"),
            pl.when(pl.col("swap")).then(pl.col("away_score")).otherwise(pl.col("home_score")).alias("home_score"),
            pl.when(pl.col("swap")).then(pl.col("home_score")).otherwise(pl.col("away_score")).alias("away_score"),
            
            # Swap ELO features
            pl.when(pl.col("swap")).then(pl.col("elo_away")).otherwise(pl.col("elo_home")).alias("elo_home"),
            pl.when(pl.col("swap")).then(pl.col("elo_change_away")).otherwise(pl.col("elo_change_home")).alias("elo_change_home"),
            pl.when(pl.col("swap")).then(pl.col("lagged_elo_away")).otherwise(pl.col("lagged_elo_home")).alias("lagged_elo_home"),
            pl.when(pl.col("swap")).then(pl.col("lagged_elo_change_away")).otherwise(pl.col("lagged_elo_change_home")).alias("lagged_elo_change_home"),
            pl.when(pl.col("swap")).then(pl.col("elo_change_ema_away")).otherwise(pl.col("elo_change_ema_home")).alias("elo_change_ema_home"),
            pl.when(pl.col("swap")).then(pl.col("elo_home")).otherwise(pl.col("elo_away")).alias("elo_away"),
            pl.when(pl.col("swap")).then(pl.col("elo_change_home")).otherwise(pl.col("elo_change_away")).alias("elo_change_away"),
            pl.when(pl.col("swap")).then(pl.col("lagged_elo_home")).otherwise(pl.col("lagged_elo_away")).alias("lagged_elo_away"),
            pl.when(pl.col("swap")).then(pl.col("lagged_elo_change_home")).otherwise(pl.col("lagged_elo_change_away")).alias("lagged_elo_change_away"),
            pl.when(pl.col("swap")).then(pl.col("elo_change_ema_home")).otherwise(pl.col("elo_change_ema_away")).alias("elo_change_ema_away"),

            # Swap EMA features
            pl.when(pl.col("swap")).then(pl.col("ema_points_away")).otherwise(pl.col("ema_points_home")).alias("ema_points_home"),
            pl.when(pl.col("swap")).then(pl.col("ema_2p_pct_away")).otherwise(pl.col("ema_2p_pct_home")).alias("ema_2p_pct_home"),
            pl.when(pl.col("swap")).then(pl.col("ema_3p_pct_away")).otherwise(pl.col("ema_3p_pct_home")).alias("ema_3p_pct_home"),
            pl.when(pl.col("swap")).then(pl.col("ema_ft_pct_away")).otherwise(pl.col("ema_ft_pct_home")).alias("ema_ft_pct_home"),
            pl.when(pl.col("swap")).then(pl.col("ema_off_rebound_rate_away")).otherwise(pl.col("ema_off_rebound_rate_home")).alias("ema_off_rebound_rate_home"),
            pl.when(pl.col("swap")).then(pl.col("ema_def_rebound_rate_away")).otherwise(pl.col("ema_def_rebound_rate_home")).alias("ema_def_rebound_rate_home"),
            pl.when(pl.col("swap")).then(pl.col("ema_turnover_rate_away")).otherwise(pl.col("ema_turnover_rate_home")).alias("ema_turnover_rate_home"),
            pl.when(pl.col("swap")).then(pl.col("ema_assist_turnover_away")).otherwise(pl.col("ema_assist_turnover_home")).alias("ema_assist_turnover_home"),
            pl.when(pl.col("swap")).then(pl.col("ema_block_rate_away")).otherwise(pl.col("ema_block_rate_home")).alias("ema_block_rate_home"),
            pl.when(pl.col("swap")).then(pl.col("ema_offensive_rating_away")).otherwise(pl.col("ema_offensive_rating_home")).alias("ema_offensive_rating_home"),
            pl.when(pl.col("swap")).then(pl.col("ema_defensive_rating_away")).otherwise(pl.col("ema_defensive_rating_home")).alias("ema_defensive_rating_home"),
            pl.when(pl.col("swap")).then(pl.col("ema_points_home")).otherwise(pl.col("ema_points_away")).alias("ema_points_away"),
            pl.when(pl.col("swap")).then(pl.col("ema_2p_pct_home")).otherwise(pl.col("ema_2p_pct_away")).alias("ema_2p_pct_away"),
            pl.when(pl.col("swap")).then(pl.col("ema_3p_pct_home")).otherwise(pl.col("ema_3p_pct_away")).alias("ema_3p_pct_away"),
            pl.when(pl.col("swap")).then(pl.col("ema_ft_pct_home")).otherwise(pl.col("ema_ft_pct_away")).alias("ema_ft_pct_away"),
            pl.when(pl.col("swap")).then(pl.col("ema_off_rebound_rate_home")).otherwise(pl.col("ema_off_rebound_rate_away")).alias("ema_off_rebound_rate_away"),
            pl.when(pl.col("swap")).then(pl.col("ema_def_rebound_rate_home")).otherwise(pl.col("ema_def_rebound_rate_away")).alias("ema_def_rebound_rate_away"),
            pl.when(pl.col("swap")).then(pl.col("ema_turnover_rate_home")).otherwise(pl.col("ema_turnover_rate_away")).alias("ema_turnover_rate_away"),
            pl.when(pl.col("swap")).then(pl.col("ema_assist_turnover_home")).otherwise(pl.col("ema_assist_turnover_away")).alias("ema_assist_turnover_away"),
            pl.when(pl.col("swap")).then(pl.col("ema_block_rate_home")).otherwise(pl.col("ema_block_rate_away")).alias("ema_block_rate_away"),
            pl.when(pl.col("swap")).then(pl.col("ema_offensive_rating_home")).otherwise(pl.col("ema_offensive_rating_away")).alias("ema_offensive_rating_away"),
            pl.when(pl.col("swap")).then(pl.col("ema_defensive_rating_home")).otherwise(pl.col("ema_defensive_rating_away")).alias("ema_defensive_rating_away"),

            # Swap result
            pl.when(pl.col("swap")).then(~pl.col("home_win")).otherwise(pl.col("home_win")).alias("home_win")
        ])
        return df_swapped.drop("swap").sort(["Season", "DayNum"], descending=[False, False])

    def build_features(self) -> tuple:
        """Run the full EMA-based feature engineering pipeline."""
        self.generate_elo_features()
        team_stats = self.aggregate_team_stats()
        self.team_stats = team_stats
        opponent_stats = self.merge_opponent_stats(team_stats)
        efficiency_metrics = self.compute_efficiency_metrics(opponent_stats)
        ema_features = self.compute_ema(efficiency_metrics)
        self.team_features = ema_features
        merged_data = self.merge_with_matchups(ema_features)
        merged_data_randomized = self.randomize_home_away(merged_data)
        merged_data_unpacked = self.unpack_features(merged_data_randomized)
        return merged_data_randomized, merged_data_unpacked