import polars as pl

class MarchMadnessFeaturePipeline():

    def __init__(self, player_stats_df: pl.DataFrame, games_df: pl.DataFrame, 
                 initial_elo: float = 1500, k_factor: float = 20, ema_alpha: float = 0.1):
        """
        Initialize with player-level stats and game-level metadata.

        Params:
            player_stats_df: Polars dataframe containing player-level stats per game.
            games_df: Polars dataframe containing game matchups and outcomes.
            initial_elo: Default starting ELO for new teams.
            k_factor: ELO adjustment factor.
            ema_alpha: Smoothing factor for EMA.
        """
        self.player_stats_df = player_stats_df
        self.games_df = games_df
        self.initial_elo = initial_elo
        self.k_factor = k_factor
        self.ema_alpha = ema_alpha
        self.team_elos = {}

        self._validate_data()


    def _validate_data(self) -> None:
        """
        Validates input dataframes for duplicate entries and drops them.
        """
        # Drop duplicate game IDs in games_df
        self.games_df = self.games_df.unique(subset=["gameID"])

        # Drop duplicate player-team-game combinations in player_stats_df 
        self.player_stats_df = self.player_stats_df.unique(
            subset=["player", "team", "gameID"]
        )


    def _expected_result(self, rating_a: float, rating_b: float) -> float:
        """Calculate expected score for team A based on ELO ratings."""
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))


    def _update_elo(self, elo_a: float, elo_b: float, score_a: float, score_b: float) -> tuple:
        """Update ELO ratings based on game result."""
        expected_a = self._expected_result(elo_a, elo_b)
        result_a = 1 if score_a > score_b else 0  # Win = 1, Loss = 0
        new_elo_a = elo_a + self.k_factor * (result_a - expected_a)
        new_elo_b = elo_b + self.k_factor * ((1 - result_a) - (1 - expected_a))
        return new_elo_a, new_elo_b


    def generate_elo_features(self) -> None:
        """Compute ELO ratings for teams over a season, including lagged values and changes."""
        game_rows = []
        self.games_df = self.games_df.sort("date")  # Ensure chronological order

        for row in self.games_df.iter_rows(named=True):
            home, away = row["home_team"], row["away_team"]
            home_score, away_score = row["home_score"], row["away_score"]

            # Retrieve team ELOs or assign initial values
            home_elo = self.team_elos.get(home, self.initial_elo)
            away_elo = self.team_elos.get(away, self.initial_elo)

            # Compute lagged ELO before the game
            home_lagged_elo = home_elo
            away_lagged_elo = away_elo

            # compute new ELO
            new_home_elo, new_away_elo = self._update_elo(home_elo, away_elo, home_score, away_score)

            # Compute ELO changes
            home_elo_change = new_home_elo - home_elo
            away_elo_change = new_away_elo - away_elo

            # store ELO changes
            self.team_elos[home] = new_home_elo
            self.team_elos[away] = new_away_elo

            # Store game result with ELOs
            game_rows.append({
                **row,
                "home_elo": new_home_elo,
                "away_elo": new_away_elo,
                "home_lagged_elo": home_lagged_elo,
                "away_lagged_elo": away_lagged_elo,
                "home_elo_change": home_elo_change,
                "away_elo_change": away_elo_change
            })
        elo_df = pl.DataFrame(game_rows)
        elo_df = elo_df.with_columns([
            pl.col("home_elo_change").shift(1).alias("home_lagged_elo_change"),
            pl.col("away_elo_change").shift(1).alias("away_lagged_elo_change"),
            (pl.col("home_elo") - pl.col("away_elo")).alias("elo_difference"),
            (pl.col("home_elo") - pl.col("away_elo")).shift(1).alias("elo_difference_lagged"),
            pl.col("home_elo_change").shift(1).ewm_mean(alpha=self.ema_alpha).alias("home_elo_change_ema"),
            pl.col("away_elo_change").shift(1).ewm_mean(alpha=self.ema_alpha).alias("away_elo_change_ema")
        ])

        self.games_df = elo_df


    def aggregate_team_stats(self) -> pl.DataFrame:
        """Aggregate player-level stats into team-level stats per game"""
        return (
            self.player_stats_df
            .group_by(["gameID", "team", "date"])
            .agg([
                pl.sum("fieldGoalsMade").alias("fg_made"),
                pl.sum("fieldGoalsAttempted").alias("fg_attempted"),
                (pl.sum("fieldGoalsMade") - pl.sum("threePointsMade")).alias("2p_made"), 
                (pl.sum("fieldGoalsAttempted") - pl.sum("threePointsAttempted")).alias("2p_attempted"),
                pl.sum("threePointsMade").alias("3p_made"), 
                pl.sum("threePointsAttempted").alias("3p_attempted"),
                pl.sum("freeThrowsMade").alias("ft_made"), 
                pl.sum("freeThrowsAttempted").alias("ft_attempted"),
                pl.sum("points").alias("team_points"),

                pl.sum("totalRebounds").alias("total_rebounds"),
                pl.sum("offensiveRebounds").alias("off_rebounds"), 
                (pl.sum("totalRebounds") - pl.sum("offensiveRebounds")).alias("def_rebounds"),

                pl.sum("assists").alias("assists"),
                pl.sum("turnovers").alias("turnovers"),
                pl.sum("steals").alias("steals"),
                pl.sum("blockedShots").alias("blocks"),
                pl.sum("personalFouls").alias("fouls")
            ])
        )
    

    def merge_opponent_stats(self, df: pl.DataFrame) -> pl.DataFrame:
        """Merges opponent team stats to compute rebound rate, block rate, and offensive/defensive ratings."""
        df = df.join(
            df.select([
                "gameID", "team", "off_rebounds", "def_rebounds", "fg_attempted",
                "turnovers", "ft_attempted", "team_points"
            ])
            .rename({
                "team": "opponent", 
                "off_rebounds": "opponent_off_rebounds",
                "def_rebounds": "opponent_def_rebounds",
                "fg_attempted": "opponent_fg_attempted",
                "turnovers": "opponent_turnovers",
                "ft_attempted": "opponent_ft_attempted",
                "team_points": "opponent_points"
            }),
            left_on="gameID", right_on="gameID", how="left"
        ).filter(pl.col("team") != pl.col("opponent"))  # Remove self-joins

        # Compute possessions for both teams
        df = df.with_columns([
            (pl.col("fg_attempted") - pl.col("off_rebounds") + pl.col("turnovers") + (0.44 * pl.col("ft_attempted"))).alias("possessions"),
            (pl.col("opponent_fg_attempted") - pl.col("opponent_off_rebounds") + pl.col("opponent_turnovers") + (0.44 * pl.col("opponent_ft_attempted"))).alias("opponent_possessions"),
        ])

        # Compute efficiency metrics
        df = df.with_columns([
            (pl.col("team_points") / pl.col("possessions") * 100).alias("offensive_rating"),  # ORTG
            (pl.col("opponent_points") / pl.col("opponent_possessions") * 100).alias("defensive_rating"),  # DRTG
            # Rebound Rates
            (pl.col("off_rebounds") / (pl.col("off_rebounds") + pl.col("opponent_def_rebounds"))).alias("off_rebound_rate"),
            (pl.col("def_rebounds") / (pl.col("def_rebounds") + pl.col("opponent_off_rebounds"))).alias("def_rebound_rate"),
            # Block Rate
            (pl.col("blocks") / (pl.col("opponent_fg_attempted") + 1)).alias("block_rate")
        ])

        return df
    
    
    def compute_efficiency_metrics(self, df: pl.DataFrame) -> pl.DataFrame:
        """Computes field goal percentages, turnover rate, and assist/turnover ratio."""
        return df.with_columns([
            # Shooting Percentages
            (pl.col("2p_made") / pl.col("2p_attempted")).alias("2p_pct"),
            (pl.col("3p_made") / pl.col("3p_attempted")).alias("3p_pct"),
            (pl.col("ft_made") / pl.col("ft_attempted")).alias("ft_pct"),

            # Turnover Rate
            (pl.col("turnovers") / (pl.col("2p_attempted") + pl.col("3p_attempted") + 0.44 * pl.col("ft_attempted") + pl.col("turnovers")))
            .alias("turnover_rate"),

            # Assist-to-Turnover Ratio
            (pl.col("assists") / (pl.col("turnovers") + 1)).alias("assist_turnover")  # +1 to avoid div by zero
        ])
    
    
    def compute_ema(self, df: pl.DataFrame) -> pl.DataFrame:
        """Computes exponential moving averages (EMA) over past games."""
        return (
            df.sort(["team", "date"])  # Ensure chronological order within each team
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
        """Merges EMA features with the game matchups."""

        # Rename df columns before the join to enforce suffixing
        df_home = df.rename({col: col + "_home" for col in df.columns if col not in ["gameID", "team"]})
        df_away = df.rename({col: col + "_away" for col in df.columns if col not in ["gameID", "team"]})

        return (
            self.games_df
            .join(df_home, left_on=["gameID", "home_team"], right_on=["gameID", "team"])
            .join(df_away, left_on=["gameID", "away_team"], right_on=["gameID", "team"])
            .select([
                "gameID", "date", "home_team", "away_team", "home_score", "away_score", "conference_home", "conference_away",

                # ELO features
                "home_elo", "home_elo_change", "home_lagged_elo", "home_lagged_elo_change", "home_elo_change_ema", 
                "away_elo", "away_elo_change", "away_lagged_elo", "away_lagged_elo_change", "away_elo_change_ema",
                "elo_difference_lagged",
                
                # EMA Features for Home Team
                "ema_points_home", "ema_2p_pct_home", "ema_3p_pct_home", "ema_ft_pct_home", 
                "ema_off_rebound_rate_home", "ema_def_rebound_rate_home", "ema_turnover_rate_home", "ema_assist_turnover_home", "ema_block_rate_home",

                # EMA Features for Away Team
                "ema_points_away", "ema_2p_pct_away", "ema_3p_pct_away", "ema_ft_pct_away", 
                "ema_off_rebound_rate_away", "ema_def_rebound_rate_away", "ema_turnover_rate_away", "ema_assist_turnover_away", "ema_block_rate_away",

                # Matchup Features
                (pl.col("ema_points_home") - pl.col("ema_points_away")).alias("points_ema_diff"),
                (pl.col("ema_offensive_rating_home") - pl.col("ema_defensive_rating_away")).alias("offensive_rating_ema_diff"),
                (pl.col("ema_defensive_rating_home") - pl.col("ema_offensive_rating_away")).alias("defensive_rating_ema_diff"),
                (pl.col("ema_2p_pct_home") - pl.col("ema_2p_pct_away")).alias("2_pt_pct_ema_diff"),
                (pl.col("ema_3p_pct_home") - pl.col("ema_3p_pct_away")).alias("3_pt_pct_ema_diff"),
                (pl.col("ema_ft_pct_home") - pl.col("ema_ft_pct_away")).alias("ft_pct_ema_diff"),
                (pl.col("ema_off_rebound_rate_home") - pl.col("ema_def_rebound_rate_away")).alias("off_rebound_rate_ema_diff"),
                (pl.col("ema_def_rebound_rate_home") - pl.col("ema_off_rebound_rate_away")).alias("def_rebound_rate_ema_diff"),
                (pl.col("ema_turnover_rate_away") - pl.col("ema_turnover_rate_home")).alias("turnover_rate_ema_diff"),
                (pl.col("ema_assist_turnover_home") - pl.col("ema_assist_turnover_away")).alias("assist_turnover_ema_diff"),
                (pl.col("ema_block_rate_home") - pl.col("ema_block_rate_away")).alias("block_rate_ema_diff"),
            ])
        )


    def build_features(self) -> pl.DataFrame:
        """Runs the full EMA-based feature engineering pipeline."""

        self.generate_elo_features()

        team_stats = self.aggregate_team_stats()
        self.team_stats = team_stats
        opponent_stats = self.merge_opponent_stats(team_stats)

        print("=========================================================")
        print("Opponent stat columns: ", opponent_stats.columns)
        print("Opponent stat shape: ", opponent_stats.shape)
        print("=========================================================")

        efficiency_metrics = self.compute_efficiency_metrics(opponent_stats)

        print("=========================================================")
        print("Efficiency Metric Columns: ", efficiency_metrics.columns)
        print("Efficiency metric shapes, ", efficiency_metrics.shape)
        print("=========================================================")

        print("=========================================================")
        print("Schedule shape", self.games_df.shape)
        print("=========================================================")

        ema_features = self.compute_ema(efficiency_metrics)
        self.team_features = ema_features

        print("=========================================================")
        print("EMA features: ", ema_features.columns)
        # print("Ema dtypes:", ema_features.dtypes)
        print("Ema shape", ema_features.shape)
        print("=========================================================")

        return self.merge_with_matchups(ema_features)