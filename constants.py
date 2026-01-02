DATE_KEY = "date"
COUNTRY_KEY = "country"
CITY_KEY = "city"
YEAR_KEY = "year"

MONTH_KEY = "month"
SEASON_KEY = "season"
DAY_KEY = "day"
WEEKDAY_KEY = "day_of_week"

TOURNAMENT_KEY = "tournament"

FRIENDLY_TAG = "Friendly"
EURO_TAG = "UEFA Euro"
COPA_AMERICA_TAG = "Copa América"
WORLD_CUP_TAG = "FIFA World Cup"

DR_CONGO_TAG = "DR Congo"
CONGO_TAG = "Congo"
HOME_KEY = "home_team"
AWAY_KEY = "away_team"
TEAM_KEY = "team"
STATE_KEY = "state"
OPPONENT_KEY = "opponent"
GOALS_KEY = "goals"
HOME_SCORE_KEY = "home_score"
AWAY_SCORE_KEY = "away_score"
GOALS_ALLOWED_KEY = "goals_allowed"
NEUTRAL_KEY = "neutral"

SCORER_KEY = "scorer"
N_SCORERS_KEY = "scorers"

categorical_features = [
    TOURNAMENT_KEY, NEUTRAL_KEY, MONTH_KEY,
    SEASON_KEY, DAY_KEY, WEEKDAY_KEY,
    STATE_KEY
]

to_shift = [
    GOALS_KEY, GOALS_ALLOWED_KEY, N_SCORERS_KEY
]

GOAL_DIFF_KEY = "goal_diff"
POINTS_KEY = "points"

CUMMULATIVE_POINTS_KEY = "cum_points"
ROLLING_CUM_POINTS_KEY = "window_cum_points"
ROLLING_MEAN_POINTS_KEY = "window_mean_points"
MEAN_GOALS_PER_GAME_KEY = "average_goals_per_game"
MEAN_ALLOWED_GOALS_PER_GAME_KEY = "average_goals_allowed_per_game"
MEAN_N_SCORERS_KEY = "scorers_mean"
MEAN_DIFF_GOAL_KEY = "average_diff_goals_per_game"

OPPONENT_GOALS_KEY = "opponent_goals"
OPPONENT_GOALS_ALLOWED_KEY = "opponent_goals_allowed"
OPPONENT_GOAL_DIFF_KEY = "opponent_goal_diff"
OPPONENTS_POINT_KEY = "opponent_points"
OPPONENT_CUM_POINTS_KEY = "opponent_cum_points"
OPPONENT_ROLLING_CUM_POINTS_KEY = "opponent_window_cum_points"
OPPONENT_ROLLING_MEAN_POINTS_KEY = "opponent_window_mean_points"
OPPONENT_MEAN_GOALS_KEY = "opponent_average_goals_per_game"
OPPONENT_MEAN_GOALS_ALLOWED_KEY = "opponent_average_goals_allowed_per_game"
OPPONENT_MEAN_DIFF_GOALS_KEY = "opponent_average_diff_goals_per_game"
OPPONENT_SCORERS_KEY = "opponent_scorers"
OPPONENT_MEAN_SCORERS_KEY = "opponent_scorers_mean"
DIFF_CUM_POINTS_KEY = "diff_cum_points"
DIFF_ROLLING_CUM_POINTS = "diff_window_cum_points"
DIFF_ROLLING_MEAN_POINTS_KEY = "diff_window_mean_points"
TARGET_KEY = "target"

WORLD_CUP_KEY = "wc"
POSITION_KEY = "position"