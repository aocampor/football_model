"""
Monkey Football Analysis package.
"""

from constants import POSITION_KEY
from constants import MEAN_GOALS_PER_GAME_KEY
from constants import ROLLING_MEAN_POINTS_KEY
import pandas as pd
import numpy as np
import lightgbm as lgbm
import matplotlib.pyplot as plt
from constants import *


def read_data():
    df_ = pd.read_csv('data/results.csv')
    df_ = df_.drop([COUNTRY_KEY, CITY_KEY], axis=1)
    df_names_ = pd.read_csv('data/former_names.csv')
    df_wc_ = pd.read_csv('data/world_champions.csv')
    df_goals_ = pd.read_csv('data/goalscorers.csv')
    
    df_[DATE_KEY] = pd.to_datetime(df_[DATE_KEY])
    df_wc_[DATE_KEY] = pd.to_datetime(df_wc_[DATE_KEY])
    df_goals_[DATE_KEY] = pd.to_datetime(df_goals_[DATE_KEY])
    
    return df_, df_names_, df_wc_, df_goals_


def calculate_time_features(df_, df_wc_):
    df_[YEAR_KEY] = df_[DATE_KEY].dt.year
    df_[MONTH_KEY] = df_[DATE_KEY].dt.month
    def get_season(month):
        if month in [12, 1, 2]:
            return 0
        elif month in [3, 4, 5]:
            return 1
        elif month in [6, 7, 8]:
            return 2
        else:
            return 3
    df_[SEASON_KEY] = df_[MONTH_KEY].apply(get_season)
    df_[DAY_KEY] = df_[DATE_KEY].dt.day
    df_[WEEKDAY_KEY] = df_[DATE_KEY].dt.dayofweek
    return df_, df_wc_


def calculate_tournament_feature(df_):
    df_[TOURNAMENT_KEY] = np.where(
        df_.tournament == FRIENDLY_TAG, 
        0, 
        np.where(
            df_.tournament == EURO_TAG, 
            1,
            np.where(
                df_.tournament == COPA_AMERICA_TAG, 
                2,
                np.where(
                    df_.tournament == WORLD_CUP_TAG,
                    3, 
                    4
                )
            )
        )
    )
    return df_


def reformat_features(df_):
    df_.loc[df_.home_team == DR_CONGO_TAG, HOME_KEY] = CONGO_TAG
    df_.loc[df_.away_team == DR_CONGO_TAG, AWAY_KEY] = CONGO_TAG
    df_copy = df_.copy()
    df_[TEAM_KEY] = df_[HOME_KEY]
    df_[STATE_KEY] = 1
    df_[OPPONENT_KEY] = df_[AWAY_KEY]
    df_[GOALS_KEY] = df_[HOME_SCORE_KEY]
    df_[GOALS_ALLOWED_KEY] = df_[AWAY_SCORE_KEY]

    df_copy[TEAM_KEY] = df_copy[AWAY_KEY]
    df_copy[STATE_KEY] = 0
    df_copy[OPPONENT_KEY] = df_copy[HOME_KEY]
    df_copy[GOALS_KEY] = df_copy[AWAY_SCORE_KEY]
    df_copy[GOALS_ALLOWED_KEY] = df_copy[HOME_SCORE_KEY]

    df_ = pd.concat([df_, df_copy])
    df_ = df_.drop(
        columns=[
            HOME_KEY, AWAY_KEY, HOME_SCORE_KEY, 
            AWAY_SCORE_KEY
        ]
    )
    df_[NEUTRAL_KEY] = df_[NEUTRAL_KEY].astype(int) 
    return df_


def get_scorers_vectorized(df_):
    # 1. Count unique scorers per match and team
    # We use nunique() or size() depending on if your raw data has duplicate scorer rows
    scorer_counts = df_.groupby([DATE_KEY, HOME_KEY, AWAY_KEY, TEAM_KEY])[SCORER_KEY].nunique().reset_index()
    scorer_counts.rename(columns={SCORER_KEY: N_SCORERS_KEY}, inplace=True)

    # 2. Create the Home Team perspective
    home_side = scorer_counts[scorer_counts[TEAM_KEY] == scorer_counts[HOME_KEY]].copy()
    home_side[OPPONENT_KEY] = home_side[AWAY_KEY]

    # 3. Create the Away Team perspective
    away_side = scorer_counts[scorer_counts[TEAM_KEY] == scorer_counts[AWAY_KEY]].copy()
    away_side[OPPONENT_KEY] = away_side[HOME_KEY]

    # 4. Combine and clean up columns to match your desired output
    final_cols = [DATE_KEY, TEAM_KEY, OPPONENT_KEY, N_SCORERS_KEY]
    output = pd.concat([home_side[final_cols], away_side[final_cols]], ignore_index=True)
    
    return output.sort_values(DATE_KEY)


def calculate_points(df_):
    df_[GOAL_DIFF_KEY] = df_[GOALS_KEY] - df_[GOALS_ALLOWED_KEY]
    df_[POINTS_KEY] = np.where(
        df_[GOAL_DIFF_KEY] > 0, 
        3, 
        np.where(
            df_[GOAL_DIFF_KEY] == 0, 
            1, 
            0
        )
    )
    return df_


def calculate_aggregated_features_vectorized(df_):
    aggregate_window = 2
    # 1. Sort the entire dataframe first to ensure time-consistency within groups
    df_ = df_.sort_values([TEAM_KEY, STATE_KEY, DATE_KEY], ascending=True)
    
    # 2. Define the groupby object once
    grouped = df_.groupby([TEAM_KEY, STATE_KEY])
    
    # 3. Vectorized Cumulative Sum
    df_[CUMMULATIVE_POINTS_KEY] = grouped[POINTS_KEY].cumsum()
    
    # 4. Vectorized Rolling Operations
    # Note: .rolling() on a GroupBy object requires the window size
    rolling_obj = grouped[[POINTS_KEY, GOALS_KEY, GOALS_ALLOWED_KEY, N_SCORERS_KEY]].rolling(
        window=aggregate_window
    )
    
    # We use .reset_index(level=[0, 1], drop=True) because rolling on groupby 
    # adds the grouping keys back into the index
    rolling_results = rolling_obj.mean().reset_index(level=[0, 1], drop=True)
    
    # Map results back to the main dataframe
    df_[ROLLING_MEAN_POINTS_KEY] = rolling_results[POINTS_KEY]
    df_[MEAN_GOALS_PER_GAME_KEY] = rolling_results[GOALS_KEY]
    df_[MEAN_ALLOWED_GOALS_PER_GAME_KEY] = rolling_results[GOALS_ALLOWED_KEY]
    df_[MEAN_N_SCORERS_KEY] = rolling_results[N_SCORERS_KEY]
    
    # Sum is mean * window_size, or we can calculate it separately
    df_[ROLLING_CUM_POINTS_KEY] = rolling_obj[POINTS_KEY].sum().values
    
    # 5. Simple Vectorized Math (No groupby needed here)
    df_[MEAN_DIFF_GOAL_KEY] = df_[MEAN_GOALS_PER_GAME_KEY] - df_[MEAN_ALLOWED_GOALS_PER_GAME_KEY]
    
    return df_

def calculate_opponent_features(df_):
    df_ = df_.sort_values(DATE_KEY, ascending=True)
    df_ = pd.merge(
        df_, 
        df_[[
            TEAM_KEY, DATE_KEY, GOALS_KEY, 
            GOALS_ALLOWED_KEY, GOAL_DIFF_KEY, POINTS_KEY, 
            CUMMULATIVE_POINTS_KEY, ROLLING_CUM_POINTS_KEY, 
            ROLLING_MEAN_POINTS_KEY, MEAN_GOALS_PER_GAME_KEY, 
            MEAN_ALLOWED_GOALS_PER_GAME_KEY, MEAN_DIFF_GOAL_KEY, 
            N_SCORERS_KEY, MEAN_N_SCORERS_KEY
        ]].rename(
            columns={
                TEAM_KEY: OPPONENT_KEY,
                GOALS_KEY: OPPONENT_GOALS_KEY,
                GOALS_ALLOWED_KEY: OPPONENT_GOALS_ALLOWED_KEY,
                GOAL_DIFF_KEY: OPPONENT_GOAL_DIFF_KEY,
                POINTS_KEY: OPPONENTS_POINT_KEY,
                CUMMULATIVE_POINTS_KEY: OPPONENT_CUM_POINTS_KEY, 
                ROLLING_CUM_POINTS_KEY: OPPONENT_ROLLING_CUM_POINTS_KEY,
                ROLLING_MEAN_POINTS_KEY: OPPONENT_ROLLING_MEAN_POINTS_KEY,
                MEAN_GOALS_PER_GAME_KEY: OPPONENT_MEAN_GOALS_KEY,
                MEAN_ALLOWED_GOALS_PER_GAME_KEY: OPPONENT_MEAN_GOALS_ALLOWED_KEY,
                MEAN_DIFF_GOAL_KEY: OPPONENT_MEAN_DIFF_GOALS_KEY,
                N_SCORERS_KEY: OPPONENT_SCORERS_KEY,
                MEAN_N_SCORERS_KEY: OPPONENT_MEAN_SCORERS_KEY
            }
        ), 
        on=[OPPONENT_KEY, DATE_KEY], 
        how="left"
    )
    df_[DIFF_CUM_POINTS_KEY] = df_.cum_points - df_.opponent_cum_points
    df_[DIFF_ROLLING_CUM_POINTS] = df_.window_cum_points - df_.opponent_window_cum_points
    df_[DIFF_ROLLING_MEAN_POINTS_KEY] = df_.window_mean_points - df_.opponent_window_mean_points
    df_[TARGET_KEY] = df_[GOAL_DIFF_KEY]
    return df_


def get_wc_features(df_, df_wc_):
    df_ = pd.merge(
        df_, 
        df_wc_, 
        on=[TEAM_KEY, DATE_KEY], 
        how="left"
    )
    df_[WORLD_CUP_KEY] = np.where(df_[POSITION_KEY] == 1, 1, 0)
    df_[WORLD_CUP_KEY] = df_.sort_values(DATE_KEY, ascending=True).groupby(TEAM_KEY)[WORLD_CUP_KEY].cumsum()
    df_[POSITION_KEY] = np.where(df_[POSITION_KEY] < 5, df_[POSITION_KEY], 0)
    return df_


def shift_features(df_, to_shift_):
    for feat in to_shift_:
        df_[feat] = df_[feat].shift(1)
    return df_


def merge_scorers(df_, df_scores_):
    df_ = pd.merge(
        df_, 
        df_scores_, 
        how="left", 
        on=[DATE_KEY, TEAM_KEY, OPPONENT_KEY]
    )
    df_[N_SCORERS_KEY] = np.where(
        df_[N_SCORERS_KEY].isna(), 
        np.where(
            df_[GOALS_KEY] == 0,
            0,
            1
        ),
        df_[N_SCORERS_KEY]
    )
    return df_


def fillna_features(df_):
    df_[ROLLING_MEAN_POINTS_KEY] = df_[ROLLING_MEAN_POINTS_KEY].fillna(0)
    df_[ROLLING_CUM_POINTS_KEY] = df_[ROLLING_CUM_POINTS_KEY].fillna(0)
    df_[MEAN_GOALS_PER_GAME_KEY] = df_[MEAN_GOALS_PER_GAME_KEY].fillna(0)
    df_[MEAN_ALLOWED_GOALS_PER_GAME_KEY] = df_[MEAN_ALLOWED_GOALS_PER_GAME_KEY].fillna(0)
    df_[MEAN_N_SCORERS_KEY] = df_[MEAN_N_SCORERS_KEY].fillna(0)
    df_[MEAN_DIFF_GOAL_KEY] = df_[MEAN_DIFF_GOAL_KEY].fillna(0)
    return df_


def preprocessing(df_, df_wc_, df_scorers_):
    df_, df_wc_ = calculate_time_features(df_, df_wc_)
    df_ = calculate_tournament_feature(df_)
    df_ = reformat_features(df_)
    df_scores_ = get_scorers_vectorized(df_scorers_)
    df_ = merge_scorers(df_, df_scores_)
    categorical_features = [
        TOURNAMENT_KEY, NEUTRAL_KEY, MONTH_KEY,
        SEASON_KEY, DAY_KEY, WEEKDAY_KEY,
        STATE_KEY
    ]
    to_shift = [
        GOALS_KEY, GOALS_ALLOWED_KEY, N_SCORERS_KEY
    ]
    df_ = calculate_points(df_)
    to_shift.append(GOAL_DIFF_KEY)
    to_shift.append(POINTS_KEY)
    df_ = calculate_aggregated_features_vectorized(df_)
    df_ = fillna_features(df_)
    aggregated_metrics = [
        CUMMULATIVE_POINTS_KEY, ROLLING_CUM_POINTS_KEY, ROLLING_MEAN_POINTS_KEY, 
        MEAN_GOALS_PER_GAME_KEY, MEAN_ALLOWED_GOALS_PER_GAME_KEY,
        MEAN_N_SCORERS_KEY, MEAN_DIFF_GOAL_KEY
    ]
    to_shift = to_shift + aggregated_metrics
    df_ = calculate_opponent_features(df_)
    opponent_features = [
        OPPONENT_GOALS_KEY, OPPONENT_GOALS_ALLOWED_KEY, OPPONENT_GOAL_DIFF_KEY,
        OPPONENTS_POINT_KEY, OPPONENT_CUM_POINTS_KEY, OPPONENT_ROLLING_CUM_POINTS_KEY,
        OPPONENT_ROLLING_MEAN_POINTS_KEY, OPPONENT_MEAN_GOALS_KEY,
        OPPONENT_MEAN_GOALS_ALLOWED_KEY, 
        OPPONENT_MEAN_DIFF_GOALS_KEY, OPPONENT_SCORERS_KEY,
        OPPONENT_MEAN_SCORERS_KEY, DIFF_CUM_POINTS_KEY, DIFF_ROLLING_CUM_POINTS,
        DIFF_ROLLING_MEAN_POINTS_KEY
    ]
    to_shift = to_shift + opponent_features
    df_ = get_wc_features(df_, df_wc_original)
    categorical_features += [POSITION_KEY, WORLD_CUP_KEY]
    df_ = shift_features(df_, to_shift + [POSITION_KEY, WORLD_CUP_KEY])
    # df_.dropna(axis=0, inplace=True)
    feature_names_ = categorical_features + to_shift 
    return df_, df_wc_, feature_names_


def train(df_, feature_names, target, categorical_cols):
    n_min = 365*10
    n_ = 180
    n_val = n_
    n_test = n_
    n_total = len(df_.date.unique())
    df_ = df_.sort_values(DATE_KEY, ascending=True)

    val_scores = []
    test_scores = []
    baseline_scores = []

    for i in range(n_min, n_total - n_val, n_val):
        try:
            print(
                df_.date.unique()[i], 
                df_.date.unique()[i + n_val], 
                df_.date.unique()[i + n_val + n_test]
            )
        except:
            print("Last")
        df_train = df_[
            df_.date <= df_.date.unique()[i]
        ]
        df_val = df_[
            (df_.date > df_.date.unique()[i]) & 
            (df_.date <= df_.date.unique()[i + n_val])
        ]
        try:
            df_test = df_[
                (df_.date > df_.date.unique()[i + n_val]) & 
                (df_.date <= df_.date.unique()[i + n_val + n_test])
            ]
        except:
            print("Last")
        print("before features")
        features_train = df_train[feature_names].copy()
        features_val = df_val[feature_names].copy()
        try:
            features_test = df_test[feature_names].copy()
        except:
            print("Last")
            
        for col in categorical_cols:
            features_train[col] = features_train[col].astype('category')
            features_val[col] = features_val[col].astype('category')
            try:
                features_test[col] = features_test[col].astype('category')
            except:
                print("Last")
        print("before target")
        labels_train = df_train[TARGET_KEY]
        labels_val = df_val[TARGET_KEY]
        try:
            labels_test = df_test[TARGET_KEY]
        except:
            print("Last")
        
        if target == POINTS_KEY:
            labels_train = labels_train.astype('category').copy()
            labels_val = labels_val.astype('category').copy()
            try:
                labels_test = labels_test.astype('category').copy()
            except:
                print("Last")

        X_train = features_train
        y_train = labels_train
        
        X_val = features_val
        y_val = labels_val

        try:
            X_test = features_test
            y_test = labels_test
        except:
            print("Last")

        early_stopping = lgbm.early_stopping(stopping_rounds=50)
        log_evaluation = lgbm.log_evaluation(period=10)
        if target == POINTS_KEY:
            model = lgbm.LGBMClassifier(
                objective='multiclass',
                num_class=3,
                metric='multi_logloss',
                is_unbalance=True,
                n_estimators=10000, 
                device='cuda',        # Tells LightGBM to use the GPU
                gpu_use_dp=False
            )
            # early_stopping = lgbm.early_stopping(stopping_rounds=50)
            # log_evaluation = lgbm.log_evaluation(period=10)
            # model.fit(
            #     X_train, y_train,
            #     eval_set=[(X_val, y_val)],  # Crucial: Early stopping needs a validation set
            #     callbacks=[early_stopping, log_evaluation]
            # )
        #elif target == GOAL_DIFF_KEY:
        else:
            model = lgbm.LGBMRegressor(
                objective="regression",
                metric="rmse",
                device='cuda',        # Tells LightGBM to use the GPU
                gpu_use_dp=False
            )
            # model.fit(X_train, y_train)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],  # Crucial: Early stopping needs a validation set
            callbacks=[early_stopping, log_evaluation]
        )
        # y_pred = model.predict(X_val)
        print(f"Validation score: {model.score(X_val, y_val)}")
        val_scores.append(model.score(X_val, y_val))
        baseline_scores = [1 if x == 0 else 0 for x in y_val]
        try: 
            y_pred_test = model.predict(X_test)
            print(f"Test score: {model.score(X_test, y_test)}")
            test_scores.append(model.score(X_test, y_test))
        except:
            print("Last")
        plt.figure(figsize=(10, 8))
        lgbm.plot_importance(model, importance_type='gain', figsize=(10, 8))
        # plt.show()
        plt.savefig(f"images/importance_{df_.date.unique()[i + n_val]}.png")
    baseline_metric = np.mean(baseline_scores)
    plt.figure(figsize=(10, 6))
    plt.hist(
        val_scores[:-1], 
        label='Validation', alpha=0.5, 
        weights=np.ones(len(val_scores[:-1])) / len(val_scores[:-1])
    )
    plt.hist(   
        test_scores, 
        label='Test', alpha=0.5, 
        weights=np.ones(len(test_scores)) / len(test_scores)
    )
    plt.legend()
    # plt.show()
    plt.savefig("images/val_test_histo.png")
    plt.figure(figsize=(10, 6))
    plt.title(f"baseline score {baseline_metric}")
    plt.plot(
        val_scores[:-1], 
        label=f'Validation {np.mean(val_scores[:-1])}', alpha=0.5,
    )
    plt.plot(   
        test_scores, 
        label=f'Test {np.mean(test_scores)}', alpha=0.5,
    )
    plt.legend()
    # plt.show()
    plt.savefig("images/val_test_scores.png")
    return model


df_original, df_names_original, df_wc_original, df_goals_original = read_data()
df, df_wc, feature_names = preprocessing(df_original.copy(), df_wc_original.copy(), df_goals_original.copy())

for feat in feature_names + [TARGET_KEY]:
    plt.figure()
    plt.title(feat)
    to_plot = df[feat]
    plt.hist(to_plot.astype(float))
    plt.yscale('log')
    # plt.show()
    plt.savefig(f"images/{feat}.png")

model = train(df, feature_names, TARGET_KEY, categorical_features)

games = pd.DataFrame({ 
  DATE_KEY: ["2025-12-21"] + 3 * ["2025-12-22"] +
      4 * ["2025-12-23"] + 4 * ["2025-12-24"] + 
      4 * ["2025-12-26"] + 4 * ["2025-12-27"] + 
      4 * ["2025-12-28"] + 4 * ["2025-12-29"] + 
      4 * ["2025-12-30"] + 4 * ["2025-12-31"], #+
      # 2 * ["2026-01-03"] + 2 * ["2026-01-04"] + 
      # 2 * ["2026-01-05"] + 2 * ["2026-01-06"] +
      # 2 * ["2026-01-09"] + 2 * ["2026-01-10"] +
      # 2 * ["2026-01-14"] + ["2026-01-17", "2026-01-18"],
  HOME_KEY: [
      "Morocco",
      "Mali",
      "South Africa", 
      "Egypt",      
      CONGO_TAG,      
      "Senegal",    
      "Nigeria",    
      "Tunisia",
      "Burkina Faso", 
      "Algeria",
      "Ivory Coast",
      "Cameroon",
      "Angola", 
      "Egypt",
      "Zambia",
      "Morocco",
      "Benin", 
      "Senegal",
      "Uganda",
      "Nigeria",
      "Gabon", 
      "Equatorial Guinea",
      "Algeria",
      "Ivory Coast",
      "Angola",
      "Zimbabwe",
      "Comoros",
      "Zambia",
      "Tanzania",
      "Uganda",
      "Benin",
      "Botswana",
      "Equatorial Guinea",
      "Sudan",
      "Gabon",
      "Mozambique",
      # "Benin",
      # "Zambia",
      # "South Africa",
      # "Morocco",
      # "Egypt",
      # "Nigeria",
      # "Ivory Coast",
      # "Burkina Faso",
      # "Zambia",
      # "Cameroon",
      # "Burkina Faso",
      # "Comoros", 
      # "Gabon",
      # "Zambia",
      # "Comoros",
      # "Zambia"
  ], 
  AWAY_KEY: [
      "Comoros",    
      "Zambia",     
      "Angola",       
      "Zimbabwe",   
      "Benin",      
      "Botswana",   
      "Tanzania",   
      "Uganda",
      "Equatorial Guinea",
      "Sudan",
      "Mozambique", 
      "Gabon", 
      "Zimbabwe",
      "South Africa",
      "Comoros",
      "Mali",
      "Botswana",
      CONGO_TAG,
      "Tanzania", 
      "Tunisia",
      "Mozambique", 
      "Sudan",
      "Burkina Faso",
      "Cameroon",
      "Egypt",
      "South Africa",
      "Mali",
      "Marocoo",
      "Tunisia", 
      "Nigeria",
      "Senegal",
      CONGO_TAG, 
      "Algeria",
      "Burkina Faso",
      "Ivory Coast",
      "Cameroon",
      # "Angola",
      # "Tunisia",
      # "Cameroon",
      # "Sudan",
      # "Comoros",
      # "Gabon",
      # "Equatorial Guinea",
      # "Senegal",
      # "Benin",
      # "Sudan",
      # "Gabon",
      # "Equatorial Guinea",
      # "Cameroon",
      # "Comoros",
      # "Gabon",
      # "Cameroon"
  ], 
  HOME_SCORE_KEY: 36 * [np.nan], 
  AWAY_SCORE_KEY: 36 * [np.nan],
  TOURNAMENT_KEY: 36 * ["AFCON 2025"],
  NEUTRAL_KEY: [False] + 14 * [True] + [False] + 20 * [True] 
  # + [False] + 12 * [True]
})

def generate_prediction_dataset(games_, df_, df_wc_, df_goals_, model, features, categoricals):
    games_[DATE_KEY] = pd.to_datetime(games_[DATE_KEY])
    for row in games_.itertuples():
        # print(df_.columns, row._asdict().keys())
        # df_p, df_wc_p = preprocessing(pd.concat([df_, pd.DataFrame([row._asdict()])]), df_wc_)
        temp = pd.concat([df_, pd.DataFrame([row._asdict()])])
        df_p, df_wc_p, feature_names = preprocessing(temp, df_wc_.copy(), df_goals_.copy())
        to_predict = df_p[feature_names].tail(1)
        to_predict[categoricals] = to_predict[categoricals].astype("category")
        y = model.predict(to_predict)
        # print(round(y[0]))
        # if target == POINTS_KEY:
        if y[0] == 3:
            games_.loc[row.Index, HOME_SCORE_KEY] = 1
            games_.loc[row.Index, AWAY_SCORE_KEY] = 0
        elif y[0] == 1:
            games_.loc[row.Index, HOME_SCORE_KEY] = 0
            games_.loc[row.Index, AWAY_SCORE_KEY] = 0
        else:
            games_.loc[row.Index, HOME_SCORE_KEY] = 0
            games_.loc[row.Index, AWAY_SCORE_KEY] = 1
        # else:
        #     if y[0] >= 0:
        #         games_.loc[row.Index, HOME_SCORE_KEY] = round(y[0])
        #         games_.loc[row.Index, AWAY_SCORE_KEY] = 0
        #     else:
        #         games_.loc[row.Index, HOME_SCORE_KEY] = 0
        #         games_.loc[row.Index, AWAY_SCORE_KEY] = -1*round(y[0])
    return games_
        
predicted_games = generate_prediction_dataset(games, df_original.copy(), df_wc_original.copy(), df_goals_original.copy(), model, feature_names, categorical_features)
groups = {
    "group_a": ["Morocco", "Mali", "Zambia", "Comoros"],
    "group_b": ["Egypt", "South Africa", "Angola", "Zimbabwe"],
    "group_c": ["Nigeria", "Tunisia", "Tanzania", "Uganda"],
    "group_d": ["Senegal", CONGO_TAG, "Benin", "Botswana"],
    "group_e": ["Algeria", "Burkina Faso", "Sudan", "Equatorial Guinea"],
    "group_f": ["Ivory Coast", "Cameroon", "Mozambique", "Gabon"]
}

groups_points = {}
for group in groups.keys():
    groups_points[group] = {}
    for team_1 in groups[group]:
        groups_points[group][team_1] = 0

for group in groups.keys():
    for team_1 in groups[group]:
        for team_2 in groups[group]:
            if team_1 == team_2:
                continue
            # print(team_1, team_2)
            temp = predicted_games[
                (predicted_games.home_team == team_1) & (predicted_games.away_team == team_2)
            ]
            if len(temp) > 0:
                if temp.home_score.values[0] > temp.away_score.values[0]:
                    groups_points[group][team_1] += 3
                elif temp.away_score.values[0] > temp.home_score.values[0]:
                    groups_points[group][team_2] += 3
                else:
                    groups_points[group][team_1] += 1
                    groups_points[group][team_2] += 1
                    
print(groups_points)