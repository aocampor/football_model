"""
Monkey Football Analysis package.
"""

import pandas as pd
import numpy as np
import lightgbm as lgbm
import matplotlib.pyplot as plt

def read_data():
    df = pd.read_csv('results.csv')
    df_names = pd.read_csv('former_names.csv')
    df_wc = pd.read_csv('world_champions.csv')
    return df, df_names, df_wc

def calculate_time_features(df_, df_wc_):
    df_['date'] = pd.to_datetime(df_['date'])
    df_wc_['date'] = pd.to_datetime(df_wc_['date'])
    df_['year'] = df_['date'].dt.year
    df_['month'] = df_['date'].dt.month
    def get_season(month):
        if month in [12, 1, 2]:
            return "Winter"
        elif month in [3, 4, 5]:
            return "Spring"
        elif month in [6, 7, 8]:
            return "Summer"
        else:
            return "Autumn"
    df_['season'] = df_['month'].apply(get_season)
    df_['day'] = df_['date'].dt.day
    df_['day_of_week'] = df_['date'].dt.day_name()
    return df_, df_wc_

def reformat_features(df_):
    df_copy = df_.copy()
    df_["team"] = df_["home_team"]
    df_["state"] = "home"
    df_["opponent"] = df_["away_team"]
    df_["goals"] = df_["home_score"]
    df_["goals_allowed"] = df_["away_score"]

    df_copy["team"] = df_copy["away_team"]
    df_copy["state"] = "away"
    df_copy["opponent"] = df_copy["home_team"]
    df_copy["goals"] = df_copy["away_score"]
    df_copy["goals_allowed"] = df_copy["home_score"]

    df_ = pd.concat([df_, df_copy])
    df_ = df_.drop(
        columns=[
            "home_team", "away_team", "home_score", 
            "away_score"
        ]
    )
    return df_

def calculate_point_features(df_):
    df_['goal_diff'] = df_['goals'] - df_['goals_allowed']
    df_['points'] = np.where(df_['goal_diff'] > 0, 3, np.where(df_['goal_diff'] == 0, 1, 0))
    grouped_teams = df_.groupby("team")
    list_df_out = []
    for team, group in grouped_teams:
        group = group.sort_values("date", ascending=True)
        group["cum_points"] = np.cumsum(group["points"]).shift(1)
        group["window_cum_points"] = group["points"].rolling(window=10).sum().shift(1)
        group["window_mean_points"] = group["points"].rolling(window=10).mean().shift(1)
        list_df_out.append(group)
    df_out = pd.concat(list_df_out)
    df_out = df_out.sort_values("date", ascending=True)
    df_out = pd.merge(
        df_out, 
        df_out[
            ["team", "date", "cum_points", "window_cum_points"]
        ].rename(
            columns={
                "team": "opponent", 
                "cum_points": "opponent_cum_points", 
                "window_cum_points": "opponent_window_cum_points"
            }
        ), 
        on=["opponent", "date"], 
        how="left"
    )
    return df_out

def get_wc_features(df_, df_wc_):
    df_ = pd.merge(
        df_, 
        df_wc_, 
        on=["team", "date"], 
        how="left"
    )
    df_['wc'] = np.where(df_['position'] == 1, 1, 0)
    df_['wc'] = df_.groupby('team')['wc'].cumsum()
    # df_['wc'] = df_['wc'].fillna(method='ffill')
    return df_

def get_feature_and_target_names(df_):
    feature_names = [
        "team", "opponent", "cum_points", 
        "opponent_cum_points", "window_cum_points",
        "window_mean_points", "opponent_window_cum_points", 
        "tournament", "city", "country", "neutral", "state",
        "year", "month", "day", "day_of_week", "season", "wc"
    ]

    targets = ["goals", "goals_allowed", "goals_diff", "points"]

    target = "points"
    return feature_names, targets, target

def train(df_, feature_names, target):
    n_min = 365*30
    n_ = 30
    n_val = n_
    n_test = n_
    n_total = len(df_.date.unique())
    df_ = df_.sort_values("date", ascending=True)

    categorical_cols = [
        "team", "opponent", "tournament", "city", "country", 
        "neutral", "state", "year", "month", 
        "day", "day_of_week", "season", "wc"
    ]

    val_scores = []
    test_scores = []

    for i in range(n_min, n_total - n_test - n_val, n_val):
        df_train = df_[df_.date <= df_.date.unique()[i]]
        df_val = df_[
            (df_.date > df_.date.unique()[i]) & 
            (df_.date <= df_.date.unique()[i + n_val])
        ]
        df_test = df_[
            (df_.date > df_.date.unique()[i + n_val]) & 
            (df_.date <= df_.date.unique()[i + n_val + n_test])
        ]
        features_train = df_train[feature_names]
        features_val = df_val[feature_names]
        features_test = df_test[feature_names]

        for col in categorical_cols:
            features_train[col] = features_train[col].astype('category')
            features_val[col] = features_val[col].astype('category')
            features_test[col] = features_test[col].astype('category')

        labels_train = df_train[target].astype('category')
        labels_val = df_val[target].astype('category')
        labels_test = df_test[target].astype('category')

        X_train = features_train
        y_train = labels_train
        X_val = features_val
        y_val = labels_val
        X_test = features_test
        y_test = labels_test
        model = lgbm.LGBMClassifier(
            objective='multiclass',
            num_class=3,
            metric='multi_logloss',
            is_unbalance=True,
            # n_estimators=1000, 
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        print(f"Validation score: {model.score(X_val, y_val)}")
        val_scores.append(model.score(X_val, y_val))
        y_pred_test = model.predict(X_test)
        print(f"Test score: {model.score(X_test, y_test)}")
        test_scores.append(model.score(X_test, y_test))
        # plt.figure(figsize=(10, 8))
        # lgbm.plot_importance(model, importance_type='split', figsize=(10, 8))
        # plt.show()
    plt.figure(figsize=(10, 6))
    plt.hist(
        val_scores, 
        label='Validation', alpha=0.5, 
        weights=np.ones(len(val_scores)) / len(val_scores)
    )
    plt.hist(   
        test_scores, 
        label='Test', alpha=0.5, 
        weights=np.ones(len(test_scores)) / len(test_scores)
    )
    plt.legend()
    plt.show()
    plt.figure(figsize=(10, 6))
    plt.plot(
        val_scores, 
        label='Validation', alpha=0.5
    )
    plt.plot(   
        test_scores, 
        label='Test', alpha=0.5
    )
    plt.legend()
    plt.show()
    return model
    

df, df_names, df_wc = read_data()
df, df_wc = calculate_time_features(df, df_wc)
df = reformat_features(df)
df = calculate_point_features(df)
df = get_wc_features(df, df_wc)
feature_names, targets, target = get_feature_and_target_names(df)
model = train(df, feature_names, target)
