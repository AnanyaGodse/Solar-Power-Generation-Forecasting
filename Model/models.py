# Importing the libraries
import numpy as np
import pandas as pd
import xgboost as xgb
import pickle

def create_features(df):
    df = df.copy()
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['week_of_year'] = df.index.isocalendar().week
    return df


#PLANT 1
plant1 = pd.read_csv("plant1_merged.csv")
plant1["DATE_TIME"] = pd.to_datetime(plant1["DATE_TIME"], format="%Y-%m-%d %H:%M:%S")
t_reduced_plant1 = plant1[["DATE_TIME","DAILY_YIELD"]]
t_reduced_plant1.set_index("DATE_TIME", inplace=True)

split_date = '2020-06-06'
plant1_train = t_reduced_plant1.loc[:split_date]
plant1_test = t_reduced_plant1.loc[split_date:]

t_reduced_plant1 = create_features(t_reduced_plant1)
plant1_train = create_features(plant1_train)
plant1_test = create_features(plant1_test)

X_p1_train_final = plant1_train.iloc[:, 1] # only hour data
y_p1_train_final = plant1_train.iloc[:, 0]

X_p1_test_final = plant1_test.iloc[:, 1]
y_p1_test_final = plant1_test.iloc[:, 0]

reg_final_p1 = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.01)
reg_final_p1.fit(X_p1_train_final, y_p1_train_final)

predictions_final_p1 = reg_final_p1.predict(X_p1_test_final)
pickle.dump(reg_final_p1, open('model1.pkl','wb'))


#PLANT 2
plant2 = pd.read_csv("plant2_merged.csv")
plant2["DATE_TIME"] = pd.to_datetime(plant2["DATE_TIME"], format="%Y-%m-%d %H:%M:%S")

t_reduced_plant2 = plant2[["DATE_TIME","DAILY_YIELD"]]
t_reduced_plant2.set_index("DATE_TIME", inplace=True)

plant2_train = t_reduced_plant2.loc[:split_date]
plant2_test = t_reduced_plant2.loc[split_date:]


t_reduced_plant2 = create_features(t_reduced_plant2)
plant2_train = create_features(plant2_train)
plant2_test = create_features(plant2_test)


X_p2_train_final = plant2_train.drop(['year', 'month', 'DAILY_YIELD'], axis=1) # using week_of_year, day_of_week, hour
y_p2_train_final = plant2_train.iloc[:, 0]

X_p2_test_final = plant2_test.drop(['year', 'month', 'DAILY_YIELD'], axis=1)
y_p2_test_final = plant2_test.iloc[:, 0]


reg_final_p2 = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.01)
reg_final_p2.fit(X_p2_train_final, y_p2_train_final)

predictions_final_p2 = reg_final_p2.predict(X_p2_test_final)
pickle.dump(reg_final_p2, open('model2.pkl','wb'))

