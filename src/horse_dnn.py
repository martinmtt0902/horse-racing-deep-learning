import tensorflow as tf
import numpy as np
import pandas as pd
import sklearn.preprocessing as preprocessing
import sklearn.model_selection as model_selection
from sklearn.metrics import confusion_matrix
import os
import time

def gen_data():
    #Read data
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        DATA_DIR = os.path.join(PROJECT_ROOT, "data")

       races_data = pd.read_csv(os.path.join(DATA_DIR, "races.csv"), index_col="race_id")
       runs_data = pd.read_csv(os.path.join(DATA_DIR, "runs.csv"))
    
    #Select target features
    factor_race = races_data.loc[:,["venue", "config", "distance", "race_class", "going"]]
    factor_runs = runs_data.loc[:,["race_id", "draw", "horse_no", "horse_age", "horse_country", "horse_type", "declared_weight", "actual_weight", "win_odds", "place_odds", "result"]]
    #Transform categorical variables to ordinal variables
    factor_runs = factor_runs.fillna(str(0))    
    encoder = preprocessing.OrdinalEncoder()
    factor_race['venue'] = encoder.fit_transform(factor_race['venue'].values.reshape(-1, 1))
    factor_race['config'] = encoder.fit_transform(factor_race['config'].values.reshape(-1, 1))
    factor_race['going'] = encoder.fit_transform(factor_race['going'].values.reshape(-1, 1))
    factor_runs['horse_country'] = encoder.fit_transform(factor_runs['horse_country'].values.reshape(-1, 1))
    factor_runs['horse_type'] = encoder.fit_transform(factor_runs['horse_type'].values.reshape(-1, 1))
    
    #Remove wrong data with draw = 15
    draw_15 = factor_runs[factor_runs['draw'] > 14].index
    factor_runs = factor_runs.drop(draw_15)
    
    #Put result columns to the right of the matrix
    def reindex_factor_result(element):
        if element[0] == 'result':
            return 13 + element[1]
        else:
            return element[1]
    
    #Join runs data with race data
    factor_runs = factor_runs.pivot(index='race_id', columns='draw', values=factor_runs.columns[2:])
    rearr_columns = sorted(list(factor_runs.columns.values), key=reindex_factor_result)
    factor_runs = factor_runs[rearr_columns]
    factor_runs = factor_runs.fillna(0)
    factor = factor_race.join(factor_runs,on='race_id', how='right')
    #Split to X and y and standardize X
    X = factor[factor.columns[:-14]]
    ss = preprocessing.StandardScaler()
    X = pd.DataFrame(ss.fit_transform(X),columns = X.columns)
    y = factor[factor.columns[-14:]]

    return (X, y)

X, y = gen_data()

#Split data into training and testing
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, train_size=0.8, test_size=0.2, random_state = 1)

y_test_result = np.asarray(y_test)
y_test = y_test[y_test.columns[-14:]].applymap(lambda x: 1.0 if 0.5 < x < 1.5 else 0.0)
y_test_arr = np.asarray(y_test)

#Fill missing values with 0
X_train = X_train.fillna(0)
X_test = X_test.fillna(0)

#Initiate K-fold validation method
kf = model_selection.KFold(n_splits = 5)
kf.get_n_splits(X_train)

first_avg_acc = 0
place_avg_acc = 0

#Train the model through K-fold validation
print("Start training... \n")
start = time.time()

for train_index, val_index in kf.split(X_train):
    print("TRAIN:", train_index, "VALIDATION:", val_index)
    X_tra, X_val = X_train.iloc[train_index], X_train.iloc[val_index]
    y_tra, y_val = y_train.iloc[train_index], y_train.iloc[val_index]
    y_val_result = np.asarray(y_val)    
    y_tra = y_tra[y_tra.columns[-14:]].applymap(lambda x: 1.0 if 0.5 < x < 1.5 else 0.0)    
    y_val = y_val[y_val.columns[-14:]].applymap(lambda x: 1.0 if 0.5 < x < 1.5 else 0.0)
    y_val_arr = np.asarray(y_val)
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(512, activation="tanh", input_shape=(117, )))
#    model.add(tf.keras.layers.BatchNormalization())
#    model.add(tf.keras.layers.Dropout(0.2))
#    model.add(tf.keras.layers.Dense(64, activation="relu", input_shape=(117, )))
#    model.add(tf.keras.layers.BatchNormalization())
#    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(14, activation="softmax"))
    model.compile(optimizer=tf.keras.optimizers.Adam(lr = 0.0005),
                  loss='mse',
                  metrics=['accuracy'])
    model.fit(X_tra, y_tra, epochs=30, batch_size=32)
    y_pred = model.predict(X_val)
    y_fit = np.argmax(y_pred,axis=1)
    y_val_vec = np.zeros(y_val_arr.shape[0])
    
    for i in range(y_val_arr.shape[0]):
       y_val_vec[i] = np.where(y_val_arr[i,:]==1)[0][0]
    y_val_vec = y_val_vec.astype(int)
    
    correct = 0
    for i in range(y_val_result.shape[0]):
        if y_val_result[i,y_fit[i]] <= 3:
            correct += 1
        
    first_avg_acc += sum(y_fit==y_val_vec)/len(y_fit)
    place_avg_acc += correct/len(y_fit)

end = time.time()
print("Done.")

#Validation performance
first_avg_acc = first_avg_acc/5
place_avg_acc = place_avg_acc/5

print("The training time is ", end - start, "s")
print("In validation, the average accuracy of predicting the first place is ", first_avg_acc)
print("In validation, the average accuracy of the prediction in the 1st-3rd place is ", place_avg_acc)

#Testing performance
first_acc = 0
place_acc = 0
y_pred = model.predict(X_test)
y_fit = np.argmax(y_pred,axis=1)
y_test_vec = np.zeros(y_test_arr.shape[0])

for i in range(y_test_arr.shape[0]):
   y_test_vec[i] = np.where(y_test_arr[i,:]==1)[0][0]
y_test_vec = y_test_vec.astype(int)

y_test_place = np.zeros(y_test_result.shape[0])

correct = 0
for i in range(y_test_result.shape[0]):
    if y_test_result[i,y_fit[i]] <= 3:
        correct += 1
        y_test_place[i] = 1
    
first_acc += sum(y_fit==y_test_vec)/len(y_fit)
place_acc += correct/len(y_fit)

print("In testing, the accuracy of predicting the first place is ", first_acc)
print("In testing, The accuracy of the prediction in the 1st-3rd place is ", place_acc)

#Confusion matrix
print(confusion_matrix(y_test_vec, y_fit))
