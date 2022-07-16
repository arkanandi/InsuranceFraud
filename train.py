import json
import pickle
import numpy as np
import pandas as pd
import scipy.stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder


def detect_fraud(hyperparams):
    """ Insurance farud detection.

    Performs gradient-boost algorithm to train the fraud_data and give the best model to test on.

    :param : Hyperparameters to train the algorithms
    :return: The best model after training on dataset
    """

    train = pd.read_csv('fraud_data/train.csv')  # Load the fraud_data file of training

    y = train['fraud']  # Create the column with fraud status
    X = train.drop(columns=['fraud'])  # Drop the column name fraud from the training fraud_data and save it to X

    # Splitting the dataset into train and validation / test during training period
    X_train_n, X_test_n, y_train, y_test = train_test_split(X, y, train_size=0.85, random_state=1234)

    #  Drop the column with claim_number for the preprocessing
    X_train = X_train_n.drop(columns=['claim_number'])
    X_test = X_test_n.drop(columns=['claim_number'])

    X_train = X_train.apply(LabelEncoder().fit_transform)
    X_test = X_test.apply(LabelEncoder().fit_transform)

    np.random.seed(123)

    randomClassifier = RandomForestClassifier()

    param_grid = {"n_estimators": np.arange(hyperparams.n_estimators_min, hyperparams.n_estimators_max).tolist(),
                  "max_depth": np.arange(hyperparams.max_depth_min, hyperparams.max_depth_max).tolist(),
                  "max_features": np.arange(hyperparams.max_features_min, hyperparams.max_features_max).tolist()}

    hparam_tuner = GridSearchCV(randomClassifier, cv=3, scoring='roc_auc', param_grid=param_grid)
    hparam_tuner = hparam_tuner.fit(X_train, y_train)

    pd.DataFrame(
        hparam_tuner.cv_results_,
        columns=[
            'n_estimators',
            'min_samples_split',
            'min_samples_leaf',
            'max_depth',
            'max_features',
            'mean_test_score',
            'std_test_score',
            'rank_test_score',
        ],
    ).sort_values(by=['rank_test_score'])

    #  Saving best model from training
    best_model = hparam_tuner.best_estimator_

    probs = best_model.predict_proba(X_test)
    df = pd.DataFrame({'claim_number': X_test_n['claim_number'], 'fraud': probs[:, 1]})
    df.to_csv("submission.csv", index=False)

    with open('models/store_best_model.pickle', 'wb') as f:
        pickle.dump(best_model, f)

    return "Training is successful. Best model has been saved at 'models/store_best_model.pickle' "


def test():
    """ Insurance fraud detection.

    Performs testing on test dataset on gradient-boost algorithm best model.

    :return: The json file with fraud prediction
    """
    with open('models/store_best_model.pickle', 'rb') as f:
        loaded_model = pickle.load(f)

    test = pd.read_csv('fraud_data/test.csv')  # Load testing fraud_data
    X_test = test.drop(columns=['claim_number'])
    test_withoutID = X_test.fillna('na')
    test_withoutID = test_withoutID.apply(LabelEncoder().fit_transform)
    final_y = loaded_model.predict(test_withoutID)  # Predict the test fraud_data

    final_report = test
    final_report['fraud_status'] = final_y
    final_report = final_report.loc[:, ['claim_number', 'fraud_status']]
    # Replace 1-0 with Yes-No to make it interpretable
    final_report = final_report.replace(1, 'Fraud')
    final_report = final_report.replace(0, 'Not Fraud')

    final_report.to_json('output/report.json', orient='records')

    f = open('output/report.json', "r")
    result = json.loads(f.read())

    return {"fraud_database": result}
