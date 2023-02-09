import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split

from src.dataloader import GeneCharacterisation


class XGBoostClassifier:
    def __init__(self, params=None, num_boost_round=100):
        self.params = params
        self.num_boost_round = num_boost_round
        self.model = None

    def fit(self, X_train, y_train):
        dtrain = xgb.DMatrix(X_train, label=y_train)
        self.model = xgb.train(self.params, dtrain, num_boost_round=self.num_boost_round)

    def predict(self, X_test):
        dtest = xgb.DMatrix(X_test)
        y_pred = self.model.predict(dtest)
        y_pred_binary = [1 if p > 0.5 else 0 for p in y_pred]
        return y_pred_binary

    def score(self, X_test, y_test):
        y_pred = self.predict(X_test)
        accuracy = sum(y_pred == y_test) / len(y_test)
        return accuracy


if __name__ == "__main__":
    df = GeneCharacterisation().tract_features[0]
    X = df.iloc[:, 2:]
    y = df.iloc[:, 1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    params = {
        'booster': 'gbtree',
        'objective': 'binary:logistic',
        'eval_metric': 'error',
        'max_depth': 4,
        'learning_rate': 0.1,
        'n_jobs': -1
    }

    model = XGBoostClassifier(params, num_boost_round=100)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    print("Accuracy: {:.2f}%".format(accuracy * 100))

    # TODO: Implement additional evaluation metrics (ROC-AUC etc.). Also, Save model and write inference code for the
    #  model in a separate script. Also figure out a way to document, keep track of and version the parameters.
    #  Maybe use mlflow?
