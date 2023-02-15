import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn import metrics


class XGBoostClassifier:
    def __init__(self, scale_pos_weight, model_type, num_boost_round=100):
        self.params = {
            'booster': 'gbtree',
            'objective': 'binary:logistic',
            'eval_metric': 'error',
            'max_depth': 5,
            'learning_rate': 0.1,
            'n_jobs': -1,
            'scale_pos_weight': scale_pos_weight
        }
        self.num_boost_round = num_boost_round
        self.model = None
        self.model_type = model_type

    def fit(self, X_train, y_train):
        dtrain = xgb.DMatrix(X_train, label=y_train)
        self.model = xgb.train(self.params, dtrain, num_boost_round=self.num_boost_round)
        return self.model

    def predict(self, X_test):
        dtest = xgb.DMatrix(X_test)
        y_pred = self.model.predict(dtest)
        y_pred_binary = [1 if p > 0.5 else 0 for p in y_pred]
        return y_pred_binary

    def score(self, X_test, y_test):
        y_pred = self.predict(X_test)
        accuracy = sum(y_pred == y_test) / len(y_test)

        confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=[False, True])
        cm_display.plot()
        plt.title(f'{self.model_type} confusion matrix')
        plt.show()

        TP = confusion_matrix[1, 1]
        FN = confusion_matrix[1, 0]
        recall = TP / (TP + FN)

        # calculate precision
        TP = confusion_matrix[1, 1]
        FP = confusion_matrix[0, 1]
        precision = TP / (TP + FP)

        # calculate F1 score
        F1 = 2 * (precision * recall) / (precision + recall)

        return accuracy, recall, F1, confusion_matrix
