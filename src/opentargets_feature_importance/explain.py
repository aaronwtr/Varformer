from sklearn.model_selection import train_test_split
import shap

from model import XGBoostClassifier
from src.dataloader import GeneCharacterisation


def make_model(X, y, model_type):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    count_0 = sum(y_train == 0)
    count_1 = sum(y_train == 1)
    if count_1 != 0:
        scale_pos_weight = count_0 / count_1
    else:
        scale_pos_weight = 1

    model = XGBoostClassifier(scale_pos_weight=scale_pos_weight, model_type=model_type, num_boost_round=100)
    booster = model.fit(X_train, y_train)

    accuracy, recall, confusion_matrix = model.score(X_test, y_test)
    print(f'{model_type} model results:')
    print("Accuracy: {:.2f}%".format(accuracy * 100))
    print("Recall: {:.2f}%".format(recall * 100))

    return model, booster, X_test


def shap_explainer(model, test_data):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(test_data)
    shap.summary_plot(shap_values, test_data, plot_type="bar")


if __name__ == '__main__':
    df_sm = GeneCharacterisation().tract_features[0]
    X_sm = df_sm.iloc[:, 2:]
    y_sm = df_sm.iloc[:, 1]
    model_sm, booster_sm, X_test_sm = make_model(X_sm, y_sm, 'Small Molecule')
    shap_explainer(booster_sm, X_test_sm)

    df_ab = GeneCharacterisation().tract_features[1]
    X_ab = df_ab.iloc[:, 2:]
    y_ab = df_ab.iloc[:, 1]
    model_ab, booster_sm, X_test_ab = make_model(X_ab, y_ab, 'Antibody')
    shap_explainer(booster_sm, X_test_ab)

    # Note: Can not make model for PROTAC because there are no FDA approved targets
    # from PROTAC based drugs
