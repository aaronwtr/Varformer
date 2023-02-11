from sklearn.model_selection import train_test_split

from model import XGBoostClassifier
from src.dataloader import GeneCharacterisation


def make_model(X, y, model_type):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    count_0 = sum(y_train == 0)
    count_1 = sum(y_train == 1)
    if count_1 != 0:
        scale_pos_weight = count_0 / count_1
        model = XGBoostClassifier(scale_pos_weight=scale_pos_weight, num_boost_round=100)
    else:
        scale_pos_weight = 1
        model = XGBoostClassifier(scale_pos_weight=scale_pos_weight, num_boost_round=100)

    model.fit(X_train, y_train)

    accuracy, recall, confusion_matrix = model.score(X_test, y_test)
    print(f'{model_type} model results:')
    print("Accuracy: {:.2f}%".format(accuracy * 100))
    print("Recall: {:.2f}%".format(recall * 100))

    return model


if __name__ == '__main__':
    df_sm = GeneCharacterisation().tract_features[0]
    X_sm = df_sm.iloc[:, 2:]
    y_sm = df_sm.iloc[:, 1]
    model_sm = make_model(X_sm, y_sm, 'Small Molecule')

    df_ab = GeneCharacterisation().tract_features[1]
    X_ab = df_ab.iloc[:, 2:]
    y_ab = df_ab.iloc[:, 1]
    model_ab = make_model(X_ab, y_ab, 'Antibody')

    # Note: Can not make model for PROTAC because there are no FDA approved targets
    # from PROTAC based drugs
