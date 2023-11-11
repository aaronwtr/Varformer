import yaml
from sklearn.model_selection import train_test_split

from preprocessing import GeneCharacterisationPreprocessor
from dataloader import DrugTargetData
from torch.utils.data import DataLoader



def main():
    with open("config.yml", 'r') as stream:
        config = yaml.safe_load(stream)

    gcp = GeneCharacterisationPreprocessor(config=config)
    print("Gene characterisation features preprocessed!\n")

    data = gcp.data
    train, test = train_test_split(data, test_size=0.2, random_state=42)
    train = DataLoader(DrugTargetData(data=train))
    test = DataLoader(DrugTargetData(data=test))

    # TODO check distribution of labels in train and test


    print("Data loaded!\n")


if __name__ == "__main__":
    main()
