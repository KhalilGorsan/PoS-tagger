"""Training script"""
import argparse
import pickle
from pathlib import Path

from sklearn_crfsuite import CRF, metrics

from data.preprocessor import make_it_dataset, ud_corpus_as_list_of_tokens

TRAIN_DATA_PATH = Path("data", "english-GUM", "en_gum-ud-train.conllu.txt")
DEV_DATA_PATH = Path("data", "english-GUM", "en_gum-ud-dev.conllu.txt")
MODEL_PATH = Path("models", "crf_model.sav")

parser = argparse.ArgumentParser(description="Process input data")
parser.add_argument(
    "--train_data", type=str, help="The training dataset", default=TRAIN_DATA_PATH
)
parser.add_argument(
    "--dev_data", type=str, help="The validation dataset", default=DEV_DATA_PATH
)


def main():

    args = parser.parse_args()

    # Read the train input data and make a dataset
    train_data = open(args.train_data)
    list_of_tokens = ud_corpus_as_list_of_tokens(train_data)
    x_train, y_train = make_it_dataset(list_of_tokens)

    # define the model
    crf_model = CRF(
        "lbfgs", c1=0.01, c2=0.1, max_iterations=100, all_possible_transitions=True
    )

    print("Start training for the Pos tagging task on the UD-GUM corpus")
    crf_model.fit(x_train, y_train)
    print("Finished training on the UD-GUM corpus")

    print("This is the F1 score computed on the train data")
    y_pred_train = crf_model.predict(x_train)
    train_f1_score = metrics.flat_f1_score(
        y_train, y_pred_train, average="weighted", labels=crf_model.classes_
    )
    print(train_f1_score)

    # save the mode to disk
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(crf_model, f)


if __name__ == "__main__":
    main()
