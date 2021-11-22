"""Evaluation script"""
import argparse
import pickle
from pathlib import Path

from sklearn_crfsuite import metrics

from data.preprocessor import make_it_dataset, ud_corpus_as_list_of_tokens
from train import MODEL_PATH

TEST_DATA_PATH = Path("data", "english-GUM", "en_gum-ud-test.conllu.txt")

parser = argparse.ArgumentParser(description="Process input data")
parser.add_argument(
    "--test_data", type=str, help="The testing dataset", default=TEST_DATA_PATH
)


def main():

    args = parser.parse_args()

    # Read the test data and make a dataset
    test_data = open(args.test_data)
    list_of_tokens = ud_corpus_as_list_of_tokens(test_data)
    x_test, y_test = make_it_dataset(list_of_tokens)

    with open(MODEL_PATH, "rb") as f:
        crf_model = pickle.load(f)

    y_pred = crf_model.predict(x_test)

    print("This is the F1 score computed on the test data")
    test_f1_score = metrics.flat_f1_score(
        y_test, y_pred, average="weighted", labels=crf_model.classes_
    )
    print(test_f1_score)


if __name__ == "__main__":
    main()
