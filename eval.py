"""Evaluation script"""
import argparse
import pickle as pkl
from pathlib import Path

from sklearn_crfsuite import metrics
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

from data.preprocessor import (
    create_encoded_dataset,
    make_it_dataset,
    ud_corpus_as_list_of_tokens,
)
from train import MODEL_PATH, create_lstm_model

TEST_DATA_PATH = Path("data", "english-GUM", "en_gum-ud-test.conllu.txt")

parser = argparse.ArgumentParser(description="Process input data")
parser.add_argument(
    "--test_data", type=str, help="The testing dataset", default=TEST_DATA_PATH
)
parser.add_argument(
    "--model",
    type=str,
    choices=["crf", "lstm"],
    help="The model used to solve Pos tag task",
    default="crf",
)


def eval_crf_model(test_data):

    # Read the test data and make a dataset
    test_data = open(test_data)
    list_of_tokens = ud_corpus_as_list_of_tokens(test_data)
    x_test, y_test = make_it_dataset(list_of_tokens)

    with open(MODEL_PATH / "crf_model.sav", "rb") as f:
        crf_model = pkl.load(f)

    y_pred = crf_model.predict(x_test)

    print("This is the F1 score computed on the test data")
    test_f1_score = metrics.flat_f1_score(
        y_test, y_pred, average="weighted", labels=crf_model.classes_
    )
    print(test_f1_score)


def eval_lstm_model(test_data):

    x_encoded, y_encoded, _ = create_encoded_dataset(test_data)

    with open(MODEL_PATH / "lstm" / "config.pkl", "rb") as f:
        info = pkl.load(f)

    # pad the sequences to max_seq_length
    x_test = pad_sequences(x_encoded, maxlen=info["max_seq_len"], padding="post")
    y_test = pad_sequences(y_encoded, maxlen=info["max_seq_len"], padding="post")

    # make tags a one-hot encode vector
    y_test = to_categorical(y_test)
    num_classes = y_test.shape[2]

    # create and load the model weights
    model = create_lstm_model(info["vocab_size"], info["max_seq_len"], num_classes)
    model.load_weights(str(MODEL_PATH / "lstm" / "lstm_checkpoint.h5"))

    loss, accuracy = model.evaluate(x_test, y_test)
    print(f"Accuracy: {accuracy}")
    print(f"Loss: {loss}")


def main():

    args = parser.parse_args()

    if args.model == "crf":
        eval_crf_model(args.test_data)
    elif args.model == "lstm":
        eval_lstm_model(args.test_data)
    else:
        raise ValueError(f"Model {args.model} not supported yet")


if __name__ == "__main__":
    main()
