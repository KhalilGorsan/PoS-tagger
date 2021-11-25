"""Training script"""
import argparse
import pickle
from pathlib import Path

from sklearn_crfsuite import CRF, metrics
from tensorflow.keras.layers import LSTM, Dense, Embedding, TimeDistributed
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

from data.preprocessor import (
    create_encoded_dataset,
    make_it_dataset,
    ud_corpus_as_list_of_tokens,
)

TRAIN_DATA_PATH = Path("data", "english-GUM", "en_gum-ud-train.conllu.txt")
DEV_DATA_PATH = Path("data", "english-GUM", "en_gum-ud-dev.conllu.txt")
MODEL_PATH = Path("models")

parser = argparse.ArgumentParser(description="Process input data")
parser.add_argument(
    "--train_data", type=str, help="The training dataset", default=TRAIN_DATA_PATH
)
parser.add_argument(
    "--dev_data", type=str, help="The validation dataset", default=DEV_DATA_PATH
)
parser.add_argument(
    "--model",
    type=str,
    choices=["crf", "lstm"],
    help="The model used to solve Pos tag task",
    default="lstm",
)


def train_crf_model(
    train_data: str,
    algo: str = "lbfgs",
    c1: int = 0.01,
    c2: int = 0.1,
    max_iterations: int = 100,
):

    # Read the train input data and make a dataset
    train_data = open(train_data)
    list_of_tokens = ud_corpus_as_list_of_tokens(train_data)
    x_train, y_train = make_it_dataset(list_of_tokens)

    # define the model
    crf_model = CRF(
        algo, c1=c1, c2=c2, max_iterations=max_iterations, all_possible_transitions=True
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
    with open(MODEL_PATH / Path("crf_model.sav"), "wb") as f:
        pickle.dump(crf_model, f)


def create_lstm_model(
    vocab_size: int, max_seq_len: int, num_classes: int
) -> Sequential:
    """Creates an LSTM model that output a probability distribution over the POS Tags.
   Args:
       vocab_size: number of words in the corpus.
       max_seq_len: the max length of sequences used for padding.
       num_classes: number of POS tags.
   Returns:
       model: A compiled Keras model
   """
    model = Sequential()
    model.add(
        Embedding(
            input_dim=vocab_size,
            output_dim=300,
            input_length=max_seq_len,
            trainable=True,
        )
    )
    model.add(LSTM(64, return_sequences=True))
    model.add(TimeDistributed(Dense(num_classes, activation="softmax")))
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["acc"])
    print(model.summary())
    return model


def train_lstm_model(train_data: str):
    x_encoded, y_encoded, info = create_encoded_dataset(train_data)
    # pad the sequences to max_seq_length
    x_train = pad_sequences(x_encoded, maxlen=info["max_seq_len"], padding="post")
    y_train = pad_sequences(y_encoded, maxlen=info["max_seq_len"], padding="post")
    # make tags a one-hot encode vector
    y_train = to_categorical(y_train)
    num_classes = y_train.shape[2]
    model = create_lstm_model(info["vocab_size"], info["max_seq_len"], num_classes)
    model.fit(x_train, y_train, batch_size=128, epochs=1)
    checkpoint_path = Path(MODEL_PATH / "lstm")
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    model.save_weights(checkpoint_path)


def main():

    args = parser.parse_args()

    if args.model == "crf":
        train_crf_model(args.train_data)
    elif args.model == "lstm":
        train_lstm_model(args.train_data)
    else:
        raise ValueError(f"Model {args.model} not supported yet")


if __name__ == "__main__":
    main()
