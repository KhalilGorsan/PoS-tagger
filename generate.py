"""Script to generate Pos tags for a given text input"""
import argparse
import pickle

from data.preprocessor import extract_features_from_text
from train import MODEL_PATH

parser = argparse.ArgumentParser(description="Generate Pos tags for input data")
parser.add_argument(
    "--target_data",
    type=str,
    help="The file containing sentences on which to generate POS tags",
)


def main():

    args = parser.parse_args()

    with open(MODEL_PATH, "rb") as f:
        crf_model = pickle.load(f)

    with open(args.target_data, "r") as f:
        list_of_sentences = f.readlines()

    for sentence in list_of_sentences:
        input_as_feat = [
            extract_features_from_text(sentence, i) for i in range(len(sentence))
        ]

        pred_tags = crf_model.predict_single(input_as_feat)
        print(f"The associated pos tags of {sentence} are {pred_tags}")


if __name__ == "__main__":
    main()
