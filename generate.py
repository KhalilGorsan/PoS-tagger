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

    with open(MODEL_PATH / "crf_model.sav", "rb") as f:
        crf_model = pickle.load(f)

    with open(args.target_data, "r") as f:
        list_of_sentences = f.readlines()

    for sentence in list_of_sentences:
        # get rid of /n that marks line break
        split_sentence = sentence[:-1].split()
        input_as_feat = [
            extract_features_from_text(split_sentence, i)
            for i in range(len(split_sentence))
        ]

        pred_tags = crf_model.predict_single(input_as_feat)
        print(f"Sentence: {sentence}")
        print(f"POS tags: {pred_tags}")
        print("-" * 100)


if __name__ == "__main__":
    # python generate.py --target_data data/test_tags.txt
    main()
