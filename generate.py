"""Script to generate Pos tags for a given text input"""
import pickle

from data.preprocessor import extract_features_from_text
from train import MODEL_PATH


def main():
    input = "The weather is great !"
    input_as_list = input.split()

    with open(MODEL_PATH, "rb") as f:
        crf_model = pickle.load(f)

    input_as_feat = [
        extract_features_from_text(input_as_list, i) for i in range(len(input_as_list))
    ]

    pred_tags = crf_model.predict_single(input_as_feat)
    print(f"The associated pos tags of {input} are {pred_tags}")


if __name__ == "__main__":
    main()
