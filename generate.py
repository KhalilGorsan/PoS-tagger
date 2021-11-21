"""Script to generate Pos tags for a given text input"""
from data.preprocessor import extract_features_from_text
from train import crf_model

input = "The weather is great !"
input_as_list = input.split()


input_as_feat = [
    extract_features_from_text(input_as_list, i) for i in range(len(input_as_list))
]

pred_tags = crf_model.predict_single(input_as_feat)
print(f"The associated pos tags of {input} are {pred_tags}")
