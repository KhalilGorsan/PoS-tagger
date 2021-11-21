"""Evaluation script"""
from sklearn_crfsuite import metrics

from data.preprocessor import make_it_dataset, ud_corpus_as_list_of_tokens
from train import crf_model, x_train, y_train

TEST_DATA_PATH = "data/english-GUM/en_gum-ud-test.conllu.txt"

# Read the test data and make a dataset
test_data = open(TEST_DATA_PATH)
list_of_tokens = ud_corpus_as_list_of_tokens(test_data)
x_test, y_test = make_it_dataset(list_of_tokens)

y_pred = crf_model.predict(x_test)

print("This is the F1 score computed on the test data")
test_f1_score = metrics.flat_f1_score(
    y_test, y_pred, average="weighted", labels=crf_model.classes_
)
print(test_f1_score)

print("This is the F1 score computed on the train data")
y_pred_train = crf_model.predict(x_train)
train_f1_score = metrics.flat_f1_score(
    y_train, y_pred_train, average="weighted", labels=crf_model.classes_
)
print(train_f1_score)
