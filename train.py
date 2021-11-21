"""Training script"""
from sklearn_crfsuite import CRF

from data.preprocessor import make_it_dataset, ud_corpus_as_list_of_tokens

TRAIN_DATA_PATH = "data/english-GUM/en_gum-ud-train.conllu.txt"
DEV_DATA_PATH = "data/english-GUM/en_gum-ud-dev.conllu.txt"

# Read the train input data and make a dataset
train_data = open(TRAIN_DATA_PATH)
list_of_tokens = ud_corpus_as_list_of_tokens(train_data)
x_train, y_train = make_it_dataset(list_of_tokens)

crf_model = CRF(
    "lbfgs", c1=0.01, c2=0.1, max_iterations=100, all_possible_transitions=True
)


print("Start training for the Pos tagging task on the UD-GUM corpus")
crf_model.fit(x_train, y_train)
print("Finished training on the UD-GUM corpus")
