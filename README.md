# PoS-tagger
Part-Of-Speech tagger
-----------------------------------------------------------------------------
It is the process of marking up a word in a corpus to a corresponding part of a speech tag, based on its context and definition
## Install
-----------------------------------------------------------------------------
We use:
- [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) to setup our environment,
- and python 3.8.3

To setup the environment:
```bash
conda --version

# Clone the repo and navigate to the project directory
git clone https://github.com/KhalilGorsan/PoS-tagger.git
cd PoS-tagger

# Create conda env
conda env create -f environment_cpu.yml
source activate pos-tagger

# Install pre-commit hooks
pre-commit install

# run pre-commit
pre-commit run --all-files
```

## Let's get started
------------------------------------------------------------------------------------
In order to train your model with the Universal Dependencies (UD) corpus, in particular the Georgetown University Multilayer Corpus, all you have to do is to call the train script with the input data and the model name.
```bash
python train.py --train_data train_file --dev_data dev_file --model model_name
```
We support **Conditional Random Fields (CRF)** and **LSTM** models as they fit well to this task, but other more advanced architecture can be added and benchmarked with our baselines.
We can cite Bidirectional LSTM, GRU, etc

Similarly, testing your trained model is triggered as follows:
```bash
python eval.p --test_data test_file --model model_name
```

And finally, to predict some tags for a given text input:
```bash
python generate.py --target_data text_file
```

We put everything related to data under the data directory and we save our models under the model directory.

## Data preprocessing
------------------------------------------------------------------------------------
You can find all related utilities to handle and preprocess the data under `data/preprocessor.py` script.

The data extracted at first is raw, so we used third party libraries and kept only the relevant information.
We assumed that only sequences and their corresponding tags are useful for our task.
We also assumed that the provided data is homogenous enough and cover largely the English corpus so that we generalise well to unseen data.

The data preparation differs from machine learning to deep learning techniques:

- CRF: In this case, we had to build our own features associated to each word in the corpus, by providing signals about the word
position is a sentence, it's type, it's components, etc
- LSTM (or any other deep learning architecture): Neural networks requires scalars as input. Thus, we used some pre-build functions to tokenize the sequences with unique integers.

## Models
------------------------------------------------------------------------------------
Looking at the literature, we distinguish 4 methods to solve Pos Tagging: Manual tagging, Rule-Based Tagging, Probabilistic Methods, Deep Learning Methods.
We would focus only on probabilistic and deep learning methods as they achieve state of the art results and are automated.

### Stochastic/Probabilistic Methods:
This method assigns the POS tags based on the probability of a particular tag sequence occurring. Conditional Random Fields (CRFs) and Hidden Markov Models (HMMs) are probabilistic approaches to assign a POS Tag.
### Deep learning methods:
Recurrent Neural Networks and their variants are a good fit for Pos tagging.

We wanted at a first stage to pick two methods from each category and benchmark them.
We picked CRF as it is widely used by researchers and provides good results and lstm as well.

CRF are also a discriminative probabilistic classifiers. Which means that it will try to model conditional probability distribution, which is the case of Pos tagging, since each word tag is conditioned by the word role in a sentence.

## Metrics and model evaluation
------------------------------------------------------------------------------------
Whether it is for train or test data, We use:
- Precision, Recall, f1_score metrics to access the CRF model
- Loss and accuracy to access LSTM.

So far, without spending any time to fine tune the models, we have these results:
| Model  | train_accuracy | test_accuracy | train f_1 score | test f1 score
| ------------- | ------------- | ------------- | ------------- | ------------- |
| CRF  |   |   | 0.98218  | 0.9337 |
| LSTM | 0.9960  | 0.8216  |  |  |