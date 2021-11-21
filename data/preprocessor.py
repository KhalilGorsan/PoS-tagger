import re
from typing import Callable, Dict, List, TextIO, Union

from conllu import parse_incr


def is_alphanumeric(s: str) -> int:
    return int(bool((re.match("^(?=.*[0-9]$)(?=.*[a-zA-Z])", s))))


# Register features here
features: Dict[str, Callable] = {
    "word": lambda text, index: text[index],
    "is_first": lambda text, index: index == 0,
    "is_last": lambda text, index: index == len(text) - 1,
    "is_capitalized": lambda text, index: text[index][0].upper() == text[index][0],
    "is_all_caps": lambda text, index: text[index].upper() == text[index],
    "is_all_lower": lambda text, index: text[index].lower() == text[index],
    "is_alphanumeric": lambda text, index: is_alphanumeric([text[index]]),
    "prefix-1": lambda text, index: text[index][0],
    "prefix-2": lambda text, index: text[index][:2],
    "prefix-3": lambda text, index: text[index][:3],
    "prefix-4": lambda text, index: text[index][:4],
    "suffix-1": lambda text, index: text[index][-1],
    "suffix-2": lambda text, index: text[index][-2:],
    "suffix-3": lambda text, index: text[index][-3:],
    "suffix-4": lambda text, index: text[index][-4:],
    "prev_word": lambda text, index: "" if index == 0 else text[index - 1],
    "next_word": lambda text, index: "" if index < len(text) else text[index + 1],
    "has_hyphen": lambda text, index: "-" in text[index],
    "is_numeric": lambda text, index: text[index].isdigit(),
    "capitals_inside": lambda text, index: text[index][1:].lower() != text[index][1:],
}


def ud_corpus_as_list_of_tokens(data_file: TextIO) -> List[Dict]:
    """Convert a raw data input from the Universal Dependencies syntax annotations from
    the GUM corpus to a word(token) tag mapping stored as python objects.

    Extra information from the corpus are discarded and only words and Pos tags are kept

    Args:
        data_file: train, dev, or test data that holds the rew data.

    Returns:
        list_of_tokens: each element contains a dict mapping the words and their tags
    """
    corpus_as_list_of_tokens = list()
    for tokens_list in parse_incr(data_file):
        token_and_tag = {token["form"]: token["upos"] for token in tokens_list}
        corpus_as_list_of_tokens.append(token_and_tag)

    return corpus_as_list_of_tokens


def extract_features_from_text(text: str, index: int) -> Dict:
    return {k: v(text, index) for k, v in features.items}


def make_it_dataset(list_of_tokens: List[Dict]) -> Union[List, List]:
    """Transforms the input into a dataset with a split of features and labels.

    This dataset will be used later to train models.

    Args:
        list_of_tokens: a list of dict where each dict holds a mapping between words and
        their associated Pos tags.

    Returns:
        features: the calculated features of all the words in the corpus.
        lables: the associated Pos tag for each word.
    """
    features, labels = [], []
    for tokens in list_of_tokens:
        features.append(list(tokens.keys()))
        labels.append(list(tokens.values()))

    return features, labels