from typing import Dict, List, Tuple
import pandas as pd

def __load_training_data(train_data_file: str) -> Tuple[List[str], List[List[str]]]:
    train_df = pd.read_csv('training_data/sentences.csv', header = None)
    train_df.columns =['sentence', 'label']
    train_df = train_df.groupby('label')['sentence'].apply(list).reset_index(name='sentences')
    labels = train_df.label.to_list()
    sentence_sets = train_df.sentences.to_list()
    return labels, sentence_sets

def __word_count(sentence: str) -> List[Dict[str, int]]:
    counts = dict()
    words = sentence.split()

    for word in words:
        if word in counts:
            counts[word] += 1
        else:
            counts[word] = 1     

    return counts

def build_simple_words_model(train_data_file: str) -> List[Dict[str,Dict[str, int]]]:
    """Builds a simple statistical sentence classification model

    Parameters:
    train_data_file (int): Link to the csv file containing the training data and the labels

    Returns:
    List[Dict[str,Dict[str, int]]]: The trained model

   """
    labels, sentence_sets = __load_training_data(train_data_file = train_data_file)
    word_counts = [__word_count(' '.join(sentence_set).lower()) for sentence_set in sentence_sets]
    simple_word_model = ([{label : word_count} for label, word_count in zip(labels, word_counts)])
    return simple_word_model