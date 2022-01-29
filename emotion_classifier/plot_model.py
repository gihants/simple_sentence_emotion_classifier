from wordcloud import WordCloud
import matplotlib.pyplot as plt
from typing import List, Dict

def generate_label_wordclouds(simple_word_model: List[Dict[str,Dict[str, int]]]) -> None:
    fig = plt.figure(figsize=(15,8))
    for index, label_dict in enumerate(simple_word_model):
        label = list(label_dict.keys())[0]
        words_dict = label_dict[label]
        fig.add_subplot(len(simple_word_model),1,index+1)
        wordcloud = WordCloud(width = 1000, height = 500).generate_from_frequencies(words_dict)
        plt.imshow(wordcloud)
        plt.axis('off')
        plt.title("Wordcloud for the sentiment class = " + label)
        
    plt.imshow(wordcloud)