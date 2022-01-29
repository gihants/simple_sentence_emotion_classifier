from typing import Dict, List, Tuple
import operator

    
def __generate_sentence_weights(sentence: str, simple_word_model: List[Dict[str,Dict[str, int]]]) -> Dict[str, Dict[str, int]]:
    words = sentence.lower().split()
    weights = dict()
    for word in words:
        word_weight = dict()
        for label_weights in simple_word_model:
            label = list(label_weights.keys())[0]
            label_dict = list(label_weights.values())[0]
            label_words = list(label_dict.keys())

            if word in label_words:
                word_weight[label] = label_dict[word]
            else:
                word_weight[label]  = 0

        weights[word] = word_weight
    return weights


def predict_label_new_sentence(sentence: str, simple_word_model: List[Dict[str,Dict[str, int]]]) -> Tuple[str, int]:
    labels = [list(label_weights.keys())[0] for label_weights in simple_word_model]
    
    sentence_weights = __generate_sentence_weights(sentence, simple_word_model)
    
    print("Weights distribution for each word:" , sentence_weights, "\n")
    
    label_scores = dict()
    label_confidences = dict()
    
    for label in labels:
        label_weight = 0
        for word, word_weights in sentence_weights.items():
            try:
                label_weight += (word_weights[label] / sum(list(word_weights.values())))
            except:
                label_weight += 0
        label_scores[label] = label_weight

    print("Label scores: ", label_scores, "\n")
    
    for label in labels:
        label_confidences[label] = round(label_scores[label] * 100 / sum(list(label_scores.values())), 2)
        
    print("Label confidences (%): ", label_confidences, "\n")

    predicted_label =  max(label_confidences.items(), key=operator.itemgetter(1))[0]
    
    prediction_confidence = label_confidences[predicted_label]
    
    print("For the [sentence: ", sentence, "] the predicted label = [", predicted_label, "] with a prediction confidence = [" , prediction_confidence, "%] \n")
    
    return predicted_label, prediction_confidence