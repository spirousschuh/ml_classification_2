import json
import pickle
import string

import pandas as pd

word_file = open('important_words.json')
important_words = json.load(word_file)
word_file.close()

def remove_punctuation(text):
    return str(text).translate(None, string.punctuation)


def get_data():
    product_data = pd.read_csv('amazon_baby_subset.csv')

    product_data.fillna({'review': ''})
    product_data['review_clean'] = product_data['review'].apply(remove_punctuation)

    product_data['sentiment'] = product_data['rating'].apply(lambda rating: 1 if rating > 3 else -1)

    for word in important_words:
        product_data[word] = product_data['review_clean'].apply(lambda review: review.split().count(word))

    return product_data


def get_array(dataframe, features, label, filename, read_from_file=False):
    if read_from_file:
        with open(filename, 'r') as my_file:
            return pickle.load(my_file)

    dataframe['constant'] = 1
    features = ['constant'] + features
    feature_array = dataframe[features].as_matrix()

    label_array = dataframe[label].as_matrix()
    with open(filename, 'w') as my_file:
        pickle.dump((feature_array, label_array), my_file)

    return feature_array, label_array