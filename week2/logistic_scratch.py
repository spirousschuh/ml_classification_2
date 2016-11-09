import json
import pickle

import numpy as np
import pandas as pd

from input_fetching import remove_punctuation

word_file = open('important_words.json')
important_words = json.load(word_file)
word_file.close()

def get_data():
    product_data = pd.read_csv('amazon_baby_subset.csv')

    product_data.fillna({'review': ''})
    product_data['review_clean'] = product_data['review'].apply(remove_punctuation)

    product_data['sentiment'] = product_data['rating'].apply(lambda rating: 1 if rating > 3 else -1)

    for word in important_words:
        product_data[word] = product_data['review_clean'].apply(lambda review: review.split().count(word))

    return product_data


def get_array(dataframe, features, label, read_from_file=False):
    if read_from_file:
        with open('matrix.pickle', 'r') as file:
            return pickle.load(file)

    dataframe['constant'] = 1
    features = ['constant'] + features
    feature_array = dataframe[features].as_matrix()

    label_array = dataframe[label].as_matrix()
    with open('matrix.pickle', 'w') as file:
        pickle.dump((feature_array, label_array), file)

    return feature_array, label_array

def predict_probability(features, coefficients):
    score = features.dot(coefficients)
    return 1 / (1 + np.exp(-score))

def feature_derivative(errors, feature_values):
    return np.dot(errors, feature_values)

def compute_log_likelihood(feature_matrix, sentiment, coefficients):
    indicator_func = sentiment == 1
    scores = feature_matrix.dot(coefficients)
    return np.sum((indicator_func - 1) * scores - np.log(1. + np.exp(-scores)))

def logistic_regression(feature_matrix, sentiment, initial_coef, step_size, max_iter):
    coefficients = initial_coef
    indicator = sentiment == +1
    for itr in range(max_iter):
        predictions = predict_probability(feature_matrix, coefficients)
        errors = indicator - predictions
        for f_number, feature in enumerate(feature_matrix.T):
            coefficients[f_number] += step_size * feature_derivative(errors, feature)

        if itr <= 15 or (itr <= 100 and itr % 10 == 0) or (itr <= 1000 and itr % 100 == 0) \
                or (itr <= 10000 and itr % 1000 == 0) or itr % 10000 == 0:
            lp = compute_log_likelihood(feature_matrix, sentiment, coefficients)
            print 'iteration %*d: log likelihood of observed labels = %.8f' % \
            (int(np.ceil(np.log10(max_iter))), itr, lp)
    with open('final_coeffs.pickle', 'w') as file:
        pickle.dump(coefficients, file)
    return coefficients

def process_data(products):
    print products['review_clean']
    print "The number of postitive reviews: " + str(len(products.loc[products['sentiment'] == +1]))
    print "The number of negative reviews: " + str(len(products.loc[products['sentiment'] == -1]))
    print "################ quiz question  ###########"
    print "Number of reviews containing perfect :" + str(len(products.loc[products['perfect'] > 0]))


if __name__ == '__main__':
    # products = get_data()
    # process_data(products)
    features, sentiment = get_array(None, important_words, 'sentiment', True)
    #result_coeffs = logistic_regression(features, sentiment, np.zeros(features.shape[1]), 1e-7, 301)
    with open('final_coeffs.pickle', 'r') as file:
        result_coeffs = pickle.load(file)
    print result_coeffs
    scores = features.dot(result_coeffs)
    labels = np.array([1 if score > 0 else -1 for score in scores])
    print "The number of positive predictions is " + str(np.sum(labels == 1))
    correct_predictions = labels == sentiment
    print "The accuracy is " + str(float(np.sum(correct_predictions)) / len(correct_predictions))
    tuples = zip(result_coeffs[1:], important_words)
    sorted_words = sorted(tuples, key=lambda x: x[0], reverse=True)
    print sorted_words
    zen_positive = [tuple[1] for tuple in sorted_words[:10]]
    print zen_positive
    zen_negative = [tuple[1] for tuple in sorted_words[-10:]]
    print zen_negative