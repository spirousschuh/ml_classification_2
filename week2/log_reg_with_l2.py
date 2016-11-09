import json
import pickle
import numpy as np

from getting_input import get_data, get_array, important_words
from logistic_scratch import predict_probability, feature_derivative, compute_log_likelihood

with open('module-4-assignment-train-idx.json') as file:
    train_ids = json.load(file)
with open('module-4-assignment-validation-idx.json') as file:
    valid_ids = json.load(file)

def write_data_to_pickles():
    df = get_data()
    train_df = df.loc[train_ids]
    valid_df = df.loc[valid_ids]
    train = get_array(train_df, important_words, 'sentiment', 'train.pickle')
    valid = get_array(valid_df, important_words, 'sentiment', 'valid.pickle')


def compute_log_likelihood_l2(feature_matrix, sentiment, coefficients, l2_penalty):
    indicator_func = sentiment == 1
    scores = feature_matrix.dot(coefficients)
    return np.sum((indicator_func - 1) * scores - np.log(1. + np.exp(-scores))) - \
           l2_penalty * np.linalg.norm(coefficients) ** 2


def logistic_regression_with_l2(feature_matrix, sentiment, initial_coef, step_size, max_iter, l2_penalty):
    coefficients = initial_coef
    indicator = sentiment == +1
    for itr in range(max_iter):
        predictions = predict_probability(feature_matrix, coefficients)
        errors = indicator - predictions
        for f_number, feature in enumerate(feature_matrix.T):
            if f_number == 0:
                delta = step_size * feature_derivative(errors, feature)
            else:
                delta = step_size * (feature_derivative(errors, feature) - 2 * l2_penalty * coefficients[f_number])

            coefficients[f_number] += delta

        # logging information
        if itr <= 15 or (itr <= 100 and itr % 10 == 0) or (itr <= 1000 and itr % 100 == 0) \
                or (itr <= 10000 and itr % 1000 == 0) or itr % 10000 == 0:
            lp = compute_log_likelihood_l2(feature_matrix, sentiment, coefficients, l2_penalty)
            print 'iteration %*d: log likelihood of observed labels = %.8f' % \
            (int(np.ceil(np.log10(max_iter))), itr, lp)
    with open('final_coeffs_l2.pickle', 'w') as file:
        pickle.dump(coefficients, file)
    return coefficients


def process_classification():
    train = get_array(None, important_words, 'sentiment', 'train.pickle', True)
    valid = get_array(None, important_words, 'sentiment', 'valid.pickle', True)
    results = []
    for l2 in [0., 4., 10., 100., 1e3, 1e5]:
        result_coeffs = logistic_regression_with_l2(train[0], train[1], np.zeros(train[0].shape[1]), 5e-6, 501, l2)
        results.append((l2, result_coeffs))
    with open('all_coeffs.pickle', 'w') as file:
        pickle.dump(results, file)
    return results


if __name__ == '__main__':
    process_classification()