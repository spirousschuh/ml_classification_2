import json

from getting_input import get_data, get_array, important_words

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



def lala():
    train = get_array(train_df, important_words, 'sentiment', 'train.pickle', True)
    valid = get_array(valid_df, important_words, 'sentiment', 'valid.pickle', True)

    print 'Hi'


if __name__ == '__main__':
    lala()