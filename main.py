'''
    Main file to drive all of the code.
    Run this file to train the NBC model
    and predict the test dataset using
    the trained NBC model.
'''

from training import train
from predict import predict
from preprocess import preprocess
from plotting import plot

import os

# specify the data path
train_data_path = "./data/train.csv"
test_data_path = "./data/test.csv"

# the dictionary to store the accuracy of the NBC model
accuracy = {}

# the amount of data to be used to train the NBC model
use_train_data_percentage = [0.1, 0.3, 0.5, 0.7, 1]

# pre-process the test data first
test_data = preprocess(test_data_path)[0]

top_20_50_file = "top_20_50_words.txt"

# remove the text file first if exists
if os.path.exists(top_20_50_file):
    os.remove(top_20_50_file)

# the loop to train the NBC and predict the test data
# and calculate the accuracy of the model
for i in use_train_data_percentage:
    # pre-process the train data first
    train_data, train_words_label_count, train_words_list = preprocess(train_data_path, i)

    # get the top 50 words that appear
    top_twenty_fifty = dict(sorted(train_words_list.items(), key=lambda item: item[1], reverse=True)[19:50])

    count = 20

    # save the top 50 words in a file
    fh = open(top_20_50_file, 'a')
    fh.write(f"Training Data Set Use: {(i * 100):.0f}%\n")
    for k, v in top_twenty_fifty.items():
        fh.write(f'{count}. {k}: {v}\n')
        count += 1

    fh.write("\n\n")
    fh.flush()
    fh.close()

    # train the NBC model
    likelihood_pos, likelihood_neg, prior_prob = train(train_data, train_words_label_count, train_words_list)

    # predict the label from the trained NBC and calculate the accuracy of the model
    temp_accuracy = predict(likelihood_pos, likelihood_neg, prior_prob, test_data)

    # save the accuracy
    accuracy[(i * 100)] = temp_accuracy

plot(accuracy)
