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

# remove the text file first if exists
if os.path.exists("top_50.txt"):
    os.remove("top_50.txt")

# the loop to train the NBC and predict the test data
# and calculate the accuracy of the model
for i in use_train_data_percentage:
    # pre-process the train data first
    train_data, train_words_label_count, train_words_list = preprocess(train_data_path, i)

    # get the top 50 words that appear
    top_fifty = dict(sorted(train_words_list.items(), key=lambda item: item[1], reverse=True)[:50])

    # save the top 50 words in a file
    fh = open("top_50.txt", 'a')
    fh.write(f"Training Data Set Use: {(i * 100):.0f}%\n")
    for k, v in top_fifty.items():
        fh.write(f'{k}: {v}\n')

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
