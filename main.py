from training import train
from predict import predict
from preprocess import preprocess
from plotting import plot

import os

train_data_path = "./data/train.csv"
test_data_path = "./data/test.csv"

accuracy = {}

use_train_data_percentage = [0.1, 0.3, 0.5, 0.7, 1]

test_data = preprocess(test_data_path)[0]

if os.path.exists("top_50.txt"):
    os.remove("top_50.txt")

for i in use_train_data_percentage:
    train_data, train_words_label_count, train_words_list = preprocess(train_data_path, i)

    top_fifty = dict(sorted(train_words_list.items(), key=lambda item: item[1], reverse=True)[:50])

    fh = open("top_50.txt", 'a')
    fh.write(f"Training Data Set Use: {(i * 100):.0f}%\n")
    for k, v in top_fifty.items():
        fh.write(f'{k}: {v}\n')

    fh.write("\n\n")
    fh.flush()
    fh.close()

    likelihood_pos, likelihood_neg, prior_prob = train(train_data, train_words_label_count, train_words_list)
    temp_accuracy = predict(likelihood_pos, likelihood_neg, prior_prob, test_data)

    accuracy[(i * 100)] = temp_accuracy

plot(accuracy)