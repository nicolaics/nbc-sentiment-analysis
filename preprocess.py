'''
    The file to pre-process the text
'''

import csv
import re

from collections import defaultdict

# convert the row data from the csv file
def convert_row(headers, row):
    header_item_dictionary = {}

    for header, item in zip(headers, row):
        header_item_dictionary[header] = item

    return header_item_dictionary

# get stopwords
def get_stopwords() -> list:
    file_name = "stopwords.txt"

    fh = open(file_name, 'r')
    lines = fh.readlines()
    fh.close()

    stopwords = [i.strip() for i in lines]

    return stopwords

# main driver to pre-process the data
def preprocess(file_path, data_use=1) -> tuple[list, dict, dict]:
    # open and read the csv file
    fh = open(file_path, "r")

    csv_reader = csv.reader(fh)
    headers = next(csv_reader)

    data = []

    for row in csv_reader:
        item_dictionary = convert_row(headers, row)
        data.append(item_dictionary)

    fh.close()

    stopwords = get_stopwords()

    # variable to store all the word features
    word_features = defaultdict(int)

    # total data's labels count
    data_labels_count = {'pos': 0, 'neg': 0}

    data_count = 0

    # number of training/test data to be utilize
    target_data_count = round(data_use * len(data))

    # pre-process the data
    for row in data:
        text = row['text']

        # lowercase the text
        text = text.lower()
        
        # remove special chars
        text = re.sub('[!\"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~]+', ' ', text)
        
        label = int(row['stars'].strip())

        # if the sentiment is positive sentiment
        if label == 5:
            data_labels_count['pos'] += 1
        else: # negative sentiment
            data_labels_count['neg'] += 1

        split_text = text.split()

        cleaned_word = []

        for word in split_text:
            word = word.strip()

            if (word != '') and (word not in stopwords):
                word_features[word] += 1

                # save the pre-process word in the list
                cleaned_word.append(word)

        # change the text into the pre-processed words list
        row['text'] = cleaned_word

        data_count += 1

        if data_count == target_data_count:
            break

    return (data, data_labels_count, word_features)
