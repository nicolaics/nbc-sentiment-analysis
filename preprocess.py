import csv
import re

from collections import defaultdict

def convert_row(headers, row):
    header_item_dictionary = {}

    for header, item in zip(headers, row):
        header_item_dictionary[header] = item

    return header_item_dictionary

def get_stopwords() -> list:
    file_name = "stopwords.txt"

    fh = open(file_name, 'r')
    lines = fh.readlines()
    fh.close()

    stopwords = [i.strip() for i in lines]

    return stopwords

def preprocess(file_path, data_use=1) -> tuple[list, dict, dict]:
    fh = open(file_path, "r")

    csv_reader = csv.reader(fh)
    headers = next(csv_reader)

    train_data = []

    for row in csv_reader:
        item_dictionary = convert_row(headers, row)
        train_data.append(item_dictionary)

    fh.close()

    stopwords = get_stopwords()

    word_features = defaultdict(int)

    data_labels_count = {'pos': 0, 'neg': 0}

    data_count = 0
    target_data_count = data_use * len(train_data)

    for row in train_data:
        text = row['text']
        text = text.lower()
        
        text = re.sub('[!\"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~]+', ' ', text)
        
        label = int(row['stars'].strip())

        if label == 5:
            data_labels_count['pos'] += 1
        else:
            data_labels_count['neg'] += 1

        split_text = text.split()

        cleaned_word = []

        for word in split_text:
            word = word.strip()

            if (word != '') and (word not in stopwords):
                word_features[word] += 1
                cleaned_word.append(word)

        row['text'] = cleaned_word

        data_count += 1

        if data_count == target_data_count:
            break

    return (train_data, data_labels_count, word_features)

if __name__ == "__main__":
    train_data_path = "./data/train.csv"
    preprocess(train_data_path)