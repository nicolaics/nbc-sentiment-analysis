import csv
import re

class Row:
    def __init__(self) -> None:
        self.stars = 0
        self.text = ""
        self.words = []

class Features:
    def __init__(self) -> None:
        self.word = ""
        self.freq = 0
        self.pos = 0
        self.neg = 0
        
'''
    special chars: " !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"
'''


def convert_row(headers, row):
    header_item_dictionary = {}

    for header, item in zip(headers, row):
        header_item_dictionary[header] = item.lower()

    return header_item_dictionary

def get_stopwords() -> list:
    file_name = "stopwords.txt"

    fh = open(file_name, 'r')
    lines = fh.readlines()
    fh.close()

    stopwords = [i.strip() for i in lines]

    return stopwords

def preprocess() -> list:
    train_data_path = "./data/train.csv"

    fh = open(train_data_path, "r")

    csv_reader = csv.reader(fh)
    headers = next(csv_reader)

    csv_rows = []

    for row in csv_reader:
        item_dictionary = convert_row(headers, row)
        csv_rows.append(item_dictionary)

    stopwords = get_stopwords()

    row_list = []
    features = {}

    word_features = []

    for row in csv_rows:
        if int(row['stars'].strip()) == 5:
            sentiment = 1
        else:
            sentiment = 0
        
        # row_data = Row()

        # row_data.stars = int(row['stars'].strip())
        text = re.sub('[!\"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~]+', ' ', row['text'])

        for word in text.split(' '):
            word = word.strip()

            if (word != '') and (word not in stopwords):
                # row_data.words.append(i)
                # if word in features.keys():
                #     features[word] += 1
                # else:
                #     features[word] = 1

                is_found = False

                if len(word_features) != 0:
                    for j in range(len(word_features)):
                        if word in word_features[j].word:
                            if sentiment == 1:
                                word_features[j].pos += 1
                            else:
                                word_features[j].neg += 1

                            word_features[j].freq += 1
                            is_found = True
                            break
                
                if len(word_features) == 0 or is_found is False:
                    word_feature = Features()
                    word_feature.freq += 1
                    word_feature.word = word

                    if sentiment == 1:
                        word_feature.pos += 1
                    else:
                        word_feature.neg += 1
                            
                    word_features.append(word_feature)
                
                                    
        # row_list.append(row_data)

    fh.close()

    # print(row_list[0].stars)
    # print(row_list[0].text)
    # print(row_list[0].words)

    count = 0

    # final_features = {}
    # top_fifty = {}

    final_features = []
    top_fifty = []
    
    for feat in sorted(word_features, key=lambda item: item.freq, reverse=True):
        final_features.append(feat)

        if count < 50:
            top_fifty.append(feat)

        count += 1

        if count == 1000:
            break

    # for k, v in dict(sorted(features.items(), key=lambda item: item[1], reverse=True)).items():
    #     final_features[k] = v

    #     if count < 50:
    #         top_fifty[k] = v
        
    #     count += 1

    #     if count == 1000:
    #         break

    # for k, v in final_features.items():
    #     print(k, v)

    print("TOP 50 WORDS")
    # for k, v in top_fifty.items():
    #     print(k, v)

    for i in top_fifty:
        print(i.word, i.freq, i.pos, i.neg)
    # print(len(final_features))
    # print(len(top_fifty))

    return final_features

if __name__ == "__main__":
    preprocess()