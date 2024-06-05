# from preprocess import preprocess

def calc_likelihood(word_features: dict, smoothing=1) -> dict:
    likelihood = {}
    
    total_sample = sum(word_features.values())
    total_word_count = len(word_features)

    for word, count in word_features.items():
        likelihood[word] = ((count + smoothing) / (total_sample + (smoothing * total_word_count)))

    return likelihood

def calc_prior_prob(data_labels_cnt: dict) -> dict:
    pos_cnt = data_labels_cnt['pos']
    neg_cnt = data_labels_cnt['neg']

    total_data = pos_cnt + neg_cnt

    pos_prior_prob = (pos_cnt / total_data)
    neg_prior_prob = (neg_cnt / total_data)

    prior_prob = {'pos': pos_prior_prob, 'neg': neg_prior_prob}
    
    return prior_prob

def train(train_data: list, train_data_count: dict, train_words: dict) -> tuple[dict, dict, dict]:
    # train_data_path = "./data/train.csv"

    # train_data, train_data_count, train_words = preprocess(train_data_path)

    top_thousand = dict(sorted(train_words.items(), key=lambda item: item[1], reverse=True)[:1000])

    train_words_pos = {k: 0 for k in top_thousand.keys()}
    train_words_neg = {k: 0 for k in top_thousand.keys()}

    for row in train_data:
        label = int(row['stars'].strip())

        for word in row['text']:
            if word in top_thousand.keys():
                if label == 5:
                    train_words_pos[word] += 1
                else:
                    train_words_neg[word] += 1


    # top_fifty = dict(sorted(train_words.items(), key=lambda item: item[1], reverse=True)[:50])

    # fh = open("top_50.txt", 'a')
    # fh.write(f"Training Data Set Use: {data}")
    # print("TOP 50")
    # for k, v in top_fifty.items():
    #     print(k, v)

    likelihood_pos = calc_likelihood(train_words_pos)
    likelihood_neg = calc_likelihood(train_words_neg)

    prior_prob = calc_prior_prob(train_data_count)

    return (likelihood_pos, likelihood_neg, prior_prob)


# def predict(likelihood_pos: dict, likelihood_neg: dict, prior_prob: dict,
#             test_data_path):
#     # test_data_path = "./data/test.csv"

#     test_data = preprocess(test_data_path)[0]

#     acc_count = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}

#     for row in test_data:
#         true_label = int(row['stars'].strip())

#         pos_prob = prior_prob['pos']
#         neg_prob = prior_prob['neg']

#         for word in row['text']:
#             if word in likelihood_pos.keys():
#                 pos_prob *= likelihood_pos[word]

#         for word in row['text']:
#             if word in likelihood_neg.keys():
#                 neg_prob *= likelihood_neg[word]

#         if pos_prob > neg_prob:
#             pred_label = 5
#         else:
#             pred_label = 1

#         if pred_label == 5:
#             if pred_label == true_label:
#                 acc_count['tp'] += 1
#             else:
#                 acc_count['fp'] += 1
#         else:
#             if pred_label == true_label:
#                 acc_count['tn'] += 1
#             else:
#                 acc_count['fn'] += 1

#     return calc_accuracy(acc_count)

