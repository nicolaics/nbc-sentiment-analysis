'''
    This file is to train the NBC model
'''

# function to calculate the likelihood probability
def calc_likelihood(word_features: dict, smoothing=1) -> dict:
    likelihood = {}
    
    total_sample = sum(word_features.values())
    total_word_count = len(word_features)

    for word, count in word_features.items():
        likelihood[word] = ((count + smoothing) / (total_sample + (smoothing * total_word_count)))

    return likelihood

# function to calculate the prior probability
def calc_prior_prob(data_labels_cnt: dict) -> dict:
    pos_cnt = data_labels_cnt['pos']
    neg_cnt = data_labels_cnt['neg']

    total_data = pos_cnt + neg_cnt

    pos_prior_prob = (pos_cnt / total_data)
    neg_prior_prob = (neg_cnt / total_data)

    prior_prob = {'pos': pos_prior_prob, 'neg': neg_prior_prob}
    
    return prior_prob

# main driver to train the NBC model
def train(train_data: list, train_data_count: dict, train_words: dict) -> tuple[dict, dict, dict]:
    # select top 1000 features 
    top_thousand = dict(sorted(train_words.items(), key=lambda item: item[1], reverse=True)[:1000])

    # store the frequency of the top 1000 words for positive and negative sentiment
    train_words_pos = {k: 0 for k in top_thousand.keys()}
    train_words_neg = {k: 0 for k in top_thousand.keys()}

    # get the number of frequency of word with respect to the sentiment they gave
    for row in train_data:
        label = int(row['stars'].strip())

        for word in row['text']:
            if word in top_thousand.keys():
                if label == 5:
                    train_words_pos[word] += 1
                else:
                    train_words_neg[word] += 1

    # calculate the likelihood probability
    likelihood_pos = calc_likelihood(train_words_pos)
    likelihood_neg = calc_likelihood(train_words_neg)

    # calculate the prior probability
    prior_prob = calc_prior_prob(train_data_count)

    # return the model
    return (likelihood_pos, likelihood_neg, prior_prob)
