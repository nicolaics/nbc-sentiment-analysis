'''
    This file is to predict the test data,
    from a trained NBC model
'''

# function to calculate the accuracy
def calc_accuracy(acc_count: dict):
    return ((acc_count['tn'] + acc_count['tp']) / sum(acc_count.values()))

def calc_prob(words: list, likelihood: dict, prior_prob):
    prob = prior_prob

    for word in words:
        if word in likelihood.keys():
            prob *= likelihood[word]

    return prob

# function to predict the label of test data
def predict(likelihood_pos: dict, likelihood_neg: dict, prior_prob: dict,
            test_data: list):
    # to calculate the accuracy using:
    # tp = true positive
    # tn = true negative
    # fp = false positive
    # fn = false negative
    acc_count = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}

    # start predicting the label
    for row in test_data:
        true_label = int(row['stars'].strip())
        
        # calculate the positive sentiment probability
        pos_prob = calc_prob(row['text'], likelihood_pos, prior_prob['pos'])
        
        # calculate the negative sentiment probability
        neg_prob = calc_prob(row['text'], likelihood_neg, prior_prob['neg'])
        
        # predict the label using the probability
        if pos_prob > neg_prob:
            pred_label = 5
        else:
            pred_label = 1

        # add the accuracy
        if pred_label == 5:
            if pred_label == true_label:
                acc_count['tp'] += 1
            else:
                acc_count['fp'] += 1
        else:
            if pred_label == true_label:
                acc_count['tn'] += 1
            else:
                acc_count['fn'] += 1

    # calculate the accuracy and return it
    return calc_accuracy(acc_count)