'''
    This file is to predict the test data,
    from a trained NBC model
'''

def calc_accuracy(acc_count: dict):
    return ((acc_count['tn'] + acc_count['tp']) / sum(acc_count.values()))

def predict(likelihood_pos: dict, likelihood_neg: dict, prior_prob: dict,
            test_data: list):
    acc_count = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}

    for row in test_data:
        true_label = int(row['stars'].strip())

        pos_prob = prior_prob['pos']
        neg_prob = prior_prob['neg']

        for word in row['text']:
            if word in likelihood_pos.keys():
                pos_prob *= likelihood_pos[word]

        for word in row['text']:
            if word in likelihood_neg.keys():
                neg_prob *= likelihood_neg[word]

        if pos_prob > neg_prob:
            pred_label = 5
        else:
            pred_label = 1

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

    return calc_accuracy(acc_count)