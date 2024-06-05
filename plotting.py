'''
    The file to plot the result of the accuracy
'''

import matplotlib.pyplot as plt

def plot(accuracy: dict):
    plt.figure(figsize=(12, 8))

    # plot the values of the accuracy
    plt.plot(accuracy.keys(), accuracy.values(), marker='o', linestyle='-')

    plt.xlabel('Training Data Set (%)')
    plt.ylabel('Accuracy')

    # set the label at each point
    for i, (xi, yi) in enumerate(zip(accuracy.keys(), accuracy.values())):
        plt.annotate(f'({xi:.0f}%, {yi})', (xi, yi), textcoords="offset points",
                     xytext=(0, 8), ha='center')
    
    # show the line chart
    plt.show()
