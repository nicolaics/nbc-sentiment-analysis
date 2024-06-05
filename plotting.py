import matplotlib.pyplot as plt
import numpy as np

def plot(accuracy: dict):
    plt.figure(figsize=(12, 8))
    plt.plot(accuracy.keys(), accuracy.values(), marker='o', linestyle='-')

    plt.xlabel('Training Data Set (%)')
    plt.ylabel('Accuracy')

    for i, (xi, yi) in enumerate(zip(accuracy.keys(), accuracy.values())):
        plt.annotate(f'({xi:.0f}%, {yi})', (xi, yi), textcoords="offset points",
                     xytext=(0, 8), ha='center')
    plt.show()

if __name__ == "__main__":
    accuracy = {10: 0.2, 30: 0.4, 50: 0.5, 70: 0.6, 100: 0.8}
    plot(accuracy)
