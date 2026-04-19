import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter

def plot_class_inbalance(y):
    plt.figure(figsize=(8, 5))

    sns.countplot(x=y, hue=y, palette='viridis', legend=False)

    plt.title('Class Distribution: No-Cry vs. Cry')
    plt.xlabel('Class (0: No-Cry, 1: Cry)')
    plt.ylabel('Number of Samples')

    # This adds the counts on top of the bars
    for i, count in enumerate([sum(y == 0), sum(y == 1)]):
        plt.text(i, count + 100, str(int(count)), ha='center', fontweight='bold')

    plt.show()


def plot_confusion_matrix(conf):
    pass


def plot_normalized_confusion_matrix(conf):
    pass


def plot_losses_over_epochs(train_losses, val_losses, title):
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses)
    plt.plot(epochs, val_losses)
    plt.title(title)
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.legend(["train loss", "validation loss"])
    plt.show()


def plot_accuracies_over_epochs(train_acc, val_acc, title):
    epochs = range(1, len(train_acc) + 1)
    plt.plot(epochs, train_acc)
    plt.plot(epochs, val_acc)
    plt.title(title)
    plt.ylabel("Accuracy (%)")
    plt.xlabel("Epochs")
    plt.legend(["train accuracy", "validation accuracy"])
    plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=1))
    plt.show()
