import matplotlib.pyplot as plt
import seaborn as sns


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


def plot_losses_over_epochs(train_losses, val_losses):
    pass


def plot_accuracies_over_epochs(train_acc, val_acc):
    pass
