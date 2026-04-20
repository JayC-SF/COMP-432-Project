import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter
import numpy as np
import src.variables as v


def plot_class_inbalance(y, dataset_type=""):
    plt.figure(figsize=(8, 5))

    sns.countplot(x=y, hue=y, palette='viridis', legend=False)

    plt.title(f"{dataset_type + ' ' if dataset_type is not None else ''}Class Distribution: No-Cry vs. Cry")
    plt.xlabel('Class (0: No-Cry, 1: Cry)')
    plt.ylabel('Number of Samples')

    # This adds the counts on top of the bars
    for i, count in enumerate([sum(y == 0), sum(y == 1)]):
        plt.text(i, count + 100, str(int(count)), ha='center', fontweight='bold')

    plt.show()


def plot_confusion_matrix(cm, model_display_name):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=v.CLASSES,
                yticklabels=v.CLASSES)

    # 3. Add labels for clarity
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.title(f"{model_display_name}: Confusion Matrix")
    plt.show()


def plot_normalized_confusion_matrix(conf):
    pass


def plot_losses_over_epochs(losses, legend, title):
    epoch_len = -1
    for loss in losses:
        epoch_len = max(epoch_len, len(loss))

    epochs = range(1, epoch_len + 1)
    for loss in losses:
        plt.plot(epochs, loss)

    plt.title(title)
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.legend(legend)
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


def plot_durations(durations, title):
    plt.figure(figsize=(10, 6))
    plt.hist(durations, bins=30, color='skyblue', edgecolor='black')
    plt.title(title)
    plt.xlabel("Durations (s)")
    plt.ylabel("Counts")
    plt.grid(axis='y', alpha=0.75)
    plt.show()


def plot_orchestrator_training_summary(orchestrator, model_display_name):

    plot_losses_over_epochs(
        [orchestrator.th.train_loss, orchestrator.th.val_loss],
        ["train loss", "validation loss"],
        f"{model_display_name}: Training and Validation Losses over Epochs"
    )

    best_epoch = np.argmin(np.array(orchestrator.th.val_loss))+1
    print(f"Best epoch's validation loss ({min(orchestrator.th.val_loss)}) achieved at epoch {best_epoch}")
    print(f"Best epoch's training loss ({orchestrator.th.train_loss[best_epoch-1]})")
    print(f"Total epochs ({len(orchestrator.th.val_loss)})")

    plot_losses_over_epochs(
        [orchestrator.th.train_acc, orchestrator.th.val_acc],
        ["train accuracy", "validation accuracy"],
        f"{model_display_name}: Training and Validation Accuracies over Epochs"
    )

    print(f"Best epoch's validation accuracy: {orchestrator.th.val_acc[best_epoch-1]:%}")
    print(f"Best epoch's training accuracy: {orchestrator.th.train_acc[best_epoch-1]:%}")


def plot_test_results(results, model_display_name):
    print(results['classification_report'])
    plot_confusion_matrix(results['confusion_matrix'], model_display_name)
    print(f"Test Loss: {results['test_loss']:.4f} | Test Acc: {results['test_acc']:.4%}")
