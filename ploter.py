import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.calibration import calibration_curve
from scipy.stats import entropy
import seaborn as sns
import torch


# 1. DISPERSION (VARIANCE)
def plot_dispersion_over_epochs(history):
    dispersions = [np.var(h["mini_proba"], axis=0) for h in history]
    dispersions = np.array(dispersions)

    plt.figure(figsize=(10, 6))
    for i in range(6):
        plt.plot(range(1, len(history)+1), dispersions[:, i], marker="o", label=f"Sample {i}")
    plt.title("MC Dropout Variance per Sample over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Variance")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# 2. ENTROPY
def plot_entropy_over_epochs(history):
    entropies = []
    for h in history:
        p = np.clip(h["mini_proba"], 1e-6, 1 - 1e-6)
        ent = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
        mean_ent = np.mean(ent, axis=0)
        entropies.append(mean_ent)
    entropies = np.array(entropies)

    plt.figure(figsize=(10, 6))
    for i in range(6):
        plt.plot(range(1, len(history)+1), entropies[:, i], marker="o", label=f"Sample {i}")
    plt.title("Entropy per Sample over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Entropy (bits)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# 3. MARGIN |p - 0.5|
def plot_margin_over_epochs(history):
    margins = []
    for h in history:
        probs = np.mean(h["mini_proba"], axis=0)
        margin = np.abs(probs - 0.5)
        margins.append(margin)
    margins = np.array(margins)

    plt.figure(figsize=(10, 6))
    for i in range(6):
        plt.plot(range(1, len(history)+1), margins[:, i], marker="o", label=f"Sample {i}")
    plt.title("Prediction Margin per Sample over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("|p - 0.5|")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# 4. VIOLIN KDE
def plot_violin_kde(history, epoch=-1):
    probs = history[epoch]["mini_proba"]
    fig, axs = plt.subplots(2, 3, figsize=(18, 8))
    for i in range(6):
        ax = axs[i // 3, i % 3]
        sns.violinplot(y=probs[:, i], ax=ax, inner="point", cut=0)
        ax.set_title(f"Sample {i}")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Probability")
    fig.suptitle(f"Violin Plots of Mini-Test Probs (Epoch {history[epoch]['epoch']})", fontsize=16)
    plt.tight_layout()
    plt.show()

# 5. CALIBRATION
def plot_reliability_diagram(model, val_loader, device="cpu"):
    model.eval()
    all_probs, all_targets = [], []
    with torch.no_grad():
        for X, y in val_loader:
            X = X.to(device)
            probs = torch.sigmoid(model(X)).squeeze().cpu().numpy()
            all_probs.append(probs)
            all_targets.append(y.numpy())
    probs = np.concatenate(all_probs)
    targets = np.concatenate(all_targets)
    prob_true, prob_pred = calibration_curve(targets, probs, n_bins=10)

    plt.figure(figsize=(6, 6))
    plt.plot(prob_pred, prob_true, marker='o', label="Model")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.title("Reliability Diagram")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Empirical Frequency")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# 6. ROC
def plot_roc_curve(model, val_loader, device="cpu"):
    model.eval()
    all_probs, all_targets = [], []
    with torch.no_grad():
        for X, y in val_loader:
            X = X.to(device)
            probs = torch.sigmoid(model(X)).squeeze().cpu().numpy()
            all_probs.append(probs)
            all_targets.append(y.numpy())
    probs = np.concatenate(all_probs)
    targets = np.concatenate(all_targets)
    fpr, tpr, _ = roc_curve(targets, probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, marker="o", label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# 7. ACCURACY + F1
def plot_acc_f1_over_epochs(history):
    epochs = [h["epoch"] for h in history]
    val_acc = [h["val_acc"] for h in history]
    val_f1 = [h["val_f1"] for h in history]
    mc_acc = [h.get("mc_accuracy_mean") for h in history]
    mc_f1 = [h.get("mc_f1_mean") for h in history]
    mc_acc_low = [h.get("mc_accuracy_ci_low") for h in history]
    mc_acc_high = [h.get("mc_accuracy_ci_high") for h in history]
    mc_f1_low = [h.get("mc_f1_ci_low") for h in history]
    mc_f1_high = [h.get("mc_f1_ci_high") for h in history]

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, val_acc, marker="o", label="Val Accuracy")
    plt.plot(epochs, val_f1, marker="s", label="Val F1")
    if all(x is not None for x in mc_acc):
        plt.plot(epochs, mc_acc, marker="^", label="MC Accuracy")
        plt.fill_between(epochs, mc_acc_low, mc_acc_high, alpha=0.2, label="MC Acc CI")
    if all(x is not None for x in mc_f1):
        plt.plot(epochs, mc_f1, marker="v", label="MC F1")
        plt.fill_between(epochs, mc_f1_low, mc_f1_high, alpha=0.2, label="MC F1 CI")
    plt.title("Accuracy & F1 over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# 8. PROBABILITY HISTOGRAMS
def plot_mini_proba_distributions(history, max_epochs=None):
    num_samples = 6
    sample_indices = list(range(num_samples))
    epochs_to_plot = history if max_epochs is None else history[:max_epochs]

    n_epochs = len(epochs_to_plot)

    fig, axs = plt.subplots(2, 3, figsize=(18, 8), sharey=True)

    for i, idx in enumerate(sample_indices):
        ax = axs[i // 3, i % 3]
        for epoch_data in epochs_to_plot:
            probs = epoch_data["mini_proba"][:, idx]
            ax.hist(
                probs, bins=50, alpha=0.5,
                label=f"Epoch {epoch_data['epoch']}",
                density=True, histtype='step'
            )
        true_label = epochs_to_plot[0]["mini_labels"][idx]
        ax.set_title(f"Sample {idx} (Label={true_label})")
        ax.set_xlim(0, 1)
        ax.set_xlabel("Probability")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)

    plt.suptitle("MC Dropout Probability Distributions (6 mini-test samples)", fontsize=16)
    plt.tight_layout()
    plt.show()


def visualize_all(history, model=None, val_loader=None, device="cpu", epoch_for_violin=-1):
    fig = plt.figure(figsize=(22, 14))

    # Subplot 1: Accuracy & F1
    ax1 = fig.add_subplot(3, 2, 1)
    epochs = [h["epoch"] for h in history]
    val_acc = [h["val_acc"] for h in history]
    val_f1 = [h["val_f1"] for h in history]
    mc_acc = [h.get("mc_accuracy_mean") for h in history]
    mc_f1 = [h.get("mc_f1_mean") for h in history]
    mc_acc_low = [h.get("mc_accuracy_ci_low") for h in history]
    mc_acc_high = [h.get("mc_accuracy_ci_high") for h in history]
    mc_f1_low = [h.get("mc_f1_ci_low") for h in history]
    mc_f1_high = [h.get("mc_f1_ci_high") for h in history]
    ax1.plot(epochs, val_acc, marker="o", label="Val Accuracy")
    ax1.plot(epochs, val_f1, marker="s", label="Val F1")
    if all(x is not None for x in mc_acc):
        ax1.plot(epochs, mc_acc, marker="^", label="MC Accuracy")
        ax1.fill_between(epochs, mc_acc_low, mc_acc_high, alpha=0.2)
    if all(x is not None for x in mc_f1):
        ax1.plot(epochs, mc_f1, marker="v", label="MC F1")
        ax1.fill_between(epochs, mc_f1_low, mc_f1_high, alpha=0.2)
    ax1.set_title("Accuracy & F1 over Epochs")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Score")
    ax1.set_ylim(0, 1)
    ax1.grid(True)
    ax1.legend()

    # Subplot 2: Dispersion
    ax2 = fig.add_subplot(3, 2, 2)
    dispersions = [np.var(h["mini_proba"], axis=0) for h in history]
    dispersions = np.array(dispersions)
    for i in range(6):
        ax2.plot(epochs, dispersions[:, i], marker="o", label=f"Sample {i}")
    ax2.set_title("Variance per Sample")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Variance")
    ax2.legend(fontsize=8)
    ax2.grid(True)

    # Subplot 3: Entropy
    ax3 = fig.add_subplot(3, 2, 3)
    entropies = []
    for h in history:
        p = np.clip(h["mini_proba"], 1e-6, 1 - 1e-6)
        ent = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
        mean_ent = np.mean(ent, axis=0)
        entropies.append(mean_ent)
    entropies = np.array(entropies)
    for i in range(6):
        ax3.plot(epochs, entropies[:, i], marker="o", label=f"Sample {i}")
    ax3.set_title("Entropy per Sample")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Entropy (bits)")
    ax3.legend(fontsize=8)
    ax3.grid(True)

    # Subplot 4: Margin
    ax4 = fig.add_subplot(3, 2, 4)
    margins = [np.abs(np.mean(h["mini_proba"], axis=0) - 0.5) for h in history]
    margins = np.array(margins)
    for i in range(6):
        ax4.plot(epochs, margins[:, i], marker="o", label=f"Sample {i}")
    ax4.set_title("Margin |p - 0.5| per Sample")
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("Margin")
    ax4.legend(fontsize=8)
    ax4.grid(True)

    # Subplot 5: Violin plots
    ax5 = fig.add_subplot(3, 2, 5)
    sns.violinplot(data=history[epoch_for_violin]["mini_proba"], inner="point", ax=ax5)
    ax5.set_ylim(0, 1)
    ax5.set_title(f"Violin plot of mini-test probs (Epoch {history[epoch_for_violin]['epoch']})")
    ax5.set_ylabel("Probability")

    plt.tight_layout()
    plt.show()

    # Separated histogram grid
    fig2, axs = plt.subplots(2, 3, figsize=(18, 8), sharey=True)
    epochs_to_plot = history
    for i, ax in enumerate(axs.flat):
        for h in epochs_to_plot:
            ax.hist(
                h["mini_proba"][:, i], bins=30, density=True,
                alpha=0.3, label=f"Epoch {h['epoch']}", histtype='step'
            )
        ax.set_xlim(0, 1)
        ax.set_title(f"Sample {i} (Label={history[0]['mini_labels'][i]})")
        ax.set_xlabel("Probability")
        ax.set_ylabel("Density")
        ax.legend(fontsize=6)

    fig2.suptitle("MC Dropout Probability Distributions (6 mini-test samples)", fontsize=16)
    plt.tight_layout()
    plt.show()
