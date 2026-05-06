import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import random
import numpy as np
from model import QuotientTransformer, QuotientSpaceDataset, MAX_OBSTACLES

def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

def compute_class_weights(dataset, num_classes=5, device='cpu'):
    counts = np.zeros(num_classes, dtype=np.float64)
    for item in dataset.data:
        counts[int(item['infeasibility_link'])] += 1

    print("\n  Class distribution in full dataset:")
    for c, name in enumerate(['Feasible', 'Link1', 'Link2', 'Link3', 'Link4']):
        pct = 100.0 * counts[c] / counts.sum()
        print(f"    {name:<10}: {int(counts[c]):6d}  ({pct:.1f}%)")

    counts = np.where(counts == 0, 1, counts)
    weights = counts.sum() / (num_classes * counts)
    print(f"\n  Class weights: {np.round(weights, 3)}")
    return torch.tensor(weights, dtype=torch.float32).to(device)


def per_class_accuracy(all_targets, all_preds, num_classes=5):
    names   = ['Feasible', 'Link1', 'Link2', 'Link3', 'Link4']
    correct = np.zeros(num_classes)
    total   = np.zeros(num_classes)
    for t, p in zip(all_targets, all_preds):
        total[t]   += 1
        correct[t] += int(t == p)
    return {
        names[c]: (100.0 * correct[c] / total[c]) if total[c] > 0 else float('nan')
        for c in range(num_classes)
    }

def plot_training_results(history):
    epochs = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-',  linewidth=2, label='Train Loss')
    plt.plot(epochs, history['val_loss'],   'r--', linewidth=2, label='Val Loss')
    plt.title('Model Convergence: Loss', fontsize=14)
    plt.xlabel('Epochs'); plt.ylabel('Cross-Entropy Loss')
    plt.grid(True, linestyle='--', alpha=0.7); plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], 'g-',  linewidth=2, label='Train Accuracy')
    plt.plot(epochs, history['val_acc'],   'm--', linewidth=2, label='Val Accuracy')
    plt.title('Model Performance: Accuracy', fontsize=14)
    plt.xlabel('Epochs'); plt.ylabel('Accuracy (%)')
    plt.grid(True, linestyle='--', alpha=0.7); plt.legend()

    plt.tight_layout()
    plt.savefig('thesis_training_graphs.png', dpi=300)
    print("\nGraphs saved as 'thesis_training_graphs.png'")

def train_model():
    DATASET_FILE  = 'dataset.json'
    OUTPUT_FILE   = 'output.txt'
    WEIGHTS_BEST  = 'best_model_weights.pth'
    WEIGHTS_FINAL = 'final_model_weights.pth'
    SEED          = 42
    BATCH_SIZE    = 32
    EPOCHS        = 60
    LR            = 1e-3
    WEIGHT_DECAY  = 1e-4
    EARLY_STOP_PATIENCE = 10

    D_MODEL    = 64
    N_HEADS    = 4
    NUM_LAYERS = 2
    DROPOUT    = 0.3

    set_random_seed(SEED)

    if not os.path.exists(DATASET_FILE):
        print(f"Error: {DATASET_FILE} not found.")
        return

    dataset = QuotientSpaceDataset(DATASET_FILE, max_obstacles=MAX_OBSTACLES)
    if len(dataset) < 10:
        print("Error: Dataset too small.")
        return

    train_size = int(0.8 * len(dataset))
    val_size   = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED)
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              generator=torch.Generator().manual_seed(SEED))
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    model = QuotientTransformer(
        d_model=D_MODEL, n_heads=N_HEADS,
        num_layers=NUM_LAYERS, dropout=DROPOUT
    ).to(device)

    class_weights = compute_class_weights(dataset, device=device)
    criterion     = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR,
                                 weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc  = 0.0
    epochs_no_imp = 0

    with open(OUTPUT_FILE, 'w') as f:
        f.write("Quotient Transformer Training Log\n")
        f.write("=" * 60 + "\n")
        f.write(f"Seed            : {SEED}\n")
        f.write(f"Dataset file    : {DATASET_FILE}\n")
        f.write(f"Total samples   : {len(dataset)}\n")
        f.write(f"Train samples   : {len(train_ds)}\n")
        f.write(f"Val samples     : {len(val_ds)}\n")
        f.write(f"Max obstacles   : {MAX_OBSTACLES}\n")
        f.write(f"Batch size      : {BATCH_SIZE}\n")
        f.write(f"Epochs          : {EPOCHS}\n")
        f.write(f"LR              : {LR}\n")
        f.write(f"Class weights   : {class_weights.cpu().numpy().round(3).tolist()}\n")
        f.write(f"d_model         : {D_MODEL}\n")
        f.write(f"n_heads         : {N_HEADS}\n")
        f.write(f"num_layers      : {NUM_LAYERS}\n")
        f.write(f"dropout         : {DROPOUT}\n")
        f.write("=" * 60 + "\n")

    print("\nStarting Training...")
    print("=" * 60)

    for epoch in range(EPOCHS):

        # Train
        model.train()
        t_loss, t_correct, t_total = 0.0, 0, 0
        for batch in train_loader:
            state        = batch['state'].to(device)
            obstacles    = batch['obstacles'].to(device)
            padding_mask = batch['padding_mask'].to(device)
            target       = batch['target'].to(device)

            optimizer.zero_grad()
            logits = model(state, obstacles, padding_mask)
            loss   = criterion(logits, target)
            loss.backward()
            optimizer.step()

            t_loss    += loss.item()
            t_correct += (logits.argmax(1) == target).sum().item()
            t_total   += target.size(0)

        ep_train_loss = t_loss / len(train_loader)
        ep_train_acc  = 100.0 * t_correct / t_total

        # Validate
        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        all_t, all_p = [], []
        with torch.no_grad():
            for batch in val_loader:
                state        = batch['state'].to(device)
                obstacles    = batch['obstacles'].to(device)
                padding_mask = batch['padding_mask'].to(device)
                target       = batch['target'].to(device)

                logits = model(state, obstacles, padding_mask)
                loss   = criterion(logits, target)

                v_loss    += loss.item()
                preds      = logits.argmax(1)
                v_correct += (preds == target).sum().item()
                v_total   += target.size(0)
                all_t.extend(target.cpu().numpy())
                all_p.extend(preds.cpu().numpy())

        ep_val_loss = v_loss / len(val_loader)
        ep_val_acc  = 100.0 * v_correct / v_total

        scheduler.step(ep_val_loss)

        if ep_val_acc > best_val_acc:
            best_val_acc  = ep_val_acc
            epochs_no_imp = 0
            torch.save(model.state_dict(), WEIGHTS_BEST)
        else:
            epochs_no_imp += 1

        history['train_loss'].append(ep_train_loss)
        history['train_acc'].append(ep_train_acc)
        history['val_loss'].append(ep_val_loss)
        history['val_acc'].append(ep_val_acc)

        per_class   = per_class_accuracy(all_t, all_p)
        pc_str      = '  '.join(f"{k}:{v:.1f}%" for k, v in per_class.items())
        epoch_line  = (
            f"Epoch [{epoch+1:02d}/{EPOCHS}] | "
            f"Train Loss: {ep_train_loss:.4f}, Acc: {ep_train_acc:.1f}% | "
            f"Val Loss: {ep_val_loss:.4f}, Acc: {ep_val_acc:.1f}% | "
            f"LR: {optimizer.param_groups[0]['lr']:.2e}"
        )
        print(epoch_line)
        if (epoch + 1) % 10 == 0:
            print(f"  Per-class val → {pc_str}")

        with open(OUTPUT_FILE, 'a') as f:
            f.write(epoch_line + "\n")
            if (epoch + 1) % 10 == 0:
                f.write(f"  Per-class: {pc_str}\n")

        if epochs_no_imp >= EARLY_STOP_PATIENCE:
            msg = (f"\nEarly stopping at epoch {epoch+1} "
                   f"(no improvement for {EARLY_STOP_PATIENCE} epochs).")
            print(msg)
            with open(OUTPUT_FILE, 'a') as f:
                f.write(msg + "\n")
            break

    per_class = per_class_accuracy(all_t, all_p)
    pc_str    = '  '.join(f"{k}:{v:.1f}%" for k, v in per_class.items())
    print(f"\nFinal per-class val → {pc_str}")

    print("=" * 60)
    plot_training_results(history)
    torch.save(model.state_dict(), WEIGHTS_FINAL)

    summary = (
        f"\nBest val accuracy   : {best_val_acc:.2f}%\n"
        f"Best weights saved  : {WEIGHTS_BEST}\n"
        f"Final weights saved : {WEIGHTS_FINAL}\n"
        f"Final per-class acc : {pc_str}\n"
    )
    print(summary)
    with open(OUTPUT_FILE, 'a') as f:
        f.write("=" * 60 + "\n")
        f.write(summary)

    print(f"Training log saved to '{OUTPUT_FILE}'")

if __name__ == "__main__":
    train_model()