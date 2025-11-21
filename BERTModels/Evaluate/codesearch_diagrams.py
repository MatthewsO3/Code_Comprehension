import json
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

# Stílus beállítása
plt.style.use('seaborn-v0_8-darkgrid')

# Színek definiálása (hasonlóan az előzőhöz)
COLORS = {
    'total': '#06b6d4',  # Cyan (Total Loss)
    'ce': '#8b5cf6',  # Violet (Cross Entropy)
    'neg': '#ec4899',  # Pink (Negative Loss)
    'lr': '#fbbf24',  # Amber (Learning Rate)
    'batch_raw': '#93c5fd',  # Halvány kék (Batch raw)
    'batch_avg': '#1d4ed8'  # Sötét kék (Batch avg)
}


def load_history(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)


def plot_comprehensive_metrics(history_file, output_file='codesearch_comprehensive.png'):
    """
    Generál egy összefoglaló ábrát (All-in-One) a CodeSearch training history-ból.
    Mivel nincs validációs adat, a komponenseket (CE vs Neg) hasonlítja össze.
    """
    history = load_history(history_file)
    epochs = history['epoch']

    # Adatok előkészítése
    train_loss = history['train_total_loss']
    ce_loss = history['train_ce_loss']
    neg_loss = history['train_neg_loss']
    lr = history['learning_rate']

    # Batch adatok és mozgóátlag
    batch_losses = history['train_batch_losses']
    window = 50  # Simításhoz
    batch_series = pd.Series(batch_losses)
    moving_avg = batch_series.rolling(window=window).mean()

    # Batch komponensek mozgóátlaga
    ce_batch_avg = pd.Series(history['train_ce_batch_losses']).rolling(window=window).mean()
    neg_batch_avg = pd.Series(history['train_neg_batch_losses']).rolling(window=window).mean()

    # --- ÁBRA LÉTREHOZÁSA ---
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1], hspace=0.4, wspace=0.25)

    fig.suptitle('CodeSearch Training: Comprehensive Metrics', fontsize=20, fontweight='bold', y=0.95)

    # 1. TOTAL LOSS PER EPOCH (Teljes szélességben fent)
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(epochs, train_loss, marker='o', linewidth=3, markersize=10, label='Total Train Loss',
             color=COLORS['total'])

    # Értékek kiírása a pontok fölé
    for e, loss in zip(epochs, train_loss):
        ax1.annotate(f'{loss:.3f}', (e, loss), textcoords="offset points", xytext=(0, 10),
                     ha='center', fontsize=10, fontweight='bold', color=COLORS['total'])

    ax1.set_title('Total Training Loss per Epoch', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Loss', fontweight='bold')
    ax1.set_xlabel('Epoch', fontweight='bold')
    ax1.set_xticks(epochs)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # 2. LOSS COMPONENTS BREAKDOWN (Train vs Val helyett CE vs Neg)
    ax2 = fig.add_subplot(gs[1, 0])

    # Vonalas ábrázolás a komponensekhez
    ax2.plot(epochs, ce_loss, marker='s', linewidth=2.5, label='CE Loss (Doc-Code)', color=COLORS['ce'])
    ax2.plot(epochs, neg_loss, marker='^', linewidth=2.5, label='Negative Loss (Contrastive)', color=COLORS['neg'])

    ax2.set_title('Loss Components: Cross-Entropy vs Negative', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontweight='bold')
    ax2.set_ylabel('Loss', fontweight='bold')
    ax2.set_xticks(epochs)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # 3. LEARNING RATE SCHEDULE
    ax3 = fig.add_subplot(gs[1, 1])
    # Skálázás, hogy olvashatóbb legyen (pl. 1e-5 egységben)
    lr_scaled = [l * 1e5 for l in lr]

    ax3.plot(epochs, lr_scaled, marker='o', linewidth=2.5, markersize=8, color=COLORS['lr'], label='LR')
    ax3.fill_between(epochs, lr_scaled, alpha=0.1, color=COLORS['lr'])

    for e, l in zip(epochs, lr_scaled):
        ax3.annotate(f'{l:.1f}', (e, l), textcoords="offset points", xytext=(0, 8), ha='center', fontsize=9)

    ax3.set_title('Learning Rate Schedule (x1e-5)', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Epoch', fontweight='bold')
    ax3.set_ylabel('LR', fontweight='bold')
    ax3.set_xticks(epochs)
    ax3.grid(True, alpha=0.3)

    # 4. TRAINING BATCH DYNAMICS (Total)
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.plot(batch_losses, color=COLORS['total'], alpha=0.15, label='Raw Batch Loss', linewidth=0.5)
    ax4.plot(moving_avg, color=COLORS['batch_avg'], linewidth=2, label=f'Moving Avg ({window})')

    ax4.set_title('Training Batch Dynamics (Total Loss)', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Global Step', fontweight='bold')
    ax4.set_ylabel('Loss', fontweight='bold')
    ax4.legend(loc='upper right', fontsize=9)
    ax4.grid(True, alpha=0.3)
    # Limitáljuk az Y tengelyt, hogy a kiugró tüskék ne torzítsák el az ábrát
    ax4.set_ylim(0, max(moving_avg.dropna()) * 2.5)

    # 5. BATCH COMPONENTS DYNAMICS (CE vs Neg - Moving Average only)
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.plot(ce_batch_avg, color=COLORS['ce'], linewidth=2, label='CE Moving Avg', alpha=0.9)
    ax5.plot(neg_batch_avg, color=COLORS['neg'], linewidth=2, label='Neg Moving Avg', alpha=0.9)

    ax5.set_title('Batch Dynamics: Components Breakdown', fontsize=14, fontweight='bold')
    ax5.set_xlabel('Global Step', fontweight='bold')
    ax5.legend(loc='upper right', fontsize=9)
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim(0, max(ce_batch_avg.dropna()) * 1.5)

    # Mentés
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Hagyunk helyet a főcímnek
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Comprehensive diagram saved to: {output_file}")
    plt.show()


if __name__ == "__main__":
    # Fájl neve (feltételezve, hogy ugyanott van a script, mint a json)
    json_path = '/Users/czapmate/Desktop/szakdoga/GraphCodeBert_CPP/BERTModels/GraphCodeBert/CodeSearch/graphcodebert-cpp-codesearch/training_history.json'

    try:
        plot_comprehensive_metrics(json_path)
    except FileNotFoundError:
        print(f"Error: {json_path} not found. Please define the correct path.")