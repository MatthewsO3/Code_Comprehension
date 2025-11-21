import json
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import numpy as np
import argparse
from pathlib import Path

# Stílus beállítása
plt.style.use('seaborn-v0_8-darkgrid')

COLORS = {
    'train_total': '#06b6d4',  # Cyan
    'val_total': '#ef4444',  # Red
    'train_ce': '#8b5cf6',  # Violet
    'val_ce': '#c4b5fd',  # Light Violet
    'train_neg': '#ec4899',  # Pink
    'val_neg': '#fbcfe8',  # Light Pink
    'lr': '#fbbf24',  # Amber
    'batch': '#93c5fd'  # Light Blue
}


def load_history(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)


def plot_validation_analysis(history_file, output_file='training_analysis.png'):
    try:
        history = load_history(history_file)
    except FileNotFoundError:
        print(f"Hiba: A fájl nem található: {history_file}")
        return

    epochs = history['epoch']

    # Ellenőrizzük, hogy van-e validációs adat a JSON-ben
    has_val = 'val_total_loss' in history

    # Ábra elrendezés létrehozása
    fig = plt.figure(figsize=(16, 14))
    gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1], hspace=0.35, wspace=0.2)

    fig.suptitle('CodeSearch Finomhangolás Elemzés', fontsize=20, fontweight='bold', y=0.96)

    # --- 1. TOTAL LOSS (Train vs Val) ---
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(epochs, history['train_total_loss'], marker='o', linewidth=3,
             label='Train Total Loss', color=COLORS['train_total'])

    if has_val:
        ax1.plot(epochs, history['val_total_loss'], marker='s', linewidth=3,
                 label='Validation Total Loss', color=COLORS['val_total'], linestyle='--')
        # Annotációk a validációs loss-hoz
        for e, v_loss in zip(epochs, history['val_total_loss']):
            ax1.annotate(f'{v_loss:.3f}', (e, v_loss), textcoords="offset points",
                         xytext=(0, 10), ha='center', fontsize=9, fontweight='bold', color=COLORS['val_total'])

    ax1.set_title('Total Loss Alakulása (Overfitting Ellenőrzés)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Veszteség (Loss)', fontweight='bold')
    ax1.set_xlabel('Epoch', fontweight='bold')
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)

    # --- 2. COMPONENT: CROSS-ENTROPY ---
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(epochs, history['train_ce_loss'], marker='o', label='Train CE', color=COLORS['train_ce'])
    if has_val:
        ax2.plot(epochs, history['val_ce_loss'], marker='s', label='Val CE', color=COLORS['val_ce'], linestyle='--')
    ax2.set_title('Cross-Entropy Loss (Pozitív Párok)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # --- 3. COMPONENT: NEGATIVE LOSS ---
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(epochs, history['train_neg_loss'], marker='o', label='Train Neg', color=COLORS['train_neg'])
    if has_val:
        ax3.plot(epochs, history['val_neg_loss'], marker='s', label='Val Neg', color=COLORS['val_neg'], linestyle='--')
    ax3.set_title('Negative Loss (Kontrasztív)', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Epoch', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # --- 4. BATCH DYNAMICS ---
    ax4 = fig.add_subplot(gs[2, 0])
    batch_losses = history.get('train_batch_losses', [])
    if batch_losses:
        ax4.plot(batch_losses, color=COLORS['batch'], alpha=0.3, label='Nyers Batch Loss')
        window = min(50, len(batch_losses) // 10) if len(batch_losses) > 10 else 1
        if window > 0:
            ma = pd.Series(batch_losses).rolling(window=window).mean()
            ax4.plot(ma, color='#2563eb', linewidth=2, label=f'Mozgóátlag ({window})')

        ax4.set_ylim(0, max(pd.Series(batch_losses).dropna().quantile(0.95), 0.1) * 1.5)  # Outlierek levágása

    ax4.set_title('Tréning Stabilitás (Batch Loss)', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Lépés (Step)', fontweight='bold')
    ax4.set_ylabel('Loss', fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # --- 5. LEARNING RATE ---
    ax5 = fig.add_subplot(gs[2, 1])
    lrs = [l * 1e5 for l in history.get('learning_rate', [])]
    if lrs:
        ax5.plot(epochs, lrs, marker='o', color=COLORS['lr'], linewidth=2)
        ax5.fill_between(epochs, lrs, alpha=0.1, color=COLORS['lr'])
    ax5.set_title('Tanulási Ráta (x1e-5)', fontsize=14, fontweight='bold')
    ax5.set_xlabel('Epoch', fontweight='bold')
    ax5.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Diagram elmentve: {output_file}")
    plt.show()


if __name__ == "__main__":
    # Itt add meg a JSON fájl nevét
    plot_validation_analysis('/Users/czapmate/Desktop/szakdoga/GraphCodeBert_CPP/BERTModels/GraphCodeBert/CodeSearch/graphcodebert-cpp-codesearch/training_history.json')