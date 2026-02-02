import matplotlib.pyplot as plt
import numpy as np

# Set publication-quality parameters
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 0.8

# Data
models = ['M1\nBaseline', 'M2\nWide/Deep', 'M3\nLeakyReLU\n+AvgPool', 'M4\n5×5 Kernel', 'M5\nTanh+Stride']
train_acc = [91.43, 98.26, 83.08, 98.45, 93.17]
val_acc = [71.34, 77.18, 67.75, 77.25, 62.42]

# Calculate positions
x = np.arange(len(models))
width = 0.35

# Create figure with specific size for two-column format
fig, ax = plt.subplots(figsize=(7, 4))

# Create bars
bars1 = ax.bar(x - width/2, train_acc, width, label='Training Accuracy', 
               color='#2E86AB', edgecolor='black', linewidth=0.7)
bars2 = ax.bar(x + width/2, val_acc, width, label='Validation Accuracy', 
               color='#A23B72', edgecolor='black', linewidth=0.7)

# Customize plot
ax.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
ax.set_xlabel('Model Architecture', fontsize=11, fontweight='bold')
ax.set_title('CNN Architecture Performance Comparison', fontsize=12, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=9)
ax.legend(loc='upper left', frameon=True, edgecolor='black', fontsize=9)

# Add grid for readability
ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
ax.set_axisbelow(True)

# Set y-axis limits with more space (no legend overlap)
ax.set_ylim([55, 111])

# Add value labels on bars
def autolabel(bars):
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}',
                ha='center', va='bottom', fontsize=7, fontweight='bold')

autolabel(bars1)
autolabel(bars2)

# Tight layout
plt.tight_layout()

# Save figure
plt.savefig('task1_architecture_comparison.pdf', dpi=300, bbox_inches='tight')
plt.savefig('task1_architecture_comparison.png', dpi=300, bbox_inches='tight')

plt.show()

print("Figure saved as 'task1_architecture_comparison.pdf' and '.png'")