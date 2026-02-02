import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Set publication quality
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 9

# Create figure
fig, ax = plt.subplots(figsize=(8, 4))
ax.set_xlim(0, 16)
ax.set_ylim(0, 6)
ax.axis('off')

# Define colors
color_input = '#E8F4F8'
color_conv = '#4A90E2'
color_pool = '#E74C3C'
color_fc = '#2ECC71'
color_output = '#F39C12'

# Layer positions and sizes
layers = [
    # (x, y, width, height, color, label, details)
    (0.5, 2, 0.8, 2, color_input, 'Input\n32×32×3', ''),
    
    # Block 1: Conv-Conv-Pool
    (2, 2.5, 0.6, 1.5, color_conv, 'Conv1\n64', '3×3, s=1\nReLU'),
    (2.8, 2.5, 0.6, 1.5, color_conv, 'Conv2\n64', '3×3, s=1\nReLU'),
    (3.7, 2.7, 0.4, 1.1, color_pool, 'MaxPool', '2×2, s=2'),
    
    # Block 2: Conv-Conv-Pool
    (4.5, 2.8, 0.6, 1.3, color_conv, 'Conv3\n128', '3×3, s=1\nReLU'),
    (5.3, 2.8, 0.6, 1.3, color_conv, 'Conv4\n128', '3×3, s=1\nReLU'),
    (6.2, 3.0, 0.4, 0.9, color_pool, 'MaxPool', '2×2, s=2'),
    
    # Block 3: Conv-Conv-Pool
    (7.0, 3.1, 0.6, 1.1, color_conv, 'Conv5\n256', '3×3, s=1\nReLU'),
    (7.8, 3.1, 0.6, 1.1, color_conv, 'Conv6\n256', '3×3, s=1\nReLU'),
    (8.7, 3.3, 0.4, 0.7, color_pool, 'MaxPool', '2×2, s=2'),
    
    # Block 4: Conv-Pool
    (9.5, 3.4, 0.6, 0.9, color_conv, 'Conv7\n512', '3×3, s=1\nReLU'),
    (10.3, 3.5, 0.4, 0.6, color_pool, 'MaxPool', '2×2, s=2'),
    
    # Flatten + FC layers
    (11.2, 3.2, 0.3, 1.2, color_fc, 'Flatten', '2048'),
    (11.8, 3.3, 0.5, 1.0, color_fc, 'FC1\n512', 'ReLU'),
    (12.6, 3.5, 0.5, 0.6, color_fc, 'FC2\n10', 'Softmax'),
    
    # Output
    (13.4, 3.6, 0.5, 0.4, color_output, 'Output\n10 classes', ''),
]

# Draw layers
for i, (x, y, w, h, color, label, details) in enumerate(layers):
    # Main box
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05", 
                         edgecolor='black', facecolor=color, linewidth=1.5)
    ax.add_patch(box)
    
    # Layer name
    ax.text(x + w/2, y + h/2 + 0.15, label, 
            ha='center', va='center', fontsize=8, fontweight='bold')
    
    # Details
    if details:
        ax.text(x + w/2, y + h/2 - 0.2, details, 
                ha='center', va='center', fontsize=6.5, style='italic')

# Draw arrows between layers
arrow_props = dict(arrowstyle='->', lw=1.5, color='black')
for i in range(len(layers) - 1):
    x1, y1, w1, h1 = layers[i][:4]
    x2, y2, w2, h2 = layers[i+1][:4]
    
    # Connect center-right of current to center-left of next
    start_x = x1 + w1
    start_y = y1 + h1/2
    end_x = x2
    end_y = y2 + h2/2
    
    arrow = FancyArrowPatch((start_x, start_y), (end_x, end_y),
                           **arrow_props)
    ax.add_patch(arrow)

# Add dimension labels below key stages
dimension_labels = [
    (1.3, 1.5, '32×32×3'),
    (3.2, 1.8, '16×16×64'),
    (5.9, 2.1, '8×8×128'),
    (8.2, 2.4, '4×4×256'),
    (9.9, 2.7, '2×2×512'),
]

for x, y, text in dimension_labels:
    ax.text(x, y, text, ha='center', va='top', fontsize=7, 
            color='darkblue', style='italic', fontweight='bold')

# Add block labels
block_labels = [
    (2.4, 4.3, 'Block 1', color_conv),
    (5.5, 4.5, 'Block 2', color_conv),
    (7.9, 4.6, 'Block 3', color_conv),
    (9.8, 4.7, 'Block 4', color_conv),
    (12.2, 4.8, 'Classifier', color_fc),
]

for x, y, text, color in block_labels:
    ax.text(x, y, text, ha='center', va='bottom', fontsize=8, 
            fontweight='bold', color=color,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                     edgecolor=color, linewidth=1.2))

# Add title
ax.text(7.5, 5.5, 'Model 2 Architecture: VGG-Style CNN', 
        ha='center', va='center', fontsize=12, fontweight='bold')

# Add parameter count
ax.text(7.5, 0.3, 'Total Parameters: 3.38M | Input: CIFAR-10 (32×32×3) | Output: 10 classes', 
        ha='center', va='center', fontsize=8, style='italic')

# Add legend
legend_elements = [
    patches.Patch(facecolor=color_conv, edgecolor='black', label='Convolution'),
    patches.Patch(facecolor=color_pool, edgecolor='black', label='Max Pooling'),
    patches.Patch(facecolor=color_fc, edgecolor='black', label='Fully Connected'),
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=7, frameon=True)

plt.tight_layout()
plt.savefig('model2_architecture.pdf', dpi=300, bbox_inches='tight')
plt.savefig('model2_architecture.png', dpi=300, bbox_inches='tight')
plt.show()

print("Architecture diagram saved!")