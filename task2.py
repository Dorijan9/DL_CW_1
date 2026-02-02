import matplotlib.pyplot as plt
import numpy as np

# Set publication quality
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 9
plt.rcParams['axes.linewidth'] = 0.8

# Data for all 4 architectures
epochs = np.arange(1, 101)

# Architecture 1: No regularization
arch1_train = [93.61, 71.39, 72.70, 72.00, 72.50, 71.97, 71.24, 68.50, 68.92, 66.92,
               62.98, 60.43, 64.15, 59.43, 73.23, 70.81, 68.10, 62.32, 60.54, 61.39,
               60.13, 56.26, 58.29, 58.18, 58.04, 55.50, 52.90, 52.90, 54.46, 54.29,
               50.71, 51.30, 49.06, 47.25, 47.07, 45.14, 43.67, 41.86, 42.61, 41.20,
               41.69, 40.88, 39.78, 37.77, 35.35, 38.37, 33.03, 37.26, 32.05, 35.31,
               35.35, 31.90, 32.22, 30.94, 31.31, 29.36, 29.66, 27.36, 27.36, 26.43,
               25.47, 24.40, 24.66, 22.08, 20.52, 19.98, 20.27, 22.15, 19.51, 19.64,
               20.00, 18.05, 18.54, 18.50, 19.02, 17.31, 17.21, 17.48, 16.21, 16.61,
               16.84, 15.93, 14.77, 14.62, 16.35, 15.17, 16.35, 17.67, 17.42, 21.39,
               20.65, 16.40, 14.11, 14.39, 13.60, 12.17, 14.12, 14.39, 13.24, 15.65]

arch1_val = [68.87, 64.86, 66.22, 68.32, 66.32, 70.02, 64.89, 63.59, 63.12, 60.72,
             58.44, 66.58, 53.44, 68.87, 65.43, 64.65, 51.30, 52.84, 49.80, 64.55,
             51.18, 52.96, 52.49, 57.25, 51.93, 52.57, 51.40, 55.26, 54.55, 53.69,
             53.52, 57.54, 56.44, 53.17, 58.16, 55.54, 55.23, 64.91, 62.79, 60.70,
             59.52, 55.97, 58.16, 58.71, 56.17, 54.67, 61.76, 57.03, 65.83, 61.13,
             59.50, 57.67, 58.70, 73.63, 59.84, 68.31, 61.33, 59.08, 65.20, 62.98,
             51.61, 61.97, 61.22, 67.43, 61.91, 70.02, 64.97, 68.21, 63.66, 76.24,
             66.62, 61.76, 69.02, 65.24, 70.04, 73.87, 75.14, 70.81, 60.07, 76.52,
             79.50, 72.46, 74.26, 76.63, 67.57, 86.02, 66.70, 77.32, 80.98, 63.40,
             64.00, 79.11, 71.27, 70.42, 82.39, 67.93, 69.87, 81.38, 82.14, 82.11]

# Architecture 2: BatchNorm + Moderate Dropout
arch2_train = [1144.50, 235.73, 175.31, 156.54, 94.75, 117.22, 76.56, 80.27, 79.77, 78.59,
               81.67, 74.21, 71.14, 64.86, 73.97, 69.44, 63.68, 62.19, 59.56, 63.52,
               59.48, 57.62, 61.78, 57.57, 55.66, 54.66, 56.98, 51.04, 53.98, 56.08,
               57.69, 52.93, 50.23, 50.41, 51.81, 52.58, 51.03, 49.20, 49.42, 48.56,
               48.17, 54.11, 54.43, 49.79, 48.21, 45.71, 46.52, 55.20, 55.62, 50.75,
               50.24, 47.33, 48.06, 49.30, 45.23, 48.50, 50.64, 48.15, 45.05, 46.94,
               39.35, 40.73, 42.64, 43.46, 48.18, 43.61, 41.51, 40.28, 38.71, 39.72,
               41.81, 37.91, 39.09, 44.46, 43.85, 40.30, 43.42, 43.71, 44.70, 43.22,
               40.49, 44.96, 38.08, 40.41, 38.34, 39.57, 36.51, 38.56, 34.64, 35.13,
               35.15, 38.02, 37.19, 36.79, 37.35, 35.04, 38.20, 33.43, 33.29, 34.87]

arch2_val = [123.57, 100.69, 70.11, 64.58, 61.40, 64.88, 56.19, 53.83, 61.43, 55.65,
             82.05, 64.08, 54.65, 56.52, 58.54, 77.80, 59.78, 58.33, 66.83, 64.83,
             64.86, 62.42, 58.78, 56.32, 54.65, 59.01, 56.76, 56.03, 58.06, 62.87,
             59.51, 55.15, 51.24, 51.78, 62.77, 52.74, 54.25, 53.27, 55.87, 54.86,
             61.45, 53.34, 54.04, 51.93, 56.03, 57.15, 57.56, 53.14, 54.96, 59.61,
             55.85, 57.19, 58.73, 53.22, 53.97, 54.88, 64.11, 52.43, 51.53, 53.76,
             51.30, 56.09, 51.99, 92.18, 63.91, 58.65, 48.22, 56.46, 53.67, 51.96,
             51.69, 50.61, 56.70, 57.74, 51.75, 53.84, 52.48, 51.12, 51.18, 52.86,
             53.90, 51.54, 51.25, 49.53, 55.88, 52.02, 51.62, 59.77, 51.79, 51.41,
             57.89, 50.66, 54.74, 68.64, 56.92, 57.63, 52.60, 53.25, 61.55, 63.64]

# Architecture 3: Aggressive Dropout
arch3_train = [839.40, 277.15, 141.33, 117.44, 124.42, 104.06, 84.03, 76.48, 70.27, 86.25,
               86.14, 68.07, 64.66, 70.47, 63.01, 59.55, 60.82, 59.19, 58.83, 57.87,
               57.67, 57.75, 57.65, 57.30, 55.63, 56.80, 57.28, 53.01, 55.88, 52.76,
               51.71, 52.28, 51.27, 50.66, 55.11, 50.65, 50.62, 52.51, 52.57, 47.49,
               49.78, 46.50, 45.11, 47.75, 47.61, 49.80, 46.88, 48.09, 47.46, 44.76,
               44.31, 49.19, 45.83, 44.94, 43.57, 41.84, 43.37, 44.35, 44.81, 44.97,
               41.79, 43.40, 45.00, 43.97, 46.74, 47.25, 43.72, 43.97, 43.49, 42.39,
               46.99, 45.54, 46.64, 43.67, 44.45, 42.92, 40.71, 40.22, 39.89, 40.06,
               37.42, 38.85, 41.09, 44.05, 45.26, 48.88, 46.62, 41.34, 40.05, 39.61,
               38.66, 39.81, 36.78, 38.20, 38.79, 36.37, 41.79, 40.06, 40.64, 38.45]

arch3_val = [244.47, 63.36, 97.99, 115.69, 103.08, 110.52, 80.95, 61.87, 79.32, 91.11,
             57.50, 62.73, 62.92, 66.67, 64.42, 56.28, 58.33, 56.06, 59.21, 63.22,
             67.10, 63.66, 61.49, 54.75, 59.79, 63.20, 58.58, 55.93, 53.75, 57.17,
             54.74, 55.91, 55.31, 63.26, 54.23, 57.42, 55.90, 58.28, 50.34, 48.98,
             49.68, 57.32, 50.89, 55.86, 50.98, 54.90, 59.40, 51.82, 52.25, 58.92,
             54.00, 50.61, 57.07, 55.07, 79.85, 52.73, 56.94, 52.36, 61.95, 50.29,
             50.03, 55.67, 56.02, 51.41, 53.23, 53.82, 49.00, 57.02, 56.21, 53.28,
             81.01, 61.86, 58.30, 61.95, 48.58, 51.77, 50.09, 49.70, 51.13, 56.61,
             49.38, 73.16, 55.80, 50.76, 96.50, 58.02, 59.04, 59.15, 47.11, 50.88,
             77.14, 53.82, 51.24, 51.40, 61.40, 75.21, 60.09, 47.03, 48.25, 50.54]

# Architecture 4: Simplified depth
arch4_train = [198.81, 99.78, 79.09, 76.03, 67.15, 71.26, 69.30, 68.23, 67.91, 66.33,
               60.14, 58.16, 58.65, 61.95, 54.89, 50.99, 53.79, 48.16, 50.82, 52.71,
               53.36, 45.41, 48.21, 41.42, 44.97, 43.13, 46.43, 45.32, 43.00, 42.55,
               38.47, 37.96, 37.74, 37.52, 36.76, 37.23, 37.70, 37.05, 32.84, 32.12,
               34.41, 32.71, 33.17, 31.59, 31.14, 30.78, 33.02, 31.64, 31.40, 29.16,
               29.90, 28.78, 30.00, 29.67, 29.99, 28.82, 29.97, 28.25, 27.23, 30.60,
               29.83, 30.29, 28.84, 26.79, 27.71, 25.71, 26.70, 25.78, 24.47, 26.22,
               27.02, 25.74, 23.33, 24.76, 24.56, 24.90, 24.74, 25.78, 23.83, 22.68,
               23.66, 21.82, 24.06, 22.53, 24.54, 25.63, 26.95, 22.78, 22.18, 24.03,
               22.47, 22.43, 20.66, 20.22, 19.66, 23.50, 21.15, 20.56, 20.16, 21.21]

arch4_val = [69.21, 64.14, 62.90, 51.36, 49.48, 49.63, 49.12, 59.73, 65.80, 54.95,
             49.84, 49.99, 47.73, 54.37, 47.88, 51.69, 60.68, 43.80, 64.45, 57.05,
             49.32, 81.17, 53.11, 46.53, 46.24, 45.72, 55.18, 86.71, 50.90, 60.30,
             43.93, 50.24, 66.74, 48.38, 49.63, 55.36, 53.68, 46.41, 46.50, 52.66,
             68.18, 52.28, 49.41, 70.49, 46.68, 71.70, 48.72, 64.01, 46.14, 78.54,
             46.32, 49.27, 46.17, 52.91, 54.49, 41.63, 44.99, 43.76, 105.83, 68.30,
             49.75, 58.67, 43.22, 48.41, 51.92, 42.88, 48.82, 130.69, 69.81, 48.86,
             54.50, 58.73, 63.29, 74.25, 46.55, 48.20, 46.30, 44.59, 41.08, 47.28,
             50.80, 55.72, 57.23, 69.77, 44.30, 70.61, 50.23, 49.68, 47.44, 58.98,
             46.92, 50.38, 47.11, 60.26, 49.43, 44.35, 45.23, 54.30, 48.71, 52.53]

# Create figure with 2x2 subplots
fig, axes = plt.subplots(2, 2, figsize=(7.5, 6))
fig.suptitle('Training and Validation MAPE Across Different CNN Architectures', 
             fontsize=11, fontweight='bold', y=0.995)

# Colors
color_train = '#2E86AB'
color_val = '#E63946'

# Architecture 1
ax = axes[0, 0]
ax.plot(epochs, arch1_train, color=color_train, linewidth=1.2, label='Training', alpha=0.8)
ax.plot(epochs, arch1_val, color=color_val, linewidth=1.2, label='Validation', alpha=0.8)
ax.set_title('Arch 1: No Regularization', fontsize=9, fontweight='bold')
ax.set_ylabel('MAPE (%)', fontsize=8)
ax.grid(alpha=0.3, linewidth=0.5)
ax.legend(fontsize=7, loc='upper right')
ax.set_ylim([0, 100])
ax.axhline(y=75, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
ax.text(5, 78, 'Target: 75%', fontsize=6, color='gray')

# Architecture 2
ax = axes[0, 1]
ax.plot(epochs, arch2_train, color=color_train, linewidth=1.2, label='Training', alpha=0.8)
ax.plot(epochs, arch2_val, color=color_val, linewidth=1.2, label='Validation', alpha=0.8)
ax.set_title('Arch 2: BatchNorm + Moderate Dropout', fontsize=9, fontweight='bold')
ax.grid(alpha=0.3, linewidth=0.5)
ax.legend(fontsize=7, loc='upper right')
ax.set_ylim([0, 130])
ax.axhline(y=75, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)

# Architecture 3
ax = axes[1, 0]
ax.plot(epochs, arch3_train, color=color_train, linewidth=1.2, label='Training', alpha=0.8)
ax.plot(epochs, arch3_val, color=color_val, linewidth=1.2, label='Validation', alpha=0.8)
ax.set_title('Arch 3: Aggressive Dropout', fontsize=9, fontweight='bold')
ax.set_xlabel('Epoch', fontsize=8)
ax.set_ylabel('MAPE (%)', fontsize=8)
ax.grid(alpha=0.3, linewidth=0.5)
ax.legend(fontsize=7, loc='upper right')
ax.set_ylim([0, 260])
ax.axhline(y=75, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)

# Architecture 4
ax = axes[1, 1]
ax.plot(epochs, arch4_train, color=color_train, linewidth=1.2, label='Training', alpha=0.8)
ax.plot(epochs, arch4_val, color=color_val, linewidth=1.2, label='Validation', alpha=0.8)
ax.set_title('Arch 4: Simplified Depth + Dropout', fontsize=9, fontweight='bold')
ax.set_xlabel('Epoch', fontsize=8)
ax.grid(alpha=0.3, linewidth=0.5)
ax.legend(fontsize=7, loc='upper right')
ax.set_ylim([0, 140])
ax.axhline(y=75, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)

plt.tight_layout()
plt.savefig('task2_loss_curves.pdf', dpi=300, bbox_inches='tight')
plt.savefig('task2_loss_curves.png', dpi=300, bbox_inches='tight')
plt.show()

print("Loss curves saved as 'task2_loss_curves.pdf' and '.png'")

# Print summary statistics
print("\n=== Performance Summary ===")
architectures = ['Arch 1', 'Arch 2', 'Arch 3', 'Arch 4']
best_vals = [min(arch1_val), min(arch2_val), min(arch3_val), min(arch4_val)]
best_epochs = [arch1_val.index(min(arch1_val))+1, arch2_val.index(min(arch2_val))+1,
               arch3_val.index(min(arch3_val))+1, arch4_val.index(min(arch4_val))+1]
final_vals = [arch1_val[-1], arch2_val[-1], arch3_val[-1], arch4_val[-1]]
final_trains = [arch1_train[-1], arch2_train[-1], arch3_train[-1], arch4_train[-1]]
gaps = [final_vals[i] - final_trains[i] for i in range(4)]

for i, arch in enumerate(architectures):
    print(f"\n{arch}:")
    print(f"  Best Val MAPE: {best_vals[i]:.2f}% (Epoch {best_epochs[i]})")
    print(f"  Final Val MAPE: {final_vals[i]:.2f}%")
    print(f"  Final Train MAPE: {final_trains[i]:.2f}%")
    print(f"  Train-Val Gap: {gaps[i]:.2f}%")
    print(f"  Meets target (<75%): {'✓' if best_vals[i] < 75 else '✗'}")