import matplotlib.pyplot as plt
import numpy as np

# Set plotting style
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

# Replace these with the actual statistics after running your evaluations
methods = ['No Defense', 'Whole Rejection', 'Approach B (Cascade)']

asr = [0.0, 0.0, 0.0]            # Attack Success Rate (ASR %) [Lower is better]
em = [0.0, 0.0, 0.0]
    fpr = [0.0, 0.0, 0.0]
    context_retention = [100.0, 88.7, 88.7]
    f1 = [53.3, 77.6, 77.6]           # Answer Token F1 (%) [Higher is better]
abstention = [0.0, 0.0, 0.0]    # Abstention Rate (%) [Lower is better]
rescue_rate = [0.0, 0.0, 0.0]    # Evidence Rescue Rate (%) [Higher is better]

x = np.arange(len(methods))
width = 0.18

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

# ==================== Figure 1: Four Metrics Bar Chart ====================
colors = ['#c0392b', '#2980b9', '#27ae60']

rects1 = ax1.bar(x - 1.5*width, asr, width, label='ASR (Downstream Attack %)', color=colors[0], alpha=0.9)
rects2 = ax1.bar(x - 0.5*width, f1, width, label='Token F1 (Utility %)', color=colors[1], alpha=0.9)
rects3 = ax1.bar(x + 0.5*width, abstention, width, label='Abstention Rate (%)', color=colors[2], alpha=0.9)
rects4 = ax1.bar(x + 1.5*width, rescue_rate, width, label='Evidence Rescue Rate (%)', color=colors[3], alpha=0.9)

ax1.set_ylabel('Percentage (%)', fontsize=11, fontweight='bold')
ax1.set_title('(a) Multi-Metric Evaluation on Unseen BIPIA Dataset', fontsize=12, pad=12, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(methods, rotation=15, ha='right', fontsize=9.5)
ax1.set_ylim(0, 100)
ax1.grid(axis='y', linestyle='--', alpha=0.4)
ax1.legend(frameon=True, facecolor='white', edgecolor='#e0e0e0', fontsize=8.5, loc='upper right')

# Add value labels to bars
for rects in [rects1, rects2, rects3, rects4]:
    for rect in rects:
        h = rect.get_height()
        if h > 0:
            ax1.annotate(f'{h:.1f}',
                        xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=7)

# ==================== Figure 2: Security vs Utility Trade-off Scatter Plot ====================
mitigation_rate = [100 - val for val in asr]  # Defense Success Rate / Mitigation Rate (%)
scatter_colors = ['#7f8c8d', '#e74c3c', '#2ecc71']

for i, method in enumerate(methods):
    # Size maps to Evidence Rescue Rate (Base size 120)
    size = 120 + rescue_rate[i] * 12
    ax2.scatter(mitigation_rate[i], f1[i], s=size, color=scatter_colors[i], 
                alpha=0.85, edgecolors='black', linewidth=1.2, zorder=3)
    
    # Annotate point name and rescue rate
    offset_y = 1.2 if i != 1 else -2.2
    label_txt = f"{method}\n(Rescue: {rescue_rate[i]:.1f}%)" if rescue_rate[i] > 0 else method
    ax2.annotate(label_txt, (mitigation_rate[i], f1[i] + offset_y),
                 ha='center', fontsize=8.5, weight='semibold',
                 bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#cccccc", alpha=0.7))

ax2.set_xlabel('Attack Defense / Mitigation Rate (100 - ASR %)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Utility Retention (Token F1 %)', fontsize=11, fontweight='bold')
ax2.set_title('(b) Security vs. Utility Frontier', fontsize=12, pad=12, fontweight='bold')
ax2.set_xlim(0, 105)
ax2.set_ylim(20, 60)
ax2.grid(True, linestyle='--', alpha=0.4)

# Draw Pareto optimal direction arrow
ax2.annotate('Optimal Direction\n(High Defense & High Utility)', 
             xy=(98, 54), xytext=(65, 52),
             arrowprops=dict(facecolor='#2c3e50', shrink=0.08, width=1, headwidth=6),
             fontsize=8, color='#2c3e50', style='italic')

plt.tight_layout()
plt.savefig('unseen_dataset_evaluation.png', dpi=300)
print("Plot successfully saved as unseen_dataset_evaluation.png")
