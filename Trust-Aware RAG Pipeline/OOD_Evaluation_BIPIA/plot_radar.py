import matplotlib.pyplot as plt
import numpy as np
from math import pi

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']

# 6 Academic Standard Dimensions
categories = [
    'Attack Mitigation\n(100 - ASR)', 
    'F1 Score\n(Answer Utility)', 
    'Exact Match\n(Strict Accuracy)', 
    'Clean Retention\n(100 - FPR)', 
    'Response Rate\n(100 - Abstention)', 
    'Evidence Rescue\nRate'
]
N = len(categories)

# Real 7B Evaluation Data
# Categories: Mitigation(100-ASR), F1, EM, Clean Retention(100-FPR), Response(100-Abs), Rescue
values_no_defense = [100.0, 53.3, 0.0, 100.0, 100.0, 0.0] 
values_whole_rejection = [100.0, 77.6, 0.0, 100.0, 100.0, 0.0] 
values_cascade = [100.0, 77.6, 0.0, 100.0, 100.0, 0.0] 

angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True), dpi=300)

# Offset to start from top
ax.set_theta_offset(pi / 2)
ax.set_theta_direction(-1)

# Draw axes
plt.xticks(angles[:-1], categories, size=11, fontweight='bold', color='#333333')
ax.tick_params(axis='x', pad=25)

# Draw ylabels
ax.set_rlabel_position(0)
plt.yticks([20, 40, 60, 80], ["20", "40", "60", "80"], color="grey", size=8)
plt.ylim(0, 100)

def plot_radar(values, color, label):
    values += values[:1]
    ax.plot(angles, values, color=color, linewidth=2, linestyle='solid', label=label)
    ax.fill(angles, values, color=color, alpha=0.15)

# Plot data
plot_radar(values_no_defense, '#c0392b', 'No Defense (Baseline)')
plot_radar(values_whole_rejection, '#2980b9', 'Whole Rejection (Coarse Drop)')
plot_radar(values_cascade, '#27ae60', 'Approach B (Cascade Sanitizer)')

# Add legend
plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1), fontsize=10, frameon=True, edgecolor='#cccccc')

plt.title('Multi-Dimensional Security vs. Utility Evaluation', size=14, fontweight='bold', y=1.1)

plt.tight_layout()
plt.savefig('radar_evaluation_chart.png', bbox_inches='tight')
print("Radar chart saved as radar_evaluation_chart.png")
