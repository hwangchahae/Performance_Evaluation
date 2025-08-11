import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
from datetime import datetime

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Read the performance data
df = pd.read_csv('performance_summary.csv')

# Create figure with subplots
fig = plt.figure(figsize=(20, 12))
fig.suptitle('Comprehensive Model Performance Analysis: Pre-Training vs Fine-Tuning', 
             fontsize=20, fontweight='bold', y=0.98)

# 1. TF-IDF Performance Comparison
ax1 = plt.subplot(2, 3, 1)
x = np.arange(len(df['Model']))
width = 0.35

bars1 = ax1.bar(x - width/2, df['TF-IDF_Pre'], width, label='Pre-Training', color='#3498db', alpha=0.8)
bars2 = ax1.bar(x + width/2, df['TF-IDF_Post'], width, label='Fine-Tuning', color='#e74c3c', alpha=0.8)

ax1.set_xlabel('Model Size', fontweight='bold')
ax1.set_ylabel('TF-IDF Cosine Similarity', fontweight='bold')
ax1.set_title('TF-IDF Performance Comparison', fontweight='bold', fontsize=12)
ax1.set_xticks(x)
ax1.set_xticklabels(df['Model'])
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)

# 2. Embedding Performance Comparison
ax2 = plt.subplot(2, 3, 2)
bars3 = ax2.bar(x - width/2, df['Embedding_Pre'], width, label='Pre-Training', color='#2ecc71', alpha=0.8)
bars4 = ax2.bar(x + width/2, df['Embedding_Post'], width, label='Fine-Tuning', color='#f39c12', alpha=0.8)

ax2.set_xlabel('Model Size', fontweight='bold')
ax2.set_ylabel('Embedding Cosine Similarity', fontweight='bold')
ax2.set_title('Embedding Performance Comparison', fontweight='bold', fontsize=12)
ax2.set_xticks(x)
ax2.set_xticklabels(df['Model'])
ax2.legend(loc='lower right')
ax2.grid(True, alpha=0.3)
ax2.set_ylim([0.89, 0.93])  # Zoom in to show differences better

# Add value labels
for bars in [bars3, bars4]:
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.0005,
                f'{height:.4f}', ha='center', va='bottom', fontsize=9)

# 3. Improvement Percentage Analysis
ax3 = plt.subplot(2, 3, 3)
improvements = pd.DataFrame({
    'Model': df['Model'],
    'TF-IDF': df['TF-IDF_Improvement_%'],
    'Embedding': df['Embedding_Improvement_%']
})

x_imp = np.arange(len(improvements['Model']))
bars5 = ax3.bar(x_imp - width/2, improvements['TF-IDF'], width, label='TF-IDF', color='#9b59b6', alpha=0.8)
bars6 = ax3.bar(x_imp + width/2, improvements['Embedding'], width, label='Embedding', color='#1abc9c', alpha=0.8)

ax3.set_xlabel('Model Size', fontweight='bold')
ax3.set_ylabel('Improvement (%)', fontweight='bold')
ax3.set_title('Performance Improvement Analysis', fontweight='bold', fontsize=12)
ax3.set_xticks(x_imp)
ax3.set_xticklabels(improvements['Model'])
ax3.legend()
ax3.grid(True, alpha=0.3)

# Add value labels
for bars in [bars5, bars6]:
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9)

# 4. Combined Performance Score (Radar Chart)
ax4 = plt.subplot(2, 3, 4, projection='polar')
categories = ['TF-IDF Pre', 'TF-IDF Post', 'Embedding Pre', 'Embedding Post']
num_vars = len(categories)

angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]

for idx, model in enumerate(df['Model']):
    values = [
        df.loc[idx, 'TF-IDF_Pre'],
        df.loc[idx, 'TF-IDF_Post'],
        df.loc[idx, 'Embedding_Pre'],
        df.loc[idx, 'Embedding_Post']
    ]
    values += values[:1]
    
    ax4.plot(angles, values, 'o-', linewidth=2, label=model, markersize=8)
    ax4.fill(angles, values, alpha=0.15)

ax4.set_xticks(angles[:-1])
ax4.set_xticklabels(categories, size=9)
ax4.set_ylim(0, 1)
ax4.set_title('Multi-Dimensional Performance Profile', fontweight='bold', fontsize=12, pad=20)
ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
ax4.grid(True)

# 5. Performance Delta Heatmap
ax5 = plt.subplot(2, 3, 5)
delta_data = np.array([
    [df.loc[0, 'TF-IDF_Post'] - df.loc[0, 'TF-IDF_Pre'], 
     df.loc[0, 'Embedding_Post'] - df.loc[0, 'Embedding_Pre']],
    [df.loc[1, 'TF-IDF_Post'] - df.loc[1, 'TF-IDF_Pre'], 
     df.loc[1, 'Embedding_Post'] - df.loc[1, 'Embedding_Pre']],
    [df.loc[2, 'TF-IDF_Post'] - df.loc[2, 'TF-IDF_Pre'], 
     df.loc[2, 'Embedding_Post'] - df.loc[2, 'Embedding_Pre']]
])

im = ax5.imshow(delta_data, cmap='RdYlGn', aspect='auto', vmin=-0.01, vmax=0.08)
ax5.set_xticks(np.arange(2))
ax5.set_yticks(np.arange(3))
ax5.set_xticklabels(['TF-IDF', 'Embedding'])
ax5.set_yticklabels(df['Model'])
ax5.set_title('Performance Delta Matrix', fontweight='bold', fontsize=12)

# Add text annotations
for i in range(3):
    for j in range(2):
        text = ax5.text(j, i, f'{delta_data[i, j]:.4f}',
                       ha="center", va="center", color="black", fontweight='bold')

plt.colorbar(im, ax=ax5, label='Improvement')

# 6. Summary Statistics Box
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')

# Calculate summary statistics
avg_tfidf_imp = df['TF-IDF_Improvement_%'].mean()
avg_emb_imp = df['Embedding_Improvement_%'].mean()
best_model_tfidf = df.loc[df['TF-IDF_Improvement_%'].idxmax(), 'Model']
best_model_emb = df.loc[df['Embedding_Improvement_%'].idxmax(), 'Model']
best_overall = df.loc[(df['TF-IDF_Post'] + df['Embedding_Post']).idxmax(), 'Model']

summary_text = f"""
📊 PERFORMANCE SUMMARY
{'='*40}

Average Improvements:
• TF-IDF: {avg_tfidf_imp:.2f}%
• Embedding: {avg_emb_imp:.2f}%

Best Performers:
• TF-IDF Improvement: {best_model_tfidf} ({df[df['Model']==best_model_tfidf]['TF-IDF_Improvement_%'].values[0]:.1f}%)
• Embedding Improvement: {best_model_emb} ({df[df['Model']==best_model_emb]['Embedding_Improvement_%'].values[0]:.2f}%)
• Overall Performance: {best_overall}

Key Insights:
✓ Fine-tuning shows significant TF-IDF gains
  for 4B and 8B models (36-37%)
✓ Embedding improvements are minimal (<0.5%)
  due to high baseline performance
✓ Larger models (4B, 8B) benefit more from
  fine-tuning than smaller models (1.7B)

Recommendation:
Deploy {best_overall} model for optimal performance
"""

ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes, fontsize=11,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Add timestamp
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
fig.text(0.99, 0.01, f'Generated: {timestamp}', ha='right', fontsize=8, color='gray')

plt.tight_layout()
plt.savefig('enhanced_performance_analysis.png', dpi=300, bbox_inches='tight')
plt.savefig('enhanced_performance_analysis.pdf', bbox_inches='tight')
print("Enhanced visualization saved as 'enhanced_performance_analysis.png' and '.pdf'")
plt.show()

# Create detailed comparison table
print("\n" + "="*80)
print("DETAILED PERFORMANCE COMPARISON TABLE")
print("="*80)

comparison_df = pd.DataFrame({
    'Model': df['Model'],
    'TF-IDF Pre': df['TF-IDF_Pre'].round(4),
    'TF-IDF Post': df['TF-IDF_Post'].round(4),
    'TF-IDF Δ': (df['TF-IDF_Post'] - df['TF-IDF_Pre']).round(4),
    'TF-IDF %': df['TF-IDF_Improvement_%'].round(2),
    'Emb Pre': df['Embedding_Pre'].round(4),
    'Emb Post': df['Embedding_Post'].round(4),
    'Emb Δ': (df['Embedding_Post'] - df['Embedding_Pre']).round(4),
    'Emb %': df['Embedding_Improvement_%'].round(2)
})

print(comparison_df.to_string(index=False))

# Calculate and display correlation
print("\n" + "="*80)
print("CORRELATION ANALYSIS")
print("="*80)

model_sizes = [1.7, 4.0, 8.0]
tfidf_improvements = df['TF-IDF_Improvement_%'].values
emb_improvements = df['Embedding_Improvement_%'].values

from scipy import stats

corr_size_tfidf, p_value_tfidf = stats.pearsonr(model_sizes, tfidf_improvements)
corr_size_emb, p_value_emb = stats.pearsonr(model_sizes, emb_improvements)

print(f"Correlation between model size and TF-IDF improvement: {corr_size_tfidf:.3f} (p={p_value_tfidf:.3f})")
print(f"Correlation between model size and Embedding improvement: {corr_size_emb:.3f} (p={p_value_emb:.3f})")

if corr_size_tfidf > 0.7:
    print("→ Strong positive correlation: Larger models benefit more from fine-tuning (TF-IDF)")
elif corr_size_tfidf > 0.3:
    print("→ Moderate positive correlation: Some benefit from model size (TF-IDF)")
    
if abs(corr_size_emb) < 0.3:
    print("→ Weak correlation: Model size has minimal impact on embedding improvements")