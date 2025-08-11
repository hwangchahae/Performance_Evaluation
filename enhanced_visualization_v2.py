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

# Create figure with subplots (adjusted layout for separate improvement charts)
fig = plt.figure(figsize=(22, 14))
fig.suptitle('Comprehensive Model Performance Analysis: Pre-Training vs Fine-Tuning', 
             fontsize=20, fontweight='bold', y=0.98)

# 1. TF-IDF Performance Comparison
ax1 = plt.subplot(3, 3, 1)
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
ax2 = plt.subplot(3, 3, 2)
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

# 3. TF-IDF Improvement Percentage (Separate)
ax3 = plt.subplot(3, 3, 4)
colors_tfidf = ['#FF6B6B', '#4ECDC4', '#45B7D1']
bars_tfidf = ax3.bar(df['Model'], df['TF-IDF_Improvement_%'], color=colors_tfidf, alpha=0.8, edgecolor='black', linewidth=2)

ax3.set_xlabel('Model Size', fontweight='bold')
ax3.set_ylabel('Improvement (%)', fontweight='bold')
ax3.set_title('TF-IDF Performance Improvement', fontweight='bold', fontsize=12)
ax3.grid(True, alpha=0.3, axis='y')

# Add value labels and improvement indicators
for i, bar in enumerate(bars_tfidf):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
            f'{height:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Add arrow indicators
    if height > 30:
        ax3.text(bar.get_x() + bar.get_width()/2., height/2,
                '⬆⬆⬆', ha='center', va='center', fontsize=14, color='white', fontweight='bold')
    elif height > 10:
        ax3.text(bar.get_x() + bar.get_width()/2., height/2,
                '⬆⬆', ha='center', va='center', fontsize=14, color='white', fontweight='bold')
    else:
        ax3.text(bar.get_x() + bar.get_width()/2., height/2,
                '⬆', ha='center', va='center', fontsize=14, color='white', fontweight='bold')

# 4. Embedding Improvement Percentage (Separate)
ax4 = plt.subplot(3, 3, 5)
colors_emb = ['#FFD93D', '#6BCB77', '#4D96FF']
bars_emb = ax4.bar(df['Model'], df['Embedding_Improvement_%'], color=colors_emb, alpha=0.8, edgecolor='black', linewidth=2)

ax4.set_xlabel('Model Size', fontweight='bold')
ax4.set_ylabel('Improvement (%)', fontweight='bold')
ax4.set_title('Embedding Performance Improvement', fontweight='bold', fontsize=12)
ax4.grid(True, alpha=0.3, axis='y')
ax4.set_ylim([0, 0.6])  # Adjusted scale for small improvements

# Add value labels
for i, bar in enumerate(bars_emb):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
            f'{height:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Add status indicator
    ax4.text(bar.get_x() + bar.get_width()/2., height/2,
            '✓', ha='center', va='center', fontsize=16, color='white', fontweight='bold')

# 5. Combined Performance Score (Radar Chart)
ax5 = plt.subplot(3, 3, 3, projection='polar')
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
    
    ax5.plot(angles, values, 'o-', linewidth=2, label=model, markersize=8)
    ax5.fill(angles, values, alpha=0.15)

ax5.set_xticks(angles[:-1])
ax5.set_xticklabels(categories, size=9)
ax5.set_ylim(0, 1)
ax5.set_title('Multi-Dimensional Performance Profile', fontweight='bold', fontsize=12, pad=20)
ax5.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
ax5.grid(True)

# 6. Performance Delta Heatmap
ax6 = plt.subplot(3, 3, 6)
delta_data = np.array([
    [df.loc[0, 'TF-IDF_Post'] - df.loc[0, 'TF-IDF_Pre'], 
     df.loc[0, 'Embedding_Post'] - df.loc[0, 'Embedding_Pre']],
    [df.loc[1, 'TF-IDF_Post'] - df.loc[1, 'TF-IDF_Pre'], 
     df.loc[1, 'Embedding_Post'] - df.loc[1, 'Embedding_Pre']],
    [df.loc[2, 'TF-IDF_Post'] - df.loc[2, 'TF-IDF_Pre'], 
     df.loc[2, 'Embedding_Post'] - df.loc[2, 'Embedding_Pre']]
])

im = ax6.imshow(delta_data, cmap='RdYlGn', aspect='auto', vmin=-0.01, vmax=0.08)
ax6.set_xticks(np.arange(2))
ax6.set_yticks(np.arange(3))
ax6.set_xticklabels(['TF-IDF', 'Embedding'])
ax6.set_yticklabels(df['Model'])
ax6.set_title('Performance Delta Matrix', fontweight='bold', fontsize=12)

# Add text annotations
for i in range(3):
    for j in range(2):
        text = ax6.text(j, i, f'{delta_data[i, j]:.4f}',
                       ha="center", va="center", color="black", fontweight='bold')

plt.colorbar(im, ax=ax6, label='Improvement')

# 7. Absolute Performance Gains
ax7 = plt.subplot(3, 3, 7)
x_pos = np.arange(len(df['Model']))
width = 0.35

# Calculate absolute gains
tfidf_gains = df['TF-IDF_Post'] - df['TF-IDF_Pre']
emb_gains = df['Embedding_Post'] - df['Embedding_Pre']

bars7 = ax7.bar(x_pos - width/2, tfidf_gains, width, label='TF-IDF Gain', color='#9b59b6', alpha=0.8)
bars8 = ax7.bar(x_pos + width/2, emb_gains, width, label='Embedding Gain', color='#1abc9c', alpha=0.8)

ax7.set_xlabel('Model Size', fontweight='bold')
ax7.set_ylabel('Absolute Gain', fontweight='bold')
ax7.set_title('Absolute Performance Gains', fontweight='bold', fontsize=12)
ax7.set_xticks(x_pos)
ax7.set_xticklabels(df['Model'])
ax7.legend()
ax7.grid(True, alpha=0.3)

# Add value labels
for bars in [bars7, bars8]:
    for bar in bars:
        height = bar.get_height()
        ax7.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{height:.4f}', ha='center', va='bottom', fontsize=9)

# 8. Model Ranking
ax8 = plt.subplot(3, 3, 8)

# Calculate combined scores
combined_scores = (df['TF-IDF_Post'] + df['Embedding_Post']) / 2
ranking_data = pd.DataFrame({
    'Model': df['Model'],
    'Score': combined_scores
}).sort_values('Score', ascending=True)

colors_rank = ['#FFB6C1', '#87CEEB', '#98D8C8']
bars_rank = ax8.barh(ranking_data['Model'], ranking_data['Score'], color=colors_rank, alpha=0.8, edgecolor='black', linewidth=2)

ax8.set_xlabel('Combined Score', fontweight='bold')
ax8.set_title('Overall Model Ranking', fontweight='bold', fontsize=12)
ax8.grid(True, alpha=0.3, axis='x')

# Add value labels and medals
for i, bar in enumerate(bars_rank):
    width = bar.get_width()
    ax8.text(width + 0.005, bar.get_y() + bar.get_height()/2,
            f'{width:.4f}', ha='left', va='center', fontsize=10, fontweight='bold')
    
    # Add ranking medals
    if i == 2:  # Best (last in sorted order)
        ax8.text(0.02, bar.get_y() + bar.get_height()/2,
                '🥇', ha='left', va='center', fontsize=16)
    elif i == 1:  # Second
        ax8.text(0.02, bar.get_y() + bar.get_height()/2,
                '🥈', ha='left', va='center', fontsize=16)
    else:  # Third
        ax8.text(0.02, bar.get_y() + bar.get_height()/2,
                '🥉', ha='left', va='center', fontsize=16)

# 9. Summary Statistics Box
ax9 = plt.subplot(3, 3, 9)
ax9.axis('off')

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

ax9.text(0.1, 0.9, summary_text, transform=ax9.transAxes, fontsize=11,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Add timestamp
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
fig.text(0.99, 0.01, f'Generated: {timestamp}', ha='right', fontsize=8, color='gray')

plt.tight_layout()
plt.savefig('enhanced_performance_analysis.png', dpi=300, bbox_inches='tight')
plt.savefig('enhanced_performance_analysis.pdf', bbox_inches='tight')
print("Enhanced visualization (v2) saved as 'enhanced_performance_analysis.png' and '.pdf'")
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

# Additional analysis: Improvement ratio
print("\n" + "="*80)
print("IMPROVEMENT RATIO ANALYSIS")
print("="*80)

for idx, model in enumerate(df['Model']):
    tfidf_ratio = df.loc[idx, 'TF-IDF_Improvement_%'] 
    emb_ratio = df.loc[idx, 'Embedding_Improvement_%']
    ratio = tfidf_ratio / emb_ratio if emb_ratio != 0 else float('inf')
    
    print(f"{model} Model:")
    print(f"  TF-IDF improvement is {ratio:.1f}x greater than Embedding improvement")
    print(f"  TF-IDF: {tfidf_ratio:.2f}% | Embedding: {emb_ratio:.2f}%")
    print()