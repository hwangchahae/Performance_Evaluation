import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class ComprehensivePerformanceAnalyzer:
    def __init__(self, base_path="."):
        self.base_path = Path(base_path)
        self.model_sizes = ['1.7B', '4B', '8B']
        self.pre_results_path = self.base_path / "Pre_Training" / "pre_similarity_results"
        self.post_results_path = self.base_path / "Post_Training" / "post_similarity_results"
        
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['axes.unicode_minus'] = False
        
        self.results = {
            'pre': {},
            'post': {}
        }
        
    def load_results(self):
        """Load all evaluation results"""
        for model_size in self.model_sizes:
            # Pre-training results
            pre_json_path = self.pre_results_path / f"{model_size}_pretrain_similarity_results.json"
            if pre_json_path.exists():
                with open(pre_json_path, 'r', encoding='utf-8') as f:
                    self.results['pre'][model_size] = json.load(f)
            
            # Post-training results  
            post_json_path = self.post_results_path / f"{model_size}_post_similarity_results.json"
            if post_json_path.exists():
                with open(post_json_path, 'r', encoding='utf-8') as f:
                    self.results['post'][model_size] = json.load(f)
    
    def calculate_improvement(self):
        """Calculate improvement metrics between pre and post training"""
        improvements = {}
        
        for model_size in self.model_sizes:
            if model_size in self.results['pre'] and model_size in self.results['post']:
                pre_data = self.results['pre'][model_size]
                post_data = self.results['post'][model_size]
                
                # Calculate improvements
                tfidf_pre = pre_data['summary']['mean_tfidf_cosine']
                tfidf_post = post_data['summary']['mean_tfidf_cosine']
                embed_pre = pre_data['summary']['mean_embedding_cosine']
                embed_post = post_data['summary']['mean_embedding_cosine']
                
                improvements[model_size] = {
                    'tfidf': {
                        'pre': tfidf_pre,
                        'post': tfidf_post,
                        'absolute_improvement': tfidf_post - tfidf_pre,
                        'relative_improvement': ((tfidf_post - tfidf_pre) / tfidf_pre * 100) if tfidf_pre > 0 else 0
                    },
                    'embedding': {
                        'pre': embed_pre,
                        'post': embed_post,
                        'absolute_improvement': embed_post - embed_pre,
                        'relative_improvement': ((embed_post - embed_pre) / embed_pre * 100) if embed_pre > 0 else 0
                    }
                }
        
        return improvements
    
    def create_comparison_plots(self, improvements):
        """Create visualization plots comparing performance"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Model Performance Comparison: Pre-training vs Fine-tuning', fontsize=16, fontweight='bold')
        
        # Plot 1: TF-IDF scores comparison
        ax = axes[0, 0]
        models = list(improvements.keys())
        pre_tfidf = [improvements[m]['tfidf']['pre'] for m in models]
        post_tfidf = [improvements[m]['tfidf']['post'] for m in models]
        
        x = np.arange(len(models))
        width = 0.35
        
        ax.bar(x - width/2, pre_tfidf, width, label='Pre-training', color='lightblue')
        ax.bar(x + width/2, post_tfidf, width, label='Fine-tuning', color='darkblue')
        ax.set_xlabel('Model Size')
        ax.set_ylabel('TF-IDF Cosine Similarity')
        ax.set_title('TF-IDF Performance')
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Plot 2: Embedding scores comparison
        ax = axes[0, 1]
        pre_embed = [improvements[m]['embedding']['pre'] for m in models]
        post_embed = [improvements[m]['embedding']['post'] for m in models]
        
        ax.bar(x - width/2, pre_embed, width, label='Pre-training', color='lightgreen')
        ax.bar(x + width/2, post_embed, width, label='Fine-tuning', color='darkgreen')
        ax.set_xlabel('Model Size')
        ax.set_ylabel('Embedding Cosine Similarity')
        ax.set_title('Embedding Performance')
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Plot 3: Relative improvement
        ax = axes[0, 2]
        tfidf_improvement = [improvements[m]['tfidf']['relative_improvement'] for m in models]
        embed_improvement = [improvements[m]['embedding']['relative_improvement'] for m in models]
        
        ax.bar(x - width/2, tfidf_improvement, width, label='TF-IDF', color='coral')
        ax.bar(x + width/2, embed_improvement, width, label='Embedding', color='purple')
        ax.set_xlabel('Model Size')
        ax.set_ylabel('Relative Improvement (%)')
        ax.set_title('Performance Improvement After Fine-tuning')
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Plot 4: Score distribution (1.7B model)
        if '1.7B' in self.results['pre'] and '1.7B' in self.results['post']:
            ax = axes[1, 0]
            pre_scores = [item['embedding_cosine'] 
                         for item in self.results['pre']['1.7B']['details']]
            post_scores = [item['embedding_cosine'] 
                          for item in self.results['post']['1.7B']['details']]
            
            ax.hist(pre_scores, bins=20, alpha=0.5, label='Pre-training', color='blue')
            ax.hist(post_scores, bins=20, alpha=0.5, label='Fine-tuning', color='red')
            ax.set_xlabel('Embedding Similarity Score')
            ax.set_ylabel('Frequency')
            ax.set_title('Score Distribution (1.7B Model)')
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
        
        # Plot 5: Combined performance metric
        ax = axes[1, 1]
        combined_pre = [(improvements[m]['tfidf']['pre'] + improvements[m]['embedding']['pre']) / 2 
                       for m in models]
        combined_post = [(improvements[m]['tfidf']['post'] + improvements[m]['embedding']['post']) / 2 
                        for m in models]
        
        ax.plot(models, combined_pre, 'o-', label='Pre-training', linewidth=2, markersize=8)
        ax.plot(models, combined_post, 's-', label='Fine-tuning', linewidth=2, markersize=8)
        ax.set_xlabel('Model Size')
        ax.set_ylabel('Combined Score (Average)')
        ax.set_title('Overall Performance Trend')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 6: Improvement by model size
        ax = axes[1, 2]
        absolute_improvements = [improvements[m]['embedding']['absolute_improvement'] for m in models]
        colors = ['green' if x > 0 else 'red' for x in absolute_improvements]
        
        bars = ax.bar(models, absolute_improvements, color=colors, alpha=0.7)
        ax.set_xlabel('Model Size')
        ax.set_ylabel('Absolute Improvement')
        ax.set_title('Embedding Score Improvement')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, val in zip(bars, absolute_improvements):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.3f}', ha='center', va='bottom' if val > 0 else 'top')
        
        plt.tight_layout()
        return fig
    
    def generate_summary_report(self, improvements):
        """Generate a comprehensive summary report"""
        report = []
        report.append("=" * 80)
        report.append("COMPREHENSIVE PERFORMANCE EVALUATION REPORT")
        report.append("=" * 80)
        report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Executive Summary
        report.append("EXECUTIVE SUMMARY")
        report.append("-" * 40)
        
        best_improvement = max(improvements.items(), 
                              key=lambda x: x[1]['embedding']['relative_improvement'])
        report.append(f"Best performing model: {best_improvement[0]}")
        report.append(f"Maximum improvement: {best_improvement[1]['embedding']['relative_improvement']:.2f}% (Embedding)")
        
        avg_tfidf_improvement = np.mean([improvements[m]['tfidf']['relative_improvement'] 
                                        for m in improvements])
        avg_embed_improvement = np.mean([improvements[m]['embedding']['relative_improvement'] 
                                        for m in improvements])
        
        report.append(f"Average TF-IDF improvement: {avg_tfidf_improvement:.2f}%")
        report.append(f"Average Embedding improvement: {avg_embed_improvement:.2f}%")
        
        # Detailed Results by Model
        report.append("\n" + "=" * 80)
        report.append("DETAILED RESULTS BY MODEL")
        report.append("=" * 80)
        
        for model_size in sorted(improvements.keys()):
            report.append(f"\n{model_size} Model")
            report.append("-" * 40)
            
            # TF-IDF Results
            tfidf = improvements[model_size]['tfidf']
            report.append("TF-IDF Cosine Similarity:")
            report.append(f"  Pre-training:  {tfidf['pre']:.4f}")
            report.append(f"  Fine-tuning:   {tfidf['post']:.4f}")
            report.append(f"  Improvement:   {tfidf['absolute_improvement']:.4f} ({tfidf['relative_improvement']:.2f}%)")
            
            # Embedding Results
            embed = improvements[model_size]['embedding']
            report.append("\nEmbedding Cosine Similarity:")
            report.append(f"  Pre-training:  {embed['pre']:.4f}")
            report.append(f"  Fine-tuning:   {embed['post']:.4f}")
            report.append(f"  Improvement:   {embed['absolute_improvement']:.4f} ({embed['relative_improvement']:.2f}%)")
            
            # Performance Analysis
            if embed['relative_improvement'] > 30:
                report.append("\n✓ Excellent improvement - Fine-tuning highly effective")
            elif embed['relative_improvement'] > 15:
                report.append("\n✓ Good improvement - Fine-tuning moderately effective")
            elif embed['relative_improvement'] > 0:
                report.append("\n✓ Positive improvement - Fine-tuning somewhat effective")
            else:
                report.append("\n⚠ No improvement or degradation observed")
        
        # Statistical Summary
        report.append("\n" + "=" * 80)
        report.append("STATISTICAL SUMMARY")
        report.append("=" * 80)
        
        # Calculate statistics for each model
        for model_size in self.model_sizes:
            if model_size in self.results['pre'] and model_size in self.results['post']:
                report.append(f"\n{model_size} Model Statistics:")
                
                # Pre-training stats
                pre_embed_scores = [item['embedding_cosine'] 
                                   for item in self.results['pre'][model_size]['details']]
                post_embed_scores = [item['embedding_cosine'] 
                                    for item in self.results['post'][model_size]['details']]
                
                report.append("  Pre-training:")
                report.append(f"    Mean: {np.mean(pre_embed_scores):.4f}")
                report.append(f"    Std:  {np.std(pre_embed_scores):.4f}")
                report.append(f"    Min:  {np.min(pre_embed_scores):.4f}")
                report.append(f"    Max:  {np.max(pre_embed_scores):.4f}")
                
                report.append("  Fine-tuning:")
                report.append(f"    Mean: {np.mean(post_embed_scores):.4f}")
                report.append(f"    Std:  {np.std(post_embed_scores):.4f}")
                report.append(f"    Min:  {np.min(post_embed_scores):.4f}")
                report.append(f"    Max:  {np.max(post_embed_scores):.4f}")
        
        # Key Findings
        report.append("\n" + "=" * 80)
        report.append("KEY FINDINGS")
        report.append("=" * 80)
        
        # Find best and worst performing samples
        for model_size in ['1.7B']:  # Focus on 1.7B for detailed analysis
            if model_size in self.results['post']:
                post_results = self.results['post'][model_size]['details']
                
                # Sort by embedding score
                sorted_results = sorted(post_results, 
                                      key=lambda x: x['embedding_cosine'],
                                      reverse=True)
                
                report.append(f"\n{model_size} Model - Top Performing Samples (Fine-tuned):")
                for i, item in enumerate(sorted_results[:3], 1):
                    report.append(f"  {i}. {item['file_name']}")
                    report.append(f"     Embedding: {item['embedding_cosine']:.4f}")
                    report.append(f"     TF-IDF: {item['tfidf_cosine']:.4f}")
                
                report.append(f"\n{model_size} Model - Lowest Performing Samples (Fine-tuned):")
                for i, item in enumerate(sorted_results[-3:], 1):
                    report.append(f"  {i}. {item['file_name']}")
                    report.append(f"     Embedding: {item['embedding_cosine']:.4f}")
                    report.append(f"     TF-IDF: {item['tfidf_cosine']:.4f}")
        
        # Recommendations
        report.append("\n" + "=" * 80)
        report.append("RECOMMENDATIONS")
        report.append("=" * 80)
        
        # Analyze overall trend
        embed_improvements = [improvements[m]['embedding']['relative_improvement'] for m in improvements]
        
        if all(imp > 20 for imp in embed_improvements):
            report.append("\n✓ Fine-tuning shows consistent significant improvements across all model sizes")
            report.append("✓ Current fine-tuning approach is highly effective")
        elif all(imp > 10 for imp in embed_improvements):
            report.append("\n✓ Fine-tuning shows moderate improvements across all model sizes")
            report.append("• Consider further optimization of fine-tuning parameters")
        else:
            report.append("\n⚠ Fine-tuning shows mixed results")
            report.append("• Review training data quality and diversity")
            report.append("• Consider adjusting hyperparameters")
        
        # Model-specific recommendations
        best_model = max(improvements.items(), 
                        key=lambda x: (x[1]['embedding']['post'] + x[1]['tfidf']['post']) / 2)
        report.append(f"\nRecommended model for deployment: {best_model[0]}")
        report.append(f"  Combined score: {(best_model[1]['embedding']['post'] + best_model[1]['tfidf']['post']) / 2:.4f}")
        
        return "\n".join(report)
    
    def save_results(self, improvements, report, fig):
        """Save all analysis results"""
        # Save report
        report_path = self.base_path / "comprehensive_analysis_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"Report saved to: {report_path}")
        
        # Save improvements data
        improvements_path = self.base_path / "performance_improvements.json"
        with open(improvements_path, 'w', encoding='utf-8') as f:
            json.dump(improvements, f, indent=2, ensure_ascii=False)
        print(f"Improvements data saved to: {improvements_path}")
        
        # Save plot
        plot_path = self.base_path / "performance_comparison_plots.png"
        fig.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Plots saved to: {plot_path}")
        
        # Create summary DataFrame and save as CSV
        summary_data = []
        for model in improvements:
            summary_data.append({
                'Model': model,
                'TF-IDF_Pre': improvements[model]['tfidf']['pre'],
                'TF-IDF_Post': improvements[model]['tfidf']['post'],
                'TF-IDF_Improvement_%': improvements[model]['tfidf']['relative_improvement'],
                'Embedding_Pre': improvements[model]['embedding']['pre'],
                'Embedding_Post': improvements[model]['embedding']['post'],
                'Embedding_Improvement_%': improvements[model]['embedding']['relative_improvement']
            })
        
        df = pd.DataFrame(summary_data)
        csv_path = self.base_path / "performance_summary.csv"
        df.to_csv(csv_path, index=False)
        print(f"Summary CSV saved to: {csv_path}")
    
    def run_analysis(self):
        """Run the complete analysis pipeline"""
        print("=" * 60)
        print("COMPREHENSIVE PERFORMANCE ANALYSIS")
        print("=" * 60)
        
        print("\n1. Loading evaluation results...")
        self.load_results()
        
        print("2. Calculating improvements...")
        improvements = self.calculate_improvement()
        
        print("3. Creating visualizations...")
        fig = self.create_comparison_plots(improvements)
        
        print("4. Generating report...")
        report = self.generate_summary_report(improvements)
        
        print("5. Saving results...")
        self.save_results(improvements, report, fig)
        
        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETE")
        print("=" * 60)
        
        # Print summary
        print("\nQuick Summary:")
        for model in sorted(improvements.keys()):
            embed_imp = improvements[model]['embedding']['relative_improvement']
            print(f"  {model}: {embed_imp:+.2f}% embedding improvement")
        
        return improvements, report, fig

if __name__ == "__main__":
    analyzer = ComprehensivePerformanceAnalyzer()
    improvements, report, fig = analyzer.run_analysis()
    
    # Display report
    print("\n" + "=" * 60)
    print("REPORT PREVIEW (First 50 lines)")
    print("=" * 60)
    print("\n".join(report.split("\n")[:50]))
    print("\n... (See comprehensive_analysis_report.txt for full report)")