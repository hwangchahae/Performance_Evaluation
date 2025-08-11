from flask import Flask, render_template, jsonify, request
import json
import os
from pathlib import Path
import glob
import pandas as pd

app = Flask(__name__)

# Configuration
BASE_DIR = Path(r"C:\Users\Playdata\Desktop\Performance_Evaluation")
PRE_TRAINING_DIR = BASE_DIR / "Pre_Training"
POST_TRAINING_DIR = BASE_DIR / "Post_Training"  
GOLD_STANDARD_DIR = BASE_DIR / "Gold_Standard_Data"

def load_json_file(filepath):
    """Load and return JSON file content"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        return {"error": str(e)}

def get_gold_standard_files():
    """Get list of all gold standard files"""
    files = []
    for filepath in GOLD_STANDARD_DIR.glob("**/*.json"):
        rel_path = filepath.relative_to(GOLD_STANDARD_DIR)
        files.append({
            "name": str(rel_path),
            "path": str(filepath),
            "size": filepath.stat().st_size
        })
    return sorted(files, key=lambda x: x["name"])

def get_model_results(model_size, training_type="pre"):
    """Get results for a specific model size and training type"""
    if training_type == "pre":
        base_dir = PRE_TRAINING_DIR / f"{model_size}_base_model_results"
    else:
        base_dir = POST_TRAINING_DIR / f"{model_size}_Fine_Tuning_results"
    
    results = []
    if base_dir.exists():
        for filepath in base_dir.glob("**/result.json"):
            folder_name = filepath.parent.name
            data = load_json_file(filepath)
            results.append({
                "folder": folder_name,
                "data": data
            })
    
    return sorted(results, key=lambda x: x["folder"])

def match_gold_standard(result_folder_name):
    """Find matching gold standard file for a result folder"""
    # Extract the base name from result folder
    # Format: result_<gold_file_name>_chunk_X or result_<gold_file_name>
    if result_folder_name.startswith("result_"):
        base_name = result_folder_name[7:]  # Remove "result_" prefix
        
        # Remove chunk suffix if present
        if "_chunk_" in base_name:
            base_name = base_name.rsplit("_chunk_", 1)[0]
        
        # Search for matching gold standard file
        for gold_file in GOLD_STANDARD_DIR.glob("**/*.json"):
            gold_name = gold_file.stem
            if gold_name == base_name:
                return {
                    "found": True,
                    "path": str(gold_file),
                    "data": load_json_file(gold_file)
                }
    
    return {"found": False}

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('dashboard.html')

@app.route('/api/overview')
def get_overview():
    """Get overview statistics"""
    stats = {
        "gold_standard": len(list(GOLD_STANDARD_DIR.glob("**/*.json"))),
        "pre_training": {
            "1.7B": len(list((PRE_TRAINING_DIR / "1.7B_base_model_results").glob("**/result.json"))) if (PRE_TRAINING_DIR / "1.7B_base_model_results").exists() else 0,
            "4B": len(list((PRE_TRAINING_DIR / "4B_base_model_results").glob("**/result.json"))) if (PRE_TRAINING_DIR / "4B_base_model_results").exists() else 0,
            "8B": len(list((PRE_TRAINING_DIR / "8B_base_model_results").glob("**/result.json"))) if (PRE_TRAINING_DIR / "8B_base_model_results").exists() else 0
        },
        "post_training": {
            "1.7B": len(list((POST_TRAINING_DIR / "1.7B_Fine_Tuning_results").glob("**/result.json"))) if (POST_TRAINING_DIR / "1.7B_Fine_Tuning_results").exists() else 0,
            "4B": len(list((POST_TRAINING_DIR / "4B_Fine_Tuning_results").glob("**/result.json"))) if (POST_TRAINING_DIR / "4B_Fine_Tuning_results").exists() else 0,
            "8B": len(list((POST_TRAINING_DIR / "8B_Fine_Tuning_results").glob("**/result.json"))) if (POST_TRAINING_DIR / "8B_Fine_Tuning_results").exists() else 0
        }
    }
    return jsonify(stats)

@app.route('/api/gold_standard')
def get_gold_standard():
    """Get list of gold standard files"""
    return jsonify(get_gold_standard_files())

@app.route('/api/gold_standard/<path:filename>')
def get_gold_standard_content(filename):
    """Get content of specific gold standard file"""
    filepath = GOLD_STANDARD_DIR / filename
    if filepath.exists():
        return jsonify(load_json_file(filepath))
    return jsonify({"error": "File not found"}), 404

@app.route('/api/results/<model_size>/<training_type>')
def get_results(model_size, training_type):
    """Get results for specific model and training type"""
    results = get_model_results(model_size, training_type)
    return jsonify(results)

@app.route('/api/compare/<model_size>/<result_folder>')
def compare_results(model_size, result_folder):
    """Compare pre-training, post-training, and gold standard for a specific result"""
    comparison = {
        "result_folder": result_folder,
        "model_size": model_size,
        "pre_training": None,
        "post_training": None,
        "gold_standard": None
    }
    
    # Get pre-training result
    pre_dir = PRE_TRAINING_DIR / f"{model_size}_base_model_results" / result_folder / "result.json"
    if pre_dir.exists():
        comparison["pre_training"] = load_json_file(pre_dir)
    
    # Get post-training result
    post_dir = POST_TRAINING_DIR / f"{model_size}_Fine_Tuning_results" / result_folder / "result.json"
    if post_dir.exists():
        comparison["post_training"] = load_json_file(post_dir)
    
    # Get matching gold standard
    comparison["gold_standard"] = match_gold_standard(result_folder)
    
    return jsonify(comparison)

@app.route('/api/performance_summary')
def get_performance_summary():
    """Get performance summary if available"""
    summary_file = BASE_DIR / "performance_summary.csv"
    if summary_file.exists():
        df = pd.read_csv(summary_file)
        return jsonify(df.to_dict(orient='records'))
    return jsonify([])

@app.route('/api/search', methods=['POST'])
def search_data():
    """Search across all data"""
    query = request.json.get('query', '').lower()
    results = []
    
    # Search in gold standard files
    for filepath in GOLD_STANDARD_DIR.glob("**/*.json"):
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read().lower()
            if query in content:
                results.append({
                    "type": "gold_standard",
                    "file": str(filepath.relative_to(BASE_DIR)),
                    "match": True
                })
    
    # Search in pre-training results
    for model in ["1.7B", "4B", "8B"]:
        model_dir = PRE_TRAINING_DIR / f"{model}_base_model_results"
        if model_dir.exists():
            for filepath in model_dir.glob("**/result.json"):
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read().lower()
                    if query in content:
                        results.append({
                            "type": "pre_training",
                            "model": model,
                            "file": str(filepath.relative_to(BASE_DIR)),
                            "match": True
                        })
    
    # Search in post-training results
    for model in ["1.7B", "4B", "8B"]:
        model_dir = POST_TRAINING_DIR / f"{model}_Fine_Tuning_results"
        if model_dir.exists():
            for filepath in model_dir.glob("**/result.json"):
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read().lower()
                    if query in content:
                        results.append({
                            "type": "post_training",
                            "model": model,
                            "file": str(filepath.relative_to(BASE_DIR)),
                            "match": True
                        })
    
    return jsonify(results[:100])  # Limit to 100 results

if __name__ == '__main__':
    # Create templates directory
    templates_dir = Path(__file__).parent / "templates"
    templates_dir.mkdir(exist_ok=True)
    
    print("="*60)
    print("Performance Evaluation Web Dashboard")
    print("="*60)
    print(f"Base Directory: {BASE_DIR}")
    print(f"Pre-Training: {PRE_TRAINING_DIR}")
    print(f"Post-Training: {POST_TRAINING_DIR}")
    print(f"Gold Standard: {GOLD_STANDARD_DIR}")
    print("="*60)
    print("Starting server at: http://localhost:5000")
    print("Press Ctrl+C to stop the server")
    print("="*60)
    
    app.run(debug=True, port=5000)