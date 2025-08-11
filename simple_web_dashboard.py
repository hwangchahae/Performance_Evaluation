from flask import Flask, render_template_string, jsonify
import json
import os
from pathlib import Path
import glob

app = Flask(__name__)

# Configuration
BASE_DIR = Path(r"C:\Users\Playdata\Desktop\Performance_Evaluation")
PRE_TRAINING_DIR = BASE_DIR / "Pre_Training"
POST_TRAINING_DIR = BASE_DIR / "Post_Training"  
GOLD_STANDARD_DIR = BASE_DIR / "Gold_Standard_Data"

# HTML template embedded in Python
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Performance Evaluation Dashboard</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
        }

        .header {
            background: white;
            border-radius: 10px;
            padding: 30px;
            margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }

        .header h1 {
            color: #333;
            margin-bottom: 10px;
        }

        .header p {
            color: #666;
        }

        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }

        .stat-card {
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            transition: transform 0.3s;
        }

        .stat-card:hover {
            transform: translateY(-5px);
        }

        .stat-card h3 {
            color: #666;
            font-size: 14px;
            margin-bottom: 10px;
        }

        .stat-card .value {
            font-size: 32px;
            font-weight: bold;
            color: #667eea;
        }

        .main-content {
            display: grid;
            grid-template-columns: 300px 1fr;
            gap: 30px;
        }

        .sidebar {
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            max-height: 70vh;
            overflow-y: auto;
        }

        .content-area {
            background: white;
            border-radius: 10px;
            padding: 30px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            max-height: 70vh;
            overflow-y: auto;
        }

        .model-selector {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
        }

        .model-btn {
            padding: 8px 16px;
            border: 2px solid #667eea;
            background: white;
            color: #667eea;
            border-radius: 5px;
            cursor: pointer;
            transition: all 0.3s;
        }

        .model-btn.active {
            background: #667eea;
            color: white;
        }

        .file-list {
            list-style: none;
        }

        .file-item {
            padding: 10px;
            margin-bottom: 5px;
            background: #f8f9fa;
            border-radius: 5px;
            cursor: pointer;
            transition: all 0.3s;
            font-size: 12px;
            word-break: break-all;
        }

        .file-item:hover {
            background: #e9ecef;
            transform: translateX(5px);
        }

        .json-viewer {
            background: #f8f9fa;
            border-radius: 5px;
            padding: 20px;
            overflow-x: auto;
            font-family: 'Courier New', monospace;
            font-size: 14px;
            max-height: 500px;
            overflow-y: auto;
        }

        .info-message {
            background: #d1ecf1;
            color: #0c5460;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔬 Performance Evaluation Dashboard</h1>
            <p>Compare Pre-Training, Post-Training, and Gold Standard Data</p>
        </div>

        <div class="stats-grid" id="statsGrid">
            <div class="stat-card">
                <h3>Gold Standard Files</h3>
                <div class="value" id="goldCount">-</div>
            </div>
            <div class="stat-card">
                <h3>Pre-Training (1.7B)</h3>
                <div class="value" id="pre17Count">-</div>
            </div>
            <div class="stat-card">
                <h3>Post-Training (1.7B)</h3>
                <div class="value" id="post17Count">-</div>
            </div>
        </div>

        <div class="main-content">
            <div class="sidebar">
                <h3>Select Model</h3>
                <div class="model-selector">
                    <button class="model-btn active" onclick="loadModel('1.7B')">1.7B</button>
                    <button class="model-btn" onclick="loadModel('4B')">4B</button>
                    <button class="model-btn" onclick="loadModel('8B')">8B</button>
                </div>
                <h3>Files</h3>
                <ul class="file-list" id="fileList">
                    <li class="file-item">Loading...</li>
                </ul>
            </div>

            <div class="content-area">
                <div class="info-message">
                    <strong>사용 방법:</strong><br>
                    1. 왼쪽에서 모델 선택 (1.7B, 4B, 8B)<br>
                    2. 파일 목록에서 확인하고 싶은 파일 클릭<br>
                    3. 아래에서 데이터 확인
                </div>
                <div id="contentDisplay">
                    <div class="json-viewer">
                        <p>파일을 선택하면 여기에 데이터가 표시됩니다.</p>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let currentModel = '1.7B';

        // Load statistics on page load
        window.onload = function() {
            loadStats();
            loadModel('1.7B');
        };

        function loadStats() {
            fetch('/api/stats')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('goldCount').textContent = data.gold_standard;
                    document.getElementById('pre17Count').textContent = data.pre_training_17B;
                    document.getElementById('post17Count').textContent = data.post_training_17B;
                })
                .catch(error => console.error('Error loading stats:', error));
        }

        function loadModel(model) {
            currentModel = model;
            
            // Update button styles
            document.querySelectorAll('.model-btn').forEach(btn => {
                btn.classList.remove('active');
                if (btn.textContent === model) {
                    btn.classList.add('active');
                }
            });

            // Load file list for selected model
            fetch(`/api/files/${model}`)
                .then(response => response.json())
                .then(data => {
                    const fileList = document.getElementById('fileList');
                    if (data.length === 0) {
                        fileList.innerHTML = '<li class="file-item">No files found</li>';
                    } else {
                        fileList.innerHTML = data.map(file => 
                            `<li class="file-item" onclick="loadFile('${model}', '${file}')">${file}</li>`
                        ).join('');
                    }
                })
                .catch(error => {
                    console.error('Error loading files:', error);
                    document.getElementById('fileList').innerHTML = '<li class="file-item">Error loading files</li>';
                });
        }

        function loadFile(model, filename) {
            fetch(`/api/data/${model}/${filename}`)
                .then(response => response.json())
                .then(data => {
                    const display = document.getElementById('contentDisplay');
                    display.innerHTML = `
                        <h3>File: ${filename}</h3>
                        <div class="json-viewer">
                            <pre>${JSON.stringify(data, null, 2)}</pre>
                        </div>
                    `;
                })
                .catch(error => {
                    console.error('Error loading file:', error);
                    document.getElementById('contentDisplay').innerHTML = 
                        '<div class="json-viewer">Error loading file data</div>';
                });
        }
    </script>
</body>
</html>
"""

def load_json_file(filepath):
    """Load and return JSON file content"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        return {"error": str(e)}

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/stats')
def get_stats():
    """Get basic statistics"""
    stats = {
        "gold_standard": len(list(GOLD_STANDARD_DIR.glob("**/*.json"))) if GOLD_STANDARD_DIR.exists() else 0,
        "pre_training_17B": 0,
        "post_training_17B": 0
    }
    
    # Count pre-training files
    pre_17b_dir = PRE_TRAINING_DIR / "1.7B_base_model_results"
    if pre_17b_dir.exists():
        stats["pre_training_17B"] = len(list(pre_17b_dir.glob("**/result.json")))
    
    # Count post-training files
    post_17b_dir = POST_TRAINING_DIR / "1.7B_Fine_Tuning_results"
    if post_17b_dir.exists():
        stats["post_training_17B"] = len(list(post_17b_dir.glob("**/result.json")))
    
    return jsonify(stats)

@app.route('/api/files/<model>')
def get_files(model):
    """Get list of files for a specific model"""
    files = []
    
    # Check pre-training directory
    pre_dir = PRE_TRAINING_DIR / f"{model}_base_model_results"
    if pre_dir.exists():
        for filepath in pre_dir.glob("**/result.json"):
            folder_name = filepath.parent.name
            files.append(folder_name)
    
    return jsonify(sorted(files[:50]))  # Limit to first 50 files

@app.route('/api/data/<model>/<filename>')
def get_data(model, filename):
    """Get data for a specific file"""
    # Try pre-training first
    pre_path = PRE_TRAINING_DIR / f"{model}_base_model_results" / filename / "result.json"
    if pre_path.exists():
        return jsonify(load_json_file(pre_path))
    
    # Try post-training
    post_path = POST_TRAINING_DIR / f"{model}_Fine_Tuning_results" / filename / "result.json"
    if post_path.exists():
        return jsonify(load_json_file(post_path))
    
    return jsonify({"error": "File not found"}), 404

if __name__ == '__main__':
    print("="*60)
    print("Performance Evaluation Simple Dashboard")
    print("="*60)
    print("Server starting at: http://localhost:5000")
    print("Press Ctrl+C to stop")
    print("="*60)
    
    # Run without debug mode for stability
    app.run(host='127.0.0.1', port=5000, debug=False)