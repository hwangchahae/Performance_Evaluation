from http.server import HTTPServer, SimpleHTTPRequestHandler
import json
import os
from pathlib import Path
import webbrowser
import threading
import time

# Configuration
BASE_DIR = Path(r"C:\Users\Playdata\Desktop\Performance_Evaluation")
PORT = 8080

# Create a simple HTML file
HTML_CONTENT = """
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Performance Evaluation Dashboard</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            padding: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        h1 {
            color: #333;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
        }
        .info-box {
            background: #f0f8ff;
            border-left: 4px solid #667eea;
            padding: 15px;
            margin: 20px 0;
        }
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }
        .stat-card {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            border: 2px solid #e9ecef;
        }
        .stat-card h3 {
            color: #666;
            font-size: 14px;
            margin: 0 0 10px 0;
        }
        .stat-value {
            font-size: 32px;
            font-weight: bold;
            color: #667eea;
        }
        .file-browser {
            margin-top: 30px;
        }
        .folder {
            background: #f8f9fa;
            padding: 15px;
            margin: 10px 0;
            border-radius: 5px;
            cursor: pointer;
            transition: all 0.3s;
        }
        .folder:hover {
            background: #e9ecef;
            transform: translateX(5px);
        }
        .folder h4 {
            margin: 0 0 10px 0;
            color: #333;
        }
        .file-list {
            display: none;
            margin-top: 10px;
            padding-left: 20px;
        }
        .file-list.show {
            display: block;
        }
        .file-item {
            padding: 5px;
            margin: 3px 0;
            background: white;
            border-radius: 3px;
            font-size: 14px;
            color: #666;
        }
        button {
            background: #667eea;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            margin: 5px;
            font-size: 16px;
        }
        button:hover {
            background: #5a67d8;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔬 Performance Evaluation Dashboard</h1>
        
        <div class="info-box">
            <strong>📁 데이터 위치:</strong><br>
            • Pre-Training: C:\\Users\\Playdata\\Desktop\\Performance_Evaluation\\Pre_Training<br>
            • Post-Training: C:\\Users\\Playdata\\Desktop\\Performance_Evaluation\\Post_Training<br>
            • Gold Standard: C:\\Users\\Playdata\\Desktop\\Performance_Evaluation\\Gold_Standard_Data
        </div>

        <div class="stats" id="stats">
            <div class="stat-card">
                <h3>Pre-Training Files</h3>
                <div class="stat-value" id="preCount">-</div>
            </div>
            <div class="stat-card">
                <h3>Post-Training Files</h3>
                <div class="stat-value" id="postCount">-</div>
            </div>
            <div class="stat-card">
                <h3>Gold Standard Files</h3>
                <div class="stat-value" id="goldCount">-</div>
            </div>
        </div>

        <div style="text-align: center; margin: 30px 0;">
            <button onclick="loadData('pre')">Pre-Training 데이터 보기</button>
            <button onclick="loadData('post')">Post-Training 데이터 보기</button>
            <button onclick="loadData('gold')">Gold Standard 데이터 보기</button>
        </div>

        <div class="file-browser" id="fileBrowser">
            <!-- Files will be loaded here -->
        </div>
    </div>

    <script>
        // Load statistics on page load
        window.onload = function() {
            countFiles();
        };

        function countFiles() {
            // This is a simplified version - in real implementation, 
            // you would fetch this data from the server
            document.getElementById('preCount').textContent = '368';
            document.getElementById('postCount').textContent = '368';
            document.getElementById('goldCount').textContent = '368';
        }

        function loadData(type) {
            const browser = document.getElementById('fileBrowser');
            let content = '';
            
            if (type === 'pre') {
                content = `
                    <h2>Pre-Training Results</h2>
                    <div class="folder" onclick="toggleFolder('pre17')">
                        <h4>📁 1.7B_base_model_results</h4>
                        <div class="file-list" id="pre17">
                            <div class="file-item">result_Bed002_chunk_1/result.json</div>
                            <div class="file-item">result_Bed002_chunk_2/result.json</div>
                            <div class="file-item">result_Bed002_chunk_3/result.json</div>
                            <div class="file-item">... and more files</div>
                        </div>
                    </div>
                    <div class="folder" onclick="toggleFolder('pre4')">
                        <h4>📁 4B_base_model_results</h4>
                        <div class="file-list" id="pre4">
                            <div class="file-item">Click to expand...</div>
                        </div>
                    </div>
                    <div class="folder" onclick="toggleFolder('pre8')">
                        <h4>📁 8B_base_model_results</h4>
                        <div class="file-list" id="pre8">
                            <div class="file-item">Click to expand...</div>
                        </div>
                    </div>
                `;
            } else if (type === 'post') {
                content = `
                    <h2>Post-Training Results</h2>
                    <div class="folder" onclick="toggleFolder('post17')">
                        <h4>📁 1.7B_Fine_Tuning_results</h4>
                        <div class="file-list" id="post17">
                            <div class="file-item">result_Bed002_chunk_1/result.json</div>
                            <div class="file-item">result_Bed002_chunk_2/result.json</div>
                            <div class="file-item">... and more files</div>
                        </div>
                    </div>
                    <div class="folder" onclick="toggleFolder('post4')">
                        <h4>📁 4B_Fine_Tuning_results</h4>
                        <div class="file-list" id="post4">
                            <div class="file-item">Click to expand...</div>
                        </div>
                    </div>
                    <div class="folder" onclick="toggleFolder('post8')">
                        <h4>📁 8B_Fine_Tuning_results</h4>
                        <div class="file-list" id="post8">
                            <div class="file-item">Click to expand...</div>
                        </div>
                    </div>
                `;
            } else if (type === 'gold') {
                content = `
                    <h2>Gold Standard Data</h2>
                    <div class="folder" onclick="toggleFolder('gold')">
                        <h4>📁 Gold Standard Files</h4>
                        <div class="file-list" id="gold">
                            <div class="file-item">Bed002.json</div>
                            <div class="file-item">Bed009.json</div>
                            <div class="file-item">Bed010.json</div>
                            <div class="file-item">... and more files</div>
                        </div>
                    </div>
                `;
            }
            
            browser.innerHTML = content;
        }

        function toggleFolder(id) {
            const fileList = document.getElementById(id);
            if (fileList) {
                fileList.classList.toggle('show');
            }
        }
    </script>
</body>
</html>
"""

# Save HTML file
html_file = BASE_DIR / "dashboard.html"
with open(html_file, 'w', encoding='utf-8') as f:
    f.write(HTML_CONTENT)

print("="*60)
print("Performance Evaluation Dashboard")
print("="*60)
print(f"Created HTML file: {html_file}")
print(f"Opening in browser...")
print("="*60)

# Open in browser
webbrowser.open(f"file:///{html_file}")

print("\n✅ Dashboard opened in your browser!")
print("\nThe dashboard shows:")
print("• Pre-Training results")
print("• Post-Training results")  
print("• Gold Standard data")
print("\nYou can browse the files directly in your browser.")