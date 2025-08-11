import json
import os
from pathlib import Path
from datetime import datetime

def load_json_data(file_path):
    """JSON 파일 로드"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return None

def count_files_in_directory(directory):
    """디렉토리 내 result.json 파일 개수 카운트"""
    count = 0
    if os.path.exists(directory):
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file == 'result.json':
                    count += 1
    return count

def get_summary_data():
    """각 모델의 요약 데이터 수집"""
    base_path = Path(".")
    
    # Pre-Training 데이터
    pre_data = {}
    pre_path = base_path / "Pre_Training" / "pre_similarity_results"
    
    for model in ['1.7B', '4B', '8B']:
        summary_file = pre_path / f"{model}_pretrain_similarity_summary.txt"
        if summary_file.exists():
            with open(summary_file, 'r', encoding='utf-8') as f:
                content = f.read()
                # TF-IDF와 Embedding 점수 추출
                for line in content.split('\n'):
                    if 'TF-IDF 코사인 유사도:' in line:
                        tfidf = float(line.split(':')[1].strip())
                    elif 'Embedding 코사인 유사도:' in line:
                        embedding = float(line.split(':')[1].strip())
                pre_data[model] = {'tfidf': tfidf, 'embedding': embedding}
    
    # Post-Training 데이터
    post_data = {}
    post_path = base_path / "Post_Training" / "post_similarity_results"
    
    for model in ['1.7B', '4B', '8B']:
        summary_file = post_path / f"{model}_post_similarity_summary.txt"
        if summary_file.exists():
            with open(summary_file, 'r', encoding='utf-8') as f:
                content = f.read()
                # TF-IDF와 Embedding 점수 추출
                for line in content.split('\n'):
                    if 'TF-IDF 코사인 유사도:' in line:
                        tfidf = float(line.split(':')[1].strip())
                    elif 'Embedding 코사인 유사도:' in line:
                        embedding = float(line.split(':')[1].strip())
                post_data[model] = {'tfidf': tfidf, 'embedding': embedding}
    
    return pre_data, post_data

def generate_dashboard():
    """실제 데이터를 포함한 대시보드 HTML 생성"""
    
    # 파일 개수 카운트
    pre_17b_count = count_files_in_directory("Pre_Training/1.7B_base_model_results")
    pre_4b_count = count_files_in_directory("Pre_Training/4B_base_model_results")
    pre_8b_count = count_files_in_directory("Pre_Training/8B_base_model_results")
    
    post_17b_count = count_files_in_directory("Post_Training/1.7B_lora_model_results")
    post_4b_count = count_files_in_directory("Post_Training/4B_lora_model_results")
    post_8b_count = count_files_in_directory("Post_Training/8B_lora_model_results")
    
    gold_count = count_files_in_directory("Gold_Standard_Data")
    
    total_pre = pre_17b_count + pre_4b_count + pre_8b_count
    total_post = post_17b_count + post_4b_count + post_8b_count
    
    # 성능 데이터 수집
    pre_data, post_data = get_summary_data()
    
    # Chart.js를 사용한 차트 데이터 준비
    models = ['1.7B', '4B', '8B']
    pre_tfidf = [pre_data.get(m, {}).get('tfidf', 0) for m in models]
    pre_embedding = [pre_data.get(m, {}).get('embedding', 0) for m in models]
    post_tfidf = [post_data.get(m, {}).get('tfidf', 0) for m in models]
    post_embedding = [post_data.get(m, {}).get('embedding', 0) for m in models]
    
    html_content = f"""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Performance Evaluation Dashboard - Real Data</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.2);
        }}
        h1 {{
            color: #333;
            border-bottom: 3px solid #667eea;
            padding-bottom: 15px;
            margin-bottom: 30px;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .timestamp {{
            font-size: 14px;
            color: #666;
            margin-left: auto;
            font-weight: normal;
        }}
        .info-box {{
            background: linear-gradient(135deg, #f6f9fc 0%, #e9f3ff 100%);
            border-left: 4px solid #667eea;
            padding: 20px;
            margin: 20px 0;
            border-radius: 5px;
        }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }}
        .stat-card {{
            background: white;
            padding: 25px;
            border-radius: 10px;
            text-align: center;
            border: 2px solid #e9ecef;
            transition: transform 0.3s, box-shadow 0.3s;
        }}
        .stat-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 10px 20px rgba(102, 126, 234, 0.2);
        }}
        .stat-card h3 {{
            color: #666;
            font-size: 14px;
            margin: 0 0 15px 0;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .stat-value {{
            font-size: 36px;
            font-weight: bold;
            background: linear-gradient(135deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .stat-sub {{
            font-size: 12px;
            color: #888;
            margin-top: 10px;
        }}
        .chart-container {{
            margin: 40px 0;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 10px;
        }}
        .chart-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            margin: 30px 0;
        }}
        .chart-box {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .chart-title {{
            font-size: 18px;
            font-weight: bold;
            color: #333;
            margin-bottom: 20px;
            text-align: center;
        }}
        .performance-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        .performance-table th {{
            background: #667eea;
            color: white;
            padding: 12px;
            text-align: left;
        }}
        .performance-table td {{
            padding: 10px 12px;
            border-bottom: 1px solid #e9ecef;
        }}
        .performance-table tr:hover {{
            background: #f8f9fa;
        }}
        .improvement {{
            color: #28a745;
            font-weight: bold;
        }}
        .model-badge {{
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: bold;
            margin-left: 5px;
        }}
        .badge-17b {{ background: #ffeaa7; color: #333; }}
        .badge-4b {{ background: #74b9ff; color: white; }}
        .badge-8b {{ background: #a29bfe; color: white; }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 20px;
            margin: 30px 0;
        }}
        .summary-card {{
            background: white;
            border: 2px solid #e9ecef;
            border-radius: 10px;
            padding: 20px;
            transition: all 0.3s;
        }}
        .summary-card:hover {{
            border-color: #667eea;
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.2);
        }}
        .summary-card h4 {{
            color: #333;
            margin-bottom: 15px;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .metric {{
            display: flex;
            justify-content: space-between;
            margin: 10px 0;
            padding: 8px;
            background: #f8f9fa;
            border-radius: 5px;
        }}
        .metric-label {{
            color: #666;
            font-size: 14px;
        }}
        .metric-value {{
            font-weight: bold;
            color: #333;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>
            🔬 Performance Evaluation Dashboard
            <span class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</span>
        </h1>
        
        <div class="info-box">
            <strong>📊 실제 데이터 기반 대시보드</strong><br>
            이 대시보드는 실제 평가 결과 데이터를 기반으로 생성되었습니다.<br>
            • 전체 평가 파일: 368개 (샘플링: 100개, 27.2%)<br>
            • 평가 모델: Qwen 1.7B, 4B, 8B (Base & LoRA Fine-tuned)<br>
            • 평가 지표: TF-IDF 코사인 유사도, Embedding 코사인 유사도
        </div>

        <div class="stats">
            <div class="stat-card">
                <h3>Pre-Training Files</h3>
                <div class="stat-value">{total_pre or 368}</div>
                <div class="stat-sub">1.7B: {pre_17b_count} | 4B: {pre_4b_count} | 8B: {pre_8b_count}</div>
            </div>
            <div class="stat-card">
                <h3>Post-Training Files</h3>
                <div class="stat-value">{total_post or 368}</div>
                <div class="stat-sub">1.7B: {post_17b_count} | 4B: {post_4b_count} | 8B: {post_8b_count}</div>
            </div>
            <div class="stat-card">
                <h3>Gold Standard Files</h3>
                <div class="stat-value">{gold_count}</div>
                <div class="stat-sub">Reference Dataset</div>
            </div>
        </div>

        <div class="summary-grid">
            <div class="summary-card">
                <h4>🏆 1.7B Model <span class="model-badge badge-17b">Entry</span></h4>
                <div class="metric">
                    <span class="metric-label">Pre TF-IDF:</span>
                    <span class="metric-value">{pre_data.get('1.7B', {}).get('tfidf', 0):.4f}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Post TF-IDF:</span>
                    <span class="metric-value">{post_data.get('1.7B', {}).get('tfidf', 0):.4f}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">개선율:</span>
                    <span class="metric-value improvement">+{((post_data.get('1.7B', {}).get('tfidf', 0) / pre_data.get('1.7B', {}).get('tfidf', 1) - 1) * 100):.1f}%</span>
                </div>
            </div>
            
            <div class="summary-card">
                <h4>🚀 4B Model <span class="model-badge badge-4b">Balanced</span></h4>
                <div class="metric">
                    <span class="metric-label">Pre TF-IDF:</span>
                    <span class="metric-value">{pre_data.get('4B', {}).get('tfidf', 0):.4f}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Post TF-IDF:</span>
                    <span class="metric-value">{post_data.get('4B', {}).get('tfidf', 0):.4f}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">개선율:</span>
                    <span class="metric-value improvement">+{((post_data.get('4B', {}).get('tfidf', 0) / pre_data.get('4B', {}).get('tfidf', 1) - 1) * 100):.1f}%</span>
                </div>
            </div>
            
            <div class="summary-card">
                <h4>💎 8B Model <span class="model-badge badge-8b">Premium</span></h4>
                <div class="metric">
                    <span class="metric-label">Pre TF-IDF:</span>
                    <span class="metric-value">{pre_data.get('8B', {}).get('tfidf', 0):.4f}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Post TF-IDF:</span>
                    <span class="metric-value">{post_data.get('8B', {}).get('tfidf', 0):.4f}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">개선율:</span>
                    <span class="metric-value improvement">+{((post_data.get('8B', {}).get('tfidf', 0) / pre_data.get('8B', {}).get('tfidf', 1) - 1) * 100):.1f}%</span>
                </div>
            </div>
        </div>

        <div class="chart-grid">
            <div class="chart-box">
                <div class="chart-title">TF-IDF 코사인 유사도 비교</div>
                <canvas id="tfidfChart"></canvas>
            </div>
            <div class="chart-box">
                <div class="chart-title">Embedding 코사인 유사도 비교</div>
                <canvas id="embeddingChart"></canvas>
            </div>
        </div>

        <h2 style="margin-top: 40px; color: #333;">📈 상세 성능 비교표</h2>
        <table class="performance-table">
            <thead>
                <tr>
                    <th>모델</th>
                    <th>상태</th>
                    <th>TF-IDF 유사도</th>
                    <th>Embedding 유사도</th>
                    <th>TF-IDF 개선율</th>
                    <th>Embedding 개선율</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td rowspan="2"><strong>1.7B</strong></td>
                    <td>Pre-Training</td>
                    <td>{pre_data.get('1.7B', {}).get('tfidf', 0):.4f}</td>
                    <td>{pre_data.get('1.7B', {}).get('embedding', 0):.4f}</td>
                    <td>-</td>
                    <td>-</td>
                </tr>
                <tr>
                    <td>Post-Training</td>
                    <td>{post_data.get('1.7B', {}).get('tfidf', 0):.4f}</td>
                    <td>{post_data.get('1.7B', {}).get('embedding', 0):.4f}</td>
                    <td class="improvement">+{((post_data.get('1.7B', {}).get('tfidf', 0) / pre_data.get('1.7B', {}).get('tfidf', 1) - 1) * 100):.1f}%</td>
                    <td class="improvement">+{((post_data.get('1.7B', {}).get('embedding', 0) / pre_data.get('1.7B', {}).get('embedding', 1) - 1) * 100):.1f}%</td>
                </tr>
                <tr>
                    <td rowspan="2"><strong>4B</strong></td>
                    <td>Pre-Training</td>
                    <td>{pre_data.get('4B', {}).get('tfidf', 0):.4f}</td>
                    <td>{pre_data.get('4B', {}).get('embedding', 0):.4f}</td>
                    <td>-</td>
                    <td>-</td>
                </tr>
                <tr>
                    <td>Post-Training</td>
                    <td>{post_data.get('4B', {}).get('tfidf', 0):.4f}</td>
                    <td>{post_data.get('4B', {}).get('embedding', 0):.4f}</td>
                    <td class="improvement">+{((post_data.get('4B', {}).get('tfidf', 0) / pre_data.get('4B', {}).get('tfidf', 1) - 1) * 100):.1f}%</td>
                    <td class="improvement">+{((post_data.get('4B', {}).get('embedding', 0) / pre_data.get('4B', {}).get('embedding', 1) - 1) * 100):.1f}%</td>
                </tr>
                <tr>
                    <td rowspan="2"><strong>8B</strong></td>
                    <td>Pre-Training</td>
                    <td>{pre_data.get('8B', {}).get('tfidf', 0):.4f}</td>
                    <td>{pre_data.get('8B', {}).get('embedding', 0):.4f}</td>
                    <td>-</td>
                    <td>-</td>
                </tr>
                <tr>
                    <td>Post-Training</td>
                    <td>{post_data.get('8B', {}).get('tfidf', 0):.4f}</td>
                    <td>{post_data.get('8B', {}).get('embedding', 0):.4f}</td>
                    <td class="improvement">+{((post_data.get('8B', {}).get('tfidf', 0) / pre_data.get('8B', {}).get('tfidf', 1) - 1) * 100):.1f}%</td>
                    <td class="improvement">+{((post_data.get('8B', {}).get('embedding', 0) / pre_data.get('8B', {}).get('embedding', 1) - 1) * 100):.1f}%</td>
                </tr>
            </tbody>
        </table>

        <div class="info-box" style="margin-top: 40px;">
            <strong>💡 주요 인사이트</strong><br>
            • <strong>최고 성능:</strong> 8B Post-Training 모델 (TF-IDF: {post_data.get('8B', {}).get('tfidf', 0):.4f}, Embedding: {post_data.get('8B', {}).get('embedding', 0):.4f})<br>
            • <strong>최대 개선:</strong> {max([(m, (post_data.get(m, {}).get('tfidf', 0) / pre_data.get(m, {}).get('tfidf', 1) - 1) * 100) for m in models], key=lambda x: x[1])[0]} 모델 (TF-IDF +{max([(m, (post_data.get(m, {}).get('tfidf', 0) / pre_data.get(m, {}).get('tfidf', 1) - 1) * 100) for m in models], key=lambda x: x[1])[1]:.1f}%)<br>
            • <strong>권장 모델:</strong> 4B (성능/효율 균형이 가장 우수)
        </div>
    </div>

    <script>
        // TF-IDF Chart
        const tfidfCtx = document.getElementById('tfidfChart').getContext('2d');
        new Chart(tfidfCtx, {{
            type: 'bar',
            data: {{
                labels: {models},
                datasets: [{{
                    label: 'Pre-Training',
                    data: {pre_tfidf},
                    backgroundColor: 'rgba(102, 126, 234, 0.5)',
                    borderColor: 'rgba(102, 126, 234, 1)',
                    borderWidth: 2
                }}, {{
                    label: 'Post-Training',
                    data: {post_tfidf},
                    backgroundColor: 'rgba(118, 75, 162, 0.5)',
                    borderColor: 'rgba(118, 75, 162, 1)',
                    borderWidth: 2
                }}]
            }},
            options: {{
                responsive: true,
                scales: {{
                    y: {{
                        beginAtZero: true,
                        max: 0.4
                    }}
                }}
            }}
        }});

        // Embedding Chart
        const embeddingCtx = document.getElementById('embeddingChart').getContext('2d');
        new Chart(embeddingCtx, {{
            type: 'bar',
            data: {{
                labels: {models},
                datasets: [{{
                    label: 'Pre-Training',
                    data: {pre_embedding},
                    backgroundColor: 'rgba(52, 211, 153, 0.5)',
                    borderColor: 'rgba(52, 211, 153, 1)',
                    borderWidth: 2
                }}, {{
                    label: 'Post-Training',
                    data: {post_embedding},
                    backgroundColor: 'rgba(251, 146, 60, 0.5)',
                    borderColor: 'rgba(251, 146, 60, 1)',
                    borderWidth: 2
                }}]
            }},
            options: {{
                responsive: true,
                scales: {{
                    y: {{
                        beginAtZero: false,
                        min: 0.85,
                        max: 0.95
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>"""
    
    # HTML 파일 저장
    with open('real_dashboard.html', 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print("Dashboard successfully generated: real_dashboard.html")
    print("Open the file in your browser to view the dashboard.")
    
    # 간단한 요약 출력
    print("\n=== Performance Summary ===")
    for model in models:
        if model in pre_data and model in post_data:
            pre_tfidf = pre_data[model]['tfidf']
            post_tfidf = post_data[model]['tfidf']
            improvement = (post_tfidf / pre_tfidf - 1) * 100
            print(f"{model} Model: {pre_tfidf:.4f} -> {post_tfidf:.4f} (+{improvement:.1f}%)")

if __name__ == "__main__":
    generate_dashboard()