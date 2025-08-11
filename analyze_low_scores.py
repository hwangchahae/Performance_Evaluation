import json
import os
from pathlib import Path

def load_similarity_results(model_size, training_phase):
    """유사도 평가 결과 로드"""
    if training_phase == 'pre':
        file_path = f"Pre_Training/pre_similarity_results/{model_size}_pretrain_similarity_results.json"
    else:
        file_path = f"Post_Training/post_similarity_results/{model_size}_post_similarity_results.json"
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        print(f"Error loading {file_path}")
        return None

def get_bottom_10_files(results, metric='tfidf'):
    """하위 10개 파일 추출"""
    if not results:
        return []
    
    # 새로운 JSON 구조 처리
    if 'details' in results:
        if metric == 'tfidf':
            sorted_results = sorted(results['details'], key=lambda x: x.get('tfidf_cosine', 0))
        else:
            sorted_results = sorted(results['details'], key=lambda x: x.get('embedding_cosine', 0))
        
        # 데이터 형식 통일
        formatted_results = []
        for item in sorted_results[:10]:
            formatted_results.append({
                'file': item.get('file_name', ''),
                'scores': {
                    'tfidf': item.get('tfidf_cosine', 0),
                    'embedding': item.get('embedding_cosine', 0)
                }
            })
        return formatted_results
    
    # 이전 형식 처리 (results 키가 있는 경우)
    elif 'results' in results:
        if metric == 'tfidf':
            sorted_results = sorted(results['results'], key=lambda x: x['scores']['tfidf'])
        else:
            sorted_results = sorted(results['results'], key=lambda x: x['scores']['embedding'])
        return sorted_results[:10]
    
    return []

def load_json_content(file_path):
    """JSON 파일 내용 로드"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        return {"error": str(e)}

def get_file_paths(val_file_name, model_size):
    """각 파일의 경로 찾기"""
    paths = {}
    
    # val_XXX 형식에서 파일명 추출
    # 예: val_001_result_ES2014a -> ES2014a
    parts = val_file_name.split('_result_')
    if len(parts) > 1:
        base_name = parts[1]
    else:
        base_name = val_file_name
    
    # Gold Standard 경로
    gold_path = f"Gold_Standard_Data/{val_file_name}/result.json"
    if os.path.exists(gold_path):
        paths['gold'] = gold_path
    
    # Pre-Training 경로
    pre_base = f"Pre_Training/{model_size}_base_model_results"
    pre_path = f"{pre_base}/result_{base_name}/result.json"
    if os.path.exists(pre_path):
        paths['pre'] = pre_path
    
    # Post-Training 경로
    post_base = f"Post_Training/{model_size}_lora_model_results"
    post_path = f"{post_base}/result_{base_name}/result.json"
    if os.path.exists(post_path):
        paths['post'] = post_path
    
    return paths

def create_comparison_report():
    """비교 리포트 생성"""
    models = ['1.7B', '4B', '8B']
    metrics = ['tfidf', 'embedding']
    
    report = {}
    
    for model in models:
        report[model] = {}
        
        for phase in ['pre', 'post']:
            results = load_similarity_results(model, phase)
            if not results:
                continue
            
            for metric in metrics:
                key = f"{phase}_{metric}"
                bottom_10 = get_bottom_10_files(results, metric)
                report[model][key] = []
                
                for item in bottom_10:
                    file_name = item['file']
                    scores = item['scores']
                    paths = get_file_paths(file_name, model)
                    
                    file_info = {
                        'file': file_name,
                        'tfidf_score': scores['tfidf'],
                        'embedding_score': scores['embedding'],
                        'paths': paths
                    }
                    
                    # 실제 JSON 내용 로드 (첫 번째 파일만 샘플로)
                    if len(report[model][key]) == 0:  # 첫 번째 파일만
                        contents = {}
                        for data_type, path in paths.items():
                            contents[data_type] = load_json_content(path)
                        file_info['sample_contents'] = contents
                    
                    report[model][key].append(file_info)
    
    return report

def save_detailed_report():
    """상세 리포트 저장"""
    report = create_comparison_report()
    
    # JSON으로 저장
    with open('low_score_analysis.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    # HTML 리포트 생성
    html_content = """<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Low Score Analysis Report</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background: #f5f5f5;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
        }
        h2 {
            color: #555;
            margin-top: 30px;
        }
        h3 {
            color: #667eea;
            margin-top: 20px;
        }
        .model-section {
            margin: 30px 0;
            padding: 20px;
            background: #f9f9f9;
            border-radius: 8px;
        }
        .metric-section {
            margin: 20px 0;
            padding: 15px;
            background: white;
            border-left: 4px solid #667eea;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }
        th {
            background: #667eea;
            color: white;
            padding: 10px;
            text-align: left;
        }
        td {
            padding: 8px;
            border-bottom: 1px solid #ddd;
        }
        .score-low {
            color: #e74c3c;
            font-weight: bold;
        }
        .score-medium {
            color: #f39c12;
            font-weight: bold;
        }
        .file-paths {
            font-size: 12px;
            color: #666;
            margin-top: 5px;
        }
        .json-content {
            background: #f4f4f4;
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
            font-family: monospace;
            font-size: 12px;
            max-height: 300px;
            overflow-y: auto;
        }
        .expandable {
            cursor: pointer;
            background: #e8f0ff;
            padding: 5px;
            border-radius: 3px;
            margin: 5px 0;
        }
        .expandable:hover {
            background: #d0e2ff;
        }
        .content-preview {
            display: none;
            margin-top: 10px;
        }
        .badge {
            display: inline-block;
            padding: 3px 8px;
            border-radius: 3px;
            font-size: 11px;
            margin: 0 3px;
        }
        .badge-gold { background: #ffd700; color: #333; }
        .badge-pre { background: #74b9ff; color: white; }
        .badge-post { background: #a29bfe; color: white; }
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Low Score Analysis Report</h1>
        <p>각 모델별 하위 10개 파일의 상세 분석 리포트입니다.</p>
"""
    
    for model in ['1.7B', '4B', '8B']:
        html_content += f"""
        <div class="model-section">
            <h2>🔍 {model} Model Analysis</h2>
"""
        
        for phase in ['pre', 'post']:
            for metric in ['tfidf', 'embedding']:
                key = f"{phase}_{metric}"
                if key not in report.get(model, {}):
                    continue
                
                phase_label = "Pre-Training" if phase == "pre" else "Post-Training"
                metric_label = "TF-IDF" if metric == "tfidf" else "Embedding"
                
                html_content += f"""
            <div class="metric-section">
                <h3>{phase_label} - {metric_label} 하위 10개</h3>
                <table>
                    <thead>
                        <tr>
                            <th>순위</th>
                            <th>파일명</th>
                            <th>TF-IDF Score</th>
                            <th>Embedding Score</th>
                            <th>데이터 경로</th>
                        </tr>
                    </thead>
                    <tbody>
"""
                
                for idx, item in enumerate(report[model][key][:10], 1):
                    tfidf_class = "score-low" if item['tfidf_score'] < 0.15 else "score-medium"
                    embed_class = "score-low" if item['embedding_score'] < 0.85 else "score-medium"
                    
                    paths_html = ""
                    if 'paths' in item:
                        for data_type, path in item['paths'].items():
                            badge_class = f"badge-{data_type}"
                            paths_html += f'<span class="badge {badge_class}">{data_type.upper()}</span> '
                    
                    html_content += f"""
                        <tr>
                            <td>{idx}</td>
                            <td>{item['file']}</td>
                            <td class="{tfidf_class}">{item['tfidf_score']:.4f}</td>
                            <td class="{embed_class}">{item['embedding_score']:.4f}</td>
                            <td>{paths_html}</td>
                        </tr>
"""
                
                html_content += """
                    </tbody>
                </table>
            </div>
"""
        
        html_content += """
        </div>
"""
    
    html_content += """
        <div style="margin-top: 40px; padding: 20px; background: #e8f5ff; border-radius: 8px;">
            <h3>📁 데이터 파일 위치</h3>
            <p><strong>JSON 파일 내용 확인 방법:</strong></p>
            <ul>
                <li><strong>Gold Standard:</strong> Gold_Standard_Data/val_XXX_result_파일명/result.json</li>
                <li><strong>Pre-Training:</strong> Pre_Training/[모델크기]_base_model_results/result_파일명/result.json</li>
                <li><strong>Post-Training:</strong> Post_Training/[모델크기]_lora_model_results/result_파일명/result.json</li>
            </ul>
            <p style="margin-top: 15px;">
                <strong>상세 JSON 데이터:</strong> <code>low_score_analysis.json</code> 파일에서 확인 가능<br>
                <strong>CSV 데이터:</strong> 각 similarity_results 폴더의 CSV 파일에서 확인 가능
            </p>
        </div>
    </div>
</body>
</html>
"""
    
    with open('low_score_analysis_report.html', 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print("Analysis complete!")
    print("Generated files:")
    print("1. low_score_analysis.json - Detailed JSON data with file paths")
    print("2. low_score_analysis_report.html - Visual HTML report")
    
    # 샘플 출력
    print("\n=== Sample: Lowest TF-IDF scores for 4B Post-Training ===")
    if '4B' in report and 'post_tfidf' in report['4B']:
        for i, item in enumerate(report['4B']['post_tfidf'][:3], 1):
            print(f"\n{i}. {item['file']}")
            print(f"   TF-IDF: {item['tfidf_score']:.4f}, Embedding: {item['embedding_score']:.4f}")
            if 'paths' in item:
                for data_type, path in item['paths'].items():
                    print(f"   {data_type.upper()}: {path}")

if __name__ == "__main__":
    save_detailed_report()