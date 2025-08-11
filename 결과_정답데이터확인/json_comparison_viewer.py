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
        return None

def get_bottom_files(results, metric='tfidf', count=10):
    """하위 파일 추출"""
    if not results or 'details' not in results:
        return []
    
    if metric == 'tfidf':
        sorted_results = sorted(results['details'], key=lambda x: x.get('tfidf_cosine', 0))
    else:
        sorted_results = sorted(results['details'], key=lambda x: x.get('embedding_cosine', 0))
    
    return sorted_results[:count]

def get_top_files(results, metric='tfidf', count=10):
    """상위 파일 추출"""
    if not results or 'details' not in results:
        return []
    
    if metric == 'tfidf':
        sorted_results = sorted(results['details'], key=lambda x: x.get('tfidf_cosine', 0), reverse=True)
    else:
        sorted_results = sorted(results['details'], key=lambda x: x.get('embedding_cosine', 0), reverse=True)
    
    return sorted_results[:count]

def load_json_content(file_path):
    """JSON 파일 내용 로드"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        return {"error": f"파일을 읽을 수 없습니다: {str(e)}"}

def get_file_paths(val_file_name, model_size):
    """각 파일의 경로 찾기"""
    paths = {}
    
    # val_XXX_result_파일명 형식에서 파일명 추출
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

def create_comparison_viewer():
    """JSON 비교 뷰어 생성"""
    
    # 각 모델의 하위/상위 성능 파일 수집
    comparison_data = {}
    
    models = ['1.7B', '4B', '8B']
    for model in models:
        comparison_data[model] = {
            'tfidf_bottom': {},
            'tfidf_top': {},
            'embedding_bottom': {},
            'embedding_top': {}
        }
        
        # Post-Training 결과 기준으로 파일 선택
        post_results = load_similarity_results(model, 'post')
        if not post_results:
            continue
        
        # TF-IDF 기준 하위 10개
        bottom_tfidf = get_bottom_files(post_results, 'tfidf', 10)
        
        for idx, item in enumerate(bottom_tfidf):
            file_name = item.get('file_name', '')
            scores = {
                'tfidf': item.get('tfidf_cosine', 0),
                'embedding': item.get('embedding_cosine', 0)
            }
            
            # 파일 경로 찾기
            paths = get_file_paths(file_name, model)
            
            # 실제 JSON 내용 로드
            contents = {}
            for data_type, path in paths.items():
                contents[data_type] = load_json_content(path)
            
            comparison_data[model]['tfidf_bottom'][f"file_{idx}"] = {
                'name': file_name,
                'scores': scores,
                'paths': paths,
                'contents': contents
            }
        
        # TF-IDF 기준 상위 10개
        top_tfidf = get_top_files(post_results, 'tfidf', 10)
        
        for idx, item in enumerate(top_tfidf):
            file_name = item.get('file_name', '')
            scores = {
                'tfidf': item.get('tfidf_cosine', 0),
                'embedding': item.get('embedding_cosine', 0)
            }
            
            # 파일 경로 찾기
            paths = get_file_paths(file_name, model)
            
            # 실제 JSON 내용 로드
            contents = {}
            for data_type, path in paths.items():
                contents[data_type] = load_json_content(path)
            
            comparison_data[model]['tfidf_top'][f"file_{idx}"] = {
                'name': file_name,
                'scores': scores,
                'paths': paths,
                'contents': contents
            }
        
        # Embedding 기준 하위 10개
        bottom_embedding = get_bottom_files(post_results, 'embedding', 10)
        
        for idx, item in enumerate(bottom_embedding):
            file_name = item.get('file_name', '')
            scores = {
                'tfidf': item.get('tfidf_cosine', 0),
                'embedding': item.get('embedding_cosine', 0)
            }
            
            # 파일 경로 찾기
            paths = get_file_paths(file_name, model)
            
            # 실제 JSON 내용 로드
            contents = {}
            for data_type, path in paths.items():
                contents[data_type] = load_json_content(path)
            
            comparison_data[model]['embedding_bottom'][f"file_{idx}"] = {
                'name': file_name,
                'scores': scores,
                'paths': paths,
                'contents': contents
            }
        
        # Embedding 기준 상위 10개
        top_embedding = get_top_files(post_results, 'embedding', 10)
        
        for idx, item in enumerate(top_embedding):
            file_name = item.get('file_name', '')
            scores = {
                'tfidf': item.get('tfidf_cosine', 0),
                'embedding': item.get('embedding_cosine', 0)
            }
            
            # 파일 경로 찾기
            paths = get_file_paths(file_name, model)
            
            # 실제 JSON 내용 로드
            contents = {}
            for data_type, path in paths.items():
                contents[data_type] = load_json_content(path)
            
            comparison_data[model]['embedding_top'][f"file_{idx}"] = {
                'name': file_name,
                'scores': scores,
                'paths': paths,
                'contents': contents
            }
    
    # HTML 생성
    html_content = """<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>JSON Content Comparison Viewer</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #f0f2f5;
            padding: 20px;
        }
        .header {
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            font-size: 28px;
            margin-bottom: 10px;
        }
        .model-selector {
            display: flex;
            gap: 20px;
            margin: 20px 0;
            flex-wrap: wrap;
        }
        .model-btn {
            padding: 12px 24px;
            background: #667eea;
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            cursor: pointer;
            transition: all 0.3s;
        }
        .model-btn:hover {
            background: #5a67d8;
            transform: translateY(-2px);
        }
        .model-btn.active {
            background: #48bb78;
        }
        .file-selector {
            display: flex;
            gap: 10px;
            margin: 20px 0;
            flex-wrap: wrap;
        }
        .file-btn {
            padding: 8px 16px;
            background: #e2e8f0;
            color: #333;
            border: 2px solid transparent;
            border-radius: 6px;
            cursor: pointer;
            transition: all 0.3s;
            font-size: 14px;
        }
        .file-btn:hover {
            background: #cbd5e0;
        }
        .file-btn.active {
            background: #667eea;
            color: white;
            border-color: #5a67d8;
        }
        .scores-info {
            background: #fef5e7;
            padding: 15px;
            border-radius: 8px;
            margin: 20px 0;
            border-left: 4px solid #f39c12;
        }
        .comparison-container {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 20px;
            margin-top: 20px;
        }
        .json-panel {
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            max-height: 70vh;
            overflow-y: auto;
        }
        .panel-header {
            font-weight: bold;
            font-size: 18px;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 2px solid #e2e8f0;
        }
        .gold-header {
            color: #f39c12;
            border-bottom-color: #f39c12;
        }
        .pre-header {
            color: #3498db;
            border-bottom-color: #3498db;
        }
        .post-header {
            color: #9b59b6;
            border-bottom-color: #9b59b6;
        }
        .json-content {
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 13px;
            line-height: 1.6;
            white-space: pre-wrap;
            word-wrap: break-word;
            background: #f8f9fa;
            padding: 15px;
            border-radius: 6px;
            border: 1px solid #dee2e6;
        }
        .highlight {
            background: #fff3cd;
            padding: 2px 4px;
            border-radius: 3px;
        }
        .error-msg {
            color: #e74c3c;
            font-style: italic;
            padding: 10px;
            background: #ffe6e6;
            border-radius: 5px;
        }
        .no-data {
            color: #7f8c8d;
            font-style: italic;
            text-align: center;
            padding: 20px;
        }
        .badge {
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: bold;
            margin-left: 10px;
        }
        .badge-tfidf {
            background: #ff6b6b;
            color: white;
        }
        .badge-embedding {
            background: #4ecdc4;
            color: white;
        }
        @media (max-width: 1200px) {
            .comparison-container {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>📊 JSON Content Comparison Viewer</h1>
        <p>각 모델의 하위 성능 파일들의 실제 JSON 내용을 Gold Standard, Pre-Training, Post-Training 순으로 비교</p>
    </div>

    <div class="model-selector">
        <button class="model-btn active" onclick="selectModel('1.7B')">1.7B Model</button>
        <button class="model-btn" onclick="selectModel('4B')">4B Model</button>
        <button class="model-btn" onclick="selectModel('8B')">8B Model</button>
    </div>
    
    <div class="metric-selector" style="margin: 20px 0; display: flex; gap: 20px; flex-wrap: wrap;">
        <button class="metric-btn active" onclick="selectMetric('tfidf_bottom')" style="padding: 10px 20px; background: #ff6b6b; color: white; border: none; border-radius: 6px; cursor: pointer; font-size: 15px;">TF-IDF 하위 10개</button>
        <button class="metric-btn" onclick="selectMetric('tfidf_top')" style="padding: 10px 20px; background: #e2e8f0; color: #333; border: none; border-radius: 6px; cursor: pointer; font-size: 15px;">TF-IDF 상위 10개</button>
        <button class="metric-btn" onclick="selectMetric('embedding_bottom')" style="padding: 10px 20px; background: #e2e8f0; color: #333; border: none; border-radius: 6px; cursor: pointer; font-size: 15px;">Embedding 하위 10개</button>
        <button class="metric-btn" onclick="selectMetric('embedding_top')" style="padding: 10px 20px; background: #e2e8f0; color: #333; border: none; border-radius: 6px; cursor: pointer; font-size: 15px;">Embedding 상위 10개</button>
    </div>

    <div id="fileSelector" class="file-selector"></div>
    
    <div id="scoresInfo" class="scores-info"></div>

    <div class="comparison-container">
        <div class="json-panel">
            <div class="panel-header gold-header">🏆 Gold Standard (정답)</div>
            <div id="goldContent" class="json-content"></div>
        </div>
        <div class="json-panel">
            <div class="panel-header pre-header">📘 Pre-Training (학습 전)</div>
            <div id="preContent" class="json-content"></div>
        </div>
        <div class="json-panel">
            <div class="panel-header post-header">📗 Post-Training (학습 후)</div>
            <div id="postContent" class="json-content"></div>
        </div>
    </div>

    <script>
        const comparisonData = """ + json.dumps(comparison_data, ensure_ascii=False) + """;
        
        let currentModel = '1.7B';
        let currentMetric = 'tfidf_bottom';
        let currentFile = 'file_0';

        function selectModel(model) {
            currentModel = model;
            currentFile = 'file_0';
            
            // 버튼 스타일 업데이트
            document.querySelectorAll('.model-btn').forEach(btn => {
                btn.classList.remove('active');
                if (btn.textContent.includes(model)) {
                    btn.classList.add('active');
                }
            });
            
            updateFileSelector();
            displayComparison();
        }
        
        function selectMetric(metric) {
            currentMetric = metric;
            currentFile = 'file_0';
            
            // 버튼 스타일 업데이트
            document.querySelectorAll('.metric-btn').forEach(btn => {
                btn.classList.remove('active');
                btn.style.background = '#e2e8f0';
                btn.style.color = '#333';
            });
            
            const activeBtn = document.querySelector(`.metric-btn[onclick="selectMetric('${metric}')"]`);
            if (activeBtn) {
                activeBtn.classList.add('active');
                if (metric.includes('tfidf_bottom')) {
                    activeBtn.style.background = '#ff6b6b';
                    activeBtn.style.color = 'white';
                } else if (metric.includes('tfidf_top')) {
                    activeBtn.style.background = '#ff9999';
                    activeBtn.style.color = 'white';
                } else if (metric.includes('embedding_bottom')) {
                    activeBtn.style.background = '#4ecdc4';
                    activeBtn.style.color = 'white';
                } else if (metric.includes('embedding_top')) {
                    activeBtn.style.background = '#66e6de';
                    activeBtn.style.color = 'white';
                }
            }
            
            updateFileSelector();
            displayComparison();
        }

        function selectFile(fileKey) {
            currentFile = fileKey;
            
            // 버튼 스타일 업데이트
            document.querySelectorAll('.file-btn').forEach(btn => {
                btn.classList.remove('active');
                if (btn.getAttribute('data-file') === fileKey) {
                    btn.classList.add('active');
                }
            });
            
            displayComparison();
        }

        function updateFileSelector() {
            const selector = document.getElementById('fileSelector');
            selector.innerHTML = '';
            
            const modelData = comparisonData[currentModel];
            if (!modelData || !modelData[currentMetric]) return;
            
            const metricData = modelData[currentMetric];
            
            Object.keys(metricData).forEach((fileKey, idx) => {
                const fileData = metricData[fileKey];
                const btn = document.createElement('button');
                btn.className = 'file-btn' + (fileKey === currentFile ? ' active' : '');
                btn.setAttribute('data-file', fileKey);
                btn.onclick = () => selectFile(fileKey);
                btn.innerHTML = `File ${idx + 1}`;
                selector.appendChild(btn);
            });
        }

        function formatJson(obj) {
            if (!obj) return '<div class="no-data">데이터 없음</div>';
            if (obj.error) return '<div class="error-msg">' + obj.error + '</div>';
            
            // JSON을 보기 좋게 포맷팅
            let formatted = JSON.stringify(obj, null, 2);
            
            // 주요 키워드 하이라이트
            formatted = formatted.replace(/"summary":/g, '<span class="highlight">"summary":</span>');
            formatted = formatted.replace(/"topics":/g, '<span class="highlight">"topics":</span>');
            formatted = formatted.replace(/"decisions":/g, '<span class="highlight">"decisions":</span>');
            formatted = formatted.replace(/"action_items":/g, '<span class="highlight">"action_items":</span>');
            
            return formatted;
        }

        function displayComparison() {
            const modelData = comparisonData[currentModel];
            if (!modelData || !modelData[currentMetric] || !modelData[currentMetric][currentFile]) {
                document.getElementById('goldContent').innerHTML = '<div class="no-data">데이터를 선택하세요</div>';
                document.getElementById('preContent').innerHTML = '<div class="no-data">데이터를 선택하세요</div>';
                document.getElementById('postContent').innerHTML = '<div class="no-data">데이터를 선택하세요</div>';
                return;
            }
            
            const fileData = modelData[currentMetric][currentFile];
            
            // 점수 정보 표시
            const scoresInfo = document.getElementById('scoresInfo');
            let metricLabel = '';
            if (currentMetric === 'tfidf_bottom') metricLabel = 'TF-IDF 하위 10개';
            else if (currentMetric === 'tfidf_top') metricLabel = 'TF-IDF 상위 10개';
            else if (currentMetric === 'embedding_bottom') metricLabel = 'Embedding 하위 10개';
            else if (currentMetric === 'embedding_top') metricLabel = 'Embedding 상위 10개';
            
            scoresInfo.innerHTML = `
                <strong>📁 파일명:</strong> ${fileData.name}<br>
                <strong>📊 성능 점수:</strong> 
                <span class="badge badge-tfidf">TF-IDF: ${fileData.scores.tfidf.toFixed(4)}</span>
                <span class="badge badge-embedding">Embedding: ${fileData.scores.embedding.toFixed(4)}</span>
                <br><strong>🔍 현재 정렬 기준:</strong> ${metricLabel}
            `;
            
            // JSON 내용 표시
            document.getElementById('goldContent').innerHTML = formatJson(fileData.contents.gold);
            document.getElementById('preContent').innerHTML = formatJson(fileData.contents.pre);
            document.getElementById('postContent').innerHTML = formatJson(fileData.contents.post);
        }

        // 초기 표시
        updateFileSelector();
        displayComparison();
    </script>
</body>
</html>"""
    
    # HTML 파일 저장
    with open('json_comparison_viewer.html', 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print("JSON Comparison Viewer created successfully!")
    print("Open 'json_comparison_viewer.html' in your browser to view the content.")
    
    # 첫 번째 파일 정보 출력
    print("\n=== Sample File Info ===")
    if '4B' in comparison_data and 'tfidf_bottom' in comparison_data['4B'] and 'file_0' in comparison_data['4B']['tfidf_bottom']:
        sample = comparison_data['4B']['tfidf_bottom']['file_0']
        print(f"TF-IDF Lowest File: {sample['name']}")
        print(f"TF-IDF Score: {sample['scores']['tfidf']:.4f}")
        print(f"Embedding Score: {sample['scores']['embedding']:.4f}")
    
    if '4B' in comparison_data and 'tfidf_top' in comparison_data['4B'] and 'file_0' in comparison_data['4B']['tfidf_top']:
        sample = comparison_data['4B']['tfidf_top']['file_0']
        print(f"\nTF-IDF Highest File: {sample['name']}")
        print(f"TF-IDF Score: {sample['scores']['tfidf']:.4f}")
        print(f"Embedding Score: {sample['scores']['embedding']:.4f}")

if __name__ == "__main__":
    create_comparison_viewer()