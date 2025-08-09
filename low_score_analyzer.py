"""
Post Training 모델 낮은 점수 파일 분석기
TF-IDF와 임베딩 점수가 낮은 파일들을 찾아서 분석
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd

def load_model_results(model_size: str) -> List[Dict]:
    """모델 결과 로드"""
    json_file = f"{model_size}_post_similarity_results.json"
    json_path = Path("Post_Training") / json_file
    
    if not json_path.exists():
        print(f"⚠️  {json_file} 파일이 없습니다.")
        return []
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data.get('details', [])
    except Exception as e:
        print(f"❌ {json_file} 로드 실패: {e}")
        return []

def find_low_score_files(results: List[Dict], score_type: str, count: int = 10) -> List[Dict]:
    """낮은 점수 파일들 찾기"""
    if not results:
        return []
    
    # 점수 기준으로 정렬 (오름차순 - 낮은 점수부터)
    sorted_results = sorted(results, key=lambda x: x.get(score_type, 0))
    return sorted_results[:count]

def analyze_model(model_size: str) -> Dict[str, Any]:
    """모델 분석"""
    print(f"\n🔍 {model_size} 모델 분석 중...")
    
    results = load_model_results(model_size)
    if not results:
        return {}
    
    # 전체 통계
    tfidf_scores = [r['tfidf_cosine'] for r in results]
    embedding_scores = [r['embedding_cosine'] for r in results]
    
    analysis = {
        'model_size': model_size,
        'total_files': len(results),
        'tfidf_stats': {
            'mean': sum(tfidf_scores) / len(tfidf_scores),
            'min': min(tfidf_scores),
            'max': max(tfidf_scores)
        },
        'embedding_stats': {
            'mean': sum(embedding_scores) / len(embedding_scores),
            'min': min(embedding_scores),
            'max': max(embedding_scores)
        },
        'low_tfidf_files': find_low_score_files(results, 'tfidf_cosine', 10),
        'low_embedding_files': find_low_score_files(results, 'embedding_cosine', 10)
    }
    
    print(f"✅ {model_size} 분석 완료 - 총 {len(results)}개 파일")
    return analysis

def generate_html_report(analyses: List[Dict]) -> str:
    """HTML 보고서 생성"""
    html = """
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>낮은 점수 파일 분석 보고서</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 20px; background-color: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1 { color: #2c3e50; text-align: center; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
        h2 { color: #34495e; border-left: 4px solid #3498db; padding-left: 10px; margin-top: 30px; }
        h3 { color: #2c3e50; }
        .model-section { margin: 30px 0; padding: 20px; border: 1px solid #ddd; border-radius: 8px; background: #fafafa; }
        .stats { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 20px 0; }
        .stat-box { background: white; padding: 15px; border-radius: 5px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
        .stat-title { font-weight: bold; color: #2c3e50; margin-bottom: 10px; }
        .stat-value { font-size: 1.2em; color: #3498db; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; background: white; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background-color: #34495e; color: white; font-weight: bold; }
        tr:nth-child(even) { background-color: #f8f9fa; }
        tr:hover { background-color: #e8f4f8; }
        .score-low { color: #e74c3c; font-weight: bold; }
        .score-medium { color: #f39c12; font-weight: bold; }
        .score-high { color: #27ae60; font-weight: bold; }
        .file-name { font-family: monospace; background: #ecf0f1; padding: 2px 4px; border-radius: 3px; }
        .summary { background: #3498db; color: white; padding: 20px; border-radius: 8px; margin: 20px 0; }
        .tabs { display: flex; margin-bottom: 20px; }
        .tab { padding: 10px 20px; background: #ecf0f1; border: 1px solid #ddd; cursor: pointer; margin-right: 5px; border-radius: 5px 5px 0 0; }
        .tab.active { background: #3498db; color: white; }
        .tab-content { display: none; }
        .tab-content.active { display: block; }
    </style>
    <script>
        function showTab(tabName, element) {
            // 모든 탭 비활성화
            document.querySelectorAll('.tab').forEach(tab => tab.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));
            
            // 선택된 탭 활성화
            element.classList.add('active');
            document.getElementById(tabName).classList.add('active');
        }
        
        function getScoreClass(score, type) {
            if (type === 'tfidf') {
                if (score < 0.15) return 'score-low';
                if (score < 0.25) return 'score-medium';
                return 'score-high';
            } else {
                if (score < 0.85) return 'score-low';
                if (score < 0.90) return 'score-medium';
                return 'score-high';
            }
        }
    </script>
</head>
<body>
    <div class="container">
        <h1>🔍 낮은 점수 파일 분석 보고서</h1>
        <div class="summary">
            <h3>📊 분석 개요</h3>
            <p>Post Training 모델들에서 TF-IDF와 임베딩 점수가 낮은 파일들을 분석하여 모델 성능 개선 포인트를 찾습니다.</p>
        </div>
"""

    for analysis in analyses:
        if not analysis:
            continue
            
        model_size = analysis['model_size']
        html += f"""
        <div class="model-section">
            <h2>🤖 {model_size} 모델</h2>
            
            <div class="stats">
                <div class="stat-box">
                    <div class="stat-title">📈 TF-IDF 통계</div>
                    <div>평균: <span class="stat-value">{analysis['tfidf_stats']['mean']:.4f}</span></div>
                    <div>최소: <span class="stat-value">{analysis['tfidf_stats']['min']:.4f}</span></div>
                    <div>최대: <span class="stat-value">{analysis['tfidf_stats']['max']:.4f}</span></div>
                </div>
                <div class="stat-box">
                    <div class="stat-title">🎯 임베딩 통계</div>
                    <div>평균: <span class="stat-value">{analysis['embedding_stats']['mean']:.4f}</span></div>
                    <div>최소: <span class="stat-value">{analysis['embedding_stats']['min']:.4f}</span></div>
                    <div>최대: <span class="stat-value">{analysis['embedding_stats']['max']:.4f}</span></div>
                </div>
            </div>
            
            <div class="tabs">
                <div class="tab active" onclick="showTab('tfidf-{model_size}', this)">낮은 TF-IDF 점수</div>
                <div class="tab" onclick="showTab('embedding-{model_size}', this)">낮은 임베딩 점수</div>
            </div>
            
            <div id="tfidf-{model_size}" class="tab-content active">
                <h3>📉 TF-IDF 점수가 낮은 상위 10개 파일</h3>
                <table>
                    <thead>
                        <tr>
                            <th>순위</th>
                            <th>파일명</th>
                            <th>TF-IDF 점수</th>
                            <th>임베딩 점수</th>
                        </tr>
                    </thead>
                    <tbody>
"""
        
        for i, file_info in enumerate(analysis['low_tfidf_files'], 1):
            tfidf_score = file_info['tfidf_cosine']
            embedding_score = file_info['embedding_cosine']
            tfidf_class = 'score-low' if tfidf_score < 0.15 else 'score-medium' if tfidf_score < 0.25 else 'score-high'
            embedding_class = 'score-low' if embedding_score < 0.85 else 'score-medium' if embedding_score < 0.90 else 'score-high'
            
            html += f"""
                        <tr>
                            <td>{i}</td>
                            <td><span class="file-name">{file_info['file_name']}</span></td>
                            <td><span class="{tfidf_class}">{tfidf_score:.4f}</span></td>
                            <td><span class="{embedding_class}">{embedding_score:.4f}</span></td>
                        </tr>
"""
        
        html += f"""
                    </tbody>
                </table>
            </div>
            
            <div id="embedding-{model_size}" class="tab-content">
                <h3>📉 임베딩 점수가 낮은 상위 10개 파일</h3>
                <table>
                    <thead>
                        <tr>
                            <th>순위</th>
                            <th>파일명</th>
                            <th>TF-IDF 점수</th>
                            <th>임베딩 점수</th>
                        </tr>
                    </thead>
                    <tbody>
"""
        
        for i, file_info in enumerate(analysis['low_embedding_files'], 1):
            tfidf_score = file_info['tfidf_cosine']
            embedding_score = file_info['embedding_cosine']
            tfidf_class = 'score-low' if tfidf_score < 0.15 else 'score-medium' if tfidf_score < 0.25 else 'score-high'
            embedding_class = 'score-low' if embedding_score < 0.85 else 'score-medium' if embedding_score < 0.90 else 'score-high'
            
            html += f"""
                        <tr>
                            <td>{i}</td>
                            <td><span class="file-name">{file_info['file_name']}</span></td>
                            <td><span class="{tfidf_class}">{tfidf_score:.4f}</span></td>
                            <td><span class="{embedding_class}">{embedding_score:.4f}</span></td>
                        </tr>
"""
        
        html += """
                    </tbody>
                </table>
            </div>
        </div>
"""

    html += """
        <div class="summary">
            <h3>💡 분석 결과 요약</h3>
            <ul>
                <li><strong>낮은 점수 기준:</strong> TF-IDF < 0.15, 임베딩 < 0.85</li>
                <li><strong>개선 포인트:</strong> 낮은 점수를 받은 파일들의 공통점을 찾아 모델 튜닝에 활용</li>
                <li><strong>활용 방법:</strong> 해당 파일들의 내용을 분석하여 모델이 어려워하는 패턴 파악</li>
            </ul>
        </div>
    </div>
</body>
</html>
"""
    return html

def main():
    """메인 실행 함수"""
    print("🚀 Post Training 낮은 점수 파일 분석 시작...")
    
    # 작업 디렉토리를 Performance_Evaluation로 변경
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    models = ['1.7B', '4B', '8B']
    analyses = []
    
    for model in models:
        analysis = analyze_model(model)
        if analysis:
            analyses.append(analysis)
    
    if not analyses:
        print("❌ 분석할 데이터가 없습니다.")
        return
    
    # HTML 보고서 생성
    print("\n📄 HTML 보고서 생성 중...")
    html_content = generate_html_report(analyses)
    
    # 보고서 저장
    output_file = "low_score_analysis_report.html"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ 분석 완료! 보고서 저장: {output_file}")
    
    # 요약 출력
    print("\n" + "="*60)
    print("📊 분석 요약")
    print("="*60)
    
    for analysis in analyses:
        model = analysis['model_size']
        print(f"\n🤖 {model} 모델:")
        print(f"  - 총 파일 수: {analysis['total_files']}")
        print(f"  - TF-IDF 평균: {analysis['tfidf_stats']['mean']:.4f}")
        print(f"  - 임베딩 평균: {analysis['embedding_stats']['mean']:.4f}")
        
        # 가장 낮은 점수 파일
        lowest_tfidf = analysis['low_tfidf_files'][0] if analysis['low_tfidf_files'] else None
        lowest_embedding = analysis['low_embedding_files'][0] if analysis['low_embedding_files'] else None
        
        if lowest_tfidf:
            print(f"  - 최저 TF-IDF: {lowest_tfidf['tfidf_cosine']:.4f} ({lowest_tfidf['file_name']})")
        if lowest_embedding:
            print(f"  - 최저 임베딩: {lowest_embedding['embedding_cosine']:.4f} ({lowest_embedding['file_name']})")

if __name__ == "__main__":
    main()