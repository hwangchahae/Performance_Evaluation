"""
낮은 점수 파일들의 정답데이터, 학습전, 학습후 내용 비교 분석기
웹에서 한눈에 비교할 수 있는 HTML 보고서 생성
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import re

class ContentComparisonAnalyzer:
    def __init__(self):
        self.base_dir = Path(__file__).parent
        os.chdir(self.base_dir)
        
    def load_post_training_results(self, model_size: str) -> List[Dict]:
        """학습 후 결과 로드"""
        json_file = f"Post_Training/post_similarity_results/{model_size}_post_similarity_results.json"
        json_path = Path(json_file)
        
        if not json_path.exists():
            print(f"[WARNING] {json_file} 파일이 없습니다.")
            return []
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data.get('details', [])
        except Exception as e:
            print(f"[ERROR] {json_file} 로드 실패: {e}")
            return []
    
    def load_pre_training_results(self, model_size: str) -> Dict[str, Dict]:
        """학습 전 결과 로드"""
        json_file = f"Pre_Training/pre_similarity_results/{model_size}_pretrain_similarity_results.json"
        json_path = Path(json_file)
        
        if not json_path.exists():
            print(f"[WARNING] {json_file} 파일이 없습니다.")
            return {}
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # 파일명을 키로 하는 딕셔너리로 변환
            results = {}
            for item in data.get('details', []):
                results[item['file_name']] = item
            return results
        except Exception as e:
            print(f"[ERROR] {json_file} 로드 실패: {e}")
            return {}
    
    def get_actual_id_from_post_training(self, file_name: str, model_size: str) -> str:
        """학습 후 파일에서 실제 ID 추출"""
        try:
            # 파일명에서 폴더명 생성
            name_parts = file_name.split("_")
            if len(name_parts) >= 4:
                meeting_id = name_parts[3]
                if len(name_parts) >= 6:
                    chunk_num = name_parts[5]
                    target_folder = f"result_{meeting_id}_chunk_{chunk_num}"
                else:
                    target_folder = f"result_{meeting_id}"
                
                model_folder = f"{model_size}_lora_model_results"
                post_path = Path(f"Post_Training/{model_folder}") / target_folder / "result.json"
                
                if post_path.exists():
                    with open(post_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    actual_id = data.get("id", "")
                    print(f"[DEBUG] {file_name} -> 실제 ID: {actual_id}")
                    return actual_id
        except Exception as e:
            print(f"[ERROR] ID 추출 실패 ({file_name}): {e}")
        return ""

    def load_actual_content(self, file_name: str, content_type: str = "gold", model_size: str = "1.7B") -> str:
        """실제 파일 내용 로드"""
        try:
            if content_type == "gold":
                # 학습 후 파일에서 실제 ID를 가져와서 매칭
                if hasattr(self, '_actual_ids') and file_name in self._actual_ids:
                    actual_id = self._actual_ids[file_name]
                    if actual_id:
                        # actual_id 형태: val_028_chunk_3 -> val_028_result_XXX_chunk_3 형태로 변환해서 찾기
                        id_parts = actual_id.split("_")
                        if len(id_parts) >= 3:
                            val_num = id_parts[1]  # 028
                            chunk_info = "_".join(id_parts[2:])  # chunk_3
                            
                            # 정답 데이터에서 해당 ID로 시작하는 폴더 찾기
                            gold_base = Path("Gold_Standard_Data")
                            for gold_folder in gold_base.iterdir():
                                if gold_folder.is_dir() and gold_folder.name.startswith(f"val_{val_num}_"):
                                    # chunk 정보 확인
                                    if chunk_info in gold_folder.name or chunk_info.replace("_", "") in gold_folder.name:
                                        gold_path = gold_folder / "result.json"
                                        if gold_path.exists():
                                            with open(gold_path, 'r', encoding='utf-8') as f:
                                                data = json.load(f)
                                            notion = data.get("notion_output", "")
                                            if isinstance(notion, dict):
                                                return json.dumps(notion, ensure_ascii=False, indent=2)
                                            return str(notion)
                
                # 기존 방식 (fallback)
                gold_path = Path("Gold_Standard_Data") / file_name / "result.json"
                if gold_path.exists():
                    with open(gold_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    notion = data.get("notion_output", "")
                    if isinstance(notion, dict):
                        return json.dumps(notion, ensure_ascii=False, indent=2)
                    return str(notion)
            
            elif content_type == "pre_training":
                # 학습 전 데이터 로드 - JSON 파일 형태
                # 파일명 변환: val_010_result_Bro003_chunk_4 -> Bro003_chunk4
                name_parts = file_name.split("_")
                if len(name_parts) >= 4:
                    # val_XXX_result_ 부분 제거하고 나머지 추출
                    meeting_id = name_parts[3]  # Bro003
                    if len(name_parts) >= 6:  # chunk 정보가 있는 경우
                        chunk_num = name_parts[5]  # 4
                        target_pattern = f"{meeting_id}_chunk{chunk_num}"
                    else:
                        target_pattern = meeting_id
                    
                    # 모델 크기에 맞는 폴더 선택
                    model_folder_mapping = {
                        "1.7B": "1.7B_model_test_results",
                        "4B": "4B_GPT_pretrain_similarity_results", 
                        "8B": "8B_model_test_results"
                    }
                    model_folder = model_folder_mapping.get(model_size, "1.7B_model_test_results")
                    folder_path = Path(f"../0805_학습전모델유사도결과/{model_folder}")
                    if folder_path.exists():
                        # Qwen3_*_AWQ_transformers_result_{target_pattern}.json 패턴 찾기
                        for json_file in folder_path.glob(f"*result_{target_pattern}.json"):
                            with open(json_file, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                            # 학습 전 결과 추출
                            result = data.get("result", {}).get("generation_result", {}).get("result", "")
                            if isinstance(result, dict):
                                return json.dumps(result, ensure_ascii=False, indent=2)
                            return str(result)
            
            elif content_type == "post_training":
                # 학습 후 데이터 로드 - 폴더 구조
                # 파일명 변환: val_021_result_Bro011_chunk_1 -> result_Bro011_chunk_1
                name_parts = file_name.split("_")
                if len(name_parts) >= 4:
                    # val_XXX_result_ 부분 제거하고 나머지 추출
                    meeting_id = name_parts[3]  # Bro011
                    if len(name_parts) >= 6:  # chunk 정보가 있는 경우
                        chunk_num = name_parts[5]  # 1
                        target_folder = f"result_{meeting_id}_chunk_{chunk_num}"
                    else:
                        target_folder = f"result_{meeting_id}"
                    
                    # 모델 크기에 맞는 폴더 선택
                    model_folder = f"{model_size}_lora_model_results"
                    post_path = Path(f"Post_Training/{model_folder}") / target_folder / "result.json"
                    if post_path.exists():
                        with open(post_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        notion = data.get("notion_output", "")
                        if isinstance(notion, dict):
                            return json.dumps(notion, ensure_ascii=False, indent=2)
                        return str(notion)
            
            return "내용을 찾을 수 없습니다."
            
        except Exception as e:
            print(f"[ERROR] {content_type} 내용 로드 실패 ({file_name}): {e}")
            return f"내용 로드 중 오류가 발생했습니다: {str(e)}"
    
    def get_low_score_files(self, model_size: str, count: int = 10) -> List[Dict]:
        """낮은 점수 파일들 가져오기"""
        results = self.load_post_training_results(model_size)
        if not results:
            return []
        
        # TF-IDF 기준으로 정렬 (낮은 순)
        sorted_results = sorted(results, key=lambda x: x.get('tfidf_cosine', 0))
        return sorted_results[:count]
    
    def analyze_model_content(self, model_size: str) -> List[Dict]:
        """모델별 내용 분석"""
        print(f"\\n[INFO] {model_size} 모델 내용 분석 중...")
        
        low_score_files = self.get_low_score_files(model_size)
        pre_training_results = self.load_pre_training_results(model_size)
        
        # 먼저 모든 파일의 실제 ID 추출
        print(f"  [INFO] 실제 ID 추출 중...")
        self._actual_ids = {}
        for file_info in low_score_files:
            folder_name = file_info['file_name']
            actual_id = self.get_actual_id_from_post_training(folder_name, model_size)
            if actual_id:
                self._actual_ids[folder_name] = actual_id
        
        content_analyses = []
        
        for i, file_info in enumerate(low_score_files, 1):
            folder_name = file_info['file_name']
            print(f"  [INFO] {i}/10: {folder_name} - 내용 로드 중...")
            
            # 실제 내용 로드 (모델 크기 전달)
            gold_content = self.load_actual_content(folder_name, "gold", model_size)
            pre_training_content = self.load_actual_content(folder_name, "pre_training", model_size) 
            post_training_content = self.load_actual_content(folder_name, "post_training", model_size)
            
            # 학습 전 점수 찾기
            pre_training_score = pre_training_results.get(folder_name, {})
            
            analysis = {
                'rank': i,
                'file_name': folder_name,
                'post_training_scores': {
                    'tfidf': file_info['tfidf_cosine'],
                    'embedding': file_info['embedding_cosine']
                },
                'pre_training_scores': {
                    'tfidf': pre_training_score.get('tfidf_cosine', 0.0),
                    'embedding': pre_training_score.get('embedding_cosine', 0.0)
                },
                'content': {
                    'gold': gold_content,
                    'pre_training': pre_training_content,
                    'post_training': post_training_content
                }
            }
            
            content_analyses.append(analysis)
        
        print(f"[OK] {model_size} 모델 내용 분석 완료")
        return content_analyses
    
    def generate_html_report(self, all_analyses: Dict[str, List[Dict]]) -> str:
        """HTML 보고서 생성"""
        html = """
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>낮은 점수 파일 분석</title>
    <style>
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            margin: 0; 
            padding: 20px; 
            background-color: #f5f7fa;
            line-height: 1.6;
        }
        .container { 
            max-width: 1400px; 
            margin: 0 auto; 
            background: white; 
            border-radius: 12px; 
            box-shadow: 0 4px 20px rgba(0,0,0,0.1); 
            overflow: hidden;
        }
        .header { 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            color: white; 
            padding: 30px; 
            text-align: center; 
        }
        .header h1 { margin: 0; font-size: 2.5em; font-weight: 300; }
        .header p { margin: 10px 0 0 0; opacity: 0.9; font-size: 1.1em; }
        
        .model-selector { 
            display: flex; 
            background: #f8f9fa; 
            border-bottom: 1px solid #dee2e6;
        }
        .model-tab { 
            flex: 1; 
            padding: 15px; 
            text-align: center; 
            cursor: pointer; 
            border: none;
            background: transparent;
            font-size: 1.1em;
            font-weight: 600;
            color: #6c757d;
            transition: all 0.3s ease;
        }
        .model-tab:hover { background: #e9ecef; }
        .model-tab.active { 
            background: #007bff; 
            color: white; 
            box-shadow: inset 0 3px 0 rgba(255,255,255,0.3);
        }
        
        .model-content { display: none; padding: 30px; }
        .model-content.active { display: block; }
        
        .file-item { 
            margin: 30px 0; 
            border: 1px solid #e9ecef; 
            border-radius: 12px; 
            overflow: hidden;
            background: #fff;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }
        .file-header { 
            background: #f8f9fa; 
            padding: 20px; 
            border-bottom: 1px solid #e9ecef; 
        }
        .file-title { 
            font-size: 1.3em; 
            font-weight: 600; 
            color: #2c3e50; 
            margin: 0 0 15px 0;
        }
        .scores { 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); 
            gap: 15px; 
            margin-top: 15px;
        }
        .score-box { 
            background: white; 
            padding: 15px; 
            border-radius: 8px; 
            border: 1px solid #dee2e6;
            text-align: center;
        }
        .score-title { 
            font-size: 0.9em; 
            color: #6c757d; 
            margin-bottom: 8px; 
            text-transform: uppercase;
            font-weight: 600;
        }
        .score-value { 
            font-size: 1.4em; 
            font-weight: bold; 
        }
        .score-low { color: #dc3545; }
        .score-medium { color: #ffc107; }
        .score-high { color: #28a745; }
        
        .content-comparison { margin-top: 20px; }
        .content-tabs { display: flex; margin-bottom: 10px; }
        .content-tab { 
            padding: 8px 16px; 
            background: #e9ecef; 
            border: 1px solid #dee2e6;
            cursor: pointer; 
            margin-right: 2px;
            font-size: 0.9em;
        }
        .content-tab.active { background: #007bff; color: white; }
        .content-panel { 
            display: none; 
            background: #f8f9fa; 
            border: 1px solid #dee2e6; 
            padding: 15px; 
            max-height: 300px; 
            overflow-y: auto;
            font-family: monospace;
            font-size: 0.85em;
            white-space: pre-wrap;
        }
        .content-panel.active { display: block; }
        .score-comparison { display: flex; gap: 10px; margin-top: 10px; }
        .score-change { font-size: 0.8em; padding: 2px 6px; border-radius: 3px; }
        .score-change.improved { background: #d4edda; color: #155724; }
        .score-change.declined { background: #f8d7da; color: #721c24; }
    </style>
    <script>
        function showModel(modelSize, element) {
            // 모든 탭과 콘텐츠 비활성화
            document.querySelectorAll('.model-tab').forEach(tab => tab.classList.remove('active'));
            document.querySelectorAll('.model-content').forEach(content => content.classList.remove('active'));
            
            // 선택된 탭과 콘텐츠 활성화
            element.classList.add('active');
            document.getElementById(`model-${modelSize}`).classList.add('active');
        }
        
        function showContent(fileId, contentType, element) {
            // 해당 파일의 모든 콘텐츠 탭과 패널 비활성화
            const fileContainer = element.closest('.file-item');
            fileContainer.querySelectorAll('.content-tab').forEach(tab => tab.classList.remove('active'));
            fileContainer.querySelectorAll('.content-panel').forEach(panel => panel.classList.remove('active'));
            
            // 선택된 탭과 패널 활성화
            element.classList.add('active');
            document.getElementById(`${fileId}-${contentType}`).classList.add('active');
        }
    </script>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>낮은 점수 파일 분석 보고서</h1>
            <p>Post Training 모델들의 낮은 점수 파일 분석</p>
        </div>
        
        <div class="model-selector">
"""

        # 모델 탭 생성
        for i, model_size in enumerate(['1.7B', '4B', '8B']):
            active_class = 'active' if i == 0 else ''
            html += f"""
            <button class="model-tab {active_class}" onclick="showModel('{model_size}', this)">
                {model_size} 모델
            </button>
"""

        html += """
        </div>
"""

        # 각 모델별 내용
        for i, (model_size, analyses) in enumerate(all_analyses.items()):
            active_class = 'active' if i == 0 else ''
            
            html += f"""
        <div id="model-{model_size}" class="model-content {active_class}">
            <h2>{model_size} 모델 - 낮은 점수 상위 10개 파일</h2>
"""
            
            # 각 파일 분석
            for analysis in analyses:
                file_name = analysis['file_name']
                rank = analysis['rank']
                
                post_tfidf = analysis['post_training_scores']['tfidf']
                post_embedding = analysis['post_training_scores']['embedding']
                pre_tfidf = analysis['pre_training_scores']['tfidf']
                pre_embedding = analysis['pre_training_scores']['embedding']
                
                # 점수 변화 계산
                tfidf_change = post_tfidf - pre_tfidf
                embedding_change = post_embedding - pre_embedding
                
                tfidf_change_class = 'improved' if tfidf_change > 0 else 'declined'
                embedding_change_class = 'improved' if embedding_change > 0 else 'declined'
                
                # 파일별 고유 ID 생성
                file_id = f"{model_size}-file-{rank}"
                
                html += f"""
            <div class="file-item">
                <div class="file-header">
                    <div class="file-title">#{rank}. {file_name}</div>
                    <div class="scores">
                        <div class="score-box">
                            <div class="score-title">학습 후 TF-IDF</div>
                            <div class="score-value score-low">{post_tfidf:.4f}</div>
                            <div class="score-comparison">
                                <span>학습 전: {pre_tfidf:.4f}</span>
                                <span class="score-change {tfidf_change_class}">
                                    {'+' if tfidf_change >= 0 else ''}{tfidf_change:.4f}
                                </span>
                            </div>
                        </div>
                        <div class="score-box">
                            <div class="score-title">학습 후 임베딩</div>
                            <div class="score-value score-low">{post_embedding:.4f}</div>
                            <div class="score-comparison">
                                <span>학습 전: {pre_embedding:.4f}</span>
                                <span class="score-change {embedding_change_class}">
                                    {'+' if embedding_change >= 0 else ''}{embedding_change:.4f}
                                </span>
                            </div>
                        </div>
                    </div>
                    
                    <div class="content-comparison">
                        <div class="content-tabs">
                            <div class="content-tab active" onclick="showContent('{file_id}', 'gold', this)">
                                정답 데이터
                            </div>
                            <div class="content-tab" onclick="showContent('{file_id}', 'pre', this)">
                                학습 전 결과
                            </div>
                            <div class="content-tab" onclick="showContent('{file_id}', 'post', this)">
                                학습 후 결과
                            </div>
                        </div>
                        
                        <div id="{file_id}-gold" class="content-panel active">
{analysis['content']['gold']}
                        </div>
                        <div id="{file_id}-pre" class="content-panel">
{analysis['content']['pre_training']}
                        </div>
                        <div id="{file_id}-post" class="content-panel">
{analysis['content']['post_training']}
                        </div>
                    </div>
                </div>
            </div>
"""
            
            html += """
        </div>
"""

        html += """
    </div>
</body>
</html>
"""
        return html
    
    def run_analysis(self):
        """전체 분석 실행"""
        print("[INFO] 낮은 점수 파일 분석 시작...")
        
        models = ['1.7B', '4B', '8B']
        all_analyses = {}
        
        for model in models:
            analyses = self.analyze_model_content(model)
            if analyses:
                all_analyses[model] = analyses
        
        if not all_analyses:
            print("[ERROR] 분석할 데이터가 없습니다.")
            return
        
        # HTML 보고서 생성
        print("\\n[INFO] HTML 보고서 생성 중...")
        html_content = self.generate_html_report(all_analyses)
        
        # 보고서 저장
        output_file = "low_score_analysis_report.html"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"[OK] 분석 완료! 보고서 저장: {output_file}")

def main():
    analyzer = ContentComparisonAnalyzer()
    analyzer.run_analysis()

if __name__ == "__main__":
    main()