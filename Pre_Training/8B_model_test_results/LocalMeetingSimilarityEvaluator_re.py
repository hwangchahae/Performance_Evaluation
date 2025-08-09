import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Union
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from bert_score import score
from rouge_score import rouge_scorer
import os
import re

class LocalMeetingSimilarityEvaluator:
    def __init__(self, weights: Dict[str, float] = None):
        if weights is None:
            self.weights = {'cosine': 0.1, 'bert': 0.5, 'rouge': 0.4}
        else:
            self.weights = weights
        self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words=None
        )
        print("로컬 유사도 평가기(단일 JSON 비교) 초기화 완료")

    def evaluate_similarity(self, ground_truth: List[str], predictions: List[str], gold_names: List[str] = None) -> Dict:
        print("=== 유사도 평가 시작 ===", flush=True)
        print(f"총 {len(ground_truth)}개 파일을 개별적으로 평가합니다.\n", flush=True)
        
        cosine_scores = []
        bert_scores = []
        rouge_scores = []
        
        for i in range(len(ground_truth)):
            gt_text = ground_truth[i]
            pred_text = predictions[i]
            
            # 개별 파일의 코사인 유사도
            all_texts = [gt_text, pred_text]
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(all_texts)
            cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            cosine_scores.append(cosine_sim)
            
            # 개별 파일의 BERT 점수
            try:
                P, R, F1 = score([pred_text], [gt_text], lang='ko', verbose=False)
                bert_f1 = F1.numpy()[0]
            except:
                P, R, F1 = score([pred_text], [gt_text], verbose=False)
                bert_f1 = F1.numpy()[0]
            bert_scores.append(bert_f1)
            
            # 개별 파일의 ROUGE-L 점수
            rouge_score_val = self.rouge_scorer.score(gt_text, pred_text)
            rouge_l = rouge_score_val['rougeL'].fmeasure
            rouge_scores.append(rouge_l)
            
            # 최종 점수 계산
            final_score = (
                self.weights['cosine'] * cosine_sim +
                self.weights['bert'] * bert_f1 +
                self.weights['rouge'] * rouge_l
            )
            
            # 파일명 표시 (정답 폴더명 또는 번호)
            file_display = gold_names[i] if gold_names else f"파일 {i+1}"
            
            # 매번 실시간 출력 (진행률 + 파일명 + 점수)
            print(f"[{i+1}/{len(ground_truth)}] {file_display}: Cosine: {cosine_sim:.4f} | BERT: {bert_f1:.4f} | ROUGE: {rouge_l:.4f} | 최종: {final_score:.4f}", flush=True)
        
        final_scores = (
            np.array(cosine_scores) * self.weights['cosine'] +
            np.array(bert_scores) * self.weights['bert'] +
            np.array(rouge_scores) * self.weights['rouge']
        )
        
        results = {
            'cosine_similarity': np.array(cosine_scores),
            'bert_f1': np.array(bert_scores),
            'rouge_l': np.array(rouge_scores),
            'final_score': final_scores,
            'weights': self.weights
        }
        
        print(f"\n=== 유사도 평가 완료 ===")
        print(f"전체 평균 - Cosine: {np.mean(cosine_scores):.4f}, BERT: {np.mean(bert_scores):.4f}, ROUGE: {np.mean(rouge_scores):.4f}")
        print(f"최종 평균 점수: {np.mean(final_scores):.4f}")
        return results

def extract_gold_text(folder_path: str) -> str:
    """정답 데이터 폴더에서 result.json의 notion_output 추출"""
    json_path = os.path.join(folder_path, "result.json")
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    notion = data.get("notion_output", "")
    
    # notion이 dict인 경우 JSON 문자열로 변환
    if isinstance(notion, dict):
        return json.dumps(notion, ensure_ascii=False, indent=2)
    
    # notion이 문자열인 경우 기존 처리
    if isinstance(notion, str) and notion.startswith("```json"):
        notion = notion.replace("```json", "").replace("```", "").strip()
    
    return str(notion)

def extract_result_text(json_path: str) -> str:
    """비교 입력 데이터 JSON 파일에서 result.generation_result.result 추출"""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    result = data.get("result", {}).get("generation_result", {}).get("result", "")
    if isinstance(result, dict):
        return json.dumps(result, ensure_ascii=False, indent=2)
    return result

def extract_core_pattern(filename):
    """파일명에서 핵심 패턴 추출"""
    # 1. _result_XXX_processed_chunk_N 또는 _result_XXX_chunk_N 패턴 (영문)
    match = re.search(r'_result_([^_]+)(?:_processed)?_chunk_?(\d+)', filename)
    if match:
        prefix = match.group(1)
        chunk_num = match.group(2)
        return f"{prefix}_chunk{chunk_num}"
    
    # 2. _result_긴한글제목_chunkN 패턴 (한글 회의록)
    match = re.search(r'_result_(.+?)_chunk(\d+)', filename)
    if match:
        prefix = match.group(1)
        chunk_num = match.group(2)
        # 한글 제목은 해시로 단축해서 매칭
        import hashlib
        prefix_hash = hashlib.md5(prefix.encode('utf-8')).hexdigest()[:8]
        return f"{prefix_hash}_chunk{chunk_num}"
    
    return None

def find_matching_files(gold_base_path: str, result_base_path: str):
    """매칭되는 파일 쌍을 찾아서 반환"""
    matches = []
    
    # 정답 데이터 폴더들
    gold_path = Path(gold_base_path)
    if not gold_path.exists():
        return matches
        
    print("정답 파일 매칭 중...")
    for gold_folder in gold_path.iterdir():
        if gold_folder.is_dir() and "result_" in gold_folder.name:
            # train_003_result_IS1005b_chunk_1에서 IS1005b_chunk_1 추출
            suffix = gold_folder.name.split("result_", 1)[1]
            
            # 매칭되는 비교 파일 찾기
            result_path = Path(result_base_path)
            if result_path.exists():
                found_match = False
                for result_file in result_path.iterdir():
                    if result_file.is_file() and result_file.suffix == '.json':
                        # 1. 정확한 매칭 시도
                        if f"result_{suffix}" in result_file.name:
                            matches.append((gold_folder, result_file, suffix))
                            found_match = True
                            print(f"정확 매칭: {gold_folder.name} <-> {result_file.name}")
                            break
                        
                        # 2. 핵심 패턴으로 매칭 시도
                        gold_core = extract_core_pattern(gold_folder.name)
                        result_core = extract_core_pattern(result_file.name)
                        
                        if gold_core and result_core and gold_core == result_core:
                            matches.append((gold_folder, result_file, suffix))
                            found_match = True
                            print(f"핵심 패턴 매칭: {gold_folder.name} <-> {result_file.name}")
                            print(f"  추출된 패턴: {gold_core}")
                            break
                
                if not found_match:
                    print(f"매칭 실패: {gold_folder.name} (찾는 패턴: result_{suffix})")
    
    return matches

if __name__ == "__main__":
    evaluator = LocalMeetingSimilarityEvaluator()
    gold_base_path = "ttalkkac_gold_standard_results_output"
    result_base_path = "1.7B_model_test_results"
    
    # 매칭되는 파일 쌍들 찾기
    matches = find_matching_files(gold_base_path, result_base_path)
    
    if not matches:
        print(f"매칭되는 파일을 찾을 수 없습니다.")
        print(f"정답 경로: {gold_base_path}")
        print(f"비교 경로: {result_base_path}")
        exit()
    
    print(f"총 {len(matches)}개의 매칭 파일 쌍을 찾았습니다.\n")
    
    # 모든 텍스트 수집
    ground_truths = []
    predictions = []
    file_names = []
    gold_folder_names = []  # 정답 폴더명 추가
    
    print("파일 처리 중...")
    for gold_folder, result_file, suffix in matches:
        try:
            gt_text = extract_gold_text(str(gold_folder))
            pred_text = extract_result_text(str(result_file))
            
            ground_truths.append(gt_text)
            predictions.append(pred_text)
            file_names.append(suffix)
            gold_folder_names.append(gold_folder.name)  # 정답 폴더명 저장
            
        except Exception as e:
            print(f"파일 처리 오류 ({suffix}): {e}")
            continue
    
    if not ground_truths:
        print("처리할 수 있는 파일이 없습니다.")
        exit()
    
    print(f"실제 처리할 파일: {len(ground_truths)}개")
    
    # 유사도 평가 수행
    results = evaluator.evaluate_similarity(ground_truths, predictions, gold_folder_names)
    
    print("\n=== 회의록 유사도 평가 결과 ===")
    for i, (cs, bs, rs, fs) in enumerate(zip(
        results['cosine_similarity'], 
        results['bert_f1'], 
        results['rouge_l'], 
        results['final_score']
    )):
        print(f"파일 {i+1}: {file_names[i]}")
        print(f"  Cosine: {cs:.4f} | BERT: {bs:.4f} | ROUGE: {rs:.4f} | 최종: {fs:.4f}")
        print()
    
    # 평균 점수
    print("=== 전체 평균 ===")
    print(f"평균 Cosine Similarity: {np.mean(results['cosine_similarity']):.4f}")
    print(f"평균 BERTScore F1: {np.mean(results['bert_f1']):.4f}")
    print(f"평균 ROUGE-L: {np.mean(results['rouge_l']):.4f}")
    print(f"평균 최종 점수: {np.mean(results['final_score']):.4f}")
    
    # 간단한 TXT 파일로 저장
    print("결과를 파일로 저장 중...")
    with open("similarity_results.txt", "w", encoding="utf-8") as f:
        f.write("파일명\t최종점수\tCosine\tBERT\tROUGE\n")
        for i, (cs, bs, rs, fs) in enumerate(zip(
            results['cosine_similarity'], 
            results['bert_f1'], 
            results['rouge_l'], 
            results['final_score']
        )):
            f.write(f"{file_names[i]}\t{fs:.4f}\t{cs:.4f}\t{bs:.4f}\t{rs:.4f}\n")
    
    print(f"\n결과가 'similarity_results.txt' 파일로 저장되었습니다.")