# Improved version with better chunking and batch processing
import os, json, re
from glob import glob
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import torch
from datetime import datetime
from typing import List, Dict, Tuple
import logging
import argparse

# 회의 분석 프롬프트 임포트
from meeting_analysis_prompts import (
    generate_meeting_analysis_system_prompt,
    generate_meeting_analysis_user_prompt
)

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 모델 선택 (기본값, 나중에 argparse로 변경)
model_path = "Qwen/Qwen3-4B-AWQ"

# 전역 모델 및 토크나이저
llm = None
tokenizer = None
sampling_params = None

def initialize_model():
    """모델 초기화 (한 번만)"""
    global llm, tokenizer, sampling_params
    
    if llm is None:
        logger.info(f"🔧 모델 초기화 중...")
        
        # VLLM 엔진 초기화 - 속도 최적화
        llm = LLM(
            model=model_path,
            tensor_parallel_size=1,
            max_model_len=8192,  # 컨텍스트 길이 감소로 속도 향상
            gpu_memory_utilization=0.85,
            trust_remote_code=True,
            enforce_eager=False,  # CUDA graphs 사용
            max_num_seqs=512,  # 더 많은 동시 처리
            enable_prefix_caching=True,
            max_num_batched_tokens=8192,
            dtype="half",  # float16으로 속도 향상
        )
        
        # 토크나이저 로드
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        # 샘플링 파라미터 - 속도 최적화
        sampling_params = SamplingParams(
            temperature=0.1,  # 더 낮은 temperature로 빠른 수렴
            max_tokens=1024,  # 토큰 수 감소
            skip_special_tokens=True,
            top_p=0.9,
            repetition_penalty=1.05,
        )
        logger.info(f"✅ 모델 초기화 완료")

def chunk_text(text: str, chunk_size: int = 5000, overlap: int = 512) -> List[str]:
    """텍스트를 청킹하여 나누기 (문자 단위) - qwen3_lora와 동일"""
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        if end >= len(text):
            chunk = text[start:]
        else:
            chunk = text[start:end]
            
            # 마지막 완전한 문장에서 끊기 시도
            last_period = chunk.rfind('.')
            last_newline = chunk.rfind('\n')
            break_point = max(last_period, last_newline)
            
            if break_point > start + chunk_size // 2:
                chunk = text[start:break_point + 1]
                end = break_point + 1
        
        chunks.append(chunk.strip())
        
        if end >= len(text):
            break
            
        start = end - overlap
    
    logger.info(f"📊 {len(chunks)}개 청크 생성 (문자 기반 5000자)")
    return chunks

def clean_text(text):
    """텍스트 정리"""
    if not text:
        return ""
    text = re.sub(r'\[TGT\]|\[/TGT\]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def load_json_file(file_path):
    """JSON 파일 로드"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
            if isinstance(data, list):
                # qwen3_lora와 동일하게 clean_text 제거
                return [{"timestamp": item.get("timestamp", "Unknown"),
                        "speaker": item.get("speaker", "Unknown"), 
                        "text": item.get("text", "")} 
                       for item in data]  # 모든 item 포함 (text 없어도)
            return []
    except Exception as e:
        logger.error(f"파일 로드 오류 ({file_path}): {e}")
        return []

def batch_generate_responses(prompts: List[str]) -> List[str]:
    """배치 처리로 여러 프롬프트 동시 생성 - 최대 속도"""
    if not prompts:
        return []
    
    logger.info(f"🚀 {len(prompts)}개 프롬프트 배치 처리 중...")
    
    # 모든 프롬프트를 한 번에 처리
    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
    
    # 빠른 결과 추출
    return [output.outputs[0].text.strip() if output.outputs else "{}" for output in outputs]

def parse_json_response(response: str) -> Dict:
    """JSON 응답 파싱"""
    try:
        if "```json" in response:
            start = response.find("```json") + 7
            end = response.find("```", start)
            return json.loads(response[start:end].strip())
        elif "{" in response:
            start = response.find("{")
            end = response.rfind("}") + 1
            return json.loads(response[start:end])
    except:
        pass
    return {"raw_text": response}

def process_files_batch(files_data: List[Tuple[str, str, List[str]]]) -> List[Dict]:
    """여러 파일을 배치로 처리"""
    all_prompts = []
    metadata = []
    
    # 시스템 프롬프트 생성 (회의 분석용)
    system_prompt = generate_meeting_analysis_system_prompt(num_tasks=5)
    
    # 모든 프롬프트 준비
    for folder_name, file_path, chunks in files_data:
        for idx, chunk in enumerate(chunks):
            # 사용자 프롬프트 생성
            user_prompt = generate_meeting_analysis_user_prompt(chunk)
            
            # 시스템 프롬프트와 사용자 프롬프트 결합
            full_prompt = f"{system_prompt}\n\n{user_prompt}"
            
            all_prompts.append(full_prompt)
            metadata.append({
                "folder_name": folder_name,
                "file_path": file_path,
                "chunk_idx": idx,
                "total_chunks": len(chunks),
                "is_chunked": len(chunks) > 1,
                "prompt_type": "meeting_analysis"
            })
    
    # 배치 생성
    logger.info(f"🔄 {len(all_prompts)}개 프롬프트 배치 생성 중...")
    responses = batch_generate_responses(all_prompts)
    
    # 결과 정리
    results = []
    for response, meta in zip(responses, metadata):
        result = {
            "folder_name": meta["folder_name"],
            "chunk_idx": meta["chunk_idx"],
            "total_chunks": meta["total_chunks"],
            "response": parse_json_response(response),
            "metadata": meta
        }
        results.append(result)
    
    logger.info(f"✅ {len(results)}개 결과 생성 완료")
    return results

def save_results(results: List[Dict], output_dir: str):
    """결과 저장"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 폴더별로 그룹화
    grouped = {}
    for result in results:
        folder = result["folder_name"]
        if folder not in grouped:
            grouped[folder] = []
        grouped[folder].append(result)
    
    # 저장 - 모든 파일 개별 폴더로 저장
    saved_count = 0
    for folder_name, folder_results in grouped.items():
        total_chunks = len(folder_results)
        for result in folder_results:
            # qwen3_lora와 동일한 로직: 청크가 1개면 _chunk_X 붙이지 않음
            if result["metadata"]["is_chunked"] and total_chunks > 1:
                # 청킹된 파일이고 청크가 2개 이상일 때만 _chunk_ 붙임
                chunk_dir = os.path.join(output_dir, f"{folder_name}_chunk_{result['chunk_idx']+1}")
                chunk_id = f"{folder_name}_chunk_{result['chunk_idx']+1}"
            else:
                # 청크가 1개이거나 청킹되지 않은 파일
                chunk_dir = os.path.join(output_dir, folder_name)
                chunk_id = folder_name
            
            os.makedirs(chunk_dir, exist_ok=True)
            
            output_data = {
                "id": chunk_id,
                "source_dir": folder_name,
                "notion_output": result["response"],
                "metadata": {
                    "source_file": result["metadata"]["file_path"],
                    "is_chunk": result["metadata"]["is_chunked"],
                    "chunk_index": result["chunk_idx"] + 1 if result["metadata"]["is_chunked"] else None,
                    "total_chunks": result["total_chunks"],
                    "processing_date": datetime.now().isoformat()
                }
            }
            
            with open(os.path.join(chunk_dir, "result.json"), 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            saved_count += 1
    
    logger.info(f"✅ {saved_count}개 결과 저장 완료 ({len(grouped)}개 원본 파일)")

def main(input_dir=None, output_dir=None, model_size=None):
    """메인 실행 함수"""
    # 설정 - 속도 최적화
    base_directory = input_dir or "../Raw_Data_val"
    
    # 출력 디렉토리 설정
    if output_dir:
        output_directory = output_dir
    elif model_size:
        output_directory = f"{model_size}_meeting_analysis_results"
    else:
        output_directory = "4B_meeting_analysis_results"
    
    batch_size = 30  # 더 큰 배치 크기
    max_chunks_per_batch = 300  # 더 많은 청크를 한번에 처리
    
    logger.info(f"🚀 개선된 처리 시작")
    logger.info(f"📂 입력: {base_directory}")
    logger.info(f"📂 출력: {output_directory}")
    
    # 모델 초기화
    initialize_model()
    
    # 파일 찾기
    target_files = []
    for folder in os.listdir(base_directory):
        folder_path = os.path.join(base_directory, folder)
        if os.path.isdir(folder_path):
            json_file = os.path.join(folder_path, "05_final_result.json")
            if os.path.exists(json_file):
                target_files.append((folder, json_file))
    
    logger.info(f"📁 {len(target_files)}개 파일 발견")
    
    # 모든 파일을 한번에 처리 준비 (최대 속도)
    all_files_data = []
    total_chunks = 0
    
    for folder_name, file_path in target_files:
        utterances = load_json_file(file_path)
        if not utterances:
            logger.warning(f"⚠️ {folder_name} 파일 로드 실패, 건너뜀")
            continue
            
        # 텍스트 결합 및 청킹 (qwen3_lora와 동일)
        meeting_lines = [f"[{utt.get('timestamp', 'Unknown')}] {utt.get('speaker', 'Unknown')}: {utt.get('text', '')}" 
                        for utt in utterances]
        full_text = "\n".join(meeting_lines)
        chunks = chunk_text(full_text, chunk_size=5000, overlap=512)
        
        all_files_data.append((folder_name, file_path, chunks))
        total_chunks += len(chunks)
        logger.info(f"📄 {folder_name}: {len(chunks)}개 청크")
    
    logger.info(f"📊 총 {len(all_files_data)}개 파일, {total_chunks}개 청크 처리 시작")
    
    # 모든 파일을 한번에 배치 처리
    results = process_files_batch(all_files_data)
    save_results(results, output_directory)
    
    logger.info("🎉 모든 처리 완료!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='회의록 처리 시스템')
    parser.add_argument('--model', type=str, default="4B",
                        choices=['1.7B', '4B', '8B'],
                        help='사용할 모델 크기 (1.7B, 4B, 8B)')
    parser.add_argument('--input_dir', type=str, 
                        default="../Raw_Data_val",
                        help='입력 디렉토리 경로')
    parser.add_argument('--output_dir', type=str,
                        default=None,
                        help='출력 디렉토리 경로')
    
    args = parser.parse_args()
    
    # 모델 경로 설정
    model_paths = {
        '1.7B': "Qwen/Qwen3-1.7B-AWQ",
        '4B': "Qwen/Qwen3-4B-AWQ", 
        '8B': "Qwen/Qwen3-8B-AWQ"
    }
    
    # 전역 model_path 업데이트
    model_path = model_paths[args.model]
    logger.info(f"🚀 선택된 모델: {model_path}")
    
    main(input_dir=args.input_dir, output_dir=args.output_dir, model_size=args.model)