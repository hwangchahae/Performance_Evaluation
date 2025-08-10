# Optimized version with better error handling and performance
import os, json, re
from glob import glob
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import torch
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import logging
import gc
import traceback

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 모델 선택
model_path = "Qwen/Qwen3-4B-AWQ"
logger.info(f"🚀 선택된 모델: {model_path}")

# 전역 모델 및 토크나이저
llm = None
tokenizer = None
sampling_params = None

def initialize_model():
    """모델 초기화 (한 번만)"""
    global llm, tokenizer, sampling_params
    
    if llm is None:
        logger.info(f"🔧 모델 초기화 중...")
        
        try:
            # VLLM 엔진 초기화
            llm = LLM(
                model=model_path,
                quantization="awq" if "AWQ" in model_path else None,
                tensor_parallel_size=1,
                max_model_len=16384,
                gpu_memory_utilization=0.9,
                trust_remote_code=True,
                enforce_eager=False,
                max_num_seqs=256,
            )
            
            # 토크나이저 로드
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            
            # 샘플링 파라미터
            sampling_params = SamplingParams(
                temperature=0.2,
                max_tokens=2048,
                skip_special_tokens=True,
            )
            logger.info(f"✅ 모델 초기화 완료")
            
        except Exception as e:
            logger.error(f"❌ 모델 초기화 실패: {e}")
            raise

def generate_notion_project_prompt(meeting_transcript: str) -> str:
    """노션 기획안 생성 프롬프트"""
    return f"""다음 회의 전사본을 바탕으로 노션에 업로드할 프로젝트 기획안을 작성하세요.

**회의 전사본:**
{meeting_transcript}

**작성 지침:**
1. 회의에서 논의된 내용을 바탕으로 체계적인 기획안을 작성
2. 프로젝트명은 회의 내용을 바탕으로 적절히 명명
3. 목적과 목표는 명확하고 구체적으로 작성
4. 실행 계획은 실현 가능한 단계별로 구성
5. 기대 효과는 정량적/정성적 결과를 포함
6. 모든 내용은 한국어로 작성

**응답 형식:**
다음 JSON 형식으로 응답하세요:
{{
    "project_name": "프로젝트명",
    "project_purpose": "프로젝트의 주요 목적",
    "project_period": "예상 수행 기간",
    "project_manager": "담당자명",
    "core_objectives": ["목표 1", "목표 2", "목표 3"],
    "core_idea": "핵심 아이디어",
    "idea_description": "아이디어 설명",
    "execution_plan": "실행 계획",
    "expected_effects": ["효과 1", "효과 2", "효과 3"]
}}"""

def chunk_text(text: str, chunk_size: int = 5000, overlap: int = 512) -> List[str]:
    """텍스트를 청킹하여 나누기 (문자 단위)"""
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

def load_json_file(file_path: str) -> List[Dict]:
    """JSON 파일 로드 - timestamp와 speaker 정보 포함"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
            if isinstance(data, list):
                return [{"timestamp": item.get("timestamp", "Unknown"),
                        "speaker": item.get("speaker", "Unknown"), 
                        "text": item.get("text", "")} 
                       for item in data if "text" in item]
            return []
    except Exception as e:
        logger.error(f"파일 로드 오류 ({file_path}): {e}")
        return []

def batch_generate_responses(prompts: List[str]) -> List[str]:
    """배치 처리로 여러 프롬프트 동시 생성"""
    if not prompts:
        return []
    
    try:
        logger.info(f"🚀 {len(prompts)}개 프롬프트 배치 처리 중...")
        outputs = llm.generate(prompts, sampling_params)
        
        results = []
        for output in outputs:
            if output.outputs:
                results.append(output.outputs[0].text.strip())
            else:
                results.append("{}")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ 배치 생성 실패: {e}")
        # 실패 시 기본값 반환
        return ["{}" for _ in prompts]

def parse_json_response(response: str) -> Dict:
    """JSON 응답 파싱 with better error handling"""
    try:
        # JSON 블록 추출
        if "```json" in response:
            start = response.find("```json") + 7
            end = response.find("```", start)
            json_str = response[start:end].strip()
        elif "{" in response and "}" in response:
            start = response.find("{")
            end = response.rfind("}") + 1
            json_str = response[start:end]
        else:
            return {"error": "No JSON found", "raw_text": response}
        
        # JSON 파싱
        parsed = json.loads(json_str)
        
        # 필수 필드 검증
        required_fields = ["project_name", "project_purpose"]
        for field in required_fields:
            if field not in parsed:
                parsed[field] = "미정"
        
        return parsed
        
    except json.JSONDecodeError as e:
        logger.warning(f"JSON 파싱 실패: {e}")
        return {"error": "JSON parsing failed", "raw_text": response}
    except Exception as e:
        logger.warning(f"예상치 못한 오류: {e}")
        return {"error": str(e), "raw_text": response}

def process_files_batch(files_data: List[Tuple[str, str, List[str]]]) -> List[Dict]:
    """여러 파일을 배치로 처리"""
    all_prompts = []
    metadata = []
    
    # 모든 프롬프트 준비
    for folder_name, file_path, chunks in files_data:
        for idx, chunk in enumerate(chunks):
            prompt = generate_notion_project_prompt(chunk)
            all_prompts.append(prompt)
            metadata.append({
                "folder_name": folder_name,
                "file_path": file_path,
                "chunk_idx": idx,
                "total_chunks": len(chunks),
                "is_chunked": len(chunks) > 1
            })
    
    # 배치 생성
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
    
    return results

def save_results(results: List[Dict], output_dir: str):
    """결과 저장 with better error handling"""
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # 폴더별로 그룹화
        grouped = {}
        for result in results:
            folder = result["folder_name"]
            if folder not in grouped:
                grouped[folder] = []
            grouped[folder].append(result)
        
        # 저장
        saved_count = 0
        for folder_name, folder_results in grouped.items():
            for result in folder_results:
                try:
                    if result["total_chunks"] == 1:
                        # 단일 파일
                        chunk_dir = os.path.join(output_dir, folder_name)
                        chunk_id = folder_name
                    else:
                        # 청킹된 파일
                        chunk_dir = os.path.join(output_dir, f"{folder_name}_chunk_{result['chunk_idx']+1}")
                        chunk_id = f"{folder_name}_chunk_{result['chunk_idx']+1}"
                    
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
                            "processing_date": datetime.now().isoformat(),
                            "model_used": model_path
                        }
                    }
                    
                    output_file = os.path.join(chunk_dir, "result.json")
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(output_data, f, ensure_ascii=False, indent=2)
                    saved_count += 1
                    
                except Exception as e:
                    logger.error(f"❌ 결과 저장 실패 ({chunk_id}): {e}")
                    continue
        
        logger.info(f"✅ {saved_count}/{len(results)}개 결과 저장 완료")
        
    except Exception as e:
        logger.error(f"❌ 결과 저장 중 오류: {e}")
        raise

def process_with_memory_management(target_files: List[Tuple[str, str]], 
                                  output_directory: str,
                                  batch_size: int = 3,
                                  max_chunks_per_batch: int = 30):
    """메모리 관리를 고려한 처리"""
    
    current_batch = []
    current_chunk_count = 0
    batch_num = 1
    total_processed = 0
    
    for i, (folder_name, file_path) in enumerate(tqdm(target_files, desc="파일 처리")):
        try:
            # 파일 로드
            utterances = load_json_file(file_path)
            if not utterances:
                logger.warning(f"⚠️ 빈 파일 건너뜀: {file_path}")
                continue
            
            # 회의록 형식으로 텍스트 결합
            meeting_lines = []
            for utt in utterances:
                if utt.get("text"):
                    line = f"[{utt['timestamp']}] {utt['speaker']}: {utt['text']}"
                    meeting_lines.append(line)
            
            if not meeting_lines:
                logger.warning(f"⚠️ 텍스트 없음: {file_path}")
                continue
            
            full_text = "\n".join(meeting_lines)
            chunks = chunk_text(full_text, chunk_size=5000, overlap=512)
            
            # 청크 수 확인 및 배치 처리
            if current_chunk_count + len(chunks) > max_chunks_per_batch and current_batch:
                # 현재 배치 처리
                logger.info(f"📦 배치 {batch_num} 처리 중... ({current_chunk_count}개 청크)")
                results = process_files_batch(current_batch)
                save_results(results, output_directory)
                total_processed += len(current_batch)
                
                # 메모리 정리
                del results
                gc.collect()
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
                # 새 배치 시작
                batch_num += 1
                current_batch = [(folder_name, file_path, chunks)]
                current_chunk_count = len(chunks)
            else:
                # 현재 배치에 추가
                current_batch.append((folder_name, file_path, chunks))
                current_chunk_count += len(chunks)
                
        except Exception as e:
            logger.error(f"❌ 파일 처리 실패 ({file_path}): {e}")
            continue
    
    # 마지막 배치 처리
    if current_batch:
        try:
            logger.info(f"📦 배치 {batch_num} 처리 중... ({current_chunk_count}개 청크)")
            results = process_files_batch(current_batch)
            save_results(results, output_directory)
            total_processed += len(current_batch)
        except Exception as e:
            logger.error(f"❌ 마지막 배치 처리 실패: {e}")
    
    return total_processed

def validate_input_directory(base_directory: str) -> List[Tuple[str, str]]:
    """입력 디렉토리 검증 및 파일 목록 생성"""
    if not os.path.exists(base_directory):
        raise FileNotFoundError(f"입력 디렉토리를 찾을 수 없습니다: {base_directory}")
    
    target_files = []
    for folder in os.listdir(base_directory):
        folder_path = os.path.join(base_directory, folder)
        if os.path.isdir(folder_path):
            json_file = os.path.join(folder_path, "05_final_result.json")
            if os.path.exists(json_file):
                target_files.append((folder, json_file))
    
    if not target_files:
        raise ValueError(f"처리할 파일을 찾을 수 없습니다: {base_directory}")
    
    return target_files

def main():
    """메인 실행 함수"""
    # 설정
    base_directory = "../Raw_Data_val"
    output_directory = "4B_awq_model_results_optimized"
    batch_size = 3
    max_chunks_per_batch = 30
    
    logger.info("="*60)
    logger.info("🚀 최적화된 회의록 처리 시작")
    logger.info(f"📂 입력: {base_directory}")
    logger.info(f"📂 출력: {output_directory}")
    logger.info(f"⚙️ 배치 크기: {batch_size}, 최대 청크/배치: {max_chunks_per_batch}")
    logger.info("="*60)
    
    try:
        # 입력 검증
        target_files = validate_input_directory(base_directory)
        logger.info(f"📁 {len(target_files)}개 파일 발견")
        
        # 모델 초기화
        initialize_model()
        
        # 처리 시작
        start_time = datetime.now()
        processed_count = process_with_memory_management(
            target_files, 
            output_directory,
            batch_size,
            max_chunks_per_batch
        )
        
        # 완료
        elapsed_time = datetime.now() - start_time
        logger.info("="*60)
        logger.info(f"🎉 처리 완료!")
        logger.info(f"📊 처리된 파일: {processed_count}/{len(target_files)}")
        logger.info(f"⏱️ 소요 시간: {elapsed_time}")
        logger.info(f"📂 결과 저장 위치: {output_directory}")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"❌ 처리 중 오류 발생: {e}")
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()