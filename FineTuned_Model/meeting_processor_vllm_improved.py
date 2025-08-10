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
        
        # VLLM 엔진 초기화
        llm = LLM(
            model=model_path,
            quantization="awq",  # AWQ 양자화 사용
            tensor_parallel_size=1,
            max_model_len=16384,
            gpu_memory_utilization=0.85,  # GPU 메모리 여유 확보 (44.95GB 가용)
            trust_remote_code=True,
            enforce_eager=False,
            max_num_seqs=256,  # 동시 처리 수
            enable_prefix_caching=True,  # 프리픽스 캐싱 활성화
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
    """배치 처리로 여러 프롬프트 동시 생성"""
    if not prompts:
        return []
    
    logger.info(f"🚀 {len(prompts)}개 프롬프트 배치 처리 중...")
    
    # 더 큰 배치 크기로 한 번에 처리
    try:
        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)  # tqdm 비활성화로 속도 개선
    except Exception as e:
        logger.error(f"배치 생성 실패: {e}")
        # 실패 시 작은 배치로 재시도
        results = []
        for i in range(0, len(prompts), 10):
            sub_batch = prompts[i:i+10]
            sub_outputs = llm.generate(sub_batch, sampling_params, use_tqdm=False)
            for output in sub_outputs:
                if output.outputs:
                    results.append(output.outputs[0].text.strip())
                else:
                    results.append("{}")
        return results
    
    results = []
    for output in outputs:
        if output.outputs:
            results.append(output.outputs[0].text.strip())
        else:
            results.append("{}")
    
    return results

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
        for result in folder_results:
            # 청킹 여부와 관계없이 모든 파일에 대해 폴더 생성
            if result["metadata"]["is_chunked"]:
                # 청킹된 파일
                chunk_dir = os.path.join(output_dir, f"{folder_name}_chunk_{result['chunk_idx']+1}")
                chunk_id = f"{folder_name}_chunk_{result['chunk_idx']+1}"
            else:
                # 청킹되지 않은 파일도 폴더 생성
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

def main():
    """메인 실행 함수"""
    # 설정 - 상대 경로 사용
    base_directory = "../Raw_Data_val"  # Performance_Evaluation/Raw_Data_val
    output_directory = "4B_awq_model_results_improved"  # 현재 폴더에 생성
    batch_size = 10  # 한 번에 처리할 파일 수 (증가시켜 속도 개선)
    max_chunks_per_batch = 100  # 배치당 최대 청크 수 (증가시켜 속도 개선)
    
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
    
    # 배치 단위로 처리 (청크 수 제한 추가)
    current_batch = []
    current_chunk_count = 0
    batch_num = 1
    total_processed = 0
    
    for folder_name, file_path in target_files:
        utterances = load_json_file(file_path)
        if not utterances:
            logger.warning(f"⚠️ {folder_name} 파일 로드 실패, 건너뜀")
            continue
            
        # 텍스트 결합 및 청킹 (qwen3_lora와 완전히 동일한 방식)
        meeting_lines = []
        for utt in utterances:
            timestamp = utt.get('timestamp', 'Unknown')
            speaker = utt.get('speaker', 'Unknown')
            text = utt.get('text', '')
            meeting_lines.append(f"[{timestamp}] {speaker}: {text}")
        full_text = "\n".join(meeting_lines)
        chunks = chunk_text(full_text, chunk_size=5000, overlap=512)
        
        logger.info(f"📄 {folder_name}: {len(chunks)}개 청크 생성 (원본 {len(full_text)}자)")
        
        # 청크 수 확인
        if current_chunk_count + len(chunks) > max_chunks_per_batch and current_batch:
            # 현재 배치 처리
            logger.info(f"📦 배치 {batch_num} 처리 중... ({len(current_batch)}개 파일, {current_chunk_count}개 청크)")
            results = process_files_batch(current_batch)
            save_results(results, output_directory)
            total_processed += len(current_batch)
            
            # 새 배치 시작
            batch_num += 1
            current_batch = [(folder_name, file_path, chunks)]
            current_chunk_count = len(chunks)
        else:
            # 현재 배치에 추가
            current_batch.append((folder_name, file_path, chunks))
            current_chunk_count += len(chunks)
    
    # 마지막 배치 처리
    if current_batch:
        logger.info(f"📦 배치 {batch_num} 처리 중... ({len(current_batch)}개 파일, {current_chunk_count}개 청크)")
        results = process_files_batch(current_batch)
        save_results(results, output_directory)
        total_processed += len(current_batch)
    
    logger.info(f"📊 총 {total_processed}개 파일 처리 완료")
    
    logger.info("🎉 모든 처리 완료!")

if __name__ == "__main__":
    main()