# !pip install vllm

import os, json, re
from glob import glob
from tqdm import tqdm
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import torch
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing as mp
from functools import partial
import threading

# Import prompt generation functions from other modules
from meeting_analysis_prompts import (
    generate_meeting_analysis_user_prompt,
    MEETING_ANALYSIS_SCHEMA
)

#  1. 모델 선택
model_path = "Qwen/Qwen3-4B-AWQ"
print(f"🚀 선택된 모델: {model_path}")

# 전역 모델 및 토크나이저 (프로세스별로 초기화)
llm = None
tokenizer = None
sampling_params = None

def initialize_model():
    """각 프로세스에서 모델 초기화"""
    global llm, tokenizer, sampling_params
    
    if llm is None:
        print(f"🔧 프로세스 {os.getpid()}에서 모델 초기화 중...")
        
        # VLLM 엔진 초기화
        llm = LLM(
            model=model_path,
            quantization="awq_marlin" if "AWQ" in model_path else None,
            tensor_parallel_size=1,
            max_model_len=16384,
            gpu_memory_utilization=0.7,  # 병렬 처리 시 메모리 사용량 조정
            trust_remote_code=True,
            enforce_eager=False,
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
            stop=None,
            skip_special_tokens=True,
        )
        print(f"✅ 프로세스 {os.getpid()} 모델 초기화 완료")

def clean_text(text):
    if not text:
        return ""
    
    # 특정 태그들만 제거
    text = re.sub(r'\[TGT\]', '', text)
    text = re.sub(r'\[/TGT\]', '', text)

    # 공백 정리
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def load_json_file(file_path):
    """JSON 파일을 로드 (qwen3_lora_meeting_generator_vllm.py 방식)"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 리스트가 아니면 빈 리스트 반환
        if not isinstance(data, list):
            print(f"⚠️  {file_path}는 리스트 형식이 아닙니다.")
            return []
            
        # 각 항목에서 timestamp, speaker, text 추출
        processed_data = []
        for item in data:
            if isinstance(item, dict):
                processed_data.append({
                    "timestamp": item.get("timestamp", "Unknown"),
                    "speaker": item.get("speaker", "Unknown"),
                    "text": item.get("text", "")
                })
        
        return processed_data
                
    except Exception as e:
        print(f"❌ 파일 로드 오류 ({file_path}): {e}")
        return []

def chunk_text_simple(text: str, chunk_size: int = 5000, overlap: int = 512) -> List[str]:
    """텍스트를 단순 문자 기반으로 청킹 (qwen3_lora_meeting_generator_vllm.py 방식)"""
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
    
    return chunks

def process_utterances_to_text(utterances, speakers_set=None):
    """발화 데이터를 텍스트로 변환 (qwen3 방식과 동일)"""
    if not utterances:
        return "", []
        
    meeting_lines = []
    speakers = speakers_set if speakers_set else set()
    
    for item in utterances:
        timestamp = item.get('timestamp', 'Unknown')
        speaker = item.get('speaker', 'Unknown')
        text = item.get('text', '')
        speakers.add(speaker)
        meeting_lines.append(f"[{timestamp}] {speaker}: {text}")
    
    full_text = '\n'.join(meeting_lines)
    return full_text, list(speakers)

# generate_chunk_summary 함수는 배치 처리 방식으로 대체됨
# 개별 청크 처리 대신 process_single_file_parallel에서 배치로 처리

def process_single_file_parallel(input_file_path, output_dir, model_used, folder_name, chunk_size=5000, overlap=512):
    """단일 파일을 처리하여 청크별로 저장 (배치 처리 방식)"""
    from pathlib import Path
    
    # 상대 경로로 표시
    rel_input_path = os.path.relpath(input_file_path, os.getcwd())
    print(f"\n📁 처리 중: {rel_input_path}")
    
    # 모델 초기화 (메인 프로세스에서)
    if llm is None:
        initialize_model()
    
    # 발화 데이터를 텍스트로 변환
    utterances = load_json_file(input_file_path)
    if not utterances:
        print(f"⚠️  {input_file_path}에서 유효한 데이터를 찾을 수 없습니다.")
        return 0, 1  # (성공, 실패) 튜플 반환
    
    # 발화를 텍스트로 변환
    full_text, speakers = process_utterances_to_text(utterances)
    
    if not full_text:
        print(f"⚠️  {input_file_path}에서 텍스트를 추출할 수 없습니다.")
        return 0, 1
    
    # 메타데이터 생성 (상대 경로로 변경)
    metadata = {
        "source_file": f"Raw_Data_val/{folder_name}/05_final_result.json",
        "utterance_count": len(utterances),
        "speakers": speakers,
        "original_length": len(full_text)
    }
    
    # 텍스트 길이에 따라 청킹 결정
    if len(full_text) > chunk_size:
        print(f"📊 긴 텍스트 감지 ({len(full_text)}자) - 청킹 처리")
        chunks = chunk_text_simple(full_text, chunk_size, overlap)
        print(f"📚 {len(chunks)}개 청크로 분할")
        metadata["chunking_info"] = {
            "is_chunked": True,
            "total_chunks": len(chunks)
        }
    else:
        print(f"📄 짧은 텍스트 ({len(full_text)}자) - 단일 처리")
        chunks = [full_text]
        metadata["chunking_info"] = {
            "is_chunked": False,
            "total_chunks": 1
        }
    
    success_count = 0
    fail_count = 0
    output_path = Path(output_dir)
    total_chunks = len(chunks)
    
    # 모든 청크에 대한 프롬프트를 한 번에 준비 (배치 처리)
    all_prompts = []
    chunk_infos = []
    summary_accum = ""
    
    for chunk_idx, chunk_text in enumerate(chunks):
        # 청크 정보 저장 - folder_name이 이미 result_로 시작함
        if total_chunks == 1:
            chunk_dir = output_path / folder_name
            chunk_id = folder_name
        else:
            chunk_dir = output_path / f"{folder_name}_chunk_{chunk_idx+1}"
            chunk_id = f"{folder_name}_chunk_{chunk_idx+1}"
        
        chunk_infos.append({
            "chunk_dir": chunk_dir,
            "chunk_id": chunk_id,
            "chunk_idx": chunk_idx,
            "chunk_text": chunk_text,
            "summary_accum": summary_accum
        })
        
        # 프롬프트 준비
        participants_str = ", ".join(speakers) if speakers else "알 수 없음"
        meeting_transcript = f"""참여자: {participants_str}

{chunk_text}"""
        
        system_prompt = """당신은 회의록을 분석하여 체계적인 프로젝트 기획안을 작성하는 전문가입니다.
회의에서 논의된 내용을 바탕으로 명확하고 실행 가능한 기획안을 작성해주세요.
응답은 반드시 요청된 JSON 형식으로만 제공하세요."""
        
        # meeting_analysis_user_prompt 사용
        if chunk_idx == 0:
            user_prompt = generate_meeting_analysis_user_prompt(meeting_transcript)
        else:
            additional_context = f"이전 분석 결과:\n{summary_accum}"
            user_prompt = generate_meeting_analysis_user_prompt(chunk_text, additional_context)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        all_prompts.append(formatted_prompt)
        summary_accum += f"청크 {chunk_idx+1} 처리 예정\n"
    
    # 배치로 모든 청크 처리
    print(f"🚀 {len(all_prompts)}개 청크 배치 처리 시작")
    outputs = llm.generate(all_prompts, sampling_params)
    
    # 결과 저장
    for idx, (output, chunk_info) in enumerate(zip(outputs, chunk_infos)):
        chunk_dir = chunk_info["chunk_dir"]
        chunk_id = chunk_info["chunk_id"]
        chunk_idx = chunk_info["chunk_idx"]
        
        chunk_dir.mkdir(parents=True, exist_ok=True)
        
        if output and output.outputs:
            result = output.outputs[0].text.strip()
            
            # JSON 응답 처리 - raw_text 없이 JSON만 추출
            try:
                # <think> 태그가 있으면 JSON 부분만 추출
                if "<think>" in result and "{" in result:
                    # <think> 태그 이후의 JSON 부분 찾기
                    json_start = result.find("{", result.find("</think>") if "</think>" in result else 0)
                    if json_start != -1:
                        # JSON 끝 찾기
                        bracket_count = 0
                        json_end = json_start
                        for i, char in enumerate(result[json_start:], json_start):
                            if char == '{':
                                bracket_count += 1
                            elif char == '}':
                                bracket_count -= 1
                                if bracket_count == 0:
                                    json_end = i + 1
                                    break
                        json_str = result[json_start:json_end]
                    else:
                        json_str = None
                elif "```json" in result:
                    start = result.find("```json") + 7
                    end = result.find("```", start)
                    if end != -1:
                        json_str = result[start:end].strip()
                    else:
                        json_str = result[start:].strip()
                elif result.startswith("{"):
                    json_str = result
                else:
                    json_str = None
                
                if json_str:
                    result_data = json.loads(json_str)
                else:
                    # 기본 구조 생성 (meeting_analysis 스키마)
                    result_data = {
                        "summary": "회의 요약 내용",
                        "topics": ["주요 주제 1", "주요 주제 2"],
                        "decisions": ["결정사항 1", "결정사항 2"],
                        "action_items": [
                            {"task": "할 일 1", "assignee": "담당자", "deadline": "기한"},
                            {"task": "할 일 2", "assignee": "담당자", "deadline": "기한"}
                        ],
                        "key_discussions": ["핵심 논의 1", "핵심 논의 2"],
                        "next_steps": ["다음 단계 1", "다음 단계 2"]
                    }
            except (json.JSONDecodeError, Exception) as e:
                # JSON 파싱 실패 시 기본 구조 생성 (meeting_analysis 스키마)
                result_data = {
                    "summary": "회의 요약 내용",
                    "topics": ["주요 주제 1", "주요 주제 2"],
                    "decisions": ["결정사항 1", "결정사항 2"],
                    "action_items": [
                        {"task": "할 일 1", "assignee": "담당자", "deadline": "기한"},
                        {"task": "할 일 2", "assignee": "담당자", "deadline": "기한"}
                    ],
                    "key_discussions": ["핵심 논의 1", "핵심 논의 2"],
                    "next_steps": ["다음 단계 1", "다음 단계 2"]
                }
            
            # 결과 저장 (원래 구조대로 복원)
            chunk_result = {
                "id": chunk_id,
                "source_dir": folder_name,  # folder_name이 이미 result_로 시작함
                "notion_output": result_data,
                "metadata": {
                    **metadata,
                    "is_chunk": total_chunks > 1,
                    "chunk_index": chunk_idx + 1 if total_chunks > 1 else None,
                    "processing_date": datetime.now().isoformat()
                }
            }
            
            with open(chunk_dir / "result.json", 'w', encoding='utf-8') as f:
                json.dump(chunk_result, f, ensure_ascii=False, indent=2)
            
            success_count += 1
            if total_chunks > 1:
                print(f"✅ 청크 {chunk_idx+1}/{total_chunks} 저장 완료: {chunk_dir.name}")
            else:
                print(f"✅ 저장 완료: {chunk_dir.name}")
        else:
            fail_count += 1
            if total_chunks > 1:
                print(f"❌ 청크 {chunk_idx+1}/{total_chunks} 생성 실패")
            else:
                print(f"❌ 생성 실패")
    
    return success_count, fail_count

# save_final_result_as_txt 함수는 더 이상 필요하지 않음 (각 청크별로 개별 JSON 저장)
# qwen3_lora_meeting_generator_vllm.py 방식으로 저장

def process_file_wrapper(args):
    """ThreadPoolExecutor를 위한 래퍼 함수"""
    folder_name, folder_path, json_file, model_used = args
    
    try:
        # 청킹 설정 (qwen3_lora_meeting_generator_vllm.py와 동일하게)
        success_count, fail_count = process_single_file_parallel(
            json_file, 
            folder_path, 
            model_used, 
            folder_name,
            chunk_size=5000,  # 문자 기반 청킹 크기
            overlap=512        # 오버랩 크기
        )
        
        if success_count > 0:
            return (folder_name, True, f"성공: {success_count}, 실패: {fail_count}")
        else:
            return (folder_name, False, f"모든 청크 처리 실패 (총 {fail_count}개)")
    except Exception as e:
        return (folder_name, False, str(e))

def batch_process_folders_parallel(base_dir, model_used, output_base_dir=None, max_workers=None):
    """순차적으로 여러 폴더 처리 (vLLM 안정성 확보)
    
    Args:
        base_dir: 입력 데이터 디렉토리
        model_used: 사용 모델명
        output_base_dir: 출력 디렉토리 (None이면 입력 디렉토리와 동일한 위치에 생성)
        max_workers: (사용하지 않음, 호환성 유지)
    """
    
    if not os.path.exists(base_dir):
        # 상대 경로로 표시
        rel_base_dir = os.path.relpath(base_dir, os.getcwd())
        print(f"❌ 기본 디렉토리가 존재하지 않습니다: {rel_base_dir}")
        return
    
    # 출력 디렉토리 설정
    if output_base_dir is None:
        # 입력 디렉토리의 부모 디렉토리에 결과 폴더 생성
        parent_dir = os.path.dirname(base_dir)
        output_base_dir = os.path.join(parent_dir, f"results_{model_used}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    # 출력 디렉토리 생성
    os.makedirs(output_base_dir, exist_ok=True)
    
    # 상대 경로로 표시
    rel_output_dir = os.path.relpath(output_base_dir, os.getcwd())
    print(f"📂 출력 디렉토리: {rel_output_dir}")
    
    # 하위 폴더들 찾기
    subfolders = []
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path):
            json_file = os.path.join(item_path, "05_final_result.json")
            if os.path.exists(json_file):
                # 출력 경로를 output_base_dir로 변경
                subfolders.append((item, output_base_dir, json_file, model_used))
            else:
                print(f"⚠️  {item} 폴더에 05_final_result.json 파일이 없습니다.")
    
    if not subfolders:
        print(f"❌ {base_dir}에서 처리할 수 있는 폴더를 찾을 수 없습니다.")
        return
    
    print(f"📂 총 {len(subfolders)}개 폴더를 순차 처리합니다:")
    for folder_name, _, _, _ in subfolders:
        print(f"  - {folder_name}")
    
    print(f"🚀 순차 처리 시작...")
    
    # 메인 프로세스에서 모델 초기화
    initialize_model()
    
    success_count = 0
    failed_folders = []
    
    # 순차 처리로 변경 (vLLM 안정성 확보)
    total_chunks_processed = 0
    with tqdm(total=len(subfolders), desc="📁 전체 폴더 처리", unit="folder") as pbar:
        for args in subfolders:
            folder_name = args[0]
            try:
                folder_name, success, message = process_file_wrapper(args)
                if success:
                    success_count += 1
                    # 성공/실패 청크 수 파싱
                    if "성공:" in message:
                        chunk_count = int(message.split("성공: ")[1].split(",")[0])
                        total_chunks_processed += chunk_count
                else:
                    failed_folders.append((folder_name, message))
            except Exception as e:
                failed_folders.append((folder_name, str(e)))
            
            pbar.update(1)
    
    # 결과 출력
    print(f"\n🎉 배치 처리 완료!")
    print(f"✅ 성공한 폴더: {success_count}/{len(subfolders)}")
    print(f"📊 처리된 총 청크 수: {total_chunks_processed}")
    
    if failed_folders:
        print(f"\n❌ 실패한 폴더들:")
        for folder, error in failed_folders:
            print(f"  - {folder}: {error}")

# 실행 부분
if __name__ == "__main__":
    # 간단한 모델명 설정 (필요시 직접 수정)
    model_used = "Qwen3_4B"  # 출력 폴더명에 사용될 이름
    
    # 배치 처리할 기본 디렉토리 (상대 경로로 설정)
    # 현재 스크립트 위치: FineTuned_Model/meeting_processor_vllm_improved.py
    # Raw_Data_val 위치: Performance_Evaluation/Raw_Data_val
    base_directory = "../Raw_Data_val"
    
    print(f"🚀 배치 처리 시작: {model_path} 모델 사용")
    print(f"📝 프롬프트: Meeting Analysis")
    print(f"📂 입력 디렉토리: {base_directory}")
    
    # 디렉토리 존재 확인
    if not os.path.exists(base_directory):
        print(f"❌ 입력 디렉토리를 찾을 수 없습니다: {base_directory}")
        print(f"📍 현재 작업 디렉토리: {os.path.relpath(os.getcwd(), '.')}")
        exit(1)
    
    # 출력 디렉토리 설정 - Pre_Training 폴더에 저장 (학습 전 base model 결과)
    output_directory = "../Pre_Training/4B_base_model_results"
    
    # 순차 배치 처리 실행
    batch_process_folders_parallel(
        base_dir=base_directory, 
        model_used=model_used, 
        output_base_dir=output_directory
    )