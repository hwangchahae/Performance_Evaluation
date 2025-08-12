import os, json, re
from glob import glob
from tqdm import tqdm
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime

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
    """JSON 파일을 로드"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 리스트가 아니면 빈 리스트 반환
        if not isinstance(data, list):
            print(f"[WARNING] {file_path}는 리스트 형식이 아닙니다.")
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
        print(f"[ERROR] 파일 로드 오류 ({file_path}): {e}")
        return []

def chunk_text_simple(text: str, chunk_size: int = 5000, overlap: int = 512) -> List[str]:
    """텍스트를 단순 문자 기반으로 청킹"""
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    print(f"\n청킹 시작:")
    print(f"  - 전체 텍스트 길이: {len(text)}자")
    print(f"  - 청크 크기: {chunk_size}자")
    print(f"  - 오버랩: {overlap}자")
    
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
        
        # 청킹 디버그 정보
        print(f"\n  청크 {len(chunks)}:")
        print(f"    - 시작 위치: {start}")
        print(f"    - 끝 위치: {end}")
        print(f"    - 청크 길이: {len(chunk)}자")
        print(f"    - 첫 50자: {chunk[:50]}...")
        print(f"    - 마지막 50자: ...{chunk[-50:]}")
        
        if end >= len(text):
            break
            
        start = end - overlap
        print(f"    - 다음 시작 위치 (오버랩 적용): {start}")
    
    print(f"\n총 {len(chunks)}개 청크 생성 완료")
    return chunks

def process_utterances_to_text(utterances, speakers_set=None):
    """발화 데이터를 텍스트로 변환"""
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

def process_single_file(input_file_path, output_dir, chunk_size=5000, overlap=512):
    """단일 파일을 처리하여 청크별로 저장"""
    from pathlib import Path
    
    # 파일명에서 폴더명 추출
    folder_name = Path(input_file_path).parent.name
    
    # 상대 경로로 표시
    rel_input_path = os.path.relpath(input_file_path, os.getcwd())
    print(f"\n처리 중: {rel_input_path}")
    print(f"원본 폴더명: {folder_name}")
    
    # 발화 데이터를 텍스트로 변환
    utterances = load_json_file(input_file_path)
    if not utterances:
        print(f"[WARNING] {input_file_path}에서 유효한 데이터를 찾을 수 없습니다.")
        return 0
    
    print(f"\n발화 데이터 정보:")
    print(f"  - 총 발화 수: {len(utterances)}")
    print(f"  - 첫 발화: {utterances[0] if utterances else 'N/A'}")
    
    # 발화를 텍스트로 변환
    full_text, speakers = process_utterances_to_text(utterances)
    
    if not full_text:
        print(f"[WARNING] {input_file_path}에서 텍스트를 추출할 수 없습니다.")
        return 0
    
    print(f"  - 화자 목록: {speakers}")
    print(f"  - 전체 텍스트 길이: {len(full_text)}자")
    
    # 메타데이터 생성
    metadata = {
        "source_file": rel_input_path,
        "utterance_count": len(utterances),
        "speakers": speakers,
        "original_length": len(full_text)
    }
    
    # 텍스트 길이에 따라 청킹 결정
    if len(full_text) > chunk_size:
        print(f"\n긴 텍스트 감지 ({len(full_text)}자) - 청킹 처리 시작")
        chunks = chunk_text_simple(full_text, chunk_size, overlap)
        metadata["chunking_info"] = {
            "is_chunked": True,
            "total_chunks": len(chunks),
            "chunk_size": chunk_size,
            "overlap": overlap
        }
    else:
        print(f"짧은 텍스트 ({len(full_text)}자) - 단일 처리")
        chunks = [full_text]
        metadata["chunking_info"] = {
            "is_chunked": False,
            "total_chunks": 1
        }
    
    output_path = Path(output_dir)
    total_chunks = len(chunks)
    
    # 각 청크를 파일로 저장
    for chunk_idx, chunk_text in enumerate(chunks):
        # 청크 디렉토리 생성
        if total_chunks == 1:
            chunk_dir = output_path / folder_name
        else:
            chunk_dir = output_path / f"{folder_name}_chunk_{chunk_idx+1}"
        
        chunk_dir.mkdir(parents=True, exist_ok=True)
        
        # 청크 메타데이터 저장
        chunk_metadata = {
            "chunk_index": chunk_idx + 1,
            "total_chunks": total_chunks,
            "chunk_size": len(chunk_text),
            "source_file": rel_input_path,
            "folder_name": folder_name,
            "speakers": speakers,
            "original_length": len(full_text),
            "chunking_info": metadata["chunking_info"],
            "processing_date": datetime.now().isoformat()
        }
        
        # 청크 텍스트 저장
        chunk_text_file = chunk_dir / "chunk_text.txt"
        with open(chunk_text_file, 'w', encoding='utf-8') as f:
            f.write(chunk_text)
        
        # 청크 메타데이터 저장
        metadata_file = chunk_dir / "chunk_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(chunk_metadata, f, ensure_ascii=False, indent=2)
        
        if total_chunks > 1:
            print(f"[OK] 청크 {chunk_idx+1}/{total_chunks} 저장 완료: {chunk_dir.name}")
        else:
            print(f"[OK] 저장 완료: {chunk_dir.name}")
        
        print(f"    - 청크 텍스트: {chunk_text_file}")
        print(f"    - 메타데이터: {metadata_file}")
    
    return total_chunks

# 실행 부분
if __name__ == "__main__":
    # 단일 파일 경로 지정
    input_file = r"C:\Users\Playdata\Desktop\Performance_Evaluation\Raw_Data_val\result_제22대국회 제427회(임시회) 제2차 산업통상자원중소벤처기업위원회(전체회의) (2025.07.15.)\05_final_result.json"
    
    # 출력 디렉토리 설정
    output_directory = "./test_chunking_output"
    
    print("=" * 50)
    print("단일 파일 청킹 테스트 시작")
    print(f"입력 파일: {input_file}")
    print(f"출력 디렉토리: {output_directory}")
    print("=" * 50)
    
    # 파일 존재 확인
    if not os.path.exists(input_file):
        print(f"[ERROR] 입력 파일을 찾을 수 없습니다: {input_file}")
        exit(1)
    
    # 출력 디렉토리 생성
    os.makedirs(output_directory, exist_ok=True)
    
    # 처리 실행 (청킹 크기를 작게 설정하여 테스트)
    total_chunks = process_single_file(
        input_file_path=input_file,
        output_dir=output_directory,
        chunk_size=5000,  # 5000자로 청킹
        overlap=512       # 512자 오버랩
    )
    
    print("\n" + "=" * 50)
    print("처리 완료!")
    print(f"총 {total_chunks}개 청크 생성")
    print(f"결과 확인: {output_directory}")
    print("=" * 50)