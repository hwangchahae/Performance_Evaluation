"""
Raw_Data_all 데이터를 ttalkkak.py 방식으로 청킹하여 저장하는 스크립트
- OpenAI API 호출 없이 청킹만 수행
- 청킹된 데이터만 저장 (청킹되지 않은 작은 파일은 제외)
"""

import json
import os
import sys
from typing import List, Dict, Any
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

# UTF-8 인코딩 설정
import locale
if sys.platform == 'win32':
    # Windows에서 UTF-8 설정
    os.system('chcp 65001 > nul 2>&1')
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')


class ChunkingProcessor:
    def __init__(self):
        """청킹 처리기 초기화"""
        self.chunk_size = 5000
        self.overlap = 512
    
    def chunk_text(self, text: str, chunk_size: int = 5000, overlap: int = 512) -> List[str]:
        """텍스트를 청킹하여 나누기 (ttalkkak.py와 동일)"""
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
    
    def load_meeting_data(self, file_path: str) -> Dict[str, Any]:
        """회의 데이터를 로드하고 텍스트로 변환"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 회의 내용을 텍스트로 변환
            meeting_text = ""
            speakers = set()
            
            for item in data:
                timestamp = item.get('timestamp', 'Unknown')
                speaker = item.get('speaker', 'Unknown')
                text = item.get('text', '')
                speakers.add(speaker)
                meeting_text += f"[{timestamp}] {speaker}: {text}\n"
            
            full_text = meeting_text.strip()
            
            # 텍스트 길이 체크 및 청킹
            if len(full_text) > self.chunk_size:
                print(f"      Long text detected ({len(full_text)} chars) - Chunking...")
                chunks = self.chunk_text(full_text, chunk_size=self.chunk_size, overlap=self.overlap)
                print(f"      Split into {len(chunks)} chunks")
                
                chunk_info = {
                    "is_chunked": True,
                    "total_chunks": len(chunks),
                    "original_length": len(full_text)
                }
                
                return {
                    "transcript": None,  # 청킹된 경우 transcript는 None
                    "all_chunks": chunks,  # 모든 청크들
                    "metadata": {
                        "source_file": file_path,
                        "utterance_count": len(data),
                        "original_transcript_length": len(full_text),
                        "speakers": list(speakers),
                        "chunking_info": chunk_info
                    }
                }
            else:
                # 청킹되지 않은 경우에도 저장
                print(f"      Short text ({len(full_text)} chars) - No chunking needed")
                
                chunk_info = {
                    "is_chunked": False,
                    "total_chunks": 1,
                    "original_length": len(full_text)
                }
                
                return {
                    "transcript": full_text,
                    "all_chunks": None,
                    "metadata": {
                        "source_file": file_path,
                        "utterance_count": len(data),
                        "original_transcript_length": len(full_text),
                        "speakers": list(speakers),
                        "chunking_info": chunk_info
                    }
                }
                
        except Exception as e:
            print(f"ERROR: 파일 로드 오류 ({file_path}): {e}")
            return None
    
    def get_meeting_files(self, base_dir: str) -> List[str]:
        """05_final_result.json 파일들을 찾아서 반환"""
        target_files = []
        
        if os.path.exists(base_dir):
            for root, dirs, files in os.walk(base_dir):
                for file in files:
                    if file == "05_final_result.json":
                        target_files.append(os.path.join(root, file))
        
        return target_files
    
    def process_all_meetings(self, input_dir: str, output_dir: str):
        """모든 회의 파일을 처리하여 청킹된 데이터만 저장"""
        
        # 출력 디렉토리 생성
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 회의 파일 찾기
        meeting_files = self.get_meeting_files(input_dir)
        
        if not meeting_files:
            print(f"ERROR: {input_dir}에서 회의 파일을 찾을 수 없습니다.")
            return
        
        print(f"[DIR] Total {len(meeting_files)} meeting files found")
        
        # 통계 초기화
        stats = {
            "total_files": len(meeting_files),
            "chunked_files": 0,
            "skipped_files": 0,
            "total_chunks": 0,
            "error_files": 0
        }
        
        chunked_results = []
        
        # 각 파일 처리
        with tqdm(total=len(meeting_files), desc="Processing meeting files") as pbar:
            for file_path in meeting_files:
                # 폴더명 추출
                parent_folder = Path(file_path).parent.name
                print(f"\n[FOLDER] Processing: {parent_folder}")
                
                # 데이터 로드 및 청킹
                meeting_data = self.load_meeting_data(file_path)
                
                if meeting_data is None:
                    # 파일 로드 오류
                    print(f"   ERROR: Failed to load file")
                    stats["error_files"] += 1
                elif meeting_data.get("all_chunks"):
                    # 청킹된 데이터 저장
                    chunks = meeting_data["all_chunks"]
                    metadata = meeting_data["metadata"]
                    
                    stats["chunked_files"] += 1
                    stats["total_chunks"] += len(chunks)
                    
                    # 각 청크를 개별 JSON 파일로 저장
                    for chunk_idx, chunk_text in enumerate(chunks):
                        # 청크 데이터 구조
                        chunk_data = {
                            "id": f"{parent_folder}_chunk_{chunk_idx+1:03d}",
                            "source_dir": parent_folder,
                            "chunk_text": chunk_text,
                            "metadata": {
                                **metadata,
                                "is_chunk": True,
                                "chunk_info": {
                                    "chunk_index": chunk_idx + 1,
                                    "total_chunks": len(chunks),
                                    "chunk_length": len(chunk_text)
                                },
                                "processing_date": datetime.now().isoformat()
                            }
                        }
                        
                        # JSON 파일로 직접 저장 (폴더 없이)
                        chunk_filename = output_path / f"{parent_folder}_chunk_{chunk_idx+1:03d}.json"
                        with open(chunk_filename, 'w', encoding='utf-8') as f:
                            json.dump(chunk_data, f, ensure_ascii=False, indent=2)
                    
                    print(f"   OK: {len(chunks)} chunks saved")
                    
                    # 전체 결과에 추가
                    chunked_results.append({
                        "source_file": file_path,
                        "parent_folder": parent_folder,
                        "chunks_count": len(chunks),
                        "original_length": metadata["original_transcript_length"],
                        "speakers": metadata["speakers"]
                    })
                elif meeting_data.get("transcript"):
                    # 청킹되지 않은 데이터도 저장
                    transcript = meeting_data["transcript"]
                    metadata = meeting_data["metadata"]
                    
                    stats["skipped_files"] += 1  # 청킹은 안 했지만 저장은 함
                    
                    # 데이터 구조
                    single_data = {
                        "id": parent_folder,
                        "source_dir": parent_folder,
                        "chunk_text": transcript,
                        "metadata": {
                            **metadata,
                            "is_chunk": False,
                            "processing_date": datetime.now().isoformat()
                        }
                    }
                    
                    # JSON 파일로 직접 저장 (폴더 없이)
                    single_filename = output_path / f"{parent_folder}.json"
                    with open(single_filename, 'w', encoding='utf-8') as f:
                        json.dump(single_data, f, ensure_ascii=False, indent=2)
                    
                    print(f"   OK: Saved without chunking (< 5000 chars)")
                    
                    # 전체 결과에 추가
                    chunked_results.append({
                        "source_file": file_path,
                        "parent_folder": parent_folder,
                        "chunks_count": 1,
                        "original_length": metadata["original_transcript_length"],
                        "speakers": metadata["speakers"]
                    })
                else:
                    stats["error_files"] += 1
                    print(f"   ERROR: Processing error")
                
                pbar.update(1)
        
        # 통계 저장
        stats_file = output_path / "chunking_statistics.json"
        final_stats = {
            "statistics": stats,
            "chunked_files_info": chunked_results,
            "processing_date": datetime.now().isoformat(),
            "settings": {
                "chunk_size": self.chunk_size,
                "overlap": self.overlap
            }
        }
        
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(final_stats, f, ensure_ascii=False, indent=2)
        
        # 결과 출력
        print(f"\n" + "="*60)
        print(f"COMPLETE! Chunking process finished!")
        print(f"STATS: Statistics:")
        print(f"  - Total files: {stats['total_files']}")
        print(f"  - Chunked files: {stats['chunked_files']}")
        print(f"  - Non-chunked files (< 5000 chars): {stats['skipped_files']}")
        print(f"  - Error files: {stats['error_files']}")
        print(f"  - Total chunks created: {stats['total_chunks']}")
        print(f"[FOLDER] Output directory: {output_path}")
        print(f"="*60)


def main():
    """메인 실행 함수"""
    # 입력 및 출력 디렉토리 설정
    input_dir = "../Raw_Data_val"
    output_dir = "../Chunked_Data_val_files"
    
    print("=" * 60)
    print("Chunking Process Started!")
    print(f"Input Directory: {input_dir}")
    print(f"Output Directory: {output_dir}")
    print("Settings: chunk_size=5000, overlap=512")
    print("=" * 60)
    
    # 처리기 생성 및 실행
    processor = ChunkingProcessor()
    processor.process_all_meetings(input_dir, output_dir)


if __name__ == "__main__":
    main()