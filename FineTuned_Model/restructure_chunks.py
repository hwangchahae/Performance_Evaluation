"""
청킹된 데이터의 디렉토리 구조를 변경하는 스크립트
변경 전: Chunked_Data_val/result_Bdb001/chunk_001/
변경 후: Chunked_Data_val/result_Bdb001_chunk_001/
"""

import os
import sys
import shutil
from pathlib import Path
import json

# UTF-8 인코딩 설정
if sys.platform == 'win32':
    # Windows에서 UTF-8 설정
    os.system('chcp 65001 > nul 2>&1')
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

def restructure_chunked_data(base_dir: str):
    """
    디렉토리 구조 변경
    
    Args:
        base_dir: 청킹된 데이터가 있는 기본 디렉토리
    """
    base_path = Path(base_dir)
    
    if not base_path.exists():
        print(f"Error: Directory {base_dir} does not exist")
        return
    
    # 통계
    stats = {
        "total_folders": 0,
        "restructured": 0,
        "skipped": 0,
        "errors": 0
    }
    
    # 모든 result_ 폴더 찾기
    result_folders = [d for d in base_path.iterdir() if d.is_dir() and d.name.startswith("result_")]
    stats["total_folders"] = len(result_folders)
    
    print(f"Found {len(result_folders)} result folders to process")
    print("=" * 60)
    
    for result_folder in result_folders:
        folder_name = result_folder.name
        
        # chunk_ 하위 폴더들 찾기
        chunk_folders = [d for d in result_folder.iterdir() if d.is_dir() and d.name.startswith("chunk_")]
        
        if not chunk_folders:
            print(f"SKIP: {folder_name} - No chunk folders found")
            stats["skipped"] += 1
            continue
        
        print(f"Processing: {folder_name} ({len(chunk_folders)} chunks)")
        
        # 각 chunk 폴더를 새로운 위치로 이동
        for chunk_folder in chunk_folders:
            chunk_name = chunk_folder.name  # e.g., "chunk_001"
            
            # 새로운 폴더명 생성
            new_folder_name = f"{folder_name}_{chunk_name}"  # e.g., "result_Bdb001_chunk_001"
            new_folder_path = base_path / new_folder_name
            
            try:
                # 이미 존재하는 경우 건너뛰기
                if new_folder_path.exists():
                    print(f"  EXISTS: {new_folder_name} already exists, skipping")
                    continue
                
                # 폴더 이동
                shutil.move(str(chunk_folder), str(new_folder_path))
                print(f"  MOVED: {chunk_folder.name} -> {new_folder_name}")
                stats["restructured"] += 1
                
            except Exception as e:
                print(f"  ERROR: Failed to move {chunk_folder.name}: {e}")
                stats["errors"] += 1
        
        # 원본 result_ 폴더가 비어있으면 삭제
        try:
            remaining_items = list(result_folder.iterdir())
            if not remaining_items:
                result_folder.rmdir()
                print(f"  REMOVED: Empty folder {folder_name}")
            else:
                print(f"  KEPT: {folder_name} still has {len(remaining_items)} items")
        except Exception as e:
            print(f"  WARNING: Could not remove {folder_name}: {e}")
    
    # 결과 출력
    print("\n" + "=" * 60)
    print("Restructuring Complete!")
    print(f"Total folders processed: {stats['total_folders']}")
    print(f"Chunks restructured: {stats['restructured']}")
    print(f"Folders skipped: {stats['skipped']}")
    print(f"Errors: {stats['errors']}")
    print("=" * 60)
    
    # 통계 저장
    stats_file = base_path / "restructure_stats.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"Statistics saved to: {stats_file}")


def verify_structure(base_dir: str):
    """
    변경된 구조 확인
    
    Args:
        base_dir: 청킹된 데이터가 있는 기본 디렉토리
    """
    base_path = Path(base_dir)
    
    # 새로운 구조의 폴더들 찾기
    new_structure_folders = [d for d in base_path.iterdir() 
                            if d.is_dir() and "_chunk_" in d.name]
    
    # 이전 구조의 폴더들 찾기
    old_structure_folders = []
    for d in base_path.iterdir():
        if d.is_dir() and d.name.startswith("result_") and "_chunk_" not in d.name:
            # chunk_ 하위 폴더가 있는지 확인
            chunk_subfolders = [sd for sd in d.iterdir() 
                              if sd.is_dir() and sd.name.startswith("chunk_")]
            if chunk_subfolders:
                old_structure_folders.append(d)
    
    print("\n" + "=" * 60)
    print("Structure Verification:")
    print(f"New structure folders (result_*_chunk_*): {len(new_structure_folders)}")
    print(f"Old structure folders (with chunk subfolders): {len(old_structure_folders)}")
    
    if old_structure_folders:
        print("\nFolders still using old structure:")
        for folder in old_structure_folders[:10]:  # 처음 10개만 표시
            print(f"  - {folder.name}")
        if len(old_structure_folders) > 10:
            print(f"  ... and {len(old_structure_folders) - 10} more")
    
    print("=" * 60)


def main():
    """메인 실행 함수"""
    chunked_data_dir = "../Chunked_Data_val"
    
    print("=" * 60)
    print("Directory Structure Restructuring")
    print(f"Target directory: {chunked_data_dir}")
    print("=" * 60)
    
    # 구조 변경 실행
    restructure_chunked_data(chunked_data_dir)
    
    # 변경된 구조 확인
    verify_structure(chunked_data_dir)


if __name__ == "__main__":
    main()