"""
청킹되지 않은 파일들의 이름에서 _chunk_001을 제거하는 스크립트
단일 청크만 있는 경우 (chunk_001만 있는 경우) _chunk_001을 제거
"""

import os
import shutil
from pathlib import Path
import json

def fix_non_chunked_folders(base_dir: str):
    """
    단일 청크 폴더들의 이름 수정
    
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
        "renamed": 0,
        "skipped": 0,
        "errors": 0
    }
    
    # 모든 result_*_chunk_* 폴더 찾기
    chunk_folders = [d for d in base_path.iterdir() 
                     if d.is_dir() and d.name.startswith("result_") and "_chunk_" in d.name]
    
    # 폴더들을 base name으로 그룹화
    folder_groups = {}
    for folder in chunk_folders:
        # result_Bdb001_chunk_001 -> result_Bdb001
        base_name = folder.name.rsplit("_chunk_", 1)[0]
        if base_name not in folder_groups:
            folder_groups[base_name] = []
        folder_groups[base_name].append(folder)
    
    print(f"Found {len(folder_groups)} unique result folders")
    print("=" * 60)
    
    # 각 그룹 처리
    for base_name, folders in folder_groups.items():
        stats["total_folders"] += 1
        
        # 단일 청크인 경우만 처리
        if len(folders) == 1 and folders[0].name.endswith("_chunk_001"):
            old_folder = folders[0]
            new_folder_path = base_path / base_name
            
            try:
                # 이미 존재하는 경우 건너뛰기
                if new_folder_path.exists():
                    print(f"EXISTS: {base_name} already exists, skipping")
                    stats["skipped"] += 1
                    continue
                
                # 폴더 이름 변경
                shutil.move(str(old_folder), str(new_folder_path))
                print(f"RENAMED: {old_folder.name} -> {base_name}")
                stats["renamed"] += 1
                
                # 내부 JSON 파일들의 id 필드도 업데이트
                update_json_ids(new_folder_path, base_name)
                
            except Exception as e:
                print(f"ERROR: Failed to rename {old_folder.name}: {e}")
                stats["errors"] += 1
        else:
            # 여러 청크가 있는 경우 건너뛰기
            print(f"SKIP: {base_name} has {len(folders)} chunks (keeping chunk numbers)")
            stats["skipped"] += 1
    
    # 결과 출력
    print("\n" + "=" * 60)
    print("Fix Non-Chunked Folders Complete!")
    print(f"Total folder groups: {stats['total_folders']}")
    print(f"Renamed folders: {stats['renamed']}")
    print(f"Skipped folders: {stats['skipped']}")
    print(f"Errors: {stats['errors']}")
    print("=" * 60)
    
    # 통계 저장
    stats_file = base_path / "fix_non_chunked_stats.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"Statistics saved to: {stats_file}")


def update_json_ids(folder_path: Path, new_id: str):
    """
    폴더 내의 JSON 파일들의 id 필드 업데이트
    
    Args:
        folder_path: 폴더 경로
        new_id: 새로운 ID
    """
    # chunk_data.json 파일 업데이트
    chunk_data_file = folder_path / "chunk_data.json"
    if chunk_data_file.exists():
        try:
            with open(chunk_data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # ID 업데이트
            if "id" in data:
                old_id = data["id"]
                data["id"] = new_id
                
                # chunk_info 제거 또는 업데이트
                if "metadata" in data and "chunk_info" in data["metadata"]:
                    data["metadata"]["is_chunk"] = False
                    del data["metadata"]["chunk_info"]
            
            # 파일 다시 저장
            with open(chunk_data_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            print(f"  Updated JSON id: {old_id} -> {new_id}")
            
        except Exception as e:
            print(f"  WARNING: Could not update JSON file: {e}")


def verify_structure(base_dir: str):
    """
    변경된 구조 확인
    
    Args:
        base_dir: 청킹된 데이터가 있는 기본 디렉토리
    """
    base_path = Path(base_dir)
    
    # 모든 result_ 폴더들 찾기
    all_result_folders = [d for d in base_path.iterdir() 
                          if d.is_dir() and d.name.startswith("result_")]
    
    # 청크 번호가 있는 폴더와 없는 폴더 구분
    chunked_folders = [d for d in all_result_folders if "_chunk_" in d.name]
    non_chunked_folders = [d for d in all_result_folders if "_chunk_" not in d.name]
    
    # 단일 청크 폴더 찾기 (chunk_001만 있는 경우)
    single_chunk_folders = []
    for folder in chunked_folders:
        if folder.name.endswith("_chunk_001"):
            base_name = folder.name.rsplit("_chunk_", 1)[0]
            # 같은 base_name의 다른 청크가 있는지 확인
            related_chunks = [f for f in chunked_folders if f.name.startswith(base_name + "_chunk_")]
            if len(related_chunks) == 1:
                single_chunk_folders.append(folder)
    
    print("\n" + "=" * 60)
    print("Structure Verification:")
    print(f"Total result folders: {len(all_result_folders)}")
    print(f"Folders with chunk numbers: {len(chunked_folders)}")
    print(f"Folders without chunk numbers: {len(non_chunked_folders)}")
    print(f"Single chunk folders (to be renamed): {len(single_chunk_folders)}")
    
    if single_chunk_folders:
        print("\nFolders that should be renamed (single chunks):")
        for folder in single_chunk_folders[:10]:  # 처음 10개만 표시
            print(f"  - {folder.name}")
        if len(single_chunk_folders) > 10:
            print(f"  ... and {len(single_chunk_folders) - 10} more")
    
    print("=" * 60)


def main():
    """메인 실행 함수"""
    chunked_data_dir = "../Chunked_Data_all"
    
    print("=" * 60)
    print("Fix Non-Chunked Folders")
    print(f"Target directory: {chunked_data_dir}")
    print("=" * 60)
    
    # 현재 구조 확인
    print("\nBefore fixing:")
    verify_structure(chunked_data_dir)
    
    # 폴더 이름 수정
    fix_non_chunked_folders(chunked_data_dir)
    
    # 변경된 구조 확인
    print("\nAfter fixing:")
    verify_structure(chunked_data_dir)


if __name__ == "__main__":
    main()