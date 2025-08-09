"""
Post_Training 폴더의 모든 _processed 제거 스크립트
"""
import os
from pathlib import Path

def remove_processed_from_folders():
    """모든 폴더명에서 _processed 제거"""
    base_path = Path("Post_Training")
    
    # 각 모델 폴더 처리
    model_folders = ["1.7B_lora_model_results", "4B_lora_model_results", "8B_lora_model_results"]
    
    for model_folder in model_folders:
        model_path = base_path / model_folder
        if not model_path.exists():
            print(f"WARNING: {model_folder} 폴더가 없습니다.")
            continue
            
        print(f"\n[INFO] {model_folder} 처리 중...")
        
        # _processed가 포함된 폴더들 찾기
        processed_folders = [d for d in model_path.iterdir() 
                           if d.is_dir() and "_processed" in d.name]
        
        print(f"  - {len(processed_folders)}개 폴더 발견")
        
        # 폴더명 변경
        renamed_count = 0
        for folder in processed_folders:
            old_name = folder.name
            new_name = old_name.replace("_processed", "")
            new_path = folder.parent / new_name
            
            try:
                folder.rename(new_path)
                print(f"  - {old_name} -> {new_name}")
                renamed_count += 1
            except Exception as e:
                print(f"  - ERROR: {old_name} 변경 실패: {e}")
        
        print(f"  - {renamed_count}개 폴더 변경 완료")
    
    print(f"\n[OK] 모든 작업 완료!")

if __name__ == "__main__":
    remove_processed_from_folders()