import os
import re

# 💡 파일이 있는 폴더 경로
folder_path = "./"  # 현재 디렉토리

# 🔁 대상 패턴: single_Qwen_Qwen3_4B_transformers_result_XX_processed_YYYYMMDD_HHMMSS.json
pattern = re.compile(r"single_Qwen_Qwen3_4B_AWQ_transformers_result_(.+?)_\d+_\d+\.json")

# ✅ 파일들을 그룹화해서 chunk 번호 붙이기
file_groups = {}

for filename in os.listdir(folder_path):
    match = pattern.match(filename)
    if match:
        base = match.group(1)  # 예: Bdb001, Bed002 등
        if base not in file_groups:
            file_groups[base] = []
        file_groups[base].append(filename)

# ✅ 정렬 및 이름 변경
for base, files in file_groups.items():
    files.sort()  # 타임스탬프 기준 정렬
    if len(files) == 1:
        # 하나뿐인 경우 chunk 붙이지 않음
        old_name = files[0]
        new_name = f"Qwen3_4B_AWQ_transformers_result_{base}.json"
        src = os.path.join(folder_path, old_name)
        dst = os.path.join(folder_path, new_name)
        os.rename(src, dst)
        print(f"✅ {old_name} → {new_name}")
    else:
        # 여러 개일 경우 chunk 붙임
        for idx, old_name in enumerate(files, 1):
            new_name = f"Qwen3_4B_AWQ_transformers_result_{base}_chunk{idx}.json"
            src = os.path.join(folder_path, old_name)
            dst = os.path.join(folder_path, new_name)
            os.rename(src, dst)
            print(f"✅ {old_name} → {new_name}")
