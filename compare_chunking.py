# 전체 66개 파일에 대해 두 방식 비교
import os
import json
import re

def clean_text(text):
    if not text:
        return ''
    text = re.sub(r'\[TGT\]|\[/TGT\]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def chunk_text(text: str, chunk_size: int = 5000, overlap: int = 512):
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

input_dir = 'Raw_Data_val'
qwen3_total = 0
improved_total = 0
differences = []

for folder in os.listdir(input_dir):
    folder_path = os.path.join(input_dir, folder)
    if os.path.isdir(folder_path):
        json_file = os.path.join(folder_path, '05_final_result.json')
        if os.path.exists(json_file):
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # qwen3_lora 방식 (clean_text 없음, 모든 item 포함)
            qwen3_lines = []
            for item in data:
                timestamp = item.get('timestamp', 'Unknown')
                speaker = item.get('speaker', 'Unknown')
                text = item.get('text', '')
                qwen3_lines.append(f'[{timestamp}] {speaker}: {text}')
            
            qwen3_text = '\n'.join(qwen3_lines)
            qwen3_chunks = chunk_text(qwen3_text)
            qwen3_total += len(qwen3_chunks)
            
            # improved 방식 (clean_text 있음, text가 있는 경우만)
            improved_lines = []
            for item in data:
                if item.get('text'):  # text가 있는 경우만 추가
                    timestamp = item.get('timestamp', 'Unknown')
                    speaker = item.get('speaker', 'Unknown')
                    text = clean_text(item.get('text', ''))
                    improved_lines.append(f'[{timestamp}] {speaker}: {text}')
            
            improved_text = '\n'.join(improved_lines)
            improved_chunks = chunk_text(improved_text)
            improved_total += len(improved_chunks)
            
            if len(qwen3_chunks) != len(improved_chunks):
                differences.append(f'{folder}: qwen3={len(qwen3_chunks)}, improved={len(improved_chunks)}')

print('='*60)
print('청킹 비교 결과:')
print('='*60)

if differences:
    print('차이가 있는 파일들:')
    for diff in differences:
        print(f'  {diff}')
else:
    print('모든 파일의 청킹 결과가 동일합니다.')

print(f'\n총합:')
print(f'qwen3_lora: {qwen3_total}개')
print(f'improved: {improved_total}개')
print(f'차이: {abs(qwen3_total - improved_total)}개')

# 주요 차이점 분석
print('\n주요 차이점:')
print('1. qwen3_lora: text가 없어도 모든 item 포함')
print('2. improved: text가 있는 item만 포함 (if item.get("text"))')
print('3. improved: clean_text로 공백 정리')