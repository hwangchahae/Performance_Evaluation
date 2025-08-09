# 회의록 유사도 평가 시스템

## 개요
이 시스템은 회의록 생성 모델의 성능을 평가하기 위한 도구입니다.
학습 전(Pre-Training)과 학습 후(Post-Training) 모델의 출력을 정답 데이터와 비교하여 유사도를 측정합니다.

## 디렉토리 구조
```
Performance_Evaluation/
├── config.json                    # 설정 파일 (경로, 평가 파라미터 등)
├── Gold_Standard_Data/            # 정답 데이터
├── Pre_Training/                  # 학습 전 평가
│   ├── pre_SimilarityEvaluator.py
│   ├── 1.7B_model_test_results/
│   ├── 4B_model_test_results/
│   └── 8B_model_test_results/
└── Post_Training/                 # 학습 후 평가
    ├── post_SimilarityEvaluator.py
    ├── qwen3_1.7B_lora_meeting_results/
    ├── qwen3_4B_lora_meeting_results/
    └── qwen3_8B_lora_meeting_results/
```

## 설정 파일 (config.json)
팀원이 자신의 환경에 맞게 수정할 수 있는 설정 파일입니다.

### 주요 설정 항목
- **paths**: 데이터 경로 설정
  - `gold_standard_data`: 정답 데이터 경로
  - `pre_training`: 학습 전 모델별 결과 경로
  - `post_training`: 학습 후 모델별 결과 경로

- **evaluation**: 평가 파라미터
  - `sample_size`: 랜덤 샘플링 크기 (기본: 100)
  - `random_seed`: 재현성을 위한 시드값 (기본: 42)
  - `tfidf_max_features`: TF-IDF 최대 특징 수 (기본: 5000)
  - `embedding_model`: OpenAI 임베딩 모델 (기본: text-embedding-3-large)

- **output**: 출력 파일명 설정

## 사용법

### 1. 환경 설정
```bash
# 필요한 패키지 설치
pip install numpy scikit-learn openai

# OpenAI API 키 설정 (환경변수)
export OPENAI_API_KEY="your-api-key"
```

### 2. 경로 설정
`config.json` 파일을 열어 자신의 환경에 맞게 경로를 수정합니다:

```json
{
    "paths": {
        "gold_standard_data": "../Gold_Standard_Data",
        "pre_training": {
            "1.7B": "./Pre_Training/1.7B_model_test_results",
            ...
        }
    }
}
```

### 3. 학습 전 모델 평가
```bash
# 1.7B 모델 평가 (기본)
python Pre_Training/pre_SimilarityEvaluator.py

# 4B 모델 평가
python Pre_Training/pre_SimilarityEvaluator.py --model 4B

# 8B 모델 평가
python Pre_Training/pre_SimilarityEvaluator.py --model 8B

# 커스텀 설정 파일 사용
python Pre_Training/pre_SimilarityEvaluator.py --model 4B --config /path/to/custom_config.json
```

### 4. 학습 후 모델 평가
```bash
# 1.7B 모델 평가 (기본)
python Post_Training/post_SimilarityEvaluator.py

# 4B 모델 평가
python Post_Training/post_SimilarityEvaluator.py --model 4B

# 8B 모델 평가
python Post_Training/post_SimilarityEvaluator.py --model 8B
```

## 출력 파일
각 평가 후 다음 3개의 파일이 생성됩니다:

1. **JSON 파일** (`*_similarity_results.json`)
   - 상세한 평가 결과
   - 각 파일별 점수

2. **CSV 파일** (`*_similarity_results.csv`)
   - 표 형식의 결과
   - Excel에서 바로 열기 가능

3. **텍스트 요약** (`*_similarity_summary.txt`)
   - 평균 점수
   - 상위/하위 5개 파일
   - 평가 메타데이터

## 평가 메트릭
- **TF-IDF Cosine Similarity**: 단어 빈도 기반 유사도
- **Embedding Cosine Similarity**: OpenAI 임베딩 기반 의미적 유사도

## 팀원을 위한 가이드

### 새로운 모델 추가하기
1. `config.json`에 새 모델 경로 추가
2. 데이터를 해당 경로에 배치
3. 위의 명령어로 평가 실행

### 경로만 변경하기
1. `config.json` 열기
2. `paths` 섹션에서 자신의 데이터 경로로 수정
3. 저장 후 평가 스크립트 실행

### 평가 파라미터 조정
1. `config.json`의 `evaluation` 섹션 수정
2. 예: 샘플 크기를 200으로 늘리기
   ```json
   "sample_size": 200
   ```

## 문제 해결

### OpenAI API 키 오류
```bash
# Windows
set OPENAI_API_KEY=your-api-key

# Linux/Mac
export OPENAI_API_KEY=your-api-key
```

### 경로 찾기 오류
- 모든 경로는 Performance_Evaluation 폴더를 기준으로 상대 경로로 작성
- 절대 경로도 사용 가능

### 메모리 부족
- `config.json`에서 `sample_size`를 줄여보세요

## 지원
문제가 있으면 팀 슬랙 채널에 문의하세요.