# JSON Comparison Viewer - 데이터 비교 도구

## 📋 개요
이 폴더는 Gold Standard, Pre-Training, Post-Training 결과를 비교하기 위한 독립적인 데이터 비교 환경입니다.

## 📁 폴더 구조
```
데이터비교/
├── json_comparison_viewer.html   # 웹 인터페이스 (브라우저에서 직접 열기)
├── json_comparison_viewer.py     # HTML 생성 스크립트
├── Gold_Standard_Data/           # 정답 데이터
├── Pre_Training/                 # 학습 전 모델 결과
└── Post_Training/                # 학습 후 모델 결과
```

## 🚀 사용 방법

### 방법 1: 기존 HTML 파일 사용
1. `json_comparison_viewer.html` 파일을 브라우저에서 직접 열기
2. 모델 선택 (1.7B, 4B, 8B)
3. 메트릭 선택 (TF-IDF/Embedding, 상위/하위 10개)
4. 파일 선택하여 비교

### 방법 2: 새로운 HTML 생성
```bash
cd C:\Users\Playdata\Desktop\Performance_Evaluation\데이터비교
python json_comparison_viewer.py
```
- 새로운 `json_comparison_viewer.html` 파일이 생성됩니다
- 최신 데이터를 반영한 비교 뷰어가 생성됩니다

## 🎯 주요 기능
- **모델별 비교**: 1.7B, 4B, 8B 모델 선택
- **메트릭별 정렬**: 
  - TF-IDF 하위 10개
  - TF-IDF 상위 10개
  - Embedding 하위 10개
  - Embedding 상위 10개
- **3개 패널 비교**: Gold Standard, Pre-Training, Post-Training 나란히 표시
- **점수 표시**: 각 파일의 TF-IDF와 Embedding 점수 확인

## 📊 데이터 소스
- **Gold_Standard_Data**: 정답 데이터 (회의록 요약 정답)
- **Pre_Training**: LoRA 파인튜닝 전 모델 결과
- **Post_Training**: LoRA 파인튜닝 후 모델 결과

## ⚙️ 필수 요구사항
- Python 3.x (json_comparison_viewer.py 실행 시)
- 최신 웹 브라우저 (Chrome, Firefox, Edge 등)

## 📝 참고사항
- 모든 데이터가 포함되어 있어 독립적으로 작동합니다
- 인터넷 연결 없이 로컬에서 사용 가능합니다
- JavaScript가 활성화된 브라우저가 필요합니다