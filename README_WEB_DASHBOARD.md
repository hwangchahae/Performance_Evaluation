# Performance Evaluation Web Dashboard

## 개요
Pre-Training, Post-Training, Gold Standard 데이터를 웹에서 한 번에 확인하고 비교할 수 있는 대시보드입니다.

## 기능

### 1. 데이터 통계 확인
- Gold Standard 파일 개수
- 각 모델별 (1.7B, 4B, 8B) Pre-Training 결과 개수
- 각 모델별 Post-Training 결과 개수

### 2. 데이터 비교 기능
- **Comparison Tab**: Pre-Training, Post-Training, Gold Standard 데이터를 나란히 비교
- **Gold Standard Tab**: 정답 데이터만 확인
- **Pre-Training Tab**: 학습 전 결과만 확인
- **Post-Training Tab**: 학습 후 결과만 확인
- **Performance Tab**: 전체 성능 요약 테이블

### 3. 모델 선택
- 1.7B, 4B, 8B 모델 간 전환
- 각 모델별 결과 파일 목록 표시

### 4. 검색 기능
- 파일명으로 빠른 검색
- 실시간 필터링

## 실행 방법

### 방법 1: 배치 파일 사용
```bash
start_dashboard.bat
```

### 방법 2: Python 직접 실행
```bash
python web_dashboard.py
```

## 접속 주소
```
http://localhost:5000
```

## 사용 방법

1. **모델 선택**: 상단의 1.7B, 4B, 8B 버튼으로 모델 선택
2. **파일 선택**: 왼쪽 사이드바에서 확인하고 싶은 결과 파일 클릭
3. **탭 전환**: 
   - Comparison: 3개 데이터 동시 비교
   - Gold Standard: 정답 데이터만
   - Pre-Training: 학습 전 결과만
   - Post-Training: 학습 후 결과만
   - Performance: 성능 요약 표
4. **검색**: 상단 검색창에서 파일명 검색

## API 엔드포인트

- `/api/overview` - 전체 통계 정보
- `/api/gold_standard` - Gold Standard 파일 목록
- `/api/results/<model>/<type>` - 모델별 결과 (type: pre/post)
- `/api/compare/<model>/<folder>` - 데이터 비교
- `/api/performance_summary` - 성능 요약
- `/api/search` - 데이터 검색

## 파일 구조
```
Performance_Evaluation/
├── web_dashboard.py          # Flask 서버
├── templates/
│   └── dashboard.html        # 웹 인터페이스
├── start_dashboard.bat       # 실행 스크립트
├── Pre_Training/            # 학습 전 데이터
├── Post_Training/           # 학습 후 데이터
└── Gold_Standard_Data/      # 정답 데이터
```

## 주요 특징

1. **실시간 데이터 비교**: Pre/Post/Gold 데이터를 한 화면에서 비교
2. **반응형 디자인**: 모바일/태블릿에서도 사용 가능
3. **빠른 검색**: 파일명 기반 실시간 검색
4. **성능 시각화**: 개선율을 색상으로 표시 (긍정: 초록, 부정: 빨강)
5. **JSON 포맷팅**: 읽기 쉬운 형태로 JSON 데이터 표시

## 종료 방법
터미널에서 `Ctrl + C`를 눌러 서버 종료