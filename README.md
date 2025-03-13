# DACON Hansoldeco Season3

본 프로젝트는 [데이콘 - 건설공사 사고 예방 및 대응책 생성 : 한솔데코 시즌3 AI 경진대회](https://dacon.io/competitions/official/236455/overview) 참가를 목적으로 개발되었습니다.

## 개발 환경

- Python 3.10+
- Poetry (의존성 관리)
- CUDA 지원 환경 (GPU 가속)
- NVIDIA H100 NVL 1ea (94 GB)

## 사용된 라이브러리

- **vllm** : LLM 추론
- **pandas** : 데이터 처리
- **tqdm** : 진행 상황 시각화
- **langchain** : LLM 애플리케이션 개발 프레임워크 (Embeddings, Vector Store Retriever 등)
- **faiss** : 벡터 유사도 검색
- **python-dotenv** : 환경 변수 관리

## 데이터셋 정보

데이터셋은 [대회에서 제공되는 데이터셋](https://dacon.io/competitions/official/236455/data)을 사용하였습니다.

```
data/
├── train.csv              # 학습 데이터셋 (사고 상황 및 재발방지대책)
├── test.csv               # 테스트 데이터셋 (사고 상황)
├── sample_submission.csv  # 제출 양식
└── pdf/*.pdf              # 건설 안전 지침 문서
```

## 프로젝트 구조

```
.
├── guideline/             # 건설 안전 지침 RAG 생성 관련 코드 (의존성 문제로 패키지 분리)
├── .env.template          # 환경 변수 템플릿 파일 (파일명 변경 후 사용 : .env)
├── main.py                # 메인 코드
├── main_v1.py             # 이전 버전 코드
├── pyproject.toml         # Poetry 프로젝트 설정 및 의존성 관리 파일
├── run.sh                 # 실행 스크립트
├── run_nohup.sh           # 백그라운드 실행 스크립트
└── view_log.sh            # 로그 확인 스크립트 (백그라운드 실행 시)
```

## 주요 기능

건설공사 사고 데이터를 분석하고, 사고 상황에 대한 재발방지대책을 LLM을 통해 생성합니다.

1. **데이터 처리**: 건설 사고 데이터 전처리 및 분석
   - 사고 정보 데이터 정제 및 통합
   - 학습 및 테스트 데이터셋 구성

2. **RAG 데이터 생성 및 임베딩**: 사고 데이터와 안전 지침에 대한 벡터 임베딩 생성
   - BGE-m3-ko 모델을 활용한 텍스트 임베딩
   - 캐시를 활용해 임베딩 데이터 재사용

3. **유사 사례 검색**: FAISS를 활용한 유사 사고 사례 검색
   - 벡터 유사도 기반 검색 (Similarity 또는 MMR 방식)
   - 건설 안전 지침 및 유사 사례 검색

4. **AI 추론 파이프라인**: 2단계 추론 프로세스
   - 1단계: Reasoning 모델(DeepSeek-R1)을 통한 사고 분석 및 예방 대책 도출 과정 수행
   - 2단계: 일반 대화 모델(Mistral)을 통한 최종 재발방지대책 생성

## 모델 구성

- **임베딩 모델**: dragonkue/BGE-m3-ko
- **추론 모델**: deepseek-ai/DeepSeek-R1-Distill-Qwen-14B
- **대화 모델**: stelterlab/Mistral-Small-24B-Instruct-2501-AWQ

## 실행 방법

1. 가상 환경 생성(Anaconda 기준):
    ```bash
    conda create -n dacon python=3.10
    conda activate dacon
    ```

2. 환경 설정:
   ```bash
   cp .env.template .env
   # .env 파일 내용 수정
   ```

3. 의존성 설치:
   ```bash
   pip install poetry==1.5.1
   poetry install
   ```

4. 실행:
   ```bash
   ./run.sh
   ```
   
   또는 백그라운드 실행:
   ```bash
   ./run_nohup.sh
   ./view_log.sh
   ```
