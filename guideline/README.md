# Guideline

## 사용된 라이브러리

- **pandas** : 데이터 처리
- **tqdm** : 진행 상황 시각화
- **langchain** : LLM 애플리케이션 개발 프레임워크 (Embeddings, Vector Store Retriever 등)
- **faiss** : 벡터 유사도 검색
- **python-dotenv** : 환경 변수 관리
- **docling** : PDF 데이터 파싱 (to Markdown)
- **pypdf** : PDF 데이터 파싱 (Docling 오류 발생 시 Fallback)

## 실행 방법

1. 가상 환경 생성(Anaconda 기준):
    ```bash
    conda create -n dacon-guideline python=3.10
    conda activate dacon-guideline
    ```

2. 의존성 설치:
   ```bash
   pip install poetry==1.5.1
   poetry install
   ```

3. 실행:
   ```bash
   ./run.sh
   ```
