import builtins
import os
import json
import glob
import torch
import logging
import pandas as pd
import numpy as np

from dotenv import load_dotenv

from typing import Literal, Optional

from string import Template

from datetime import datetime

from tqdm import tqdm

from vllm import LLM, SamplingParams

from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from sentence_transformers import SentenceTransformer


# vLLM 로그 비활성화
logger = logging.getLogger("vllm")
logger.setLevel(logging.ERROR)


#region 0. Utils

_log_file = None

def print(log: str, log_dir: str = "./logs") -> None:
    """Print log with timestamp

    Args:
        log (str): log message
    """
    
    global _log_file
    
    if _log_file is None:
        os.makedirs(log_dir, exist_ok=True)
        log_filename = datetime.now().strftime("%Y%m%d_%H%M%S") + ".log"
        _log_file = open(os.path.join(log_dir, log_filename), "w", encoding="utf-8")
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    
    builtins.print(f"\033[36m[\033[1m{timestamp}\033[0m\033[36m]\033[0m {log}")
    
    _log_file.write(f"[{timestamp}] {log}\n")
    _log_file.flush()

#endregion

#region 1. Data Processing

def load_data(data_dir: str = "./data") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train and test data

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: (train data, test data)
    """

    train = pd.read_csv(os.path.join(data_dir, "train.csv"), encoding="utf-8-sig")
    test = pd.read_csv(os.path.join(data_dir, "test.csv"), encoding="utf-8-sig")
    return train, test

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess data

    Args:
        df (pd.DataFrame): raw data

    Returns:
        pd.DataFrame: preprocessed data
    """
    
    df["공사종류(대분류)"] = df["공사종류"].str.split(" / ").str[0]
    df["공사종류(중분류)"] = df["공사종류"].str.split(" / ").str[1]
    df["공종(대분류)"] = df["공종"].str.split(" > ").str[0]
    df["공종(중분류)"] = df["공종"].str.split(" > ").str[1]
    df["사고객체(대분류)"] = df["사고객체"].str.split(" > ").str[0]
    df["사고객체(중분류)"] = df["사고객체"].str.split(" > ").str[1]
    return df

def create_combined_data(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    """Create combined data

    Args:
        df (pd.DataFrame): raw data
        is_train (bool, optional): whether the data is train data. Defaults to True.
    
    Returns:
        pd.DataFrame: combined data
    """
    
    combined_data = df.apply(
        lambda row: {
            "question": json.dumps({
                "공사종류(대분류)": row["공사종류(대분류)"],
                "공사종류(중분류)": row["공사종류(중분류)"],
                "공종(대분류)": row["공종(대분류)"],
                "공종(중분류)": row["공종(중분류)"],
                "사고객체(대분류)": row["사고객체(대분류)"],
                "사고객체(중분류)": row["사고객체(중분류)"],
                "작업프로세스": row["작업프로세스"],
                "사고원인": row["사고원인"]
            }, ensure_ascii=False, indent=2),
            **({"answer": row["재발방지대책 및 향후조치계획"]} if is_train else {})
        },
        axis=1
    )
    return pd.DataFrame(list(combined_data))

#endregion

#region 2. Model Loading

def load_embedding_model(model_name: str, **kwargs) -> Embeddings:
    """Load embedding model

    Args:
        model_name (str): model name
        **kwargs: additional parameters for embedding model

    Returns:
        Embeddings: loaded embedding model
    """
    
    return HuggingFaceEmbeddings(
        model_name=model_name,
        **kwargs
    )

def load_llm_model(model_name: str, **kwargs) -> LLM:
    """Load LLM model

    Args:
        model_name (str): model name
        **kwargs: additional parameters for vLLM

    Returns:
        LLM: loaded chat model
    """
    
    return LLM(
        model=model_name,
        tensor_parallel_size=torch.cuda.device_count(),
        disable_async_output_proc=True,
        disable_custom_all_reduce=True,
        **kwargs
    )

#endregion

#region 3. Main Tasks

def create_case_documents(data: pd.DataFrame) -> list[Document]:
    """Create train documents

    Args:
        data (pd.DataFrame): train dataset

    Returns:
        list[Document]: train documents
    """
    
    train_questions = data["question"].tolist()
    train_answers = data["answer"].tolist()
    
    return [
        Document(f"""\
# 건설공사 사고 상황 데이터
```json
{question}
```

# 재발 방지 대책 및 향후 조치 계획
{answer}\
""")
        for question, answer in zip(train_questions, train_answers)
    ]

def create_retriever(
    documents: Optional[list[Document]],
    embedding: Embeddings,
    cache_path: str,
    path: str,
    index_name: str,
    search_type: Literal["similarity", "mmr"] = "similarity",
    k: int = 10,
    lambda_mult: float = 0.5
) -> VectorStoreRetriever:
    """Create retriever

    Args:
        documents (list[Document]): documents
        embedding (Embeddings): embedding model
        cache_path (str): cache path
        path (str): path
        index_name (str): index name
        search_type (Literal["similarity", "mmr"], optional): search type. Defaults to "similarity".
        k (int, optional): k. Defaults to 10.
        lambda_mult (float, optional): lambda mult. Defaults to 0.5.

    Returns:
        VectorStoreRetriever: retriever
    """
    
    vectorstore = None
    
    if documents is None:
        # Load existing index
        faiss_index_path = f"{os.path.join(path, index_name)}.*"
        if len(glob.glob(faiss_index_path)) == 0:
            raise FileNotFoundError(f"Index file not found: {faiss_index_path}")
        
        vectorstore = FAISS.load_local(path, embedding, index_name, allow_dangerous_deserialization=True)
    else:
        # Create new index
        embedding_store = LocalFileStore(cache_path)
        cached_embedder = CacheBackedEmbeddings.from_bytes_store(embedding, embedding_store, namespace=f"{index_name}_embed-")
        
        embedding_texts = []
        embedding_datas = []
        metadatas = []
        
        for doc in tqdm(documents, desc="Embedding documents"):
            embedding_vector = cached_embedder.embed_documents([doc.page_content])
            embedding_texts.append(doc.page_content)
            embedding_datas.extend(embedding_vector)
            metadatas.append(doc.metadata)
        
        vectorstore = FAISS.from_embeddings(zip(embedding_texts, embedding_datas), embedding, metadatas=metadatas)
        vectorstore.save_local(path, index_name)
    
    # Create retriever
    search_kwargs = {"k": k}
    if search_type == "mmr":
        search_kwargs.update({
            "fetch_k": k * 2,
            "lambda_mult": lambda_mult
        })
    
    return vectorstore.as_retriever(
        search_type=search_type,
        **search_kwargs
    )

def exec_test(
    guideline_retriever: VectorStoreRetriever,
    case_retriever: VectorStoreRetriever,
    reasoning_model: LLM,
    chat_model: LLM,
    reasoning_sampling_params: SamplingParams,
    chat_sampling_params: SamplingParams,
    data: pd.DataFrame
) -> list[str]:
    """Execute test

    Args:
        retriever (VectorStoreRetriever): retriever
        reasoning_model (LLM): reasoning model
        chat_model (LLM): chat model
        reasoning_sampling_params (SamplingParams): reasoning sampling parameters
        chat_sampling_params (SamplingParams): chat sampling parameters
        data (pd.DataFrame): test dataset

    Returns:
        list[str]: results
    """
    
    REASONING_SYSTEM_PROMPT = Template("""\
# Role
You are a construction safety expert.

# Rules
- Always think in English.
- Analyze the cause of the accident and suggest specific preventive measures accordingly.
- Analyze the response patterns of previous cases provided to think about how to respond effectively.
- Refer to the provided construction safety guidelines to analyze the cause of the accident and suggest preventive measures.
- Include approaches that comply with the regulations and instructions specified in the construction safety guidelines.

# Construction Safety Guidelines
<guidelines>
${guidelines}
</guidelines>

# Cases
<cases>
${cases}
</cases>\
""")
    CHAT_SYSTEM_PROMPT = Template("""\
# Role
당신은 건설 안전 전문가입니다.

# Rules
- 항상 한국어로 답변하세요.
- 서론, 배경 설명, 추가 설명 없이 이전 사례와 동일한 형식으로 간결하게 답변하세요.
- 제공된 이전 사례의 응답 패턴을 분석하여 답변하세요.
- 추론 모델이 분석한 내용을 참고하여 제공된 예시 자료와 같은 형식으로 한 문장의 답변을 작성하세요.
- 건설 안전 지침에 명시된 규정과 지시사항을 준수하는 접근 방식을 포함하세요.
- 건설 안전 지침을 참고하여 사고 원인을 분석하고 예방 대책을 제시하세요.

# Construction Safety Guidelines
<guidelines>
${guidelines}
</guidelines>

# Cases
<cases>
${cases}
</cases>

# Reasoning Results
${reasoning}\
""")
    
    
    def _inference(question: str):
        guideline_contexts = guideline_retriever.invoke(question)
        case_contexts = case_retriever.invoke(question)
        
        guidelines = [
            f"<guideline>\n{context.page_content}\n</guideline>"
            for context in guideline_contexts
        ]
        cases = [
            f"<case>\n{context.page_content}\n</case>"
            for context in case_contexts
        ]
        
        # TODO: Reranking 모델 적용
        
        # Reasoning
        reasoning_prompt = REASONING_SYSTEM_PROMPT.substitute(
            guidelines="\n".join(guidelines),
            cases="\n".join(cases)
        )
        reasoning_output = reasoning_model.chat(
            [
                {"role": "system", "content": reasoning_prompt},
                {"role": "user", "content": question}
            ],
            sampling_params=reasoning_sampling_params,
            use_tqdm=False
        )
        reasoning_text = reasoning_output[0].outputs[0].text.strip()
        
        print(f"# Reasoning:\n{reasoning_text}")
        
        # Chat
        chat_prompt = CHAT_SYSTEM_PROMPT.substitute(
            guidelines="\n".join(guidelines),
            cases="\n".join(cases),
            reasoning=reasoning_text
        )
        chat_output = chat_model.chat(
            [
                {"role": "system", "content": chat_prompt},
                {"role": "user", "content": f"""\
# 건설공사 사고 상황 데이터
```json
{question}
```\
"""},
                {"role": "assistant", "content": f"# 재발 방지 대책 및 향후 조치 계획\n"}
            ],
            sampling_params=chat_sampling_params,
            use_tqdm=False
        )
        result_text = chat_output[0].outputs[0].text.strip()
        
        print(f"# Result:\n{result_text}")
        
        return result_text
    
    questions = data["question"].tolist()
    
    results = []
    for question in tqdm(questions, desc="Inferring ..."):
        results.append(_inference(question))
    
    return results

def create_result_embeddings(results: list[str], batch_size: int = 64, model_name: str = "jhgan/ko-sbert-sts") -> np.ndarray:
    """Create result embeddings

    Args:
        results (list[str]): results
        batch_size (int, optional): batch size. Defaults to 64.
        model_name (str, optional): model name. Defaults to "jhgan/ko-sbert-sts".

    Returns:
        np.ndarray: result embeddings
    """
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedding = SentenceTransformer(model_name).to(device)
    
    return embedding.encode(
        results,
        batch_size=batch_size,
        show_progress_bar=True,
        device=device
    )

def save_results(results: list[str], embeddings: np.ndarray, data_dir: str = "./data", submissions_dir: str = "./submissions") -> None:
    """Save results

    Args:
        results (list[str]): results
        embeddings (np.ndarray): embeddings
        data_dir (str, optional): data directory. Defaults to "./data".
        submissions_dir (str, optional): submissions directory. Defaults to "./submissions".
    """
    
    submission = pd.read_csv(os.path.join(data_dir, "sample_submission.csv"), encoding="utf-8-sig")
    submission.iloc[:,1] = results
    submission.iloc[:,2:] = embeddings
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(submissions_dir, exist_ok=True)
    filename = f"{submissions_dir}/submission_{timestamp}.csv"
    submission.to_csv(filename, index=False, encoding="utf-8-sig")

#endregion


if __name__ == "__main__":
    print("🚀 Start !")
    
    # Environment Variables Loading
    load_dotenv(override=True)
    
    DATA_DIR = os.getenv("DATA_DIR", "./data")
    SUBMISSIONS_DIR = os.getenv("SUBMISSIONS_DIR", "./submissions")
    CACHE_PATH = os.getenv("CACHE_PATH", "./cache")
    FAISS_PATH = os.getenv("FAISS_PATH", "./faiss")
    FAISS_GUIDELINE_INDEX_NAME = os.getenv("FAISS_GUIDELINE_INDEX_NAME", "dacon_guideline")
    FAISS_CASE_INDEX_NAME = os.getenv("FAISS_CASE_INDEX_NAME", "dacon_case")
    
    EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")
    GUIDELINE_SEARCH_OPTIONS = json.loads(os.getenv("GUIDELINE_SEARCH_OPTIONS"))
    CASE_SEARCH_OPTIONS = json.loads(os.getenv("CASE_SEARCH_OPTIONS"))
    
    REASONING_MODEL_NAME = os.getenv("REASONING_MODEL_NAME")
    REASONING_MODEL_OPTIONS = json.loads(os.getenv("REASONING_MODEL_OPTIONS", "{}"))
    REASONING_MODEL_SAMPLING_PARAMS = json.loads(os.getenv("REASONING_MODEL_SAMPLING_PARAMS", "{}"))
    
    CHAT_MODEL_NAME = os.getenv("CHAT_MODEL_NAME")
    CHAT_MODEL_OPTIONS = json.loads(os.getenv("CHAT_MODEL_OPTIONS", "{}"))
    CHAT_MODEL_SAMPLING_PARAMS = json.loads(os.getenv("CHAT_MODEL_SAMPLING_PARAMS", "{}"))
    
    # Data Loading
    print("🗂️ Data loading ...")
    train_data, test_data = load_data(DATA_DIR)
    
    # Data Preprocessing
    print("🔄 Preprocessing train data ...")
    train_data = preprocess_data(train_data)
    print("🔄 Preprocessing test data ...")
    test_data = preprocess_data(test_data)
    
    # Data Combination
    print("🔗 Create combined data ...")
    combined_train_data = create_combined_data(train_data, is_train=True)
    combined_test_data = create_combined_data(test_data, is_train=False)
    
    # Create Embedding Model
    print("📥 Loading embedding model ...")
    embedding = load_embedding_model(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True}
    )
    
    # Create Case Documents
    print("📄 Create case documents ...")
    case_documents = create_case_documents(combined_train_data)
    
    # Create Guideline Retriever
    print("🔍 Create guideline retriever ...")
    guideline_retriever = create_retriever(
        None,
        embedding,
        CACHE_PATH,
        FAISS_PATH,
        FAISS_GUIDELINE_INDEX_NAME,
        **GUIDELINE_SEARCH_OPTIONS
    )
    
    # Create Case Retriever
    print("🔍 Create case retriever ...")
    case_retriever = create_retriever(
        case_documents,
        embedding,
        CACHE_PATH,
        FAISS_PATH,
        FAISS_CASE_INDEX_NAME,
        **CASE_SEARCH_OPTIONS
    )
    
    # Reasoning Model Loading
    print("🧠 Loading reasoning model ...")
    reasoning_model = load_llm_model(
        model_name=REASONING_MODEL_NAME,
        **REASONING_MODEL_OPTIONS,
    )
    reasoning_sampling_params = SamplingParams(**REASONING_MODEL_SAMPLING_PARAMS)
    
    # Chat Model Loading
    print("💬 Loading chat model ...")
    chat_model = load_llm_model(
        model_name=CHAT_MODEL_NAME,
        **CHAT_MODEL_OPTIONS
    )
    chat_sampling_params = SamplingParams(**CHAT_MODEL_SAMPLING_PARAMS)
    
    # Execute Test
    print("🧪 Testing execution ...")
    results = exec_test(
        guideline_retriever=guideline_retriever,
        case_retriever=case_retriever,
        reasoning_model=reasoning_model,
        chat_model=chat_model,
        reasoning_sampling_params=reasoning_sampling_params,
        chat_sampling_params=chat_sampling_params,
        data=combined_test_data,
    )
    
    # Create Result Embeddings
    print("🔢 Create result embeddings ...")
    result_embeddings = create_result_embeddings(results)
    
    # Save Results
    print("💾 Save results ...")
    save_results(results, result_embeddings, DATA_DIR, SUBMISSIONS_DIR)
    
    print("🏁 End !")
