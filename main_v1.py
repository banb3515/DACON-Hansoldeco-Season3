import builtins
import os
import json
import torch
import logging
import pandas as pd
import numpy as np

from dotenv import load_dotenv

from typing import Literal

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
            "question": (
                f"공사종류 대분류 '{row['공사종류(대분류)']}', 중분류 '{row['공사종류(중분류)']}' 공사 중 "
                f"공종 대분류 '{row['공종(대분류)']}', 중분류 '{row['공종(중분류)']}' 작업에서 "
                f"사고객체 '{row['사고객체(대분류)']}'(중분류: '{row['사고객체(중분류)']}')와 관련된 사고가 발생했습니다. "
                f"작업 프로세스는 '{row['작업프로세스']}'이며, 사고 원인은 '{row['사고원인']}'입니다. "
                f"재발 방지 대책 및 향후 조치 계획은 무엇인가요?"
            ),
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

def create_train_documents(data: pd.DataFrame) -> list[Document]:
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
<question>{question}</question>
<answer>{answer}</answer>\
""")
        for question, answer in zip(train_questions, train_answers)
    ]

def create_retriever(
    documents: list[Document],
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
    retriever: VectorStoreRetriever,
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
<role>
You are a construction safety expert.
</role>
<rules>
- Always respond in English.
- Summarize only the core content of the answer concisely.
- Never include introductions, background, or additional explanations.
- Do not include phrases such as "We suggest taking the following actions:".
- Clearly list safety measures and action plans.
- Provide the answer in a single sentence as presented in the provided examples.
- Base your answer on the provided examples.
</rules>
<examples>
${examples}
</examples>\
""")
    CHAT_SYSTEM_PROMPT = Template("""\
<role>
당신은 건설 안전 전문가입니다.
</role>
<rules>
- 항상 한국어로 답변하세요.
- 질문에 대한 답변을 핵심 내용만 요약하여 간략하게 작성하세요.
- 서론, 배경 설명 또는 추가 설명을 절대 포함하지 마세요.
- 다음과 같은 조치를 취할 것을 제안합니다: 와 같은 내용을 포함하지 마세요.
- 안전 대책과 조치 계획을 명확하게 나열하세요.
- 제공된 예시 자료를 기반으로 답변하세요.
- <think> 태그 내용을 참조하여 제공된 예시 자료와 같이 한 문장으로 답변을 작성하세요.
- 답변 끝에 "한다.", "합니다" 등 조동사를 포함하지 마세요.
</rules>
<examples>
${examples}
</examples>
<think>
${think}
</think>
""")
    
    
    def _inference(question: str):
        contexts = retriever.invoke(question)
        examples = [
            f"<example>\n{context.page_content}\n</example>"
            for context in contexts
        ]
        
        # TODO: Reranking 모델 적용
        
        # Reasoning
        reasoning_prompt = REASONING_SYSTEM_PROMPT.substitute(examples="\n".join(examples))
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
        chat_prompt = CHAT_SYSTEM_PROMPT.substitute(examples="\n".join(examples), think=reasoning_text)
        chat_output = chat_model.chat(
            [
                {"role": "system", "content": chat_prompt},
                {"role": "user", "content": question},
                {"role": "assistant", "content": f"작업전 안전교육 실시와 안전관리자 안전점검 실시를 통한 재발 방지 대책 및 향후 조치 계획: "}
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
    FAISS_INDEX_NAME = os.getenv("FAISS_INDEX_NAME", "dacon")
    
    EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")
    SEARCH_OPTIONS = json.loads(os.getenv("SEARCH_OPTIONS"))
    
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
    
    # Create Train Documents
    print("📄 Create train documents ...")
    train_documents = create_train_documents(combined_train_data)
    
    # Create Retriever
    print("🔍 Create retriever ...")
    retriever = create_retriever(
        train_documents,
        embedding,
        CACHE_PATH,
        FAISS_PATH,
        FAISS_INDEX_NAME,
        **SEARCH_OPTIONS
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
        retriever,
        reasoning_model,
        chat_model,
        reasoning_sampling_params,
        chat_sampling_params,
        combined_test_data,
    )
    
    # Create Result Embeddings
    print("🔢 Create result embeddings ...")
    result_embeddings = create_result_embeddings(results)
    
    # Save Results
    print("💾 Save results ...")
    save_results(results, result_embeddings, DATA_DIR, SUBMISSIONS_DIR)
    
    print("🏁 End !")
