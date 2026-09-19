import re, uuid
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

emb = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5",
                            encode_kwargs={"normalize_embeddings": True})

# a heading is a line of only CAPITAL letters, like TEST RESULTS
HEADING = r"(?m)^(?=[A-Z][A-Z &/\-]{2,40}$)"

def split_by_section(text: str, source: str) -> list[Document]:
    text = text.replace("\r\n", "\n")
    fallback = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=100)
    chunks = []
    for part in re.split(HEADING, text):
        part = part.strip()
        if not part:
            continue
        first = part.splitlines()[0]
        section = first if re.fullmatch(r"[A-Z][A-Z &/\-]{2,40}", first) else "HEADER"
        # only very long sections get split further
        pieces = fallback.split_text(part) if len(part) > 900 else [part]
        for p in pieces:
            chunks.append(Document(page_content=p,
                                   metadata={"section": section, "source": source}))
    return chunks

def load_record(path: str):
    loader = PyPDFLoader(path) if path.lower().endswith(".pdf") \
             else TextLoader(path, encoding="utf-8")
    docs = loader.load()
    full_text = "\n".join(d.page_content for d in docs)

    chunks = split_by_section(full_text, path)

    sid = uuid.uuid4().hex[:8]          # one upload = one private collection
    db = Chroma.from_documents(chunks, emb, collection_name=f"s_{sid}",
                               persist_directory=f"./sessions/{sid}")
    return full_text, db.as_retriever(search_kwargs={"k": 3})