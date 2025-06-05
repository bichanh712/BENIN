from pathlib import Path
import pdfplumber
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from collections import defaultdict

base_pdf_dir = "data"
base_vector_dir = "vectorstore"
topics = ["labeling", "standard", "ads", "distribute", "price"]

def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                page_text = page_text.strip()
                if not page_text.endswith('.'):
                    page_text += ' '
                text += page_text + "\n"
    return text

import re
from langchain.text_splitter import RecursiveCharacterTextSplitter

def hybrid_split_legal_text(raw_text):
    # Tách thành các Điều luật
    dieu_blocks = re.split(r"(?=\n?Điều\s+\d+\.)", raw_text)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        length_function=len,
        separators=["\n\n", ".", "\n"]
    )

    chunks = []
    for block in dieu_blocks:
        block = block.strip()
        if not block or len(block) < 100:
            continue  # bỏ đoạn quá ngắn

        # Nếu block quá dài thì cắt nhỏ
        if len(block) > 900:
            sub_chunks = splitter.split_text(block)
            chunks.extend(sub_chunks)
        else:
            chunks.append(block)
    return chunks


def create_vector_by_topic():
    embedding_model = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-small"
    )

    for topic in topics:
        pdf_folder = Path(base_pdf_dir) / topic
        vector_path = Path(base_vector_dir) / f"{topic}_db"
        vector_path.parent.mkdir(parents=True, exist_ok=True)

        # Tạo và đọc file cache
        processed_cache_path = Path(f"processed_{topic}_files.txt")
        processed_files = set()
        if processed_cache_path.exists():
            processed_files = set(processed_cache_path.read_text(encoding="utf-8").splitlines())


        all_documents = []
        newly_processed_files = []  # <== Ghi lại các file mới

        for pdf_file in pdf_folder.glob("*.pdf"):
            if pdf_file.name in processed_files:
                print(f"Bỏ qua (đã xử lý): {pdf_file.name}")
                continue

            print(f"\n Đang xử lý file mới: {pdf_file.name}")
            raw_text = extract_text_from_pdf(pdf_file)
            print(f" → Độ dài văn bản: {len(raw_text)} ký tự")
            print(" Mẫu nội dung đầu tiên:\n", raw_text[:300], "\n")

            chunks = hybrid_split_legal_text(raw_text)

            print(f" → Tổng số đoạn context sinh ra: {len(chunks)}")

            for chunk in chunks:
                all_documents.append(
                    Document(
                        page_content=chunk,
                        metadata={"source": pdf_file.name, "topic": topic}
                    )
                )

            # Ghi vào cache và đánh dấu là file mới
            with open(processed_cache_path, "a", encoding="utf-8") as f:
                f.write(pdf_file.name + "\n")
            newly_processed_files.append(pdf_file.name)

        if not all_documents:
            print(f"Không tìm thấy đoạn văn nào trong topic: {topic}")
            continue

        # Gom context theo từng file
        docs_by_source = defaultdict(list)
        for doc in all_documents:
            source_file = doc.metadata.get("source", "Không rõ")
            docs_by_source[source_file].append(doc)

        print(f"\nTổng hợp context theo file trong topic: {topic}")
        for i, (source, docs) in enumerate(docs_by_source.items()):
            print(f"\n [{i+1}] File: {source} — tổng số đoạn: {len(docs)}")
            for j, doc in enumerate(docs[:2]):
                print(f"\n--- Đoạn {j+1} ---")
                print("Nội dung:\n", doc.page_content[:500], "...\n")

        print(f"Tổng số context toàn topic [{topic}]: {len(all_documents)}")

        db = FAISS.from_documents(all_documents, embedding_model)
        db.save_local(str(vector_path))
        print(f"Đã lưu vectorstore tại: {vector_path}")

        # 👉 In thông báo cập nhật nếu có file mới
        if newly_processed_files:
            print("Đã cập nhật file vào vectorstore.")

if __name__ == "__main__":
    create_vector_by_topic()
