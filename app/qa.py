from langchain_community.llms.ctransformers import CTransformers
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import re
# Cấu hình
model_file = "vilm/vinallama-7b-chat-GGUF"
base_vector_path = "vectorstore"
embedding_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-small")

# Chủ đề phân loại
topic_descriptions = {
    "labeling": "Câu hỏi liên quan đến ghi nhãn, thành phần, hạn dùng, hướng dẫn sử dụng, hàng hoá, sản phẩm.",
    "standard": "Câu hỏi về tiêu chuẩn chất lượng, phương pháp kiểm tra, quy trình đánh giá, LCA",
    "ads": "Câu hỏi về quảng cáo sản phẩm, nội dung quảng cáo, hình thức quảng cáo, thông điệp truyền thông, các định nghĩa liên quan đến quảng cáo",
    "distribute": "Câu hỏi liên quan đến phân phối hàng hoá, kênh phân phối, nhà phân phối, vận chuyển, kho bãi.",
    "price": "Câu hỏi liên quan đến giá cả, niêm yết giá, định giá sản phẩm, chính sách giá, khuyến mãi."
}

# Phân loại semantic
def classify_semantic_intent(question):
    q_vec = embedding_model.embed_query(question)
    best_score = -1
    best_topic = "standard"

    for topic, desc in topic_descriptions.items():
        desc_vec = embedding_model.embed_query(desc)
        score = cosine_similarity([q_vec], [desc_vec])[0][0]
        if score > best_score:
            best_score = score
            best_topic = topic

    return best_topic

# Prompt cho từng chủ đề
def create_prompt(template):
    return PromptTemplate(template=template, input_variables=["context", "question"])

prompts = {
    "labeling": create_prompt("""
<|im_start|>system
Bạn là chuyên gia pháp lý về ghi nhãn hàng hóa. Dưới đây là nội dung pháp lý liên quan:

{context}
                              
Trả lời chính xác, đầy đủ nội dung liên quan, không lặp lại, không suy đoán.
<|im_end|> 
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
"""),

    "standard": create_prompt("""
<|im_start|>system
Bạn là chuyên gia về tiêu chuẩn chất lượng sản phẩm. Dưới đây là thông tin pháp lý cần thiết:

{context}
                             
Trả lời đúng trọng tâm câu hỏi, đầy đủ nội dung liên quan, không được lặp từ, không suy đoán những câu vô nghĩa.
Kết thúc câu trả lời bằng dấu chấm. Dừng và không hỏi hoặc trả lời thêm.
<|im_end|> 
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
"""),

    "ads": create_prompt("""<|im_start|>system
Bạn là chuyên gia pháp lý về quảng cáo sản phẩm. Dưới đây là thông tin liên quan:

{context}
                         
Trả lời đầy đủ, chính xác và không suy đoán. Không lặp từ. Kết thúc bằng dấu chấm.
<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
"""),

    "distribute": create_prompt("""<|im_start|>system
Bạn là chuyên gia pháp lý về phân phối hàng hoá. Dưới đây là thông tin cần thiết:

{context}
                                
Trả lời đúng trọng tâm, rõ ràng, không lặp lại nội dung và không suy đoán.
<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
"""),

    "price": create_prompt("""<|im_start|>system
Bạn là chuyên gia pháp lý về định giá và niêm yết giá sản phẩm. Dưới đây là thông tin liên quan:

{context}
                           
Trả lời chính xác, ngắn gọn, không lặp từ và không suy đoán. Kết thúc bằng dấu chấm.
<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
""")}
#Trả lời ngắn gọn, đúng trọng tâm, kết thúc bằng dấu chấm.
#Không được bịa, không lặp từ, không suy đoán.
#Không đặt câu hỏi phụ hay yêu cầu bổ sung thông tin. Nếu cần, user sẽ tự hỏi tiếp.
#Trả lời chính xác, đầy đủ nội dung liên quan, không lặp lại, không suy đoán.
#Không đặt câu hỏi phụ hay yêu cầu bổ sung thông tin. Nếu cần, user sẽ tự hỏi tiếp
#Kết thúc câu trả lời bằng dấu chấm.
# Chỉ trả lời đúng nội dung câu hỏi, Không lặp lại hay in lại đoạn văn bản gốc.
# Tạo LLM


def load_llm():
    return CTransformers(
        model=model_file,
        model_type="llama",
        max_new_tokens=128,
        temperature=0.01,
        repetition_penalty=2,
        stop=["<|im", "<|im_end|>"]
    )
 
# Kiểm tra context có chứa từ khóa chính của câu hỏi
def context_covers_question(question, docs):
    keywords = question.lower().split()[:5]
    context_text = " ".join(doc.page_content.lower() for doc in docs)
    return all(word in context_text for word in keywords if len(word) > 3)

# Kiểm tra có "bịa" hay không
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def detect_hallucination(answer, docs, threshold=0.8):
    if not answer.strip():
        print("Trả lời rỗng.")
        return True  # Xem là không tin cậy

    try:
        answer_vec = embedding_model.embed_query(answer)
        context_vecs = [embedding_model.embed_query(doc.page_content[:1000]) for doc in docs]

        similarities = [
            cosine_similarity([answer_vec], [ctx_vec])[0][0]
            for ctx_vec in context_vecs
        ]
        avg_similarity = np.mean(similarities)
        print(f">>> Độ tương đồng giữa answer và từng đoạn:")
        for i, sim in enumerate(similarities):
            print(f"  - Context {i+1}: similarity = {sim:.3f}")
        print(f">>> TRUNG BÌNH = {avg_similarity:.3f} | Threshold = {threshold}")

        return avg_similarity < threshold

    except Exception as e:
        print("Lỗi trong detect_hallucination:", str(e))
        return True

def clean_answer(answer: str) -> str:
    if "<|im_end|>" in answer:
        answer = answer.split("<|im_end|>")[0]

    # Xoá các phần prompt còn sót lại như <|im_start|>user...
    answer = re.sub(r"<\|im_start\|>.*?<\|im_end\|>", "", answer, flags=re.DOTALL)

    # Loại lặp từ liên tiếp
    answer = re.sub(r"\b(\w{2,})\b(?:\s+\1\b){2,}", r"\1", answer)

    # Dọn dấu câu dư và khoảng trắng
    answer = re.sub(r"[.;,\s]+$", ".", answer.strip())
    answer = re.sub(r"\s{2,}", " ", answer)

    return answer.strip()

# Hàm chính: xử lý truy vấn
def answer_question(question):
    topic = classify_semantic_intent(question)
    prompt = prompts[topic]
    vector_path = f"{base_vector_path}/{topic}_db"
    db = FAISS.load_local(vector_path, embedding_model, allow_dangerous_deserialization=True)

    llm = load_llm()
    retriever = db.as_retriever(search_kwargs={"k": 1})
    docs = retriever.get_relevant_documents(question)

    print("\nCác đoạn context gần nhất:")
    for i, doc in enumerate(docs):
        print(f"[Context {i+1} - {doc.metadata.get('source', 'Không rõ nguồn')}]:")
        print(doc.page_content[:1000])
        print("")

    if not context_covers_question(question, docs):
        return topic, "Tôi chưa tìm thấy thông tin cụ thể trong tài liệu đã cung cấp."

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=False,
        chain_type_kwargs={"prompt": prompt},
        chain_type="stuff"
    )
    result = qa.invoke({"query": question})
    answer = result['result']

    if detect_hallucination(answer, docs):
        answer = "Tôi chưa tìm thấy thông tin cụ thể trong tài liệu đã cung cấp."
    else:
        answer = clean_answer(answer)
    # In tên tài liệu đã dùng
    sources = {doc.metadata.get("source", "Không rõ nguồn") for doc in docs}
    print("Nguồn tài liệu được sử dụng:")
    for src in sources:
        print(f" - {src}")

    return topic, answer #list(sources) #, clean_answer(answer)

# Test
if __name__ == "__main__":
    question = " Mặt hàng thuốc bảo vệ thực vật có chứa các hoạt chất thực hiện bình ổn giá là những mặt hàng nào?" 
    topic, answer = answer_question(question)
    print(f"[Chủ đề được phân loại: {topic}]")
    print("Câu hỏi:", question)
    print("Trả lời:", answer)