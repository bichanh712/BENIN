import streamlit as st
from pathlib import Path
from qa import answer_question  # Gọi chatbot
from vector_builder import create_vector_by_topic  # Gọi backend xử lý vector hóa

# Danh sách các topic
topics = ["labeling", "standard", "ads", "distribute", "price"]

# Cấu hình chung
st.set_page_config(page_title="TƯ VẤN PHÁP LÝ", page_icon="⚖️")
st.sidebar.title("📚 TUỲ CHỌN")
page = st.sidebar.selectbox("Chọn trang", ["Tư vấn pháp lý", "Cập nhật tài liệu"])

# ---------------------------
# ✅ Trang Chatbot Tư vấn
# ---------------------------
if page == "Tư vấn pháp lý":
    st.title("⚖️ TƯ VẤN PHÁP LÝ SẢN PHẨM")

    # Khởi tạo session_state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Hiển thị hội thoại trước đó
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Nhập câu hỏi từ người dùng
    question = st.chat_input("Hãy nhập câu hỏi pháp lý bạn cần tư vấn?")

    if question:
        with st.chat_message("user"):
            st.markdown(question)
        st.session_state.messages.append({"role": "user", "content": question})

        with st.spinner("Đang tìm câu trả lời..."):
            try:
                topic, answer = answer_question(question)
                print("DEBUG | Chủ đề:", topic)
                print("DEBUG | Trả lời:", answer)
            except Exception as e:
                answer = "Xin lỗi, tôi chưa tìm thấy thông tin cụ thể trong tài liệu đã cung cấp."
                print("LỖI:", str(e))

        with st.chat_message("assistant"):
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})

# ---------------------------
# ✅ Trang Admin Upload
# ---------------------------
elif page == "Cập nhật tài liệu":
    st.title("📥 Cập nhật tài liệu pháp lý")

    topic = st.selectbox("📂 Chọn chủ đề cần thêm tài liệu:", topics)
    uploaded_files = st.file_uploader("📎 Tải lên 1 hoặc nhiều file PDF:", type=["pdf"], accept_multiple_files=True)

    if uploaded_files:
        st.success(f"✅ Đã chọn {len(uploaded_files)} file.")

        if st.button("Cập nhật vào hệ thống"):
            save_dir = Path("data") / topic
            save_dir.mkdir(parents=True, exist_ok=True)

            new_files = []
            skipped_files = []

            for file in uploaded_files:
                file_path = save_dir / file.name
                if file_path.exists():
                    skipped_files.append(file.name)
                    continue  # Bỏ qua file đã tồn tại
                with open(file_path, "wb") as f:
                    f.write(file.read())
                new_files.append(file.name)

            if new_files:
                st.info(f"🔄 Đã lưu {len(new_files)} file mới. Đang xử lý FAISS...")
                with st.spinner("Đang tạo vectorstore..."):
                    # Gọi lại hàm create_vector_by_topic() theo topic,
                    # hàm này bên bạn đã có logic bỏ qua file đã xử lý trong cache rồi
                    create_vector_by_topic()
                st.success("Cập nhật và vector hóa thành công!")
            else:
                st.warning("Không có file mới để cập nhật.")

            if skipped_files:
                st.info(f"📄 Bỏ qua {len(skipped_files)} file đã tồn tại: {', '.join(skipped_files)}")

        # --- Hiển thị file đã xử lý ---
    processed_cache_path = Path(f"processed_{topic}_files.txt")
    if processed_cache_path.exists():
        processed_files = processed_cache_path.read_text(encoding="utf-8").splitlines()
        st.subheader("📄 Danh sách file đã xử lý")
        if processed_files:
            for i, fname in enumerate(processed_files, 1):
                st.markdown(f"{i}. `{fname}`")



