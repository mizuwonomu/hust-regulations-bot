import streamlit as st

def render_sources(sources) -> None:
    # Sau cutover HTTP: sources là plain dict theo wire shape của backend
    # ({"title", "content", "doc_id"}), không còn LangChain Document.
    if not sources:
        return
    st.divider()
    st.subheader("📚 Nguồn tài liệu tham khảo")
    for i, doc in enumerate(sources):
        source_name = doc.get("title", f"Nguồn tài liệu #{i+1}")
        with st.expander(f"📖 [{i+1}] {source_name}"):
            st.markdown("**Nội dung**")
            st.info(doc.get("content", ""))
