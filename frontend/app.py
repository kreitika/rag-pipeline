import streamlit as st
import requests
import json

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="RAG Pipeline",
    page_icon="🔍",
    layout="wide",
)


st.title("🔍 RAG Pipeline — Internal Documentation Search")
st.caption("Hybrid search with dense + sparse retrieval, cross-encoder reranking, and citation verification")

with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Chunks to retrieve", min_value=1, max_value=5, value=3)
    show_chunks = st.checkbox("Show retrieved chunks", value=False)
    show_confidence = st.checkbox("Show confidence breakdown", value=True)
    compare_mode = st.checkbox("Compare hybrid vs dense-only", value=False)

    st.divider()
    st.header("Documents")
    st.divider()
    st.header("Upload Document")

    uploaded_file = st.file_uploader(
        "Add a new document",
        type=["md", "txt", "html", "pdf"],
    )

    if uploaded_file is not None:
        if st.button("Ingest Document", type="primary"):
            with st.spinner("Indexing document..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                    response = requests.post(
                        f"{API_URL}/v1/ingest",
                        files=files,
                        timeout=120,
                    )
                    if response.status_code == 200:
                        st.success(f"✓ {uploaded_file.name} indexed successfully")
                        st.rerun()
                    else:
                        st.error(f"Failed: {response.text}")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

    try:
        docs_response = requests.get(f"{API_URL}/v1/documents")
        if docs_response.status_code == 200:
            docs = docs_response.json()["documents"]
            for doc in docs:
                st.text(f"📄 {doc['filename']} ({doc['size_kb']} KB)")
    except Exception:
        st.error("API not reachable")


query = st.text_input(
    "Ask a question about your documentation:",
    placeholder="e.g. What is the rollback process if a deployment fails?",
)

col1, col2 = st.columns([1, 4])
with col1:
    ask_button = st.button("Ask", type="primary", use_container_width=True)
with col2:
    if st.button("Clear", use_container_width=True):
        st.rerun()



if ask_button and query:
    with st.spinner("Searching documentation..."):
        try:
            if compare_mode:
                col_hybrid, col_dense = st.columns(2)

                hybrid_response = requests.post(
                    f"{API_URL}/v1/ask",
                    json={"question": query, "top_k": top_k},
                    timeout=120,
                )
                dense_response = requests.post(
                    f"{API_URL}/v1/ask/dense-only",
                    json={"question": query, "top_k": top_k},
                    timeout=120,
                )

                with col_hybrid:
                    st.markdown("### 🔀 Hybrid Search")
                    if hybrid_response.status_code == 200:
                        r = hybrid_response.json()
                        st.markdown(r["answer"])
                        st.caption(f"Confidence: {r['composite_score']:.0%}")

                with col_dense:
                    st.markdown("### 🎯 Dense Only")
                    if dense_response.status_code == 200:
                        r = dense_response.json()
                        st.markdown(r["answer"])
                        st.caption(f"Method: {r['method']}")

            else:
                response = requests.post(
                    f"{API_URL}/v1/ask",
                    json={"question": query, "top_k": top_k},
                    timeout=120,
                )

                if response.status_code == 200:
                    result = response.json()

                    if result["confident"]:
                        st.success("Answer found")
                    else:
                        st.warning("Low confidence — answer may be incomplete")

                    st.markdown("### Answer")
                    st.markdown(result["answer"])

                    if result["citations"]:
                        st.markdown("### Sources")
                        for citation in result["citations"]:
                            with st.expander(
                                f"[{citation['citation_number']}] {citation['source']} "
                                f"— cited {citation['times_cited']} time(s)"
                            ):
                                st.text(citation["chunk_text"])

                    if show_confidence:
                        st.markdown("### Confidence")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Composite Score", f"{result['composite_score']:.0%}")
                        with col2:
                            st.metric("Confident", "Yes" if result["confident"] else "No")
                        with col3:
                            st.metric("Tokens Used", result["tokens_used"])

                else:
                    st.error(f"API error: {response.status_code}")

            

        except requests.exceptions.Timeout:
            st.error("Request timed out. The pipeline is taking too long.")
        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to API. Make sure the server is running.")
        except Exception as e:
            st.error(f"Error: {str(e)}")

elif ask_button and not query:
    st.warning("Please enter a question first.")





