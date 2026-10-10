import os
import requests
import streamlit as st
from dotenv import load_dotenv
load_dotenv()
API_URL = os.getenv("API_URL", "http://localhost:8000")
st.set_page_config(page_title="EvidencePilot", page_icon="🧭", layout="wide")
st.title("🧭 EvidencePilot")
st.caption("Multimodal research copilot · RAG + vector search + independent answer verification")
with st.sidebar:
    st.header("Knowledge base")
    st.write("Upload a PDF, DOCX, TXT, or Markdown file to index its contents.")
    uploaded_doc = st.file_uploader("Add a document", type=["pdf", "docx", "txt", "md"], key="doc")
    if st.button("Index document", disabled=uploaded_doc is None, use_container_width=True):
        with st.spinner("Extracting, chunking, embedding, and indexing..."):
            try:
                res = requests.post(f"{API_URL}/ingest", files={"file": (uploaded_doc.name, uploaded_doc.getvalue(), uploaded_doc.type)}, timeout=180)
                if res.ok: st.success(f"Indexed {res.json()['chunks_indexed']} chunks from {uploaded_doc.name}.")
                else: st.error(res.text)
            except requests.RequestException as exc: st.error(f"API unavailable: {exc}")
    st.divider()
    st.subheader("Multimodal input")
    image_file = st.file_uploader("Optional image", type=["png", "jpg", "jpeg", "webp"], key="image")
    st.caption("Image analysis uses the vision-capable model configured in your API account.")

question = st.text_area("Ask a question", placeholder="Compare the approaches described in my uploaded documents and explain the trade-offs.", height=100)
if st.button("Research and verify", type="primary", use_container_width=True):
    if len(question.strip()) < 3:
        st.warning("Enter a question of at least 3 characters.")
    else:
        try:
            with st.spinner("Researcher agent is drafting; verifier agent is auditing..."):
                if image_file:
                    resp = requests.post(f"{API_URL}/ask-image", params={"question": question}, files={"file": (image_file.name, image_file.getvalue(), image_file.type)}, timeout=240)
                else:
                    resp = requests.post(f"{API_URL}/ask", json={"question": question}, timeout=240)
            if not resp.ok:
                st.error(resp.text)
            else:
                data = resp.json()
                st.subheader("Answer")
                st.markdown(data["answer"])
                c1, c2, c3 = st.columns(3)
                c1.metric("Verifier status", data["verdict"])
                c2.metric("Revision needed", "Yes" if data["revised"] else "No")
                c3.metric("Evidence chunks", len(data["sources"]))
                with st.expander("Retrieved evidence", expanded=True):
                    if not data["sources"]: st.info("No document evidence retrieved. Treat the answer as image/general analysis, not a sourced document answer.")
                    for i, src in enumerate(data["sources"], 1):
                        st.markdown(f"**[S{i}] {src['source']}** · similarity {src.get('score', 'n/a')}")
                        st.write(src["excerpt"])
                with st.expander("Agent trace"):
                    for event in data["trace"]:
                        st.markdown(f"**{event['agent']}** · {event.get('model', 'retrieval')} · {event['status']}  \n{event['details']}")
                st.caption("Verifier status is an automated check, not a guarantee of factual correctness. Review sources before relying on important claims.")
        except requests.RequestException as exc:
            st.error(f"Could not reach the API at {API_URL}: {exc}")

st.divider()
st.caption("Built with FastAPI · OpenAI Responses API · ChromaDB · Streamlit. API keys stay server-side in .env.")
