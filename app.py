"""Chat UI over the agent. Run: streamlit run app.py"""
import streamlit as st

from agent import build, flush, run, store, tracing

st.set_page_config(page_title="Veritas", page_icon="📄")
st.title("Veritas")
st.caption("Self-correcting agentic RAG over your PDFs")


@st.cache_resource
def graphs():
    vs = store()  # embedded Qdrant allows one client per process, so both graphs share it
    return {True: build(vs=vs), False: build(max_attempts=0, vs=vs)}


self_correct = st.sidebar.toggle("Self-correction (grade, rewrite, grounding check)", value=True)
st.sidebar.caption("Off = baseline: hybrid retrieve then generate.")

if "history" not in st.session_state:
    st.session_state.history = []
for role, content in st.session_state.history:
    st.chat_message(role).markdown(content)

if question := st.chat_input("Ask about the indexed PDFs"):
    st.chat_message("user").markdown(question)
    with st.chat_message("assistant"), st.spinner("Thinking..."):
        out = run(graphs()[self_correct], question)
        flush()
        st.markdown(out["answer"])
        if out.get("queries"):
            st.caption("Rewritten queries: " + " → ".join(out["queries"]))
        st.caption(f"attempts: {out['attempts']} · grounded: {out.get('grounded', 'n/a')}")
        if out["trace_id"] and tracing():
            from langfuse import get_client
            st.caption(f"[Langfuse trace]({get_client().get_trace_url(trace_id=out['trace_id'])})")
        with st.expander(f"Sources ({len(out['docs'])})"):
            for d in out["docs"]:
                st.markdown(f"**{d.metadata.get('source')} p.{d.metadata.get('page')}** · {d.metadata.get('headings')}")
                st.text(d.page_content)
    st.session_state.history += [("user", question), ("assistant", out["answer"])]
