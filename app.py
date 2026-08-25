"""Streamlit 기반 화성시 민원 RAG 챗봇."""

from __future__ import annotations

import streamlit as st

from src.config import Settings
from src.rag import NO_ANSWER, RagService


st.set_page_config(
    page_title="화성시 민원 RAG 챗봇",
    page_icon="🏛️",
    layout="centered",
)


@st.cache_resource(show_spinner=False)
def get_service(settings: Settings) -> RagService:
    return RagService(settings)


def show_sources(sources: list[str]) -> None:
    st.markdown("**출처**")
    if sources:
        for source in sources:
            st.markdown(f"- `{source}`")
    else:
        st.caption("출처 없음")


def main() -> None:
    st.title("🏛️ 화성시 민원 RAG 챗봇")
    st.caption("화성시 행정 문서에서 근거를 검색해 답변합니다.")

    settings = Settings.from_env()
    try:
        settings.validate()
    except (ValueError, FileNotFoundError) as error:
        st.error(str(error))
        st.code(
            "copy .env.example .env\n# .env 파일에 OPENAI_API_KEY 입력",
            language="powershell",
        )
        st.stop()

    with st.sidebar:
        st.header("시스템 정보")
        st.text(f"임베딩: {settings.embedding_model}")
        st.text(f"답변 모델: {settings.chat_model}")
        st.caption("API 키는 환경변수에서만 읽습니다.")
        st.divider()
        st.warning(
            "이 서비스는 행정업무 보조용입니다. 법적 판단·자격 결정·행정처분은 "
            "반드시 담당자가 원문과 최신 규정을 확인해야 합니다."
        )

    try:
        with st.spinner("문서를 확인하고 벡터 DB를 준비하고 있습니다..."):
            service = get_service(settings)
    except Exception as error:
        st.error(f"벡터 DB 준비 중 오류가 발생했습니다: {error}")
        st.info("API 키, 인터넷 연결, data 폴더의 문서 형식을 확인하세요.")
        st.stop()

    with st.sidebar:
        if st.button("문서 다시 색인", use_container_width=True):
            try:
                with st.spinner("문서를 다시 색인하고 있습니다..."):
                    stats = service.ensure_index(force=True)
                st.success(
                    f"완료: 문서 {stats['documents']}개, 청크 {stats['chunks']}개"
                )
            except Exception as error:
                st.error(f"재색인 실패: {error}")
        if service.skipped_duplicates:
            st.caption(f"중복 문서 {len(service.skipped_duplicates)}개는 제외했습니다.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.sidebar:
        if st.button("대화 내용 지우기", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                show_sources(message.get("sources", []))

    question = st.chat_input("화성시 민원 또는 관련 규정을 질문하세요")
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("관련 문서를 검색하고 있습니다..."):
                result = service.answer(question)
            st.markdown(result.answer)
            show_sources(result.sources)
            if result.evidence:
                with st.expander("검색된 문서 근거 보기"):
                    for item in result.evidence:
                        st.markdown(
                            f"**{item.source}** · 유사도 {item.similarity:.1%}"
                        )
                        st.write(item.text)
                        st.divider()
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": result.answer,
                    "sources": result.sources,
                }
            )
        except Exception as error:
            st.error(f"답변 생성 중 오류가 발생했습니다: {error}")
            st.caption("OpenAI API 사용 한도와 인터넷 연결 상태를 확인하세요.")

if __name__ == "__main__":
    main()
