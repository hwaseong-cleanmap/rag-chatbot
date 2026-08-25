"""Streamlit 기반 화성시 공개 FAQ RAG 챗봇."""

from __future__ import annotations

from collections.abc import Mapping

import streamlit as st

from src.config import Settings
from src.privacy import detect_personal_information
from src.rag import RagService


st.set_page_config(
    page_title="화성시 민원 FAQ 챗봇",
    page_icon=":material/account_balance:",
    layout="centered",
)


def streamlit_secrets() -> Mapping[str, object]:
    """로컬 Secrets 파일이 없어도 동작하도록 안전하게 읽는다."""

    try:
        return dict(st.secrets)
    except (FileNotFoundError, RuntimeError):
        return {}


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
    st.title("화성시 민원 FAQ 챗봇", anchor=False)
    st.caption("공개된 화성시 행정 문서에서 근거를 검색해 답변합니다.")
    st.warning(
        "공개 FAQ 서비스입니다. 이름·주소·전화번호·이메일·주민등록번호 등 "
        "개인정보와 실제 민원 내용을 입력하지 마세요.",
        icon=":material/privacy_tip:",
    )

    settings = Settings.from_env(streamlit_secrets())
    try:
        settings.validate()
    except (ValueError, FileNotFoundError) as error:
        st.error(str(error))
        st.code(
            "Copy-Item .env.example .env\n# .env 파일에 Cloudflare 설정 입력",
            language="powershell",
        )
        st.stop()

    with st.sidebar:
        st.header("시스템 정보")
        st.text(f"임베딩: {settings.embedding_model}")
        st.text(f"답변 모델: {settings.chat_model}")
        st.caption("Cloudflare 인증정보는 Secrets 또는 환경변수에서만 읽습니다.")
        st.divider()
        st.warning(
            "이 서비스는 공개 행정정보 안내용입니다. 법적 판단·자격 결정·행정처분은 "
            "반드시 담당자가 원문과 최신 규정을 확인해야 합니다."
        )

    try:
        with st.spinner("문서를 확인하고 벡터 DB를 준비하고 있습니다..."):
            service = get_service(settings)
    except Exception as error:
        st.error(f"벡터 DB 준비 중 오류가 발생했습니다: {error}")
        st.info(
            "Cloudflare Account ID·API Token·무료 사용량과 data 폴더의 문서 형식을 "
            "확인하세요."
        )
        st.stop()

    with st.sidebar:
        if st.button("문서 다시 색인", width="stretch", icon=":material/refresh:"):
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

    st.session_state.setdefault("messages", [])

    with st.sidebar:
        if st.button(
            "대화 내용 지우기", width="stretch", icon=":material/delete_sweep:"
        ):
            st.session_state.messages = []
            st.rerun()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                show_sources(message.get("sources", []))

    question = st.chat_input(
        "개인정보 없이 화성시 공개 행정정보를 질문하세요",
        submit_mode="disable",
    )
    if not question:
        return

    detected = detect_personal_information(question)
    if detected:
        labels = ", ".join(detected)
        st.error(
            f"입력에서 {labels} 형식이 감지되었습니다. 개인정보를 삭제하고 "
            "일반적인 FAQ 형태로 다시 질문해 주세요."
        )
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
            st.caption(
                "Cloudflare Workers AI 토큰, 모델 접근 권한, 일일 무료 사용량과 "
                "인터넷 연결을 확인하세요."
            )


if __name__ == "__main__":
    main()
