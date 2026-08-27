"""화성시 징수과 업무매뉴얼 AI."""

from __future__ import annotations

from collections.abc import Mapping

import streamlit as st

from src.config import Settings
from src.privacy import detect_personal_information
from src.rag import IndexStatusError, LocalFallbackError, LocalKeywordBackup, RagService, build_all_indexes


SUGGESTIONS = {
    "압류·매각 처리 절차": "압류·매각 처리 절차를 알려줘",
    "체납처분 중지 기준": "체납처분 중지 기준은 무엇인가요?",
    "분납 신청 처리": "분납 신청 업무처리 방법을 알려줘",
    "공매 업무 절차": "공매 업무 절차를 알려줘",
}

st.set_page_config(page_title="화성시 징수과 업무매뉴얼 AI", page_icon=":material/account_balance:", layout="centered")


def streamlit_secrets() -> Mapping[str, object]:
    try:
        return dict(st.secrets)
    except (FileNotFoundError, RuntimeError):
        return {}


@st.cache_resource(show_spinner=False)
def get_service(settings: Settings) -> RagService:
    # Cloud deployment can use the committed lightweight keyword backup even
    # when the private local Chroma index is intentionally unavailable.
    return RagService(settings, allow_create=True)


def show_sources(evidence: list[object]) -> None:
    if not evidence:
        return
    st.markdown("#### 근거자료")
    for item in evidence:
        with st.container(border=True):
            st.markdown(f":material/description: **{item.source}**")
            details = [f"업무 분류: {item.category}", f"문서 유형: {item.document_type}"]
            if item.page:
                details.append(f"페이지: {item.page}")
            st.caption(" · ".join(details))


def show_sidebar(service: RagService, settings: Settings, primary_index_ready: bool) -> None:
    stats = service.document_stats() if primary_index_ready else LocalKeywordBackup(settings).stats()
    with st.sidebar:
        st.header(":material/folder_managed: 징수과 업무자료")
        st.metric("등록 문서", f"{stats.get('documents', 0)}건")
        st.metric("생성 청크", f"{stats.get('chunks', 0):,}개")
        if stats.get("category_counts"):
            st.caption("자료 분류")
            for category, count in sorted(stats["category_counts"].items()):
                st.write(f"{category} {count}건")

        if st.button("문서 다시 색인", width="stretch", icon=":material/refresh:"):
            try:
                with st.spinner("문서를 확인하고 검색 색인을 준비하고 있습니다..."):
                    results = build_all_indexes(settings)
                get_service.clear()
                for provider, label in (("ollama", "Ollama 로컬 백업"), ("cloudflare", "Cloudflare")):
                    provider_stats = results.get(provider)
                    if isinstance(provider_stats, dict):
                        st.success(f"{label}: 문서 {provider_stats['documents']}건 · 청크 {provider_stats['chunks']:,}개")
                    else:
                        st.warning(f"{label} 색인 실패: {results.get(f'{provider}_error', '알 수 없는 오류')}")
                st.rerun()
            except Exception as error:
                st.error(f"색인 실패: {error}")
        if st.button("대화 내용 지우기", width="stretch", icon=":material/delete_sweep:"):
            st.session_state.messages = []
            st.rerun()

        with st.expander("관리자 정보", icon=":material/settings:"):
            st.caption("임베딩 모델")
            st.code(settings.embedding_model, language=None)
            st.caption("답변 모델")
            st.code(settings.chat_model, language=None)
            st.caption("Vector DB")
            st.success("정상", icon=":material/check_circle:")
            backup_ready, backup_message = service.local_backup_status()
            (st.success if backup_ready else st.warning)(backup_message, icon=":material/check_circle:" if backup_ready else ":material/warning:")


def main() -> None:
    st.title("화성시 징수과 업무매뉴얼 AI", anchor=False)
    st.caption("징수과 업무매뉴얼·법령·지침·업무자료에서 관련 근거를 찾아 답변합니다.")
    st.info(
        "본 서비스는 화성시 징수과 내부 업무지원용입니다. 등록된 자료를 바탕으로 답변하며, "
        "개인정보 또는 개별 체납자의 실제 정보를 입력하지 마세요. 최종 행정처리 전에는 관련 법령·최신 지침·담당자 확인이 필요합니다.",
        icon=":material/privacy_tip:",
    )

    settings = Settings.from_env(streamlit_secrets())
    try:
        settings.validate()
    except (ValueError, FileNotFoundError) as error:
        st.error(str(error))
        st.code("Copy-Item .env.example .env", language="powershell")
        st.stop()

    primary_index_ready = False
    try:
        service = get_service(settings)
        service.load_ready_index()
        primary_index_ready = True
    except IndexStatusError as error:
        keyword_backup = LocalKeywordBackup(settings)
        if not keyword_backup.is_ready():
            st.error(
                "검색 색인이 준비되지 않았습니다. 관리자가 VS Code 터미널에서 "
                "`python -m scripts.build_index`를 한 번 실행해야 합니다."
            )
            st.caption(f"상태: {error.code}")
            st.stop()
        st.info("온라인 경량 검색 모드로 실행 중입니다. 근거 문서의 키워드를 우선 검색합니다.")
    except Exception as error:
        st.error(f"검색 DB를 여는 중 오류가 발생했습니다: {error}")
        st.stop()
    show_sidebar(service, settings, primary_index_ready)

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if not st.session_state.messages:
        selected = st.pills("무엇을 확인하시겠습니까?", list(SUGGESTIONS), label_visibility="collapsed")
        suggested_question = SUGGESTIONS.get(selected, "") if selected else ""
    else:
        suggested_question = ""

    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar=":material/smart_toy:" if message["role"] == "assistant" else None):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                show_sources(message.get("evidence", []))

    question = suggested_question or st.chat_input("업무자료에서 확인할 내용을 입력하세요.", submit_mode="disable")
    if not question:
        return
    detected = detect_personal_information(question)
    if detected:
        st.error(f"입력에서 {', '.join(detected)} 형식이 감지되었습니다. 개인정보를 제거한 뒤 일반적인 업무 질문으로 다시 입력하세요.")
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant", avatar=":material/smart_toy:"):
        try:
            with st.spinner("관련 업무자료를 검색하고 있습니다..."):
                if primary_index_ready:
                    result = service.answer_with_local_fallback(question)
                else:
                    evidence = LocalKeywordBackup(settings).search(question, settings.top_k)
                    result = service.answer_from_evidence(question, evidence)
            if result.mode == "Ollama 로컬 백업":
                st.warning("Cloudflare를 사용할 수 없어 Ollama 로컬 백업 모드로 답변했습니다.")
            elif primary_index_ready:
                st.caption("현재 AI: Cloudflare")
            else:
                st.caption("현재 AI: Cloudflare · 온라인 경량 검색")
            st.markdown(result.answer)
            show_sources(result.evidence)
            if result.evidence:
                with st.expander("검색된 근거 내용 보기", icon=":material/visibility:"):
                    for item in result.evidence:
                        st.caption(f"{item.source} · {item.category}" + (f" · {item.page}페이지" if item.page else ""))
                        st.write(item.text)
            st.session_state.messages.append({"role": "assistant", "content": result.answer, "evidence": result.evidence})
        except LocalFallbackError as error:
            st.error(str(error))
        except Exception as error:
            st.error(f"답변 생성 중 오류가 발생했습니다: {error}")


if __name__ == "__main__":
    main()
