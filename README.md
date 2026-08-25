# 화성시 민원 FAQ RAG 챗봇

화성시 공개 행정문서를 검색하여 **검색된 문서 내용만 근거로** 답변하는 Streamlit 애플리케이션입니다. 관련 근거를 찾지 못하면 `자료에서 확인할 수 없습니다`라고 답하고, 답변에 사용한 출처 파일명을 표시합니다.

## 서비스 범위와 운영 원칙

- 공개 FAQ 안내 전용이며 실제 민원 접수 시스템이 아닙니다.
- 이름·주소·전화번호·이메일·주민등록번호 등 개인정보와 실제 민원 내용을 입력하지 않습니다.
- 전화번호·이메일·주민등록번호 형식이 감지되면 AI로 전송하지 않고 입력을 차단합니다.
- 법적 판단, 자격 결정, 민원 거부, 행정처분은 담당자가 원문과 최신 규정을 확인해야 합니다.
- 질문과 문서는 Cloudflare Workers AI API로 전송됩니다.

## 동작 구조

```text
data의 TXT/DOC 공개 문서
        ↓ 문서 읽기·중복 제거·청킹
Cloudflare Workers AI: @cf/baai/bge-m3 임베딩
        ↓
ChromaDB (vector_db/ 로컬 저장)
        ↓ 질문과 유사한 문서 근거 검색
Cloudflare Workers AI: @cf/qwen/qwen3-30b-a3b-fp8
        ↓
Streamlit: 답변 + 출처 + 검색 근거
```

문서 내용, 임베딩 모델 또는 청크 설정이 바뀌면 벡터 DB를 자동 재생성합니다. 현재 제공된 `.doc` 파일은 내부적으로 RTF 형식이며 `striprtf`로 처리합니다. 내용이 같은 중복 파일은 색인에서 제외합니다.

## 프로젝트 구조

```text
민원챗봇/
├─ app.py
├─ requirements.txt
├─ .env.example
├─ .gitignore
├─ README.md
├─ data/
├─ src/
│  ├─ config.py
│  ├─ documents.py
│  ├─ privacy.py
│  └─ rag.py
└─ tests/
```

## Windows 로컬 설치

VS Code에서 프로젝트 폴더를 열고 **터미널 → 새 터미널**에서 실행합니다.

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Cloudflare Workers AI 설정

Cloudflare Dashboard에서 `Workers AI → Use REST API → Create a Workers AI API Token`을 선택하여 Account ID와 전용 API Token을 발급합니다. Global API Key와 Zone ID는 필요하지 않습니다.

Account ID는 이메일 주소가 아니라 Workers AI의 `Use REST API` 화면에 표시되는 32자리 영문·숫자 식별자입니다.

`.env.example`을 복사합니다.

```powershell
Copy-Item .env.example .env
```

`.env`에 실제 값을 입력합니다.

```dotenv
CLOUDFLARE_ACCOUNT_ID=실제_Account_ID
CLOUDFLARE_API_TOKEN=실제_API_Token
CHAT_MODEL=@cf/qwen/qwen3-30b-a3b-fp8
EMBEDDING_MODEL=@cf/baai/bge-m3
```

`.env`는 GitHub에 업로드되지 않습니다. Token을 코드, README, 대화창, 커밋에 넣지 마세요.

## 로컬 실행

```powershell
python -m streamlit run app.py
```

브라우저에서 `http://localhost:8501`을 엽니다. 첫 실행 또는 임베딩 모델 변경 시 문서를 다시 임베딩하므로 시간이 걸리고 Cloudflare 무료 사용량을 소비합니다.

## 자동 테스트

```powershell
python -m pytest -q
```

테스트는 실제 API를 호출하지 않고 문서 로딩, 설정 검증, 개인정보 감지, 검색 임계값과 답변 생성을 검사합니다.

## Streamlit Community Cloud 배포

1. GitHub 저장소 `hwaseong-cleanmap/rag-chatbot`에 코드를 push합니다.
2. [Streamlit Community Cloud](https://share.streamlit.io/)에서 `Create app`을 선택합니다.
3. Repository는 `hwaseong-cleanmap/rag-chatbot`, Branch는 `main`, Main file은 `app.py`로 지정합니다.
4. `Advanced settings → Secrets`에 아래 내용을 등록합니다.

```toml
CLOUDFLARE_ACCOUNT_ID = "실제 Account ID"
CLOUDFLARE_API_TOKEN = "실제 API Token"
CHAT_MODEL = "@cf/qwen/qwen3-30b-a3b-fp8"
EMBEDDING_MODEL = "@cf/baai/bge-m3"
```

`.streamlit/secrets.toml`은 로컬 테스트용으로만 사용할 수 있으며 `.gitignore`에 포함되어 있습니다.

## 문서 추가와 재색인

1. `data` 폴더에 공개 가능한 UTF-8/CP949 TXT 또는 RTF 형식 DOC 파일을 넣습니다.
2. 앱을 다시 실행하면 변경된 문서 해시를 감지하여 자동 재색인합니다.
3. 실행 중에는 사이드바의 **문서 다시 색인** 버튼을 사용할 수 있습니다.

구형 Word 바이너리 `.doc` 파일은 지원하지 않습니다. Word나 LibreOffice에서 RTF 또는 TXT로 변환하세요. `.docx`는 현재 MVP 범위에 포함하지 않습니다.

## 오류 해결

### Cloudflare 설정 오류

- Account ID와 API Token이 정확한지 확인합니다.
- Base URL은 `CLOUDFLARE_ACCOUNT_ID`를 사용해 코드에서 자동 생성합니다.
- Streamlit 배포 환경에서는 `.env`가 아니라 앱의 `Advanced settings → Secrets`를 사용합니다.

### 401 또는 403 오류

Workers AI 전용 Token과 모델 접근 권한을 확인합니다. Token을 노출했다면 즉시 폐기하고 새로 발급합니다.

### 429 오류

Cloudflare 무료 일일 사용량을 소진했거나 일시적으로 처리 용량이 부족할 수 있습니다. Workers AI Dashboard에서 사용량을 확인하고 나중에 다시 시도합니다.

### ChromaDB 오류

앱을 종료한 다음 `vector_db` 폴더만 삭제하고 다시 실행합니다. Cloudflare 모델로 처음 전환할 때는 기존 OpenAI 임베딩 색인이 자동 교체됩니다.

### 관련 자료가 검색되지 않는 경우

현재 코드는 코사인 유사도 35% 이상인 결과만 답변 근거로 사용합니다. `src/config.py`의 `min_similarity`는 평가 질문 세트로 검증한 후 조정해야 합니다. 지나치게 낮추면 무관한 문서를 근거로 사용할 위험이 있습니다.

## 무료 운영 시 주의사항

- Cloudflare Workers AI와 Streamlit Community Cloud의 무료 한도와 정책은 변경될 수 있습니다.
- Streamlit Community Cloud 컨테이너가 다시 생성되면 로컬 ChromaDB를 재색인할 수 있습니다.
- 공개 저장소에는 공개 가능한 문서만 포함해야 합니다.
- 질문 로그를 별도로 저장하지 않으며, 운영 중에도 개인정보가 로그에 포함되지 않도록 점검해야 합니다.
