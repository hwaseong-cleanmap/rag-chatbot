# 화성시 민원 RAG 챗봇

화성시 민원·조례·지방세 관련 문서를 검색하여, **검색된 문서 내용만 근거로** 답변하는 Streamlit 애플리케이션입니다. 관련 근거를 찾지 못하면 `자료에서 확인할 수 없습니다`라고 답하며 출처 파일명을 함께 표시합니다.

## 행정업무 적용 범위

- 주요 사용자: 화성시 민원인 및 민원 담당자
- 입력: 자연어 민원 질문
- 처리: OpenAI 임베딩 → ChromaDB 유사 문서 검색 → 근거 제한 답변 생성
- 출력: 답변, 출처 파일명, 검색된 원문 근거
- 사람의 확인이 필요한 영역: 법적 판단, 자격 결정, 민원 거부, 행정처분, 최신 법령 여부

이 프로그램은 행정업무 보조용 MVP이며 담당자의 최종 판단을 대체하지 않습니다. 실제 민원 개인정보는 테스트 입력에 사용하지 마세요.

## 동작 구조

```text
data의 TXT/DOC 문서
        ↓ 문서 읽기·중복 제거·청킹
OpenAI text-embedding-3-small
        ↓
ChromaDB (vector_db/에 로컬 저장)
        ↓ 질문과 유사한 근거 검색
OpenAI 답변 모델
        ↓
Streamlit: 답변 + 출처 + 원문 근거
```

문서 내용, 임베딩 모델 또는 청크 설정이 바뀐 경우에만 벡터 DB를 자동으로 다시 만듭니다. 현재 제공된 `.doc` 파일은 내부적으로 RTF 형식이며 `striprtf`로 처리합니다. 내용이 완전히 같은 중복 파일은 색인에서 제외합니다.

## 프로젝트 구조

```text
민원챗봇/
├─ app.py                 # Streamlit 화면
├─ requirements.txt       # Python 패키지
├─ .env.example           # 환경변수 예시
├─ .gitignore
├─ README.md
├─ data/                  # 검색할 TXT, DOC 문서
├─ src/
│  ├─ config.py           # 환경변수와 설정
│  ├─ documents.py        # 문서 로딩·중복 제거·청킹
│  └─ rag.py              # 임베딩·ChromaDB·답변 생성
└─ tests/                 # API 호출 없는 자동 테스트
```

## Windows에서 설치

아래 명령은 **VS Code → 터미널 → 새 터미널**에서 프로젝트 폴더를 연 뒤 실행합니다.

### 1. 가상환경 생성 및 활성화

Python 3.11 이상을 사용합니다. 이 프로젝트는 Windows의 Python 3.14에서도 테스트했습니다.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

PowerShell 실행 정책 오류가 발생하면 현재 터미널에만 다음 설정을 적용한 뒤 다시 활성화합니다.

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### 2. 패키지 설치

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

주요 패키지는 다음과 같습니다.

- `streamlit`: 웹 화면
- `openai`: 임베딩 및 근거 기반 답변 생성
- `chromadb`: 로컬 벡터 데이터베이스
- `striprtf`: RTF 형식의 `.doc` 문서 읽기
- `python-dotenv`: `.env` 환경변수 로딩
- `pytest`: 자동 테스트

## OpenAI API 키 설정

API 키는 코드에 입력하지 않습니다. `.env.example`을 복사하여 `.env`를 만들고 값을 입력합니다.

```powershell
Copy-Item .env.example .env
```

`.env` 파일:

```dotenv
OPENAI_API_KEY=실제_API_키
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_CHAT_MODEL=gpt-5-mini
```

`.env`는 `.gitignore`에 포함되어 GitHub에 업로드되지 않습니다. OpenAI Python SDK는 환경변수 방식으로 API 키를 사용할 수 있습니다.

## 실행

```powershell
python -m streamlit run app.py
```

브라우저에서 `http://localhost:8501`이 자동으로 열립니다. 첫 실행 시 문서 임베딩 API를 호출하고 `vector_db/`에 ChromaDB를 생성하므로 문서 양에 따라 시간이 걸릴 수 있으며 API 비용이 발생합니다. 이후 문서가 바뀌지 않으면 저장된 DB를 재사용합니다.

## 테스트

API 키 없이 문서 로딩, RTF 변환, 청킹, 검색 임계값 로직을 확인합니다.

```powershell
python -m pytest -q
```

정상 결과 예시:

```text
9 passed
```

실제 동작 확인 질문 예시:

- `대형폐기물은 어떻게 배출하나요?`
- `동물 등록 관련 안내가 있나요?`
- `세입징수 포상금 지급 대상은 누구인가요?`
- 자료와 무관한 질문을 입력했을 때 `자료에서 확인할 수 없습니다`가 표시되는지 확인

## 문서 추가 및 재색인

1. `data` 폴더에 UTF-8/CP949 TXT 또는 RTF 형식 DOC 파일을 넣습니다.
2. 앱을 다시 실행하면 변경된 문서 해시를 감지하여 자동 재색인합니다.
3. 실행 중이라면 왼쪽 메뉴의 **문서 다시 색인** 버튼을 누릅니다.

진짜 구형 Word 바이너리 `.doc` 파일은 현재 로더가 지원하지 않습니다. Word나 LibreOffice에서 RTF 또는 TXT로 변환한 뒤 `data` 폴더에 넣으세요. `.docx`는 현재 MVP 범위에 포함하지 않았습니다.

## 오류 해결

### `OPENAI_API_KEY가 설정되지 않았습니다`

프로젝트 최상위 폴더에 `.env`가 있는지, 키 앞뒤에 불필요한 따옴표나 공백이 없는지 확인합니다.

### OpenAI 인증 또는 사용 한도 오류

API 키가 유효한지, 프로젝트 결제/사용 한도가 남아 있는지 확인합니다. 모델 접근 권한이 다르면 `.env`의 `OPENAI_CHAT_MODEL`을 계정에서 사용 가능한 모델로 변경할 수 있습니다.

### ChromaDB 오류

앱을 종료한 뒤 프로젝트의 `vector_db` 폴더만 삭제하고 다시 실행하면 재생성됩니다. 이 폴더에는 문서 조각이 저장되므로 민감한 행정자료를 사용할 경우 접근권한과 보관기간을 별도로 관리해야 합니다.

### 관련 문서가 있는데 “자료에서 확인할 수 없습니다”가 나오는 경우

현재 코드는 코사인 유사도 35% 이상인 검색 결과만 답변 근거로 사용합니다. `src/config.py`의 `min_similarity`는 평가용 질문 세트로 검증한 뒤 조정하세요. 임계값을 지나치게 낮추면 무관한 문서로 답변할 위험이 커집니다.

## 배포 시 보안 주의

- 공개 Streamlit Cloud에 민원 개인정보나 비공개 내부 문서를 올리지 마세요.
- 배포 환경의 Secret/환경변수 기능에 `OPENAI_API_KEY`를 등록하세요.
- 문서와 ChromaDB의 접근권한, 암호화, 로그, 보관기간 정책을 마련하세요.
- 최신 법령·조례 반영 여부를 담당자가 정기적으로 점검하세요.
- 외부 공개 서비스라면 질문 로그에 개인정보가 남지 않도록 별도 검토가 필요합니다.
