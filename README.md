# 화성시 징수과 업무매뉴얼 RAG 챗봇

화성시 징수과 직원이 업무매뉴얼, 법령, 지침, 업무자료를 검색하고 등록 문서의 근거만으로 답변을 받는 내부 업무지원 도구입니다. 이 서비스는 행정처분·법적 판단·개별 체납자 판단을 자동으로 수행하지 않습니다.

## 시스템 구조

~~~text
PDF / HWPX / PPTX / TXT / RTF-DOC
        ↓ 텍스트 추출
개인정보 형식 감지 및 마스킹
        ↓ 청킹 + 메타데이터
Cloudflare Workers AI (@cf/baai/bge-m3)
        ↓
ChromaDB
        ↓ 직원 질문 개인정보 차단
유사 문서 검색
        ↓
Cloudflare Workers AI (@cf/qwen/qwen3-30b-a3b-fp8)
        ↓
답변 + 근거자료(파일명·업무 분류·PDF 페이지)
~~~

## 지원 문서와 data 폴더 관리

data/ 아래에 업무 분류별 폴더를 만들고 파일을 넣습니다. 하위 폴더 이름은 코드 수정 없이 업무 분류로 사용됩니다.

~~~text
data/
├─ 법령/
│  └─ 지방세징수법.pdf
├─ 압류/
│  └─ 압류 업무매뉴얼.pdf
├─ 공매/
│  └─ 공매 업무매뉴얼.hwpx
└─ 기타/
   └─ 참고자료.txt
~~~

- 지원 형식: PDF, HWPX, PPTX, TXT, RTF 형식의 .doc
- PDF는 텍스트 기반 PDF만 처리하며, 페이지 번호를 근거자료에 표시합니다.
- 스캔·이미지 PDF는 OCR 범위가 아니므로 해당 파일만 실패 처리하고 나머지 문서는 계속 색인합니다.
- HWPX는 Python 표준 라이브러리의 ZIP/XML 처리로 읽으므로 한글·LibreOffice 설치가 필요하지 않습니다.
- PPTX는 슬라이드의 텍스트를 자동 추출합니다. 원본 PPTX 파일은 변경하지 않습니다.
- 동일한 본문은 중복 문서로 제외합니다.

문서를 추가·수정·삭제한 뒤 앱 사이드바의 **문서 다시 색인**을 누르세요. 문서 내용, 임베딩 모델, 청킹 설정이 바뀌면 기존 Vector DB를 안전하게 다시 만듭니다.

## 사전 색인(권장)

자료가 많은 경우에는 사용자가 접속하기 전에 관리자가 검색 색인을 미리 만들 수 있습니다. `data/`의 자료를 추가·수정·삭제한 뒤, VS Code 터미널에서 프로젝트 폴더를 연 상태로 실행합니다.

~~~powershell
.\.venv\Scripts\Activate.ps1
python -m scripts.build_index
~~~

명령이 `색인 완료`를 표시하면 `vector_db/`에 검색 색인이 저장됩니다. 이후 앱은 같은 `data/` 자료와 설정을 사용하면 기존 색인을 재사용하므로 Cloudflare 임베딩을 다시 만들지 않습니다. `vector_db/`는 업무자료에서 생성된 로컬 DB이므로 GitHub에 올리지 말고, 내부 서버 배포 시에는 서버의 영속 저장소에 보관하세요.

색인이 완료되면 파일 경로·크기·수정 시각만 담은 `index_manifest.json`도 함께 저장됩니다. 이후 앱을 재시작해도 이 목록만 빠르게 비교하며, 자료와 설정이 바뀌지 않았다면 PDF/HWPX 본문을 다시 읽거나 청킹·임베딩하지 않습니다. 첫 색인 또는 자료 변경 후 색인에만 시간이 걸립니다.

## 개인정보 보호

문서에서 주민등록번호, 전화번호, 이메일, 카드번호 형식을 감지해 마스킹한 텍스트만 Cloudflare Workers AI 임베딩 API로 전송합니다. 화면에는 원본 개인정보나 탐지된 값이 아니라 유형별 건수만 표시합니다.

질문 입력에서도 주민등록번호, 전화번호, 이메일, 카드번호 형식을 차단합니다. 이름·주소·계좌번호는 법령·업무 문서의 의미 있는 내용과 혼동될 위험이 있어 이 MVP에서 무조건 제거하지 않습니다. 실제 개인·체납자 정보는 입력하지 마세요.

질문과 답변은 별도 DB나 파일에 저장하지 않습니다.

## 로컬 실행

VS Code에서 이 프로젝트 폴더를 연 뒤 **터미널 → 새 터미널**에서 아래 순서로 실행합니다.

1. 가상환경 활성화

~~~powershell
.\.venv\Scripts\Activate.ps1
~~~

2. 필요한 패키지 설치

~~~powershell
python -m pip install -r requirements.txt
~~~

3. 설정 파일 만들기

~~~powershell
Copy-Item .env.example .env
~~~

4. .env에 Cloudflare Workers AI 값을 입력합니다.

~~~dotenv
CLOUDFLARE_ACCOUNT_ID=실제_Account_ID
CLOUDFLARE_API_TOKEN=실제_API_Token
CHAT_MODEL=@cf/qwen/qwen3-30b-a3b-fp8
EMBEDDING_MODEL=@cf/baai/bge-m3
~~~

5. 테스트 실행

~~~powershell
python -m pytest -q
~~~

6. 앱 실행

~~~powershell
python -m streamlit run app.py
~~~

브라우저에서 http://localhost:8501을 엽니다.

## Cloudflare Workers AI 설정

Cloudflare Dashboard에서 **Workers AI → Use REST API → Create a Workers AI API Token**을 선택해 Account ID와 Workers AI 전용 API Token을 발급합니다. Account ID는 이메일 주소가 아니라 Cloudflare 화면에 표시되는 32자리 식별자입니다.

API Token, 비밀번호, Account ID를 코드·README·커밋에 쓰지 마세요. .env와 .streamlit/secrets.toml은 .gitignore에 포함되어 있습니다.

## GitHub Private 및 Streamlit Community Cloud 배포

이 프로젝트의 data/ 폴더에는 내부 업무자료가 포함될 수 있으므로 GitHub 저장소를 **Public으로 변경하지 마세요.** 코드와 자료는 Private Repository를 통해서만 Streamlit Community Cloud에 연결합니다.

1. GitHub에서 Private Repository에 변경사항을 push합니다.
2. [Streamlit Community Cloud](https://share.streamlit.io/)에서 **Create app**을 선택합니다.
3. Repository는 해당 Private Repository, Branch는 main, Main file path는 app.py로 지정합니다.
4. **Advanced settings → Secrets**에 아래 값을 입력합니다.

~~~toml
CLOUDFLARE_ACCOUNT_ID = "실제 Account ID"
CLOUDFLARE_API_TOKEN = "실제 API Token"
CHAT_MODEL = "@cf/qwen/qwen3-30b-a3b-fp8"
EMBEDDING_MODEL = "@cf/baai/bge-m3"
~~~

Secrets는 GitHub에 올리지 않습니다. 배포 후 Streamlit이 알려주는 https://...streamlit.app 주소로 직원이 접속합니다. 내부 문서와 Cloudflare API 사용 정책에 맞는 계정·접근 통제를 별도로 확인하세요.

## 문제 해결

- **PDF에서 텍스트를 추출할 수 없음**: 스캔/이미지 PDF일 수 있습니다. OCR 처리된 PDF 또는 텍스트 원본을 사용하세요.
- **401 또는 403 오류**: Cloudflare Account ID, Token, Workers AI 모델 권한을 확인하세요.
- **429 오류**: Workers AI 사용량 또는 일시적 처리 용량 문제일 수 있습니다.
- **관련 자료를 찾지 못함**: 해당 자료가 data/에 있는지 확인한 뒤 문서 다시 색인을 실행하세요. 검색 최소 유사도는 의도적으로 낮추지 않는 것이 안전합니다.
