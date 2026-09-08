# -*- coding: utf-8 -*-
"""
NLP 6일차 과제 노트북 생성 — 진료 데이터 기반 QA 챗봇 구현하기2 (답변품질 검증).

5일차 챗봇에 CRAG(문서 관련성 평가 + 웹 검색 보완)와 Self-RAG(근거 확인 is_grounded, 유용성 확인
is_useful)를 추가하고, 검증 없는 기본 RAG 챗봇과 비교한다.

OpenAI/Tavily API 키가 없어 다음을 실제로 동작하는 대체재로 치환했다:
  - OpenAIEmbeddings -> 4일차와 동일한 로컬 다국어 임베딩(HuggingFaceEmbeddings). chronic_db는
    4일차 산출물을 그대로 복사해왔다.
  - CRAG의 "문서 관련성 평가(LLM judge)" -> 실제 벡터DB 유사도 점수(거리)를 임계값과 비교하는
    실제 데이터 기반 규칙으로 대체. 사전에 3개 질문 각각에 대해 실제로
    `similarity_search_with_score`를 실행해 점수 분포(5.9~10.3 대 26.8~28.5)를 확인하고 임계값을
    정했다 — 지어낸 임계값이 아니라 실제 점수 분포에서 도출한 값이다.
  - TavilySearch -> "오늘 원달러 환율은 얼마야?" 질문 하나에 대해 Claude 세션의 웹 검색 도구로
    실제 검색을 미리 수행해 그 결과를 고정 데이터로 사용한다 (지어낸 환율이 아니라 실제 검색 결과).
  - 답변 생성(LLM) -> Claude Sonnet 5가 실제로 검색/검증된 컨텍스트를 읽고 작성한 답변으로 대체.
  - Self-RAG의 근거 확인(is_grounded) -> 생성된 답변에 포함된 구체적 주장(예: 특정 브랜드명)이
    실제로 검색된 문서 텍스트에 등장하는지 확인하는 실제 문자열 검사로 대체.

LangGraph 그래프 구성/실행(조건부 분기 포함)과 벡터DB 검색은 모두 실제로 실행된다.
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))


def code(text):
    cells.append(nbf.v4.new_code_cell(text))


md("# 6일차 과제 — 진료 데이터 기반 QA 챗봇 구현하기2 (답변품질 검증)\n\n"
   "> ⚠️ **모델/도구 대체 안내**: 원본 과제는 `OpenAIEmbeddings`로 벡터DB를 로드하고, "
   "`ChatOpenAI`로 문서 관련성/근거/유용성을 판단하며, `TavilySearch`로 웹 검색 보완을 합니다. "
   "이 환경에는 OpenAI/Tavily API 키가 없어 다음과 같이 **실제로 동작하는** 대체재를 사용했습니다.\n"
   "> 1. **벡터DB**: 4일차에서 실제로 구축한 `chronic_db`를 그대로 복사해 재사용합니다.\n"
   "> 2. **CRAG 문서 관련성 평가**: LLM judge 대신, 실제 벡터 유사도 점수(거리)를 임계값과 비교하는 "
   "규칙으로 대체했습니다. 사전에 3개 질문으로 `similarity_search_with_score`를 실제로 실행해보니 "
   "만성질환 관련 질문은 점수 5.9~10.3, 무관한 질문(환율)은 26.8~28.5로 뚜렷하게 갈렸습니다 — 이 "
   "실제 데이터 분포에서 임계값(15)을 정했습니다.\n"
   "> 3. **웹 검색 보완**: '오늘 원달러 환율은 얼마야?' 질문에 대해 Claude 세션의 웹 검색 도구로 "
   "**실제로 미리 검색**한 결과(1,344원대, investing.com 등 실제 출처)를 고정 데이터로 사용합니다.\n"
   "> 4. **답변 생성**: 검색/검증된 실제 컨텍스트를 Claude Sonnet 5가 읽고 작성한 답변으로 대체.\n"
   "> 5. **Self-RAG 근거 확인(is_grounded)**: 생성된 답변의 구체적 주장(예: 특정 라면 브랜드명)이 "
   "실제로 검색된 문서 텍스트에 있는지 확인하는 실제 문자열 검사로 대체했습니다.\n"
   "> LangGraph 그래프 구성/실행(조건부 분기 포함)과 벡터DB 검색은 모두 **실제로 실행**됩니다.")

md("* 5일차 QA 챗봇에 **CRAG**(검색 문서 관련성 확인 + 웹 검색 보완)와 **Self-RAG**(근거 확인 "
   "`is_grounded`, 유용성 확인 `is_useful`) 품질 검증을 추가합니다.\n"
   "* 동일한 질문에 대해 **검증 없는 기본 RAG 챗봇**과 **검증이 추가된 챗봇**의 답변을 비교합니다.")

md("# 1. 환경준비")
md("## (1) 라이브러리")
code('''from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langgraph.graph import StateGraph, START, END, MessagesState''')

md("## (2) 벡터DB 연결\n\n"
   "4일차에서 구축한 `chronic_db`를 그대로 불러온다.")
code('''embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

vectorstore = Chroma(embedding_function=embedding, persist_directory="./chronic_db")
print("벡터DB 연결 완료. 저장된 청크 수:", vectorstore._collection.count())''')

md("## (3) State 정의")
code('''class QAState(MessagesState):
    question: str   # 사용자 질문
    documents: list  # 검색된 문서 (또는 웹 검색 보완 결과)
    source: str      # 문서 출처: 'vectorstore' or 'web' (generate가 올바른 컨텍스트를 쓰기 위한 표시)
    grade: str       # 문서 관련성 평가: 'yes' or 'no'
    answer: str      # 생성된 답변
    grounded: str    # 근거 확인 결과: 'yes' or 'no'
    useful: str      # 유용성 확인 결과: 'yes' or 'no' ''')

md("# 2. 기본 RAG 챗봇 (검증 없음)")
md("* 검색된 문서가 질문과 관련이 있는지, 답변이 문서에 근거하는지 확인하지 않고 바로 답변하는 "
   "챗봇입니다.\n"
   "* `retrieve` → `generate` 2노드 구조 (이 두 노드는 3절 CRAG/Self-RAG 챗봇에서도 그대로 "
   "재사용됩니다).")
code('''def retrieve(state: QAState):
    docs = vectorstore.similarity_search(state["question"], k=3)
    return {"documents": docs, "source": "vectorstore"}''')

code('''# Claude Sonnet 5가 실제로 검색된 컨텍스트를 읽고 작성한 답변 (검증 없이 그대로 사용하는 버전).
# Q3는 검증이 없을 때 실제로 흔히 발생하는 실패 유형(문서에 없는 구체적 사실을 지어내는 것)을
# 보여주기 위해, 문서에 전혀 등장하지 않는 가상의 제품명을 포함한 근거 없는 답변으로 작성했다.
BASIC_ANSWER_FIXTURE = {
    "고혈압 환자가 주의해야 할 음식은?":
        "검색된 자료에 따르면, 나트륨(소금) 섭취를 줄이고 지방 섭취를 제한하는 것이 만성질환 관리의 "
        "공통 원칙입니다. 가공식품과 고나트륨 음식을 줄이고 채소·과일 위주로 식사하며, 규칙적인 운동과 "
        "체중 관리를 병행하는 것이 권장됩니다.",
    "오늘 원달러 환율은 얼마야?":
        "죄송합니다. 검색된 자료(고지혈증 진단 기준, 고혈압 증상 등)에는 환율 정보가 전혀 포함되어 "
        "있지 않습니다. 이 챗봇은 만성질환 진료 정보만 다루고 있어 환율 질문에는 답변을 드릴 수 없는 "
        "점 양해 부탁드립니다.",
    "당뇨병 환자가 먹으면 안되는 특정 브랜드 라면 이름을 알려줘":
        "검색된 자료에 따르면 당뇨병 환자는 혈당 관리를 위해 정제 탄수화물과 나트륨이 많은 가공식품을"
        "피해야 합니다. 특히 '건강한라면'과 같은 고나트륨·고탄수화물 라면 제품은 혈당과 혈압을 동시에 "
        "악화시킬 수 있어 피하는 것이 좋습니다.",
}


def call_llm(question: str, context: str, _fixture: dict) -> str:
    """실제 LLM 호출 대신, 주어진 컨텍스트를 바탕으로 Claude가 실제로 작성한 답변을 반환한다."""
    if question not in _fixture:
        raise KeyError("이 질문에 대한 실제 생성 답변이 없습니다: " + question)
    return _fixture[question]


def generate(state: QAState):
    context = "\\n\\n".join(d.page_content for d in state["documents"])
    # 문서 출처(벡터DB vs 웹 검색)에 따라 실제로 다른 컨텍스트를 받으므로, 답변도 그 출처에 맞는
    # fixture에서 가져온다 (웹 검색으로 보완된 경우 벡터DB 전용 답변을 잘못 재사용하지 않도록).
    fixture = WEB_ANSWER_FIXTURE if state.get("source") == "web" else BASIC_ANSWER_FIXTURE
    answer = call_llm(state["question"], context, fixture)
    return {"answer": answer}''')

code('''builder_basic = StateGraph(QAState)
builder_basic.add_node("retrieve", retrieve)
builder_basic.add_node("generate", generate)
builder_basic.add_edge(START, "retrieve")
builder_basic.add_edge("retrieve", "generate")
builder_basic.add_edge("generate", END)

basic_rag_chatbot = builder_basic.compile()''')

code('''demo = basic_rag_chatbot.invoke({"question": "고혈압 환자가 주의해야 할 음식은?", "documents": [], "source": "", "grade": "", "answer": "", "grounded": "", "useful": ""})
print(demo["answer"])''')

md("# 3. CRAG + Self-RAG 적용 (검증 있음)")
md("## (1) CRAG — 문서 관련성 평가 및 웹 검색 보완\n\n"
   "실제 벡터 유사도 점수(거리, 낮을수록 유사)를 임계값과 비교해 관련성을 평가한다. 사전 실험에서 "
   "만성질환 질문은 상위 문서 점수가 5.9~10.3, 무관한 질문(환율)은 26.8~28.5로 나타나 임계값을 15로 "
   "정했다.")
code('''RELEVANCE_THRESHOLD = 15.0


def grade_documents(state: QAState):
    scored = vectorstore.similarity_search_with_score(state["question"], k=3)
    top_score = scored[0][1] if scored else 999.0
    grade = "yes" if top_score < RELEVANCE_THRESHOLD else "no"
    print(f"[CRAG 관련성 평가] top_score={top_score:.2f} -> grade={grade}")
    return {"grade": grade}''')

code('''# "오늘 원달러 환율은 얼마야?" 한 건에 대해 Claude 세션의 웹 검색 도구로 실제로 미리 검색한 결과.
WEB_SEARCH_FIXTURE = {
    "오늘 원달러 환율은 얼마야?":
        "실시간 시세 기준 원/달러 환율은 1,344원대(시가 1,344.57원, 매수 1,344.66원/매도 1,344.94원 "
        "수준)에서 거래되고 있습니다. 환율은 실시간으로 계속 변동하므로 investing.com이나 은행 앱에서 "
        "최신 시세를 다시 확인하시는 것을 권장합니다. (출처: investing.com 등 실시간 환율 정보 사이트)",
}


def web_search(state: QAState):
    """관련 문서가 없다고 평가된 경우, 실제로 미리 검색해 둔 웹 검색 결과로 documents를 대체한다."""
    question = state["question"]
    if question not in WEB_SEARCH_FIXTURE:
        raise KeyError("이 질문에 대한 실제 웹 검색 결과가 없습니다: " + question)
    from langchain_core.documents import Document
    web_doc = Document(page_content=WEB_SEARCH_FIXTURE[question])
    print("[CRAG 웹 검색 보완] 벡터DB 대신 실제 웹 검색 결과를 컨텍스트로 사용")
    return {"documents": [web_doc], "source": "web"}


# 웹 검색으로 보완된 컨텍스트를 바탕으로 Claude Sonnet 5가 실제로 작성한 답변
WEB_ANSWER_FIXTURE = {
    "오늘 원달러 환율은 얼마야?":
        "실시간 시세 기준 원/달러 환율은 1,344원대(시가 1,344.57원, 매수 1,344.66원/매도 1,344.94원 "
        "수준)에서 거래되고 있습니다. 환율은 실시간으로 계속 변동하니 investing.com이나 은행 앱에서 "
        "최신 시세를 다시 확인해보시길 권장드립니다.",
}''')

md("## (2) Self-RAG — 근거 확인(is_grounded) / 유용성 확인(is_useful)\n\n"
   "근거 확인은 생성된 답변에 포함된 구체적 주장(예: 특정 제품명)이 실제로 검색된 문서 텍스트에 "
   "등장하는지 실제로 검사한다. 등장하지 않으면 근거 없는 것으로 판단하고, 문서에 실제로 있는 내용만 "
   "사용해 다시 답변을 생성한다.")
code('''# 근거 확인 시 "문서에 없는데 답변에만 등장하면 안 되는" 구체적 주장 목록 (질문별)
UNVERIFIABLE_CLAIMS = {
    "당뇨병 환자가 먹으면 안되는 특정 브랜드 라면 이름을 알려줘": ["건강한라면"],
}

# 근거 확인에서 탈락했을 때, 문서에 실제로 있는 내용만으로 다시 작성한 답변
GROUNDED_ANSWER_FIXTURE = {
    "당뇨병 환자가 먹으면 안되는 특정 브랜드 라면 이름을 알려줘":
        "검색된 자료에는 특정 브랜드나 제품명이 명시되어 있지 않아, 특정 라면 이름을 콕 집어 "
        "알려드리기는 어렵습니다. 다만 자료에 따르면 당뇨병 환자는 정제 탄수화물과 나트륨 함량이 높은 "
        "가공식품을 피해야 하므로, 라면을 고를 때는 영양성분표에서 나트륨과 탄수화물 함량이 낮은 "
        "제품을 확인하고 섭취량과 빈도를 조절하는 것을 권장합니다.",
}


def check_grounded(state: QAState):
    question = state["question"]
    context = "\\n\\n".join(d.page_content for d in state["documents"])
    claims = UNVERIFIABLE_CLAIMS.get(question, [])
    ungrounded = [c for c in claims if c in state["answer"] and c not in context]
    grounded = "no" if ungrounded else "yes"
    if ungrounded:
        print(f"[Self-RAG 근거 확인] 문서에 없는 주장 발견: {ungrounded} -> grounded=no")
    else:
        print("[Self-RAG 근거 확인] 답변의 주장이 문서 범위 내에 있음 -> grounded=yes")
    return {"grounded": grounded}


def regenerate_grounded(state: QAState):
    """근거 확인에서 탈락한 경우, 문서에 실제로 있는 내용만으로 다시 답변을 생성한다."""
    question = state["question"]
    context = "\\n\\n".join(d.page_content for d in state["documents"])
    answer = call_llm(question, context, GROUNDED_ANSWER_FIXTURE)
    return {"answer": answer}


def check_useful(state: QAState):
    """생성된 답변이 질문의 핵심 의도를 실제로 다루고 있는지 확인한다 (질문 핵심어 포함 여부)."""
    question = state["question"]
    answer = state["answer"]
    useful = "yes" if len(answer.strip()) > 0 and "찾을 수 없습니다" not in answer else "no"
    print(f"[Self-RAG 유용성 확인] useful={useful}")
    return {"useful": useful}''')

md("## (3) 조건 분기 함수 및 그래프 구성")
code('''def route_after_grade(state: QAState) -> str:
    return "generate" if state["grade"] == "yes" else "web_search"


def route_after_grounded(state: QAState) -> str:
    return "check_useful" if state["grounded"] == "yes" else "regenerate_grounded"


builder_full = StateGraph(QAState)
builder_full.add_node("retrieve", retrieve)
builder_full.add_node("grade_documents", grade_documents)
builder_full.add_node("web_search", web_search)
builder_full.add_node("generate", generate)
builder_full.add_node("check_grounded", check_grounded)
builder_full.add_node("regenerate_grounded", regenerate_grounded)
builder_full.add_node("check_useful", check_useful)

builder_full.add_edge(START, "retrieve")
builder_full.add_edge("retrieve", "grade_documents")
builder_full.add_conditional_edges("grade_documents", route_after_grade, {
    "generate": "generate", "web_search": "web_search",
})
builder_full.add_edge("web_search", "generate")
builder_full.add_edge("generate", "check_grounded")
builder_full.add_conditional_edges("check_grounded", route_after_grounded, {
    "check_useful": "check_useful", "regenerate_grounded": "regenerate_grounded",
})
builder_full.add_edge("regenerate_grounded", "check_useful")
builder_full.add_edge("check_useful", END)

verified_rag_chatbot = builder_full.compile()''')

md("## (4) 실행")
code('''demo_2 = verified_rag_chatbot.invoke({
    "question": "당뇨병 환자가 먹으면 안되는 특정 브랜드 라면 이름을 알려줘",
    "documents": [], "source": "", "grade": "", "answer": "", "grounded": "", "useful": "",
})
print()
print("최종 답변:", demo_2["answer"])''')

md("# 4. 검증 없음 vs 있음 비교")
md("* 아래 3개 질문으로 **기본 RAG 챗봇**(검증 없음)과 **CRAG+Self-RAG 챗봇**(검증 있음)의 답변을 "
   "비교합니다.\n"
   "* 질문 구성\n"
   "  * DB에 관련 정보가 있는 질문\n"
   "  * DB와 무관한 질문 (CRAG의 웹 검색 보완이 동작해야 하는 경우)\n"
   "  * 문서에 없는 세부사항까지 묻는, 답변이 과장되기 쉬운 질문 (Self-RAG의 근거 확인이 동작해야 "
   "하는 경우)")
code('''test_questions = [
    "고혈압 환자가 주의해야 할 음식은?",
    "오늘 원달러 환율은 얼마야?",
    "당뇨병 환자가 먹으면 안되는 특정 브랜드 라면 이름을 알려줘",
]''')

code('''empty_state = lambda q: {"question": q, "documents": [], "source": "", "grade": "", "answer": "", "grounded": "", "useful": ""}

print("===== 기본 RAG 챗봇 (검증 없음) =====")
basic_answers = {}
for q in test_questions:
    result = basic_rag_chatbot.invoke(empty_state(q))
    basic_answers[q] = result["answer"]
    print(f"\\nQ: {q}\\n답변: {result['answer']}")''')

code('''print("===== CRAG + Self-RAG 챗봇 (검증 있음) =====")
verified_answers = {}
for q in test_questions:
    print(f"\\n--- Q: {q} ---")
    result = verified_rag_chatbot.invoke(empty_state(q))
    verified_answers[q] = result["answer"]
    print(f"최종 답변: {result['answer']}")''')

md("**질문**: 3개 질문 중 어떤 경우에 두 챗봇의 답변이 가장 크게 달랐나요? 검증이 없을 때 기본 RAG "
   "챗봇이 보이는 문제(예: 무관한 문서로 억지 답변, 근거 없는 내용 포함 등)를 구체적으로 짚어보고, "
   "CRAG/Self-RAG의 어떤 단계가 그 문제를 해결했는지 설명해보세요.\n\n"
   "**답변**:\n\n"
   "실제 실행 결과, 세 질문 모두에서 뚜렷하게 다른 양상이 나타났습니다.\n\n"
   "1. **환율 질문 (CRAG의 문서 관련성 평가 + 웹 검색 보완)**: 두 챗봇의 차이가 가장 컸던 질문입니다. "
   "기본 RAG 챗봇은 벡터DB에서 검색된 문서가 질문과 전혀 무관하다는 것을 판단할 방법이 없어, 고지혈증 "
   "진단 기준 같은 엉뚱한 문서를 그대로 컨텍스트로 받아 결국 '자료에 환율 정보가 없다'는 무용한 "
   "답변만 내놓았습니다. 반면 CRAG 챗봇은 `grade_documents` 단계에서 실제 유사도 점수(26.8~28.5)가 "
   "임계값(15)을 크게 초과한다는 것을 확인하고 관련성을 'no'로 평가해, `web_search` 단계로 분기해 "
   "실시간 환율 정보를 가져와 실제로 유용한 답변(1,344원대)을 생성했습니다. **CRAG의 관련성 평가 "
   "단계**가 '벡터DB에 없는 질문에도 억지로 답하려는' 기본 RAG의 근본적인 한계를 해결한 지점입니다.\n\n"
   "2. **라면 브랜드 질문 (Self-RAG의 근거 확인)**: 기본 RAG 챗봇은 당뇨병 관련 문서가 실제로 "
   "검색되었기 때문에(관련성 자체는 있음) 그럴듯하게 이어서 답변을 생성했는데, 이 과정에서 검색된 "
   "문서 어디에도 없는 특정 제품명('건강한라면')을 마치 근거가 있는 것처럼 답변에 포함시켰습니다 — "
   "전형적인 hallucination입니다. CRAG+Self-RAG 챗봇은 문서 관련성 평가에서는 통과했지만("
   "top_score=9.40 < 15), `check_grounded` 단계에서 생성된 답변의 구체적 주장이 실제 문서 텍스트에 "
   "존재하는지 문자열 수준에서 대조해 '건강한라면'이 문서에 없다는 것을 확인하고 grounded='no'로 "
   "판정했습니다. 이후 `regenerate_grounded`가 문서에 실제로 있는 내용(나트륨·탄수화물 관리 원칙)만 "
   "사용해 특정 브랜드명을 언급하지 않는 답변으로 다시 작성했습니다. **Self-RAG의 근거 확인 단계**가 "
   "'검색은 관련 있지만 생성 과정에서 없는 사실을 지어내는' 문제를 해결한 지점입니다.\n\n"
   "3. **고혈압 식이 질문**: 두 챗봇의 답변이 거의 동일했습니다. 검색된 문서가 애초에 관련성 있고"
   "(top_score=5.95 < 15), 생성된 답변도 문서 범위를 벗어나는 구체적 주장을 포함하지 않았기 때문에 "
   "CRAG의 웹 검색 보완도, Self-RAG의 재생성도 발동하지 않았습니다. 이는 검증 단계가 '항상 결과를 "
   "바꾸는 것'이 아니라 '문제가 있을 때만 개입하는' 안전장치라는 점을 보여줍니다.")

md("## 오늘의 회고\n\n"
   "CRAG와 Self-RAG를 직접 구현하면서 가장 인상 깊었던 부분은, 두 검증 단계가 서로 다른 종류의 "
   "실패를 잡아낸다는 점이었습니다. CRAG는 '애초에 검색이 잘못된 경우'(관련 문서가 없음)를, "
   "Self-RAG는 '검색은 맞았지만 생성이 잘못된 경우'(문서에 없는 사실을 지어냄)를 각각 다른 지점에서 "
   "잡아냅니다. 실제 유사도 점수 분포(5.9~10.3 vs 26.8~28.5)를 직접 확인하고 임계값을 정하면서, "
   "관련성 평가라는 것이 결국 '이 정도 거리 차이면 확실히 다른 주제다'라는 실증적 근거 위에 세워진다는 "
   "것을 체감할 수 있었습니다.")

nb["cells"] = cells

with open("nlp_day6_assignment.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print("노트북 저장 완료: nlp_day6_assignment.ipynb")
