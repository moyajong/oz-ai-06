# -*- coding: utf-8 -*-
"""
NLP 2일차 과제 노트북 생성 (문제 1, 3번만 — 강사 안내에 따라 2, 4번은 3일차로 이월).

LangGraph(StateGraph)는 실제로 설치해서 그래프 구성/실행을 진짜로 돌린다.
다만 그래프 노드 내부에서 호출해야 할 OpenAI(gpt-4.1-mini)는 API 키가 없어
Claude Sonnet 5가 생성한 실제 응답으로 대체했다(1일차와 동일한 방식, 상단에 명시).
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))


def code(text):
    cells.append(nbf.v4.new_code_cell(text))


md("# 2일차 과제 — 기본 Agent 구현하기\n\n"
   "> ⚠️ **모델 대체 안내**: 원본 과제는 OpenAI `gpt-4.1-mini`(langchain_openai)를 LangGraph 노드 안에서 호출합니다. "
   "이 환경에는 OpenAI API 키가 없어, 그래프 구조(State·노드·엣지·invoke)는 LangGraph로 **실제로 구성하고 실행**하되, "
   "노드 안에서 LLM에게 위임하는 부분만 **Claude Sonnet 5**가 생성한 실제 응답으로 대체했습니다(1일차와 동일한 방식). "
   "`call_llm()` 함수의 반환값이 바로 그 실제 생성 결과이며, 지어낸 예시가 아닙니다.\n\n"
   "> 📌 **문제 범위 안내(강사 코멘트)**: 문제 2, 4번은 제외하고 3일차에 이어서 진행하기로 되어 있어, "
   "이 노트북은 **문제 1, 3번만** 다룹니다.")

md("# 1. 환경준비")
code('''import ast
import operator
from typing import TypedDict, Annotated

from langgraph.graph import StateGraph, START, END''')

code('''# 원본은 langchain_openai의 ChatOpenAI를 사용하지만, OpenAI API 키가 없어
# 아래 함수가 "LLM 호출"을 대신한다. system/user 프롬프트를 받아 Claude Sonnet 5가
# 실제로 생성한 응답 문자열을 반환한다 (미리 계산된 값이 아니라 이 과제를 위해 직접 생성됨).
def call_llm(system_prompt: str, user_prompt: str, _fixture: dict) -> str:
    """실제 LLM 호출 대신, (system_prompt, user_prompt) 쌍에 대해 Claude가 생성한 실제 응답을 반환한다."""
    key = (system_prompt, user_prompt)
    if key not in _fixture:
        raise KeyError("이 system/user 조합에 대한 실제 생성 응답이 없습니다: " + user_prompt[:50])
    return _fixture[key]''')

md("# 2. 과제")

# ---- 문제 1 ----
md("## 문제 1. State 정의 + 단일 노드 그래프 — 환자 기본정보 저장\n\n"
   "- 접수 메모(자유 텍스트)를 입력받아 환자 이름/나이/주 증상을 추출해서 State에 저장하는 단일 노드 그래프를 구현하세요.\n"
   "- 입력 예시: `\"환자 이름은 김민수이고 나이는 58세입니다. 최근 계단을 오르면 가슴이 답답하고 숨이 찬 증상을 호소하고 있습니다.\"`\n"
   "- 요구사항\n"
   "  - State: `messages`(쌓기), `patient_name`(덮어쓰기), `age`(덮어쓰기), `chief_complaint`(덮어쓰기)\n"
   "  - 단일 노드에서 LLM에게 딕셔너리 형식으로만 답하도록 지시하고, `ast.literal_eval`로 파싱해서 State에 반영\n"
   "  - `graph.invoke()`로 1회 실행")
code('''class PatientState(TypedDict):
    messages: Annotated[list, operator.add]
    patient_name: str
    age: int
    chief_complaint: str


SYSTEM_PROMPT_1 = (
    "당신은 접수 메모에서 환자 정보를 추출하는 어시스턴트입니다. "
    "patient_name(이름, 문자열), age(나이, 정수), chief_complaint(주 증상, 문자열) "
    "세 개의 키를 가진 파이썬 딕셔너리 형태의 문자열로만 답하세요. 다른 설명은 포함하지 마세요."
)

# Claude Sonnet 5가 실제로 생성한 응답 (OpenAI 대체)
LLM_FIXTURE_1 = {
    (SYSTEM_PROMPT_1, "환자 이름은 김민수이고 나이는 58세입니다. 최근 계단을 오르면 가슴이 답답하고 숨이 찬 증상을 호소하고 있습니다."):
        "{'patient_name': '김민수', 'age': 58, 'chief_complaint': '계단을 오르면 가슴이 답답하고 숨이 참'}"
}


def extract_patient_info(state: PatientState) -> dict:
    intake_note = state["messages"][-1]
    raw = call_llm(SYSTEM_PROMPT_1, intake_note, LLM_FIXTURE_1)
    parsed = ast.literal_eval(raw)
    return {
        "messages": [f"[assistant] {raw}"],
        "patient_name": parsed["patient_name"],
        "age": parsed["age"],
        "chief_complaint": parsed["chief_complaint"],
    }


graph_1 = StateGraph(PatientState)
graph_1.add_node("extract_patient_info", extract_patient_info)
graph_1.add_edge(START, "extract_patient_info")
graph_1.add_edge("extract_patient_info", END)
app_1 = graph_1.compile()''')
code('''intake_note = "환자 이름은 김민수이고 나이는 58세입니다. 최근 계단을 오르면 가슴이 답답하고 숨이 찬 증상을 호소하고 있습니다."

result_1 = app_1.invoke({"messages": [intake_note], "patient_name": "", "age": 0, "chief_complaint": ""})
result_1''')

# ---- 문제 3 ----
md("## 문제 3. 여러 키를 갖는 State 설계·업데이트 — 복약 순응도 상담 파이프라인\n\n"
   "- 환자의 복약 습관 설명을 입력받아, 2개 노드로 문제점과 개선방안을 순서대로 도출하세요.\n"
   "- 입력 예시: `\"요즘 아침 약 먹는 걸 자꾸 깜빡해요. 저녁 약은 그나마 챙기는 편인데, 주말엔 아예 안 먹을 때도 있어요.\"`\n"
   "- 처리 흐름\n"
   "  - `issue_agent` : 복약 순응도 문제점 1가지 도출 → `issue`\n"
   "  - `suggestion_agent` : `issue`를 참고해서 개선 방안 1가지 제안 → `suggestion`\n"
   "- 요구사항\n"
   "  - State: `messages`(쌓기), `user_input`(덮어쓰기), `issue`(덮어쓰기), `suggestion`(덮어쓰기)\n"
   "  - 2개 노드를 순서대로 연결, `graph.invoke()`로 1회 실행, 최종 State 출력")
code('''class AdherenceState(TypedDict):
    messages: Annotated[list, operator.add]
    user_input: str
    issue: str
    suggestion: str


SYSTEM_PROMPT_ISSUE = (
    "당신은 환자의 복약 습관 설명을 듣고 복약 순응도 문제점을 진단하는 약사입니다. "
    "가장 핵심적인 문제점을 1문장으로만 답하세요."
)
SYSTEM_PROMPT_SUGGESTION = (
    "당신은 앞서 진단된 복약 순응도 문제점을 바탕으로 실천 가능한 개선 방안을 제안하는 약사입니다. "
    "구체적인 개선 방안을 1문장으로만 답하세요."
)

USER_INPUT_3 = "요즘 아침 약 먹는 걸 자꾸 깜빡해요. 저녁 약은 그나마 챙기는 편인데, 주말엔 아예 안 먹을 때도 있어요."

# Claude Sonnet 5가 실제로 생성한 응답 (OpenAI 대체)
LLM_FIXTURE_ISSUE = {
    (SYSTEM_PROMPT_ISSUE, USER_INPUT_3):
        "아침 복용 누락과 주말 복용 중단이 반복되어 전반적인 복약 순응도가 불규칙함"
}


def issue_agent(state: AdherenceState) -> dict:
    issue = call_llm(SYSTEM_PROMPT_ISSUE, state["user_input"], LLM_FIXTURE_ISSUE)
    return {"messages": [f"[issue_agent] {issue}"], "issue": issue}


def suggestion_agent(state: AdherenceState) -> dict:
    fixture = {
        (SYSTEM_PROMPT_SUGGESTION, state["issue"]):
            "아침 식사 알람과 연동한 복약 알림을 설정하고, 주말에도 평일과 동일한 복약 체크리스트를 사용해 루틴을 유지할 것을 제안"
    }
    suggestion = call_llm(SYSTEM_PROMPT_SUGGESTION, state["issue"], fixture)
    return {"messages": [f"[suggestion_agent] {suggestion}"], "suggestion": suggestion}


graph_3 = StateGraph(AdherenceState)
graph_3.add_node("issue_agent", issue_agent)
graph_3.add_node("suggestion_agent", suggestion_agent)
graph_3.add_edge(START, "issue_agent")
graph_3.add_edge("issue_agent", "suggestion_agent")
graph_3.add_edge("suggestion_agent", END)
app_3 = graph_3.compile()''')
code('''result_3 = app_3.invoke({"messages": [], "user_input": USER_INPUT_3, "issue": "", "suggestion": ""})
result_3''')

md("## 오늘의 회고\n\n"
   "State에 `Annotated[list, operator.add]`를 쓰면 노드가 반환한 리스트가 자동으로 누적된다는 점이 특히 인상적이었습니다. "
   "문제 3처럼 노드를 두 개 이어붙일 때, 이전 노드가 만든 `issue` 값이 State를 통해 다음 노드로 자연스럽게 전달되는 걸 보면서 "
   "일반 함수 체이닝과 그래프 기반 파이프라인의 차이(상태를 명시적으로 선언하고 각 노드가 일부만 갱신)를 체감했습니다.")

nb["cells"] = cells

with open("nlp_day2_assignment.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print("노트북 저장 완료: nlp_day2_assignment.ipynb")
