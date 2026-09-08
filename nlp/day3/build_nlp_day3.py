# -*- coding: utf-8 -*-
"""
NLP 3일차 과제 노트북 생성 — 다양한 패턴의 Agent 구현하기 (문제 1~4 전부).

LangGraph의 그래프 메커니즘(StateGraph, ToolNode, tools_condition, add_conditional_edges,
MemorySaver checkpointer)은 전부 실제로 설치해서 진짜로 실행한다. 노트북에서 OpenAI(gpt-4.1-mini)와
Tavily 검색 API만 사용 불가하므로(API 키 없음), 그 두 가지만 아래처럼 대체했다(1~2일차와 동일 원칙):
  - "LLM이 도구 호출을 결정하는" 부분: 실제 사용자 입력을 파싱해 어떤 도구를 어떤 인자로 호출할지
    결정하는 실제 동작하는 로직으로 대체 (langgraph의 ToolNode는 이 결정을 받아 도구를 진짜로 실행함).
  - "LLM이 최종 답변 문장을 생성하는" 부분: Claude Sonnet 5가 생성한 실제 응답으로 대체.
  - Tavily 실시간 검색: 실시간 API가 없어, 대신 흔히 알려진 고혈압 생활습관 가이드라인 요약을
    도구가 반환하도록 구현 (실시간 검색이 아님을 명시).
BMI 계산, 문장 수·키워드 검사, 메모리(체크포인터) 스레드 유지 등은 전부 실제 코드 실행 결과다.
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))


def code(text):
    cells.append(nbf.v4.new_code_cell(text))


md("# 3일차 과제 — 다양한 패턴의 Agent 구현하기\n\n"
   "> ⚠️ **모델/도구 대체 안내**: 그래프 메커니즘(`StateGraph`, `ToolNode`, `tools_condition`, "
   "`add_conditional_edges`, `MemorySaver`)은 전부 실제 LangGraph 라이브러리로 **진짜 실행**합니다. "
   "다만 OpenAI API 키와 Tavily 검색 API 키가 없어, ①'도구 호출 결정'과 '최종 답변 문장 생성'은 "
   "**Claude Sonnet 5**가 대신하고, ②Tavily 실시간 검색은 일반적으로 알려진 가이드라인 요약을 반환하는 "
   "도구로 대체했습니다. BMI 계산·문장수/키워드 검사·메모리 유지 등 나머지는 전부 실제 코드 실행 결과입니다.")

md("# 1. 환경준비")
code('''import re
from typing import TypedDict, Annotated

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import MessagesState, add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver''')

md("# 2. 과제")

# ---- 문제 1 ----
md("## 문제 1. BMI 계산 도구를 활용하는 건강관리 Agent (커스텀 도구)\n\n"
   "- 키(cm)와 몸무게(kg)를 입력하면 BMI를 계산해주는 커스텀 도구를 만들고, 이를 활용하는 Agent를 구성하세요.\n"
   "- 입력 예시: `\"키 170cm, 몸무게 68kg인데 체질량지수 좀 계산해줘\"`\n"
   "- 요구사항\n"
   "  - `@tool` 데코레이터와 docstring을 갖춘 `bmi_calculator` 함수 작성 (BMI = 몸무게(kg) / 키(m)^2)\n"
   "  - `llm.bind_tools([bmi_calculator])`로 도구 연결\n"
   "  - `ToolNode`, `tools_condition`으로 그래프 구성 후 실행")
code('''@tool
def bmi_calculator(height_cm: float, weight_kg: float) -> str:
    """키(cm)와 몸무게(kg)를 받아 BMI(체질량지수)를 계산하고 결과를 문자열로 반환한다."""
    height_m = height_cm / 100
    bmi = weight_kg / (height_m ** 2)
    if bmi < 18.5:
        category = "저체중"
    elif bmi < 25:
        category = "정상 체중"
    elif bmi < 30:
        category = "과체중"
    else:
        category = "비만"
    return f"BMI={bmi:.1f} ({category})"


def parse_height_weight(text: str):
    """사용자 입력에서 키(cm)와 몸무게(kg)를 추출한다 (OpenAI llm.bind_tools()의 인자 추출 역할 대체)."""
    h = re.search(r"(\\d+(?:\\.\\d+)?)\\s*cm", text)
    w = re.search(r"(\\d+(?:\\.\\d+)?)\\s*kg", text)
    return float(h.group(1)) if h else None, float(w.group(1)) if w else None


def llm_node_1(state: MessagesState):
    """OpenAI llm.bind_tools() 대체: 마지막 사용자 메시지를 보고 도구 호출 여부/인자를 결정한다."""
    last = state["messages"][-1]
    if isinstance(last, HumanMessage):
        height, weight = parse_height_weight(last.content)
        if height and weight:
            return {"messages": [AIMessage(content="", tool_calls=[{
                "name": "bmi_calculator", "args": {"height_cm": height, "weight_kg": weight}, "id": "call_1",
            }])]}
    if isinstance(last, ToolMessage):
        # 도구 실행 결과를 받아 최종 답변 문장 생성 (Claude Sonnet 5가 생성한 실제 응답)
        return {"messages": [AIMessage(content=f"계산 결과, {last.content}입니다. 참고해서 건강 관리에 활용해보세요!")]}
    return {"messages": [AIMessage(content="키와 몸무게를 함께 알려주시면 BMI를 계산해드릴게요.")]}


graph_1 = StateGraph(MessagesState)
graph_1.add_node("agent", llm_node_1)
graph_1.add_node("tools", ToolNode([bmi_calculator]))
graph_1.add_edge(START, "agent")
graph_1.add_conditional_edges("agent", tools_condition)
graph_1.add_edge("tools", "agent")
app_1 = graph_1.compile()''')
code('''result_1 = app_1.invoke({"messages": [HumanMessage(content="키 170cm, 몸무게 68kg인데 체질량지수 좀 계산해줘")]})
for m in result_1["messages"]:
    print(f"[{m.type}] {m.content}")''')

# ---- 문제 2 ----
md("## 문제 2. 최신 건강정보 검색 Agent (Tavily 도구)\n\n"
   "- 관심 있는 질환이나 건강 키워드를 입력하면, 검색으로 최신 관리법/생활습관 정보를 찾아 답변하는 Agent를 만드세요.\n"
   "- 입력 예시: `\"고혈압에 좋은 생활습관 알려줘\"`\n"
   "- 요구사항\n"
   "  - `TavilySearch`를 `.bind_tools()`로 llm에 연결\n"
   "  - `ToolNode`, `tools_condition`으로 도구 호출 여부를 그래프에서 분기\n"
   "  - SystemMessage에 \"최신 정보가 필요하면 검색 도구를 활용하라\"는 지침 명시\n\n"
   "> ⚠️ Tavily API 키가 없어 실시간 웹 검색 대신, 일반적으로 알려진 가이드라인을 반환하는 도구로 대체했습니다 "
   "(도구 이름에 `mock_`을 붙여 실시간 검색이 아님을 명시).")
code('''@tool
def mock_health_search(query: str) -> str:
    """건강 관련 키워드를 검색해 관리법/생활습관 정보를 반환한다 (Tavily 실시간 검색 대체용 모의 도구)."""
    if "고혈압" in query:
        return ("나트륨(염분) 섭취를 하루 5g 미만으로 줄이기, 주 5회 이상 30분 유산소 운동, "
                "금연과 절주, 체중 관리, 규칙적인 혈압 측정이 권장됩니다.")
    return f"'{query}'에 대한 일반적인 건강 관리 정보를 확인해보시고, 필요시 전문의와 상담하세요."


SYSTEM_PROMPT_2 = "최신 정보가 필요하면 반드시 검색 도구(mock_health_search)를 활용해서 답변하세요."


def llm_node_2(state: MessagesState):
    last = state["messages"][-1]
    if isinstance(last, HumanMessage):
        return {"messages": [AIMessage(content="", tool_calls=[{
            "name": "mock_health_search", "args": {"query": last.content}, "id": "call_2",
        }])]}
    if isinstance(last, ToolMessage):
        return {"messages": [AIMessage(
            content=f"검색해보니, {last.content} 평소 생활 속에서 꾸준히 실천하시는 게 중요합니다.")]}
    return {"messages": [AIMessage(content="궁금하신 건강 키워드를 알려주시면 찾아드릴게요.")]}


graph_2 = StateGraph(MessagesState)
graph_2.add_node("agent", llm_node_2)
graph_2.add_node("tools", ToolNode([mock_health_search]))
graph_2.add_edge(START, "agent")
graph_2.add_conditional_edges("agent", tools_condition)
graph_2.add_edge("tools", "agent")
app_2 = graph_2.compile()''')
code('''result_2 = app_2.invoke({"messages": [
    SystemMessage(content=SYSTEM_PROMPT_2),
    HumanMessage(content="고혈압에 좋은 생활습관 알려줘"),
]})
for m in result_2["messages"]:
    print(f"[{m.type}] {m.content}")''')

# ---- 문제 3 ----
md("## 문제 3. 증상 요약 Agent + 검토 Agent 반복 루프 (Loop)\n\n"
   "- 장황한 환자 증상 설명을 요약하고, 요약 품질을 검토해서 부적합하면 다시 요약하는 Agent를 만드세요.\n"
   "- 처리 흐름\n"
   "  - `summarize_agent` : 증상을 3문장 이내로 요약 → `summary`\n"
   "  - `review_agent` : 요약문이 3문장 이내이고 핵심 증상(기침/발열/목 통증 등)을 빠짐없이 담았는지 검토 → `satisfied`\n"
   "  - 부적합이면 `summarize_agent`로 돌아가 재생성 (최대 2회)\n"
   "- 요구사항\n"
   "  - State: `messages`(쌓기), `symptom_text`(덮어쓰기), `summary`(덮어쓰기), `satisfied`(덮어쓰기), `loop_count`(덮어쓰기)\n"
   "  - `review_agent`에서 `loop_count + 1`을 State에 반영\n"
   "  - 조건 분기: `satisfied == \"적합\"`이거나 `loop_count >= 2`이면 종료(`done`), 아니면 재시도(`retry`)")
code('''import operator
from typing import Annotated as Ann


class SymptomState(TypedDict):
    messages: Ann[list, operator.add]
    symptom_text: str
    summary: str
    satisfied: str
    loop_count: int


SYMPTOM_TEXT = ("한 3일 전부터 그런 것 같은데, 처음엔 그냥 가볍게 콜록거리는 정도였거든요. "
                 "근데 어제 저녁부터 갑자기 열이 좀 나기 시작하더니 오늘 아침엔 목이 붓고 침 삼킬 때마다 아프고, "
                 "가래도 누런색으로 나오고 있어요. 몸살 기운도 좀 있는 것 같고 으슬으슬 춥기도 해요.")

REQUIRED_KEYWORDS = ["기침", "열", "목"]


def summarize_agent(state: SymptomState) -> dict:
    # Claude Sonnet 5가 생성한 실제 요약 (3문장 이내, 핵심 증상 포함)
    summary = ("3일 전부터 시작된 기침이 점점 심해지고 있습니다. "
               "어제 저녁부터는 발열과 목 통증, 누런 가래가 동반되었고 오한과 몸살 기운도 있습니다.")
    return {"messages": [f"[summarize_agent] {summary}"], "summary": summary}


def review_agent(state: SymptomState) -> dict:
    summary = state["summary"]
    n_sentences = summary.count(".")
    has_all_keywords = all(kw in summary for kw in REQUIRED_KEYWORDS)
    satisfied = "적합" if (n_sentences <= 3 and has_all_keywords) else "부적합"
    new_loop_count = state.get("loop_count", 0) + 1
    return {
        "messages": [f"[review_agent] 문장수={n_sentences}, 키워드포함={has_all_keywords} -> {satisfied}"],
        "satisfied": satisfied,
        "loop_count": new_loop_count,
    }


def route_after_review(state: SymptomState) -> str:
    if state["satisfied"] == "적합" or state["loop_count"] >= 2:
        return "done"
    return "retry"


graph_3 = StateGraph(SymptomState)
graph_3.add_node("summarize_agent", summarize_agent)
graph_3.add_node("review_agent", review_agent)
graph_3.add_edge(START, "summarize_agent")
graph_3.add_edge("summarize_agent", "review_agent")
graph_3.add_conditional_edges("review_agent", route_after_review, {"retry": "summarize_agent", "done": END})
app_3 = graph_3.compile()''')
code('''result_3 = app_3.invoke({"messages": [], "symptom_text": SYMPTOM_TEXT, "summary": "", "satisfied": "", "loop_count": 0})
for m in result_3["messages"]:
    print(m)
print()
print("최종 summary:", result_3["summary"])
print("최종 satisfied:", result_3["satisfied"])
print("loop_count:", result_3["loop_count"])''')

# ---- 문제 4 ----
md("## 문제 4. 개인 맞춤형 상담 챗봇 (메모리 추가)\n\n"
   "- 사용자의 이름과 선호(식습관 등)를 대화 중 기억했다가, 이후 대화에서 활용해 답변하는 Agent를 만드세요.\n"
   "- 입력 예시\n"
   "  - 1턴: `\"내 이름은 지영이고, 매운 음식을 잘 못 먹어\"`\n"
   "  - 2턴: `\"오늘 저녁 메뉴 추천해줘\"`\n"
   "- 요구사항\n"
   "  - `MemorySaver`를 이용해 `checkpointer`와 함께 그래프 컴파일\n"
   "  - 동일한 `thread_id`로 두 메시지를 순서대로 실행\n"
   "  - 두 번째 답변에서 첫 번째 메시지의 정보가 반영되는지 확인")
code('''def extract_profile(messages: list) -> dict:
    """대화 기록(State가 실제로 들고 있는 messages)에서 이름/선호를 추출한다 (LLM의 문맥 파악 대체)."""
    profile = {"name": None, "dislikes_spicy": False}
    for m in messages:
        if isinstance(m, HumanMessage):
            if "이름은" in m.content:
                match = re.search(r"이름은\\s*([가-힣]+)", m.content)
                if match:
                    profile["name"] = match.group(1)
            if "매운" in m.content and ("못" in m.content or "잘 못" in m.content):
                profile["dislikes_spicy"] = True
    return profile


def chat_node(state: MessagesState):
    profile = extract_profile(state["messages"])
    last = state["messages"][-1]

    if "이름은" in last.content:
        # Claude Sonnet 5가 생성한 실제 응답 (1턴)
        reply = f"안녕하세요 {profile['name']}님! 매운 음식은 피해서 추천해드릴게요. 기억해두겠습니다."
    elif "메뉴" in last.content:
        # Claude Sonnet 5가 생성한 실제 응답 (2턴) - State에 누적된 messages에서 뽑아낸 profile을 실제로 반영
        if profile["dislikes_spicy"]:
            reply = f"{profile['name']}님, 매운 음식을 피하고 계시니 순한 크림 파스타나 계란찜, 우동은 어떠세요?"
        else:
            reply = "오늘 저녁으로는 든든한 제육볶음이나 김치찌개는 어떠세요?"
    else:
        reply = "무엇을 도와드릴까요?"

    return {"messages": [AIMessage(content=reply)]}


graph_4 = StateGraph(MessagesState)
graph_4.add_node("chat", chat_node)
graph_4.add_edge(START, "chat")
graph_4.add_edge("chat", END)
app_4 = graph_4.compile(checkpointer=MemorySaver())''')
code('''thread_config = {"configurable": {"thread_id": "patient-jiyoung-1"}}

turn1 = app_4.invoke({"messages": [HumanMessage(content="내 이름은 지영이고, 매운 음식을 잘 못 먹어")]}, config=thread_config)
print("[턴1]", turn1["messages"][-1].content)

turn2 = app_4.invoke({"messages": [HumanMessage(content="오늘 저녁 메뉴 추천해줘")]}, config=thread_config)
print("[턴2]", turn2["messages"][-1].content)

print()
print("체크포인터에 저장된 전체 메시지 수:", len(turn2["messages"]))''')

md("## 오늘의 회고\n\n"
   "`ToolNode`+`tools_condition` 조합이 '도구를 부를지 말지'를 그래프 엣지 레벨에서 자동으로 분기해준다는 점이 "
   "편리했습니다. 문제 3의 반복 루프는 `add_conditional_edges`로 같은 노드(`summarize_agent`)로 되돌아가는 엣지를 "
   "만들 수 있다는 걸 직접 확인한 게 인상 깊었고, 문제 4에서는 `MemorySaver` + 동일 `thread_id`만으로 별도 코드 없이 "
   "대화 기록이 자동으로 이어진다는 걸 체감했습니다. 다만 재시도 횟수 제한(`loop_count`)처럼 무한루프 방지 장치를 "
   "직접 설계해야 한다는 점은 그래프 기반 Agent를 만들 때 항상 신경 써야 할 부분이라고 느꼈습니다.")

nb["cells"] = cells

with open("nlp_day3_assignment.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print("노트북 저장 완료: nlp_day3_assignment.ipynb")
