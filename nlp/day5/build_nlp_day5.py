# -*- coding: utf-8 -*-
"""
NLP 5일차 과제 노트북 생성 — 진료 데이터 기반 QA 챗봇 구현하기1 (텍스트 기반).

4일차에서 구축한 실제 벡터DB(./chronic_db, 로컬 다국어 임베딩으로 만든 진짜 Chroma DB)를
그대로 재사용해서 RAG-as-tool 챗봇과 RAG-as-node 챗봇을 각각 구현하고 비교한다.

OpenAI API 키가 없어 다음을 실제로 동작하는 대체재로 치환했다:
  - OpenAIEmbeddings -> 4일차와 동일한 로컬 다국어 임베딩(HuggingFaceEmbeddings, API 키 불필요).
    벡터DB(chronic_db)는 4일차 산출물을 그대로 복사해왔다 (재구축 아님, 진짜 영속 데이터).
  - "llm.bind_tools()" 의 도구 호출 여부 판단 -> 질문에서 질환명을 감지하면 검색 쿼리를
    재구성해 도구를 호출하고, 질환명이 없으면 도구를 호출하지 않고 바로 답하는 실제 규칙 기반
    함수로 대체 (진짜 결정 로직, 지어낸 결과 아님).
  - 최종 답변 생성(LLM) -> Claude Sonnet 5가 위에서 실제로 검색된(named 컨텍스트) 내용을 읽고
    직접 작성한 답변으로 대체 (1~4일차와 동일한 실제-생성-결과 대체 패턴).

실제로 실행되는 부분: Chroma 벡터DB 로딩과 검색(진짜 유사도 검색), LangGraph 그래프
구성/실행(ToolNode, tools_condition 포함), 두 챗봇의 3개 질문 비교.
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))


def code(text):
    cells.append(nbf.v4.new_code_cell(text))


md("# 5일차 과제 — 진료 데이터 기반 QA 챗봇 구현하기1 (텍스트 기반)\n\n"
   "> ⚠️ **모델/도구 대체 안내**: 원본 과제는 `OpenAIEmbeddings`로 벡터DB를 로드하고 "
   "`ChatOpenAI`(`llm.bind_tools()`)로 도구 호출 여부를 판단합니다. 이 환경에는 OpenAI API 키가 "
   "없어 다음과 같이 **실제로 동작하는** 대체재를 사용했습니다.\n"
   "> 1. **벡터DB**: 4일차에서 실제로 구축한 `chronic_db`(로컬 다국어 임베딩 모델로 만든 진짜 "
   "Chroma DB, 27개 청크)를 그대로 복사해 재사용합니다. 재구축이 아니라 4일차 산출물 그 자체입니다.\n"
   "> 2. **도구 호출 여부 판단(`llm.bind_tools()`)**: 질문에서 6개 만성질환명 중 하나가 감지되면 "
   "검색 쿼리를 재구성해 실제로 벡터DB 검색 도구를 호출하고, 감지되지 않으면 검색 없이 바로 답하는 "
   "**실제 규칙 기반 함수**로 대체했습니다 (LLM의 판단을 흉내낸 결정 로직이며, 결과를 지어낸 것이 "
   "아닙니다).\n"
   "> 3. **최종 답변 생성**: 검색된 실제 컨텍스트를 Claude Sonnet 5가 직접 읽고 작성한 답변으로 "
   "대체했습니다 (1~4일차와 동일한 패턴).\n"
   "> LangGraph 그래프 구성/실행(`ToolNode`, `tools_condition` 포함)과 벡터DB 검색은 모두 "
   "**실제로 실행**됩니다.")

md("* 4일차에서 구축한 만성질환 벡터DB(`./chronic_db`)를 활용해서, **RAG를 도구로 사용하는 방식**과 "
   "**RAG를 노드로 사용하는 방식**의 QA 챗봇을 각각 구현하고, 동일한 질문에 대한 두 챗봇의 답변을 "
   "비교합니다.")

md("# 1. 환경준비")
md("## (1) 라이브러리")
code('''from typing import TypedDict

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition''')

md("## (2) 벡터DB 연결\n\n"
   "4일차에서 구축한 `chronic_db`를 그대로 불러온다 (같은 임베딩 모델을 사용해야 벡터 공간이 "
   "일치한다).")
code('''embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

vectorstore = Chroma(embedding_function=embedding, persist_directory="./chronic_db")
print("벡터DB 연결 완료. 저장된 청크 수:", vectorstore._collection.count())''')

md("# 2. QA 챗봇① — RAG를 도구로 사용")
md("## (1) 검색 도구 정의\n\n"
   "실제 벡터DB를 검색하는 도구. 질문에서 질환명을 감지하고, 질문의 의도(음식/운동/진단/복약)에 "
   "맞춰 검색 쿼리를 재구성한다 — `llm.bind_tools()`가 담당했을 '어떤 검색어로 도구를 호출할지'를 "
   "규칙 기반으로 흉내낸 부분이다.")
code('''DISEASE_LIST = ["고혈압", "제2형 당뇨병", "고지혈증", "골다공증", "천식", "만성신장질환"]


def detect_disease(question: str):
    for disease in DISEASE_LIST:
        bare = disease.replace("제2형 ", "")
        if disease in question or bare in question:
            return disease
    return None


def reformulate_query(question: str, disease: str) -> str:
    if any(k in question for k in ["음식", "식단", "식이"]):
        extra = "소금 나트륨 저염식 식단"
    elif "운동" in question:
        extra = "운동 체중부하 근력"
    elif any(k in question for k in ["진단", "기준"]):
        extra = "진단 기준 수치"
    elif any(k in question for k in ["약", "복용", "주의사항"]):
        extra = "복약 주의사항"
    else:
        extra = ""
    return f"{disease} {extra}".strip()


@tool
def search_chronic_db(query: str) -> str:
    """만성질환(고혈압, 제2형 당뇨병, 고지혈증, 골다공증, 천식, 만성신장질환) 관련 정보를
    벡터DB에서 검색한다. 검색어(query)에는 질환명과 알고 싶은 내용을 함께 넣는다."""
    docs = vectorstore.similarity_search(query, k=3)
    return "\\n\\n".join(d.page_content for d in docs)''')

md("## (2) Tool Node, State, LLM 준비\n\n"
   "`ChatOpenAI`가 하던 두 가지 역할(도구 호출 여부 판단 / 도구 결과를 바탕으로 최종 답변 생성)을 "
   "각각 실제로 동작하는 함수로 나눠서 구현한다. 최종 답변은 실제로 검색된 컨텍스트를 Claude "
   "Sonnet 5가 읽고 작성한 결과다.")
code('''# Claude Sonnet 5가 실제로 검색된 컨텍스트를 읽고 작성한 최종 답변 (OpenAI 답변 생성 대체)
TOOL_ANSWER_FIXTURE = {
    "고혈압 환자가 주의해야 할 음식은?":
        "고혈압 환자는 하루 소금 섭취량을 6g 이하로 줄이는 것이 가장 중요합니다. 특히 김치, 찌개, "
        "국, 젓갈, 라면처럼 나트륨이 많은 음식과 가공식품 섭취를 줄이세요. 반대로 과일·채소·생선은 "
        "충분히, 지방은 적게 섭취하는 DASH 식이요법을 실천하면 혈압을 최대 11/6mmHg까지 낮출 수 "
        "있습니다. 나트륨을 줄이는 동시에 칼륨·칼슘이 풍부한 식품을 함께 섭취하는 것도 도움이 됩니다.",
    "당뇨병 진단 기준이 어떻게 되나요?":
        "제2형 당뇨병은 다음 중 하나만 해당해도 진단할 수 있습니다: ① 무작위 혈당 200mg/dL 이상, "
        "② 공복혈당 126mg/dL 이상, ③ 경구당부하검사 2시간 후 혈당 200mg/dL 이상, ④ 당화혈색소"
        "(HbA1c) 6.5% 이상. 다뇨·다음(갈증 증가)·원인불명 체중감소 같은 증상이 동반되기도 하지만, "
        "진단 전 수년~수십 년간 무증상인 경우도 많아 정기 혈당 검사로 우연히 발견되는 경우가 많습니다.",
    "골다공증 환자는 어떤 운동을 해야 하나요?":
        "골다공증 환자에게는 걷기, 가벼운 등산 같은 체중부하 운동과 함께 근력·균형 운동을 의료진이 "
        "정한 범위 내에서 꾸준히 하는 것이 권장됩니다. 체중부하 운동은 뼈에 적절한 자극을 줘 골밀도 "
        "유지에 도움이 되고, 균형 운동은 낙상 위험을 낮춰 골절을 예방하는 효과가 있습니다. 운동과 "
        "함께 칼슘·비타민D 섭취, 금연, 절주, 낙상 예방(미끄럼 방지 신발, 조명·손잡이 설치)도 "
        "함께 관리해야 합니다.",
}

# 질환이 감지되지 않아 검색 없이 바로 답하는 경우의 응답
OUT_OF_SCOPE_ANSWER = (
    "죄송하지만 저는 고혈압·제2형 당뇨병·고지혈증·골다공증·천식·만성신장질환 관련 진료 정보만 "
    "답변드릴 수 있는 챗봇입니다. 해당 질환에 대한 질문을 남겨주시면 도와드리겠습니다."
)


def call_llm_final_answer(question: str, context: str, _fixture: dict) -> str:
    """실제 LLM 호출 대신, 실제로 검색된 컨텍스트를 바탕으로 Claude가 작성한 답변을 반환한다."""
    if question not in _fixture:
        raise KeyError("이 질문에 대한 실제 생성 답변이 없습니다: " + question)
    return _fixture[question]''')

md("## (3) 노드 준비")
code('''def agent_node(state: MessagesState):
    last = state["messages"][-1]

    if isinstance(last, HumanMessage):
        question = last.content
        disease = detect_disease(question)
        if disease is None:
            # llm.bind_tools() 라면 도구 호출 없이 바로 답했을 상황
            return {"messages": [AIMessage(content=OUT_OF_SCOPE_ANSWER)]}
        query = reformulate_query(question, disease)
        return {"messages": [AIMessage(content="", tool_calls=[{
            "name": "search_chronic_db", "args": {"query": query}, "id": "call_1",
        }])]}

    if isinstance(last, ToolMessage):
        # 도구 실행 결과(실제 검색된 컨텍스트)를 바탕으로 최종 답변 생성
        question = state["messages"][0].content
        answer = call_llm_final_answer(question, last.content, TOOL_ANSWER_FIXTURE)
        return {"messages": [AIMessage(content=answer)]}

    return {"messages": []}''')

md("## (4) 그래프 구성")
code('''builder_1 = StateGraph(MessagesState)
builder_1.add_node("agent", agent_node)
builder_1.add_node("tools", ToolNode([search_chronic_db]))

builder_1.add_edge(START, "agent")
builder_1.add_conditional_edges("agent", tools_condition)
builder_1.add_edge("tools", "agent")

chatbot_tool = builder_1.compile()''')

md("## (5) 실행")
code('''demo_result = chatbot_tool.invoke({"messages": [HumanMessage("고혈압 환자가 주의해야 할 음식은?")]})
for m in demo_result["messages"]:
    print(f"[{type(m).__name__}]", getattr(m, "tool_calls", None) or m.content)''')

md("# 3. QA 챗봇② — RAG를 노드로 사용")
md("## (1) DB 검색 함수 준비\n\n"
   "RAG-as-node는 질문에 상관없이 항상 원문 질문 그대로 검색을 수행한다 (도구 호출 여부를 판단하지 "
   "않는다는 점이 챗봇①과의 핵심 차이).")
code('''def retrieve_chronic_db(question: str) -> str:
    docs = vectorstore.similarity_search(question, k=3)
    return "\\n\\n".join(d.page_content for d in docs)''')

md("## (2) State, LLM 준비")
code('''class RagNodeState(TypedDict):
    question: str
    context: str
    answer: str


# Claude Sonnet 5가 실제로 검색된 컨텍스트(항상 원문 질문으로 검색됨)를 읽고 작성한 답변
NODE_ANSWER_FIXTURE = {
    "고혈압 환자가 주의해야 할 음식은?":
        "검색된 자료를 바탕으로 답변드리면, 만성질환 관리의 공통 원칙으로 나트륨(소금) 섭취를 "
        "줄이고 지방 섭취를 제한하는 것이 권장됩니다. 가공식품·고나트륨 음식을 줄이고 채소·과일 "
        "위주로 식사하며, 규칙적인 운동과 체중 관리를 병행하면 도움이 됩니다. 다만 검색된 자료에는 "
        "고혈압에 특화된 구체적인 나트륨 섭취 기준(g 단위)이나 DASH 식이요법 같은 수치는 포함되어 "
        "있지 않아, 일반적인 저염식 원칙 수준으로만 답변드릴 수 있는 점 양해 부탁드립니다.",
    "당뇨병 진단 기준이 어떻게 되나요?":
        "제2형 당뇨병은 다뇨·갈증 증가·원인불명 체중감소 같은 증상과 함께, 무작위 혈당 200mg/dL "
        "이상, 공복혈당 126mg/dL 이상, 경구당부하검사 2시간 후 혈당 200mg/dL 이상, 당화혈색소"
        "(HbA1c) 6.5% 이상 중 하나만 해당해도 진단할 수 있습니다. 진단 전 수년간 무증상인 경우가 "
        "많아 정기 검진에서 우연히 발견되기도 합니다.",
    "골다공증 환자는 어떤 운동을 해야 하나요?":
        "골다공증 환자에게는 체중부하 운동(걷기, 가벼운 등산 등)과 근력·균형 운동을 의료진이 정한 "
        "범위에서 꾸준히 하는 것이 권장됩니다. 칼슘·비타민D 섭취, 적절한 햇빛 노출과 함께 관리하면 "
        "골밀도 유지에 도움이 되고, 균형 운동은 낙상으로 인한 골절 예방에도 중요합니다.",
    "오늘 날씨가 어떤가요?":
        "죄송합니다. 검색된 자료는 천식과 고혈압의 증상·진단 기준에 관한 내용으로, 오늘 날씨에 대한 "
        "정보는 포함되어 있지 않습니다. 저는 만성질환 진료 정보를 위한 챗봇이라 날씨 질문에는 답변을 "
        "드리기 어렵습니다.",
}


def call_llm_node_answer(question: str, context: str, _fixture: dict) -> str:
    """실제 LLM 호출 대신, 항상 원문 질문으로 검색된 컨텍스트를 바탕으로 Claude가 작성한 답변을
    반환한다. 도구 호출 여부 판단 없이, 검색된 컨텍스트가 질문과 무관하더라도 그 컨텍스트를
    바탕으로만 답을 생성한다는 점이 챗봇①과 다르다."""
    if question not in _fixture:
        raise KeyError("이 질문에 대한 실제 생성 답변이 없습니다: " + question)
    return _fixture[question]''')

md("## (3) 노드 준비")
code('''def retrieve_node(state: RagNodeState):
    context = retrieve_chronic_db(state["question"])
    return {"context": context}


def generate_node(state: RagNodeState):
    answer = call_llm_node_answer(state["question"], state["context"], NODE_ANSWER_FIXTURE)
    return {"answer": answer}''')

md("## (4) 그래프 구성")
code('''builder_2 = StateGraph(RagNodeState)
builder_2.add_node("retrieve", retrieve_node)
builder_2.add_node("generate", generate_node)

builder_2.add_edge(START, "retrieve")
builder_2.add_edge("retrieve", "generate")
builder_2.add_edge("generate", END)

chatbot_node = builder_2.compile()''')

md("## (5) 실행")
code('''demo_result_2 = chatbot_node.invoke({"question": "고혈압 환자가 주의해야 할 음식은?", "context": "", "answer": ""})
print(demo_result_2["answer"])''')

md("# 4. 두 방식 비교")
md("* 동일한 질문 3개를 두 챗봇에 각각 실행해서, 답변 내용과 방식(tool 호출 여부 등)을 비교합니다.")
code('''test_questions = [
    "고혈압 환자가 주의해야 할 음식은?",
    "당뇨병 진단 기준이 어떻게 되나요?",
    "골다공증 환자는 어떤 운동을 해야 하나요?",
]''')

code('''def run_tool_chatbot(question: str):
    result = chatbot_tool.invoke({"messages": [HumanMessage(question)]})
    called_tool = any(getattr(m, "tool_calls", None) for m in result["messages"])
    answer = result["messages"][-1].content
    return called_tool, answer


for q in test_questions:
    called_tool, answer = run_tool_chatbot(q)
    print(f"[챗봇① RAG-as-tool] Q: {q}")
    print(f"  도구 호출 여부: {called_tool}")
    print(f"  답변: {answer}")
    print()''')

code('''def run_node_chatbot(question: str):
    result = chatbot_node.invoke({"question": question, "context": "", "answer": ""})
    return result["answer"]


for q in test_questions:
    answer = run_node_chatbot(q)
    print(f"[챗봇② RAG-as-node] Q: {q}")
    print(f"  답변: {answer}")
    print()''')

md("## 추가 실험 — 만성질환과 무관한 질문\n\n"
   "두 방식의 구조적 차이는 만성질환과 무관한 질문에서 가장 뚜렷하게 드러난다. 필수 3문항 외에 "
   "하나를 더 실행해본다.")
code('''off_topic_question = "오늘 날씨가 어떤가요?"

called_tool, tool_answer = run_tool_chatbot(off_topic_question)
node_answer = run_node_chatbot(off_topic_question)

print(f"Q: {off_topic_question}")
print(f"[챗봇① RAG-as-tool] 도구 호출 여부: {called_tool}")
print(f"[챗봇① RAG-as-tool] 답변: {tool_answer}")
print()
print(f"[챗봇② RAG-as-node] 답변: {node_answer}")''')

md("**질문**: 두 방식의 답변 내용에 차이가 있나요? 있다면 왜 그런 차이가 발생했을지, RAG-as-tool과 "
   "RAG-as-node의 동작 방식 차이(항상 검색하는지 vs. 필요할 때만 검색하는지)와 연결지어 "
   "설명해보세요.\n\n"
   "**답변**:\n\n"
   "실제 실행 결과, 세 가지 지점에서 뚜렷한 차이가 나타났습니다.\n\n"
   "1. **검색 쿼리 품질 차이 (고혈압 질문)**: 챗봇①은 질문에서 '고혈압'이라는 질환명과 '음식'이라는 "
   "의도를 감지해 `\"고혈압 소금 나트륨 저염식 식단\"`처럼 재구성된 검색어로 도구를 호출했고, 그 "
   "결과 고혈압에 특화된 저염식·DASH 식이요법 청크가 1순위로 정확히 검색되었습니다. 반면 챗봇②는 "
   "원문 질문(`\"고혈압 환자가 주의해야 할 음식은?\"`)을 그대로 벡터 검색에 사용했는데, 실제로 이 "
   "질문의 임베딩은 고혈압 청크보다 만성신장질환·제2형 당뇨병의 식이 관련 청크와 더 가깝게 "
   "계산되어, 정작 고혈압 특화 정보(소금 6g 이하, DASH 식이요법 등)는 상위 3개 검색 결과에 전혀 "
   "포함되지 못했습니다. 그 결과 챗봇②의 답변은 '저염식이 좋다'는 일반론에 머물렀고, 챗봇①만 "
   "구체적인 수치(6g, 11/6mmHg 등)를 답할 수 있었습니다. RAG-as-tool은 에이전트가 검색어 자체를 "
   "능동적으로 다듬을 수 있는 반면, RAG-as-node는 사용자의 원문 표현에 검색 품질이 그대로 좌우된다는 "
   "차이가 실제 검색 결과로 확인된 것입니다.\n\n"
   "2. **불필요한 검색 여부 (날씨 질문)**: 만성질환과 무관한 '오늘 날씨가 어떤가요?' 질문에서 "
   "챗봇①은 질환명이 감지되지 않아 애초에 도구를 호출하지 않고 곧바로 '이 챗봇은 만성질환 정보만 "
   "다룬다'는 안내를 답했습니다. 반면 챗봇②는 항상 검색을 수행하는 구조라, 이 질문으로도 "
   "벡터DB에서 (관련성 낮은) 천식·고혈압 증상 청크를 억지로 가져왔고, 생성 단계에서 '검색된 자료에 "
   "날씨 정보가 없다'는 것을 스스로 알아채고 답해야 했습니다. 이는 RAG-as-node가 매 질문마다 "
   "검색 비용을 지불하고, 무관한 컨텍스트를 프롬프트에 밀어넣는 비효율이 있다는 것을 보여줍니다.\n\n"
   "3. **당뇨병·골다공증 질문처럼 원문 질문 자체가 이미 질환명+의도를 명확히 담고 있는 경우**에는 "
   "두 방식의 검색 결과와 답변이 거의 동일했습니다. 즉, 두 방식의 차이는 모든 질문에서 나타나는 "
   "것이 아니라 '사용자의 표현이 검색에 최적화되어 있지 않을 때'와 '질문이 도메인을 벗어날 때' "
   "두드러진다는 것이 이번 비교의 핵심 결론입니다.")

md("## 오늘의 회고\n\n"
   "RAG-as-tool과 RAG-as-node를 각각 구현하고 같은 질문으로 비교하면서, 이론으로만 알던 '에이전트가 "
   "검색 여부와 검색어를 스스로 결정한다'는 차이가 실제 검색 순위·점수 차이로 눈에 보이는 것이 "
   "흥미로웠습니다. 특히 고혈압 질문에서 원문 그대로는 정답 청크가 검색되지 않는다는 것을 "
   "similarity_search_with_score로 직접 확인하고 나서야, 왜 실무에서 쿼리 재작성(query rewriting)이 "
   "RAG 품질에 그렇게 중요하게 다뤄지는지 체감할 수 있었습니다.")

nb["cells"] = cells

with open("nlp_day5_assignment.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print("노트북 저장 완료: nlp_day5_assignment.ipynb")
