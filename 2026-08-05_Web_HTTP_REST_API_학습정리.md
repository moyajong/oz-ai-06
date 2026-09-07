# 웹 서비스 기초와 구조 1일차 학습 정리

**날짜:** 2026-08-05

---

## 1. 웹 서비스는 어떻게 동작하는가

웹 서비스는 기본적으로 **클라이언트(Client) - 서버(Server)** 구조로 동작한다.

- **클라이언트**: 브라우저, 모바일 앱 등 사용자가 직접 사용하는 쪽. 서버에 **요청(Request)**을 보낸다.
- **서버**: 요청을 받아 처리하고 **응답(Response)**을 돌려주는 쪽.

```
클라이언트 ---- Request(요청) ----> 서버
클라이언트 <---- Response(응답) ---- 서버
```

브라우저 주소창에 URL을 입력하는 순간부터 이 흐름이 시작된다.

## 2. HTTP: 요청과 응답의 규칙

HTTP(HyperText Transfer Protocol)는 클라이언트와 서버가 데이터를 주고받을 때 따르는 **약속(프로토콜)**이다.

### 요청(Request)의 구성
- **Method**: 무엇을 하고 싶은지 (`GET`, `POST`, `PUT`, `PATCH`, `DELETE`)
- **URL**: 어떤 자원(resource)에 대한 요청인지
- **Header**: 부가 정보 (인증 토큰, 컨텐츠 타입 등)
- **Body**: 실제로 보낼 데이터 (GET에는 보통 없고, POST/PUT에 주로 있음)

### 응답(Response)의 구성
- **Status Code**: 요청이 어떻게 처리됐는지 (아래 참고)
- **Header**: 응답에 대한 부가 정보
- **Body**: 실제로 돌려주는 데이터 (HTML, JSON 등)

### 주요 상태 코드
| 코드 | 의미 |
|---|---|
| 200 | 성공 |
| 201 | 생성 성공 (Created) |
| 400 | 잘못된 요청 (Bad Request) |
| 401 | 인증 필요 (Unauthorized) |
| 403 | 권한 없음 (Forbidden) |
| 404 | 자원 없음 (Not Found) |
| 500 | 서버 내부 오류 (Internal Server Error) |

## 3. REST API 설계 원칙

REST(REpresentational State Transfer)는 웹 자원을 다루는 방식에 대한 설계 스타일이다.
핵심은 **"자원은 URL로, 행위는 HTTP Method로"** 표현하는 것.

| 행위 | Method | 예시 URL |
|---|---|---|
| 목록 조회 | GET | `/patients` |
| 단건 조회 | GET | `/patients/1` |
| 생성 | POST | `/patients` |
| 전체 수정 | PUT | `/patients/1` |
| 부분 수정 | PATCH | `/patients/1` |
| 삭제 | DELETE | `/patients/1` |

URL에는 동사(`/getPatients` 같은)를 넣지 않고, **명사(자원)** 중심으로 설계하는 것이 REST의 핵심 규칙이다.

## 4. MVC 구조

서버 내부에서 요청을 처리할 때 역할을 나누는 대표적인 패턴이 MVC다.

- **Model**: 데이터와 비즈니스 로직 (DB 테이블/ORM 모델 등)
- **View**: 사용자에게 보여지는 화면 (템플릿, JSON 응답 등)
- **Controller**: 요청을 받아 Model을 조작하고 View에 전달하는 흐름 제어

FastAPI 기준으로 보면 `router(=Controller)` → `service/CRUD(=Model 조작)` → `response schema(=View)` 흐름과 대응된다.

## 5. 오늘 배운 것 요약

- 지금까지 막연하게 쓰던 GET/POST가 "무엇을 하려는 의도"를 나타내는 약속이라는 걸 명확히 이해했다.
- URL 설계할 때 동사를 쓰지 않고 자원(명사) 중심으로 설계해야 하는 이유(일관성, 예측 가능성)를 알게 됐다.
- MVC의 각 역할이 실제 FastAPI 프로젝트의 폴더 구조(`apis`/`services`/`repositories`)와 어떻게 매칭되는지 감이 잡혔다.
