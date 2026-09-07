const OPERATORS = {
    "+": (a, b) => a + b,
    "-": (a, b) => a - b,
    "*": (a, b) => a * b,
    "/": (a, b) => a / b,
};

const HIGH_PRIORITY = ["*", "/"];

function tokenize(formula) {
    return formula.trim().split(/\s+/);
}

function isValidFormula(tokens) {
    if (tokens.length < 3 || tokens.length % 2 === 0) return false;
    return tokens.every((token, i) => {
        const isOperatorSlot = i % 2 === 1;
        if (isOperatorSlot) return token in OPERATORS;
        return !Number.isNaN(Number(token));
    });
}

function reduceByPriority(tokens, operators) {
    const result = [tokens[0]];
    for (let i = 1; i < tokens.length; i += 2) {
        const operator = tokens[i];
        const nextValue = tokens[i + 1];

        if (operators.includes(operator)) {
            const left = Number(result.pop());
            const right = Number(nextValue);
            if (operator === "/" && right === 0) {
                throw new Error("0으로 나눌 수 없습니다.");
            }
            result.push(OPERATORS[operator](left, right));
        } else {
            result.push(operator, nextValue);
        }
    }
    return result;
}

function calculate(formula) {
    const tokens = tokenize(formula);
    if (!isValidFormula(tokens)) {
        throw new Error("계산식 형식이 올바르지 않습니다. 예: 1 + 1 * 4");
    }

    const afterHighPriority = reduceByPriority(tokens, HIGH_PRIORITY);
    const [finalResult] = reduceByPriority(afterHighPriority, ["+", "-"]);
    return finalResult;
}

function inputFormula() {
    return prompt("계산식을 입력하세요. (예: 1 + 1 * 4)");
}

function runOnce(formula) {
    const input = formula || inputFormula();
    if (!input) {
        console.log("계산식이 입력되지 않아 종료합니다.");
        return false;
    }

    try {
        console.log(`결과: ${calculate(input)}`);
    } catch (err) {
        console.log(`에러 발생: ${err.message}`);
    }
    return true;
}

function start(formula) {
    if (!runOnce(formula)) return;

    const again = confirm("계속 계산하시겠습니까?");
    if (again) {
        start();
    } else {
        console.log("계산기를 종료합니다.");
    }
}
