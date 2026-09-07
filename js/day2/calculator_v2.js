const OPERATORS = {
    "+": (a, b) => a + b,
    "-": (a, b) => a - b,
    "*": (a, b) => a * b,
    "/": (a, b) => a / b,
};
const HIGH_PRIORITY = ["*", "/"];

const display = document.getElementById("display");
const powerBtn = document.getElementById("powerBtn");

const state = {
    poweredOn: true,
    formula: "",
    justCalculated: false,
};

function reduceByPriority(tokens, operators) {
    const result = [tokens[0]];
    for (let i = 1; i < tokens.length; i += 2) {
        const operator = tokens[i];
        const nextValue = tokens[i + 1];

        if (operators.includes(operator)) {
            const left = Number(result.pop());
            const right = Number(nextValue);
            if (operator === "/" && right === 0) {
                throw new Error("DivBy0");
            }
            result.push(OPERATORS[operator](left, right));
        } else {
            result.push(operator, nextValue);
        }
    }
    return result;
}

function evaluateFormula(formula) {
    const tokens = formula.trim().split(/\s+/);
    if (tokens.length < 3 || tokens.length % 2 === 0) {
        throw new Error("Error");
    }
    const afterHighPriority = reduceByPriority(tokens, HIGH_PRIORITY);
    const [finalResult] = reduceByPriority(afterHighPriority, ["+", "-"]);
    if (Number.isNaN(finalResult)) throw new Error("Error");
    return finalResult;
}

function render() {
    display.value = state.formula === "" ? "0" : state.formula;
}

function togglePower() {
    state.poweredOn = !state.poweredOn;
    powerBtn.classList.toggle("on", state.poweredOn);

    const otherButtons = document.querySelectorAll("button:not(.power)");
    otherButtons.forEach((btn) => (btn.disabled = !state.poweredOn));

    if (state.poweredOn) {
        display.style.backgroundColor = "#222";
        clearAll();
    } else {
        display.value = "";
        display.style.backgroundColor = "#111";
        state.formula = "";
        state.justCalculated = false;
    }
}

function pushNumber(digit) {
    if (!state.poweredOn) return;

    if (state.justCalculated) {
        state.formula = "";
        state.justCalculated = false;
    }

    if (state.formula === "Error" || state.formula === "DivBy0") {
        state.formula = "";
    }

    state.formula += digit;
    render();
}

function pushOperator(operator) {
    if (!state.poweredOn) return;
    if (state.formula === "Error" || state.formula === "DivBy0") return;

    state.justCalculated = false;

    if (state.formula === "") {
        state.formula = "0";
    }

    if (state.formula.endsWith(" ")) {
        state.formula = state.formula.slice(0, -3);
    }

    state.formula += ` ${operator} `;
    render();
}

function clearAll() {
    if (!state.poweredOn) return;
    state.formula = "";
    state.justCalculated = false;
    render();
}

function submitFormula() {
    if (!state.poweredOn || !state.formula) return;

    let formulaToEvaluate = state.formula;
    if (formulaToEvaluate.endsWith(" ")) {
        formulaToEvaluate = formulaToEvaluate.trim();
    }

    try {
        const result = evaluateFormula(formulaToEvaluate);
        state.formula = String(result);
    } catch (err) {
        state.formula = err.message === "DivBy0" ? "DivBy0" : "Error";
    }

    state.justCalculated = true;
    render();
}

window.onload = () => {
    powerBtn.classList.add("on");
    render();
};
