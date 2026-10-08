const bases = Array.from({ length: 63 }, (_, index) => index + 2);
const commonBases = [2, 8, 10, 16, 32, 60, 64];
const sourceSelect = document.querySelector("#source-base");
const targetSelect = document.querySelector("#target-base");
const convertForm = document.querySelector("#convert-form");
const scienceForm = document.querySelector("#science-form");
const convertNote = document.querySelector("#convert-note");
const operationSelect = document.querySelector("#operation-select");
const paramsInput = document.querySelector("#params-input");

for (const select of [sourceSelect, targetSelect]) {
  for (const base of bases) {
    const option = document.createElement("option");
    option.value = String(base);
    option.textContent = `BASE ${base}`;
    select.append(option);
  }
}
sourceSelect.value = "2";
targetSelect.value = "10";
for (const select of [document.querySelector("#expression-base"), document.querySelector("#expression-output-base")]) {
  for (const base of bases) {
    const option = document.createElement("option");
    option.value = String(base);
    option.textContent = `BASE ${base}`;
    select.append(option);
  }
  select.value = "10";
}

document.querySelector("#base-chips").innerHTML = commonBases.map((base) => `<span class="base-chip">${base}</span>`).join("");

async function postJson(path, payload) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await response.json();
  if (!response.ok) throw new Error([data.error, data.hint].filter(Boolean).join(" ") || "Ошибка запроса.");
  return data.result;
}

convertForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const error = document.querySelector("#convert-error");
  const output = document.querySelector("#converted-value");
  const badge = document.querySelector("#result-badge");
  const copy = document.querySelector("#copy-result");
  error.textContent = "";
  badge.textContent = "СЧИТАЮ…";
  try {
    const result = await postJson("/api/convert", {
      value: document.querySelector("#number-input").value,
      source: Number(sourceSelect.value),
      target: Number(targetSelect.value),
    });
    output.textContent = result.value;
    badge.textContent = result.exact ? "ТОЧНО" : "48 ЗНАКОВ";
    convertNote.textContent = result.exact
      ? `Точное значение: ${result.fraction}`
      : `Дробь ${result.fraction}; показаны первые 48 знаков, многоточие означает продолжение.`;
    copy.disabled = false;
  } catch (exception) {
    output.textContent = "—";
    badge.textContent = "ПРОВЕРЬТЕ ВВОД";
    convertNote.textContent = "";
    copy.disabled = true;
    error.textContent = exception.message;
  }
});

document.querySelector("#swap-bases").addEventListener("click", () => {
  [sourceSelect.value, targetSelect.value] = [targetSelect.value, sourceSelect.value];
  convertForm.requestSubmit();
});

document.querySelector("#copy-result").addEventListener("click", async (event) => {
  const output = document.querySelector("#converted-value").textContent;
  try {
    await navigator.clipboard.writeText(output.replace(/…$/, ""));
    event.currentTarget.textContent = "✓";
    window.setTimeout(() => { event.currentTarget.textContent = "⧉"; }, 1200);
  } catch {
    document.querySelector("#convert-error").textContent = "Буфер обмена недоступен в этом браузере.";
  }
});

scienceForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const error = document.querySelector("#science-error");
  error.textContent = "";
  document.querySelector("#exact-result").textContent = "СЧИТАЮ…";
  try {
    const params = JSON.parse(paramsInput.value || "{}");
    const result = await postJson("/api/compute", {
      expression: document.querySelector("#expression-input").value,
      operation: operationSelect.value,
      base_in: Number(document.querySelector("#expression-base").value),
      base_out: Number(document.querySelector("#expression-output-base").value),
      params,
    });
    document.querySelector("#exact-result").textContent = result.value_decimal ?? "Расчёт выполнен";
    const detail = result.data ? JSON.stringify(summarize(result.data)) : result.latex;
    const warnings = result.warnings?.length ? ` · ${result.warnings.join(" ")}` : "";
    const outputBase = result.value_base_out ? ` · base-${document.querySelector("#expression-output-base").value}: ${result.value_base_out}` : "";
    document.querySelector("#approx-result").textContent = `${result.method || "символьный результат"}${outputBase}${detail ? ` · ${detail}` : ""}${warnings}`;
  } catch (exception) {
    document.querySelector("#exact-result").textContent = "—";
    document.querySelector("#approx-result").textContent = "—";
    error.textContent = exception.message;
  }
});

document.querySelectorAll(".mode-tab").forEach((button) => {
  button.addEventListener("click", () => {
    document.querySelectorAll(".mode-tab").forEach((tab) => tab.classList.toggle("active", tab === button));
    document.querySelectorAll(".tool-panel").forEach((panel) => panel.classList.toggle("active", panel.id === button.dataset.panel));
  });
});

document.querySelectorAll("[data-expression]").forEach((button) => {
  button.addEventListener("click", () => {
    document.querySelector("#expression-input").value = button.dataset.expression;
    if (button.dataset.operation) operationSelect.value = button.dataset.operation;
    if (button.dataset.params) paramsInput.value = button.dataset.params;
    scienceForm.requestSubmit();
  });
});

function summarize(value) {
  if (Array.isArray(value)) return value.slice(0, 8).map(summarize).concat(value.length > 8 ? [`… ${value.length - 8} more`] : []);
  if (value && typeof value === "object") {
    if (Array.isArray(value.energies) && Array.isArray(value.x)) {
      return { energies: value.energies, domain: [value.x[0], value.x.at(-1)], grid_points: value.x.length,
        maximum_residual: Math.max(...value.residuals), units: value.units, method: value.method };
    }
    if (Array.isArray(value.xi) && Array.isArray(value.theta)) {
      return { polytrope_index: value.n, first_zero_xi: value.xi1, omega_n: value.omega_n,
        integration_points: value.xi.length, success: value.success };
    }
    return Object.fromEntries(Object.entries(value).map(([key, item]) => [key, summarize(item)]));
  }
  return value;
}

document.querySelectorAll("[data-insert]").forEach((button) => {
  button.addEventListener("click", () => {
    const input = document.querySelector("#expression-input");
    const insertion = button.dataset.insert;
    const cursor = input.selectionStart;
    const placeholder = insertion.indexOf(",");
    const text = placeholder < 0 ? insertion : insertion.replace(",", "");
    input.setRangeText(text, cursor, input.selectionEnd, "end");
    if (placeholder >= 0) input.setSelectionRange(cursor + placeholder, cursor + placeholder);
    input.focus();
  });
});

document.querySelector("#number-input").addEventListener("keydown", (event) => {
  if (event.key === "Enter") convertForm.requestSubmit();
});

convertForm.requestSubmit();
scienceForm.requestSubmit();
