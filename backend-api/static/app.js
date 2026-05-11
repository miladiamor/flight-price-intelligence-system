// static/app.js
console.log("✅ app.js loaded");

function el(id) {
  const node = document.getElementById(id);
  if (!node) console.error("❌ Missing element id:", id);
  return node;
}

async function postJson(url, data) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });

  const text = await res.text();
  let json;
  try {
    json = JSON.parse(text);
  } catch {
    throw new Error("Server returned non-JSON: " + text.slice(0, 120));
  }

  if (!res.ok) throw new Error(json.error || `HTTP ${res.status}`);
  return json;
}

function fmt2(x) {
  if (x === null || x === undefined || Number.isNaN(Number(x))) return "—";
  return Number(x).toFixed(2);
}

let lastPredictPayload = null; // used to refresh after feedback

function renderCurve(curveData) {
  const box = el("trendBox");
  const points = curveData?.prices || [];
  const summary = curveData?.summary || null;

  if (!points.length) {
    box.innerHTML = `<div class="muted">No curve data.</div>`;
    return;
  }

  const bestWait = summary?.best_wait_days ?? null;

  const rows = points
    .map((p) => {
      const isBest = bestWait !== null && p.wait_days === bestWait;
      return `
        <tr class="${isBest ? "best-row" : ""}">
          <td>${p.wait_days}</td>
          <td>${p.days_left_after_wait}</td>
          <td><b>${fmt2(p.predicted_price)} EUR</b></td>
        </tr>
      `;
    })
    .join("");

  const headline = summary
    ? `Best: wait <b>${summary.best_wait_days}</b> day(s) → <b>${fmt2(summary.best_price)} EUR</b>
       <span class="muted">(now: ${fmt2(summary.price_now)} EUR, savings: ${fmt2(summary.savings_vs_now)} EUR, trend: ${summary.trend_direction})</span>`
    : "";

  box.innerHTML = `
    <div style="margin-bottom:8px">${headline}</div>
    <table>
      <thead>
        <tr><th>Wait days</th><th>Days left</th><th>Predicted price</th></tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>
  `;
}

function badgeClassForDecision(decision) {
  if (decision === "BUY_NOW") return "buy";
  if (decision === "WAIT") return "wait";
  if (decision === "SKIPPED") return "skip";
  return "neutral";
}

function renderResults(data, curveData) {
  el("resultsCard").style.display = "block";

  // Model 1
  const m1 = data.model1;
  el("priceBox").innerHTML = `
    <div><b>${fmt2(m1.predicted_price)} ${m1.currency}</b></div>
    <div class="muted">Days left: ${m1.days_left}</div>
  `;

  // Model 2
  const m2 = data.model2;
  if (m2.skipped) {
    el("advisorBox").innerHTML = `
      <div class="badge skip">SKIPPED</div>
      <div style="margin-top:10px" class="muted">${m2.reason}</div>
    `;
  } else {
    const decision = m2.decision || "NO_CLEAR_SIGNAL";
    const badgeClass = badgeClassForDecision(decision);

    el("advisorBox").innerHTML = `
      <div class="badge ${badgeClass}">${decision}</div>
      <div style="margin-top:10px">
        <div>Now: <b>${fmt2(m2.price_now)} EUR</b></div>
        <div>Best (min): <b>${fmt2(m2.best_price)} EUR</b></div>
        <div>Best wait: <b>${m2.best_wait_days}</b> day(s)</div>
        <div>Expected savings: <b>${fmt2(m2.expected_savings)} EUR</b></div>
        <div class="muted" style="margin-top:6px">${m2.reason}</div>
      </div>
    `;
  }

  // Curve
  renderCurve(curveData);

  // Model 3 + Feedback buttons
  const recs = data.model3?.recommendations || [];

  const rows = recs
    .map((r, i) => {
      const why = (r.why_tags || []).join(", ");
      return `
        <tr>
          <td>${i + 1}</td>
          <td><b>${r.destination}</b></td>
          <td>${fmt2(r.score)}</td>
          <td>${why}</td>
          <td style="white-space:nowrap;">
            <button class="feedback-btn" data-dest="${r.destination}" data-val="1">👍</button>
            <button class="feedback-btn" data-dest="${r.destination}" data-val="-1">👎</button>
          </td>
        </tr>
      `;
    })
    .join("");

  el("recsBox").innerHTML = `
    <table>
      <thead>
        <tr>
          <th>#</th><th>Destination</th><th>Score</th><th>Why</th><th>Feedback</th>
        </tr>
      </thead>
      <tbody>
        ${rows || `<tr><td colspan="5" class="muted">No recommendations.</td></tr>`}
      </tbody>
    </table>
  `;

  // Attach click handlers after rendering
  document.querySelectorAll(".feedback-btn").forEach((btn) => {
    btn.addEventListener("click", async (ev) => {
      const dest = ev.currentTarget.getAttribute("data-dest");
      const val = parseInt(ev.currentTarget.getAttribute("data-val"), 10);
      await sendFeedback(dest, val);
    });
  });

  // Debug JSON
  el("debugJson").textContent = JSON.stringify({ ...data, model1_curve: curveData }, null, 2);
}

async function sendFeedback(destination, value) {
  try {
    el("status").textContent = "Saving feedback...";

    const payload = {
      user_id: el("user_id").value || "1",
      destination,
      value,
    };

    await postJson("/feedback", payload);

    // Refresh predictions + curve using last payload
    if (lastPredictPayload) {
      el("status").textContent = "Refreshing recommendations...";
      const data = await postJson("/predict", lastPredictPayload);

      const curve = await postJson("/model1_curve", {
        origin: lastPredictPayload.origin,
        destination: lastPredictPayload.destination,
        date: lastPredictPayload.date,
      });

      renderResults(data, curve);
      el("status").textContent = "Updated ✅";
    } else {
      el("status").textContent = "Feedback saved ✅";
    }
  } catch (e) {
    console.error(e);
    el("status").textContent = "Feedback error: " + e.message;
  }
}

function init() {
  console.log("✅ init()");
  const btn = el("btnPredict");
  const btnDebug = el("btnToggleDebug");
  if (!btn) return;

  let debugVisible = false;

  btn.addEventListener("click", async () => {
    el("status").textContent = "Running models...";

    try {
      const payload = {
        user_id: el("user_id").value || "1",
        origin: el("origin").value.trim(),
        destination: el("destination").value.trim(),
        date: el("date").value.trim(),
        k: 5,
      };

      lastPredictPayload = payload;

      const data = await postJson("/predict", payload);

      const curve = await postJson("/model1_curve", {
        origin: payload.origin,
        destination: payload.destination,
        date: payload.date,
      });

      renderResults(data, curve);
      el("status").textContent = "Done ✅";
    } catch (e) {
      console.error(e);
      el("status").textContent = "Error: " + e.message;
    }
  });

  if (btnDebug) {
    btnDebug.addEventListener("click", () => {
      debugVisible = !debugVisible;
      el("debugCard").style.display = debugVisible ? "block" : "none";
    });
  }
}

document.addEventListener("DOMContentLoaded", init);
