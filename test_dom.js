const fs = require("fs");
const path = require("path");
const { JSDOM } = require("jsdom");

(async () => {
  const html = fs.readFileSync(path.join(__dirname, "index.html"), "utf8");
  const dom = new JSDOM(html, {
    url: "http://localhost/",
    runScripts: "outside-only",
    resources: "usable",
    pretendToBeVisual: true,
  });
  const { window } = dom;

  // stub Chart.js since jsdom has no canvas backend
  window.Chart = function Chart() {
    return { destroy() {} };
  };
  window.fetch = async () => {
    throw new Error("network disabled in test");
  };

  const files = ["js/indicators.js", "js/data.js", "js/model.js", "js/terminal.js", "js/app.js"];
  const combined = files.map((f) => fs.readFileSync(path.join(__dirname, f), "utf8")).join("\n;\n");
  window.eval(combined);

  // fire DOMContentLoaded to run boot()
  dom.window.document.dispatchEvent(new dom.window.Event("DOMContentLoaded"));

  // give init a tick
  await new Promise((r) => setTimeout(r, 50));

  const doc = window.document;
  console.log("watchlist value sample:", doc.getElementById("watchlistInput").value.slice(0, 40));

  // trigger a scan (simulated mode, default settings)
  doc.getElementById("runScanBtn").click();

  // wait for the async scan to finish (poll button state)
  let waited = 0;
  while (doc.getElementById("runScanBtn").disabled && waited < 20000) {
    await new Promise((r) => setTimeout(r, 100));
    waited += 100;
  }

  const meta = doc.getElementById("resultsMeta").textContent;
  const rows = doc.querySelectorAll("table.results tbody tr");
  console.log("resultsMeta:", meta);
  console.log("result rows rendered:", rows.length);

  if (rows.length === 0) {
    console.error("FAIL: no result rows rendered");
    process.exit(1);
  }

  // click the first row and check the detail panel populated
  rows[0].dispatchEvent(new window.Event("click", { bubbles: true }));
  await new Promise((r) => setTimeout(r, 50));

  const detailTicker = doc.getElementById("detailTicker").textContent;
  const detailPrice = doc.getElementById("detailPrice").textContent;
  const similar = doc.getElementById("similarList").innerHTML;
  console.log("detailTicker:", detailTicker);
  console.log("detailPrice:", detailPrice);
  console.log("similar list populated:", similar.includes("similar-chip"));

  // terminal lines stream in with small delays for the "live" effect -
  // give the queue time to fully drain before checking
  await new Promise((r) => setTimeout(r, 4000));
  const termLines = doc.querySelectorAll("#terminalBody .term-line").length;
  console.log("terminal lines logged (after drain):", termLines);

  if (!detailTicker.includes("·") || !detailPrice.startsWith("$")) {
    console.error("FAIL: detail panel did not populate correctly");
    process.exit(1);
  }
  if (termLines < 10) {
    console.error("FAIL: terminal did not log expected volume of activity");
    process.exit(1);
  }

  console.log("\nALL CHECKS PASSED");
})().catch((err) => {
  console.error("TEST ERROR:", err);
  process.exit(1);
});
