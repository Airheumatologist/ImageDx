/* Disease page: tabs, filters, grouped panel cards, key findings. */
const state = { tab: null, tabs: [], panels: [], vocab: [] };

const $ = (id) => document.getElementById(id);
const q = (sel) => document.querySelector(sel);

const lb = $("lightbox");
lb.addEventListener("click", () => { lb.hidden = true; });
function openLightbox(src, caption) {
  $("lb-img").src = src;
  $("lb-caption").textContent = caption;
  lb.hidden = false;
}

function filters() {
  const p = new URLSearchParams();
  for (const [k, id] of [["subtype", "f-subtype"], ["modality", "f-modality"],
    ["finding", "f-finding"], ["typicality", "f-typicality"], ["skin_tone", "f-skin-tone"]]) {
    const v = $(id).value;
    if (v) p.set(k, v);
  }
  return p;
}

function findingKeys(p) {
  return (p.findings || []).map(f => (typeof f === "string" ? f : f.key));
}

function groupName(p, groupBy) {
  if (groupBy === "subtype") return p.subtype || "other";
  if (groupBy === "stage") return p.stage || "unknown";
  if (groupBy === "finding") return findingKeys(p)[0] || "other";
  return null;
}

function render() {
  const tabs = state.tabs;
  const active = tabs.find(t => t.key === state.tab);
  $("f-skin-tone").hidden = !(active && active.skin_tone_filter);

  let panels = state.panels;
  if (state.tab !== "all" && active) panels = panels.filter(p => p.tab === state.tab);
  if (state.tab === "other") panels = panels.filter(p => p.tab === "other");

  const content = $("content");
  content.innerHTML = "";
  if (!panels.length) { content.innerHTML = "<p class='empty'>No panels in this view.</p>"; return; }

  const groupBy = active && active.group_by;
  if (!groupBy) { content.appendChild(grid(panels)); return; }

  const order = (active.group_order || []).map(s => s.toLowerCase());
  const buckets = {};
  for (const p of panels) {
    const g = groupName(p, groupBy);
    (buckets[g] = buckets[g] || []).push(p);
  }
  const names = Object.keys(buckets).sort((a, b) => {
    const ia = order.indexOf(a.toLowerCase()), ib = order.indexOf(b.toLowerCase());
    return (ia < 0 ? 99 : ia) - (ib < 0 ? 99 : ib) || a.localeCompare(b);
  });
  for (const name of names) {
    const h = document.createElement("h2");
    h.className = "group-header";
    h.textContent = name;
    content.appendChild(h);
    content.appendChild(grid(buckets[name]));
  }
}

function captionTexts(p) {
  const captions = (p.caption_variants || [p.figure_caption]).filter(Boolean);
  return captions.map(cap => [p.figure_label, cap].filter(Boolean).join(" — "));
}

function grid(panels) {
  const g = document.createElement("div");
  g.className = "grid";
  for (const p of panels) {
    const c = document.createElement("div");
    c.className = "panel-card";
    const img = p.thumb || p.image;
    c.innerHTML = `
      ${img ? `<img class="thumb" src="${img}" alt="${p.panel_label || ""}">` : ""}
      <div class="card-body">
        <div class="badges">
          ${p.typicality ? `<span class="badge t-${p.typicality}">${p.typicality}</span>` : ""}
          ${p.skin_tone ? `<span class="badge">skin: ${p.skin_tone}</span>` : ""}
          ${p.subtype ? `<span class="badge">${p.subtype}</span>` : ""}
        </div>
        <div class="findings">${findingKeys(p).join(", ")}</div>
        <div class="meta">
          ${[p.disease_key, p.modality, p.body_site, p.stage, p.age_group].filter(Boolean).join(" · ")}
        </div>
        ${p.stated_ethnicity ? `<div class="meta">stated ethnicity: ${p.stated_ethnicity}</div>` : ""}
        ${p.study_region ? `<div class="meta">${p.study_region}</div>` : ""}
        ${captionTexts(p).map(cap => `<div class="caption">${cap}</div>`).join("")}
        ${(p.attribution_variants || [p.attribution_text || ""]).map(a =>
          `<div class="attrib">${a}
            ${p.doi_url ? `<a href="${p.doi_url}" target="_blank" rel="noopener">DOI</a>` : ""}
          </div>`).join("")}
      </div>`;
    if (img) c.querySelector(".thumb").addEventListener("click", () => {
      const parts = [...captionTexts(p), ...(p.in_text_mentions || [])];
      parts.push((p.attribution_text || "") + (p.doi_url ? " — " + p.doi_url : ""));
      openLightbox(p.image || img, parts.filter(Boolean).join("\n\n"));
    });
    g.appendChild(c);
  }
  return g;
}

function loadPanels() {
  return fetch(`/api/diseases/${DISEASE}/panels?` + filters())
    .then(r => r.json()).then(d => { state.panels = d.panels; render(); });
}

Promise.all([
  fetch("/api/diseases").then(r => r.json()),
  fetch(`/api/diseases/${DISEASE}/tabs`).then(r => r.json()),
  fetch(`/api/diseases/${DISEASE}/findings`).then(r => r.json()),
  fetch(`/api/vocab?disease=${DISEASE}`).then(r => r.json()),
]).then(([diseases, tabs, findings, vocab]) => {
  const d = diseases.find(x => x.key === DISEASE) || {};
  $("title").textContent = `${d.name || DISEASE} — visual library`;

  state.tabs = [{ key: "all", label: "All" }, ...tabs, { key: "other", label: "Other" }];
  state.tab = "all";
  const nav = $("tabs");
  for (const t of state.tabs) {
    const b = document.createElement("button");
    b.textContent = t.label;
    b.className = t.key === state.tab ? "tab active" : "tab";
    b.addEventListener("click", () => {
      state.tab = t.key;
      nav.querySelectorAll(".tab").forEach(x => x.classList.remove("active"));
      b.classList.add("active");
      render();
    });
    nav.appendChild(b);
  }

  for (const s of d.subtypes || []) {
    const o = document.createElement("option");
    o.value = s.key; o.textContent = s.label;
    $("f-subtype").appendChild(o);
  }
  const modalities = ["clinical_photo", "dermoscopy", "capillaroscopy", "histology_he",
    "histology_ihc", "immunofluorescence", "radiograph", "ct", "mri", "ultrasound",
    "echo", "pet", "endoscopy", "ophthalmic", "gross", "other"];
  for (const m of modalities) {
    const o = document.createElement("option"); o.value = m; o.textContent = m;
    $("f-modality").appendChild(o);
  }
  state.vocab = vocab;
  for (const v of vocab) {
    const o = document.createElement("option"); o.value = v.finding_key; o.textContent = v.label;
    $("f-finding").appendChild(o);
  }
  for (const id of ["f-subtype", "f-modality", "f-finding", "f-typicality", "f-skin-tone"])
    $(id).addEventListener("change", loadPanels);

  const ul = q("#key-findings ul");
  for (const f of findings) {
    const li = document.createElement("li");
    li.innerHTML = `<b>${f.finding_key || "—"}</b>
      ${f.frequency_text ? ` <span class="freq">${f.frequency_text}</span>` : ""}
      ${f.quote ? `<blockquote>${f.quote}</blockquote>` : ""}`;
    ul.appendChild(li);
  }
  if (!findings.length) ul.innerHTML = "<li>none extracted yet</li>";

  loadPanels();
});
