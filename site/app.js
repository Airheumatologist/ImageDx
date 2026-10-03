/* Disease page: clean image cards with source details available on demand. */
const state = { tab: null, tabs: [], panels: [], vocab: [], eyeEvidence: [], hasPediatric: null, tabFilters: {}, loading: false };
const FILTER_IDS = ["f-subtype", "f-modality", "f-finding", "f-typicality", "f-skin-tone"];
let panelsRequest = 0;
const $ = (id) => document.getElementById(id);
const lb = $("lightbox");
let previouslyFocused = null;

function setText(el, value) {
  if (value) el.textContent = value;
  return el;
}

function closeLightbox() {
  if (lb.hidden) return;
  lb.hidden = true;
  $("lb-media").replaceChildren();
  if (previouslyFocused?.isConnected) previouslyFocused.focus();
}

lb.addEventListener("click", (event) => {
  if (event.target === lb || event.target.closest("[data-close-lightbox]")) closeLightbox();
});
document.addEventListener("keydown", (event) => {
  if (lb.hidden) return;
  if (event.key === "Escape") { event.preventDefault(); closeLightbox(); }
  if (event.key === "Tab") {
    const focusable = [...lb.querySelectorAll("button, a[href]")].filter(el => !el.disabled);
    if (!focusable.length) return;
    const first = focusable[0], last = focusable[focusable.length - 1];
    if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus(); }
    else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus(); }
  }
});

function openLightbox(panel, src) {
  previouslyFocused = document.activeElement;
  const image = VP.panelImage(panel, "lb-img", panel.image || src);
  image.setAttribute("aria-label", displayLabel(panel));
  $("lb-media").replaceChildren(image);
  $("lb-title").textContent = displayLabel(panel);

  const details = $("lb-details");
  details.replaceChildren();
  const add = (label, value, href) => {
    if (!value) return;
    const row = document.createElement("p");
    const heading = document.createElement("strong");
    heading.textContent = `${label}: `;
    row.append(heading);
    if (href) {
      const link = document.createElement("a");
      link.href = href; link.target = "_blank"; link.rel = "noopener noreferrer";
      link.textContent = value; row.append(link);
    } else row.append(document.createTextNode(value));
    details.append(row);
  };

  const sources = panel.source_variants?.length ? panel.source_variants : [panel];
  const uniqueSources = [];
  const seen = new Set();
  for (const source of sources) {
    const identity = [source.article_url || source.doi_url, source.article_title, source.copyright || source.attribution_text, source.license_url, source.license_code].join("|");
    if (!seen.has(identity)) { seen.add(identity); uniqueSources.push(source); }
  }
  const ages = [...new Set(sources.map(source => source.age_group_label).filter(isStatedAgeLabel))];
  const ageValues = ages.length ? ages : [panel.age_group_label].filter(isStatedAgeLabel);
  if (ageValues.length) add(ageValues.length > 1 ? "Age groups noted" : "Age group noted", ageValues.map(humanize).join(" · "));
  const locations = [...new Set(sources.map(source => source.country).filter(Boolean))];
  const locationValues = locations.length ? locations : [panel.country || panel.study_region].filter(Boolean);
  if (locationValues.length) add(locationValues.length > 1 ? "Countries/regions listed in article metadata" : "Country/region listed in article metadata", locationValues.join(" · "));
  const context = panel.context || compactContext(panel);
  add("Context", context);
  if (uniqueSources.length) {
    const heading = document.createElement("h3");
    heading.className = "source-heading";
    heading.textContent = uniqueSources.length === 1 ? "Source" : "Sources";
    details.append(heading);
  }
  uniqueSources.forEach(source => {
    const section = document.createElement("section");
    section.className = "source-item";
    addSourceRow(section, "Article", source.article_title || "Article source", source.article_url || source.doi_url);
    const attribution = source.copyright || source.attribution_text;
    if (attribution) addSourceRow(section, "Copyright & attribution", attribution);
    if (source.license_code || source.license_url) {
      addSourceRow(section, "License", source.license_code || "License details", source.license_url);
    }
    if (source.figure_label) addSourceRow(section, "Figure", source.figure_label);
    if (source.figure_caption && source.figure_caption !== context) {
      addSourceRow(section, "Full figure caption", source.figure_caption);
    }
    details.append(section);
  });

  lb.hidden = false;
  $("lb-close").focus();
}

function isStatedAgeLabel(age) {
  return Boolean(age) && !["unknown", "not stated", "unspecified", "not reported", "not available", "n/a", "na"].includes(String(age).trim().toLowerCase());
}

function addSourceRow(container, label, value, href) {
  if (!value) return;
  const row = document.createElement("p");
  const heading = document.createElement("strong");
  heading.textContent = `${label}: `;
  row.append(heading);
  if (href) {
    const link = document.createElement("a");
    link.href = href; link.target = "_blank"; link.rel = "noopener noreferrer";
    link.textContent = value; row.append(link);
  } else row.append(document.createTextNode(value));
  container.append(row);
}

function compactContext(panel) {
  const parts = [];
  const caption = (panel.caption_variants || [panel.figure_caption]).filter(Boolean).join(" ");
  const mention = (panel.in_text_mentions || []).filter(Boolean).join(" ");
  const evidence = [caption, mention].filter(Boolean).join(" ").replace(/\s+/g, " ").trim();
  if (evidence) parts.push(evidence);
  return parts.join(" · ");
}

function humanize(value) {
  return String(value || "").replace(/[_-]+/g, " ").replace(/\b\w/g, ch => ch.toUpperCase());
}

function displayLabel(panel) {
  if (panel.display_label) return panel.display_label;
  const findings = (panel.findings || []).map(f => typeof f === "string" ? f : (f.label || humanize(f.key)));
  return findings.filter(Boolean).join(", ") || humanize(panel.body_site || panel.modality || "Clinical image");
}

function filters() {
  const p = new URLSearchParams();
  for (const [k, id] of [["subtype", "f-subtype"], ["modality", "f-modality"],
    ["finding", "f-finding"], ["typicality", "f-typicality"]]) {
    const v = $(id).value;
    if (v) p.set(k, v);
  }
  return p;
}

function rememberFilters() {
  state.tabFilters[state.tab] = Object.fromEntries(FILTER_IDS.map(id => [id, $(id).value]));
}

function restoreFilters(tab) {
  const values = state.tabFilters[tab] || {};
  for (const id of FILTER_IDS) $(id).value = values[id] || "";
}

function isPediatric(panel) {
  if (typeof panel.pediatric === "boolean") return panel.pediatric;
  if (panel.pediatric === true) return true;
  const ages = [...(panel.age_group_variants || []), panel.age_group, panel.age_group_label].filter(Boolean);
  return ages.some(age => String(age).toLowerCase().split(/[\s/,;|]+/).some(part =>
    ["child", "children", "pediatric", "paediatric", "infant", "adolescent"].includes(part)));
}

function groupName(panel, groupBy) {
  if (panel.plate_kind === "combined") return "Combined views";
  if (state.tab === "pediatric") {
    if (panel.clinical_group) return panel.clinical_group;
    const clinicalTab = panel.clinical_tab || panel.tab;
    return state.tabs.find(tab => tab.key === clinicalTab)?.label || "Pediatric manifestations";
  }
  if (groupBy === "clinical_group" && panel.clinical_group) return panel.clinical_group;
  if (groupBy === "clinical_group") return "Other";
  if (groupBy === "subtype") return panel.subtype || "other";
  if (groupBy === "stage") return panel.stage || "unknown";
  if (groupBy === "finding") return ((panel.findings || [])[0] || {}).key || "other";
  return null;
}

function render() {
  const active = state.tabs.find(t => t.key === state.tab);
  $("filters").hidden = Boolean(active?.evidence_only);
  $("f-skin-tone").hidden = !(active && active.skin_tone_filter);
  if (state.hasPediatric !== null) {
    const pediatricTab = $("tabs").querySelector('[data-tab-key="pediatric"]');
    if (pediatricTab) pediatricTab.hidden = !state.hasPediatric;
  }
  let panels = state.panels;
  if (state.tab === "pediatric") panels = panels.filter(isPediatric);
  else if (state.tab !== "all" && active) panels = panels.filter(p => p.tab === state.tab);
  if (state.tab === "other") panels = panels.filter(p => p.tab === "other");
  // Skin tone is local to views that expose this filter. Keep the complete
  // result set so switching sections never inherits a hidden skin filter.
  const skinTone = $("f-skin-tone").value;
  if (active?.skin_tone_filter && skinTone) panels = panels.filter(p => p.skin_tone === skinTone);

  const content = $("content");
  content.replaceChildren();
  if (active?.evidence_only) { renderEyeEvidence(content); return; }
  if (state.loading) { content.innerHTML = "<p class='empty'>Loading images…</p>"; return; }
  if (!panels.length) { content.innerHTML = "<p class='empty'>No panels in this view.</p>"; return; }
  const groupBy = state.tab === "pediatric" ? "clinical_tab" : active && active.group_by;
  if (!groupBy) {
    const combined = panels.filter(p => p.plate_kind === "combined");
    const rest = panels.filter(p => p.plate_kind !== "combined");
    if (rest.length) content.appendChild(grid(rest));
    if (combined.length) {
      const heading = document.createElement("h2");
      heading.className = "group-header";
      heading.textContent = "Combined views";
      content.append(heading, grid(combined));
    }
    return;
  }

  const order = (active?.group_order || []).map(s => s.toLowerCase());
  const buckets = {};
  for (const panel of panels) {
    const name = groupName(panel, groupBy) || (state.tab === "pediatric" ? "Pediatric manifestations" : "Other");
    (buckets[name] = buckets[name] || []).push(panel);
  }
  const names = Object.keys(buckets).sort((a, b) => {
    if (a === "Combined views") return 1;
    if (b === "Combined views") return -1;
    const ia = order.indexOf(a.toLowerCase()), ib = order.indexOf(b.toLowerCase());
    return (ia < 0 ? 99 : ia) - (ib < 0 ? 99 : ib) || a.localeCompare(b);
  });
  for (const name of names) {
    const heading = document.createElement("h2");
    heading.className = "group-header";
    heading.textContent = name;
    content.append(heading, grid(buckets[name]));
  }
}

function renderEyeEvidence(content) {
  const intro = document.createElement("p");
  intro.className = "evidence-intro";
  intro.textContent = "Documented eye manifestations. No image meeting the library's publication criteria is available yet.";
  content.append(intro);
  const list = document.createElement("div");
  list.className = "evidence-grid";
  const groups = new Map();
  for (const item of state.eyeEvidence) {
    if (!groups.has(item.finding_key)) groups.set(item.finding_key, []);
    if (groups.get(item.finding_key).length < 3) groups.get(item.finding_key).push(item);
  }
  for (const evidence of groups.values()) {
    const card = document.createElement("article");
    card.className = "evidence-card";
    const heading = document.createElement("h2");
    heading.textContent = evidence[0].label;
    card.append(heading);
    for (const item of evidence) {
      const source = document.createElement("section");
      source.className = "evidence-source";
      const quote = document.createElement("p");
      quote.textContent = item.quote;
      const link = document.createElement("a");
      link.href = item.article_url;
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      link.textContent = item.article_title || item.pmcid;
      source.append(quote, link);
      card.append(source);
    }
    list.append(card);
  }
  content.append(list);
}

function grid(panels) {
  const gridEl = document.createElement("div");
  gridEl.className = "grid";
  for (const panel of panels) {
    const src = panel.thumb || panel.image;
    if (!src) continue;
    const card = document.createElement("article");
    card.className = "panel-card";
    const open = document.createElement("button");
    open.type = "button";
    open.className = "card-open";
    open.setAttribute("aria-label", `Open image: ${displayLabel(panel)}`);
    const image = VP.panelImage(panel, "thumb", src);
    const label = document.createElement("span");
    label.className = "card-label";
    label.textContent = displayLabel(panel);
    open.append(image, label);
    open.addEventListener("click", () => openLightbox(panel, src));
    card.append(open);
    gridEl.append(card);
  }
  return gridEl;
}

function loadPanels() {
  const params = filters();
  const request = ++panelsRequest;
  state.loading = true;
  render();
  return VP.panels(DISEASE, params).then(d => {
      // A previous tab or filter request must not replace the current view.
      if (request !== panelsRequest) return;
      state.loading = false;
      state.panels = d.panels;
      const reserves = Object.values(d.reserves || {}).reduce((sum, n) => sum + n, 0);
      const note = $("reserves-note");
      if (note) {
        note.hidden = reserves === 0;
        note.textContent = reserves
          ? `${reserves} additional eligible image${reserves === 1 ? "" : "s"} held in reserve — galleries are capped at ${d.gallery_cap || 20} per finding.`
          : "";
      }
      if (!params.toString() && state.hasPediatric === null) state.hasPediatric = d.panels.some(isPediatric);
      render();
    }).catch(() => {
      if (request !== panelsRequest) return;
      state.loading = false;
      state.panels = [];
      render();
      $("content").innerHTML = "<p class='empty'>Images could not be loaded. Try changing a filter or switching sections.</p>";
    });
}

Promise.all([
  VP.diseases(),
  VP.tabs(DISEASE),
  VP.vocab(DISEASE),
  VP.eyeEvidence(DISEASE),
]).then(([diseases, tabs, vocab, eyeEvidence]) => {
  const disease = diseases.find(x => x.key === DISEASE) || {};
  $("title").textContent = `${disease.name || DISEASE} — visual library`;
  state.tabs = [{ key: "all", label: "All" }, ...tabs];
  state.eyeEvidence = eyeEvidence;
  state.tab = "all";
  const nav = $("tabs");
  for (const tab of state.tabs) {
    const button = document.createElement("button");
    button.type = "button";
    button.dataset.tabKey = tab.key;
    button.textContent = tab.label;
      button.className = tab.key === state.tab ? "tab active" : "tab";
    if (tab.key === "pediatric" && state.hasPediatric === false) button.hidden = true;
    button.setAttribute("aria-pressed", String(tab.key === state.tab));
    button.addEventListener("click", () => {
      if (state.tab === tab.key) return;
      rememberFilters();
      state.tab = tab.key;
      restoreFilters(tab.key);
      nav.querySelectorAll(".tab").forEach(el => { el.classList.remove("active"); el.setAttribute("aria-pressed", "false"); });
      button.classList.add("active"); button.setAttribute("aria-pressed", "true");
      if (tab.evidence_only) render();
      else loadPanels();
    });
    nav.append(button);
  }
  for (const subtype of disease.subtypes || []) {
    const option = document.createElement("option"); option.value = subtype.key; option.textContent = subtype.label;
    $("f-subtype").append(option);
  }
  const modalities = ["clinical_photo", "dermoscopy", "capillaroscopy", "histology_he", "histology_ihc", "immunofluorescence", "radiograph", "ct", "mri", "ultrasound", "echo", "pet", "endoscopy", "ophthalmic", "gross", "other"];
  for (const modality of modalities) {
    const option = document.createElement("option"); option.value = modality; option.textContent = humanize(modality);
    $("f-modality").append(option);
  }
  state.vocab = vocab;
  for (const item of vocab) {
    const option = document.createElement("option"); option.value = item.finding_key; option.textContent = item.label;
    $("f-finding").append(option);
  }
  for (const id of FILTER_IDS) $(id).addEventListener("change", () => {
    rememberFilters();
    if (id === "f-skin-tone") render();
    else loadPanels();
  });
  loadPanels();
});
