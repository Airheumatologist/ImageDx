/* Disease page: clean image cards with source details available on demand. */
const state = { tabs: [], active: null, panels: [], eyeEvidence: [], hasPediatric: null, loading: false };
let panelsRequest = 0;
const $ = (id) => document.getElementById(id);
const lb = $("lightbox");
// Card button -> { panel, src }, so the lightbox can step through a section's cards.
const cardPanels = new WeakMap();
let lbItems = [];
let lbIndex = 0;

function setText(el, value) {
  if (value) el.textContent = value;
  return el;
}

function closeLightbox() {
  if (lb.hidden) return;
  lb.hidden = true;
  document.body.classList.remove("lb-open");
  $("lb-media").replaceChildren();
  const trigger = lbItems[lbIndex];
  if (trigger?.isConnected) trigger.focus();
}

function stepLightbox(delta) {
  if (lbItems.length < 2) return;
  showLightboxItem((lbIndex + delta + lbItems.length) % lbItems.length);
}

lb.addEventListener("click", (event) => {
  if (event.target === lb || event.target.closest("[data-close-lightbox]")) closeLightbox();
});
$("lb-prev").addEventListener("click", () => stepLightbox(-1));
$("lb-next").addEventListener("click", () => stepLightbox(1));

// Horizontal swipe on the image steps through the section on touch screens.
let touchStart = null;
const stage = lb.querySelector(".lb-stage");
stage.addEventListener("touchstart", (event) => {
  const t = event.touches[0];
  touchStart = event.touches.length === 1 ? { x: t.clientX, y: t.clientY } : null;
}, { passive: true });
stage.addEventListener("touchend", (event) => {
  if (!touchStart) return;
  const t = event.changedTouches[0];
  const dx = t.clientX - touchStart.x, dy = t.clientY - touchStart.y;
  touchStart = null;
  if (Math.abs(dx) > 50 && Math.abs(dx) > Math.abs(dy) * 1.5) stepLightbox(dx < 0 ? 1 : -1);
});

document.addEventListener("keydown", (event) => {
  if (lb.hidden) return;
  if (event.key === "Escape") { event.preventDefault(); closeLightbox(); }
  if (event.key === "ArrowLeft") { event.preventDefault(); stepLightbox(-1); }
  if (event.key === "ArrowRight") { event.preventDefault(); stepLightbox(1); }
  if (event.key === "Tab") {
    const focusable = [...lb.querySelectorAll("button, a[href]")].filter(el => !el.disabled);
    if (!focusable.length) return;
    const first = focusable[0], last = focusable[focusable.length - 1];
    if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus(); }
    else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus(); }
  }
});

function openLightbox(trigger) {
  const scope = trigger.closest(".section-panel") || document;
  lbItems = [...scope.querySelectorAll(".card-open")];
  showLightboxItem(Math.max(0, lbItems.indexOf(trigger)));
  lb.hidden = false;
  document.body.classList.add("lb-open");
  $("lb-close").focus();
}

function showLightboxItem(index) {
  lbIndex = index;
  const { panel, src } = cardPanels.get(lbItems[index]);
  const multiple = lbItems.length > 1;
  $("lb-prev").hidden = !multiple;
  $("lb-next").hidden = !multiple;
  $("lb-counter").textContent = multiple ? `${index + 1} / ${lbItems.length}` : "";
  lb.scrollTop = 0;
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

function isPediatric(panel) {
  if (typeof panel.pediatric === "boolean") return panel.pediatric;
  if (panel.pediatric === true) return true;
  const ages = [...(panel.age_group_variants || []), panel.age_group, panel.age_group_label].filter(Boolean);
  return ages.some(age => String(age).toLowerCase().split(/[\s/,;|]+/).some(part =>
    ["child", "children", "pediatric", "paediatric", "infant", "adolescent"].includes(part)));
}

function groupName(panel, groupBy, tabKey) {
  if (panel.plate_kind === "combined") return "Combined views";
  if (tabKey === "pediatric") {
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

function sectionPanels(tab) {
  let panels = state.panels;
  if (tab.key === "pediatric") panels = panels.filter(isPediatric);
  else panels = panels.filter(p => p.tab === tab.key);
  return panels;
}

function plural(n, word) {
  return `${n} ${word}${n === 1 ? "" : "s"}`;
}

function render() {
  const content = $("content");
  const nav = $("section-nav");
  gridResize.disconnect();
  content.replaceChildren();
  nav.replaceChildren();
  if (state.loading) {
    nav.hidden = true;
    content.innerHTML = "<p class='empty'>Loading images…</p>";
    return;
  }
  const tabs = state.tabs.filter(tab => !(tab.key === "pediatric" && !state.hasPediatric));
  const entries = tabs.map(tab => {
    const panels = tab.evidence_only ? [] : sectionPanels(tab);
    const count = tab.evidence_only
      ? new Set(state.eyeEvidence.map(item => item.finding_key)).size
      : panels.length;
    return { tab, panels, count };
  });
  // One section shows at a time. Start from the URL hash, else the first section with images.
  if (!entries.some(e => e.tab.key === state.active)) {
    const hash = decodeURIComponent(location.hash.slice(1));
    const start = entries.find(e => e.tab.key === hash)
      || entries.find(e => !e.tab.evidence_only && e.count) || entries[0];
    state.active = start?.tab.key;
  }
  $("topic-meta").textContent = [state.panels.length && plural(state.panels.length, "image"), plural(entries.length, "section")]
    .filter(Boolean).join(" · ");

  for (const { tab, count } of entries) {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "section-chip";
    chip.id = `tab-${tab.key}`;
    chip.dataset.key = tab.key;
    chip.setAttribute("role", "tab");
    const selected = tab.key === state.active;
    chip.setAttribute("aria-selected", String(selected));
    chip.setAttribute("aria-controls", "section-panel");
    chip.tabIndex = selected ? 0 : -1;
    const name = document.createElement("span");
    name.textContent = tab.label;
    const badge = document.createElement("span");
    badge.className = "section-chip-count";
    badge.textContent = String(count);
    badge.setAttribute("aria-label", tab.evidence_only ? plural(count, "finding") : plural(count, "image"));
    chip.append(name, badge);
    chip.addEventListener("click", () => selectSection(tab.key));
    nav.append(chip);
  }
  nav.hidden = entries.length < 2;

  const entry = entries.find(e => e.tab.key === state.active);
  if (!entry) return;
  const panel = document.createElement("section");
  panel.className = "section-panel";
  panel.id = "section-panel";
  panel.setAttribute("role", "tabpanel");
  panel.setAttribute("aria-labelledby", `tab-${entry.tab.key}`);
  if (entry.tab.evidence_only) renderEyeEvidence(panel);
  else renderSection(panel, entry.tab, entry.panels);
  content.append(panel);
  // Keep the selected chip visible in the horizontally scrolling row.
  const active = nav.querySelector("[aria-selected='true']");
  if (active) nav.scrollLeft = active.offsetLeft - (nav.clientWidth - active.offsetWidth) / 2;
}

function selectSection(key) {
  if (key === state.active) return;
  state.active = key;
  history.replaceState(null, "", `#${encodeURIComponent(key)}`);
  render();
  // render() rebuilt the chips; keep focus on the selected one so arrow keys keep working.
  $(`tab-${key}`)?.focus({ preventScroll: true });
  // If the chip row is stuck to the top, bring the new section's first row into view.
  const nav = $("section-nav");
  const navTop = nav.getBoundingClientRect().top + window.scrollY;
  if (window.scrollY > navTop) window.scrollTo({ top: navTop });
}

// Arrow keys move between section tabs (WAI-ARIA tabs pattern).
$("section-nav").addEventListener("keydown", (event) => {
  const chips = [...event.currentTarget.querySelectorAll(".section-chip")];
  const index = chips.indexOf(document.activeElement);
  if (index < 0) return;
  const next = { ArrowRight: index + 1, ArrowLeft: index - 1, Home: 0, End: chips.length - 1 }[event.key];
  if (next === undefined) return;
  event.preventDefault();
  selectSection(chips[(next + chips.length) % chips.length].dataset.key);
});

function renderSection(content, tab, panels) {
  if (!panels.length) { content.innerHTML = "<p class='empty'>No images yet.</p>"; return; }
  const groupBy = tab.key === "pediatric" ? "clinical_tab" : tab.group_by;
  if (!groupBy) {
    const combined = panels.filter(p => p.plate_kind === "combined");
    const rest = panels.filter(p => p.plate_kind !== "combined");
    if (rest.length) content.appendChild(grid(rest));
    if (combined.length) {
      const heading = document.createElement("h3");
      heading.className = "group-header";
      heading.textContent = "Combined views";
      content.append(heading, grid(combined));
    }
    return;
  }

  const order = (tab.group_order || []).map(s => s.toLowerCase());
  const buckets = {};
  for (const panel of panels) {
    const name = groupName(panel, groupBy, tab.key) || (tab.key === "pediatric" ? "Pediatric manifestations" : "Other");
    (buckets[name] = buckets[name] || []).push(panel);
  }
  const names = Object.keys(buckets).sort((a, b) => {
    if (a === "Combined views") return 1;
    if (b === "Combined views") return -1;
    const ia = order.indexOf(a.toLowerCase()), ib = order.indexOf(b.toLowerCase());
    return (ia < 0 ? 99 : ia) - (ib < 0 ? 99 : ib) || a.localeCompare(b);
  });
  for (const name of names) {
    const heading = document.createElement("h3");
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

// Justified rows: images keep their own shape, and each full row is scaled to
// exactly fill the width; the last row stays at the target height (--row).
function cardRatio(card) {
  return parseFloat(card.style.getPropertyValue("--ratio")) || 4 / 3;
}

function justify(gridEl) {
  const width = gridEl.clientWidth;
  if (!width) return;
  const style = getComputedStyle(gridEl);
  const target = parseFloat(style.getPropertyValue("--row")) || 200;
  const gap = parseFloat(style.columnGap) || 0;
  const place = (row, height) => row.forEach(card => {
    card.style.width = `${Math.floor(cardRatio(card) * height * 100) / 100}px`;
  });
  let row = [];
  for (const card of gridEl.children) {
    row.push(card);
    const ratios = row.reduce((sum, c) => sum + cardRatio(c), 0);
    const available = width - gap * (row.length - 1);
    if (ratios * target >= available) { place(row, available / ratios); row = []; }
  }
  place(row, target);
}

const gridResize = new ResizeObserver(entries => entries.forEach(entry => justify(entry.target)));

function grid(panels) {
  const gridEl = document.createElement("div");
  gridEl.className = "grid";
  let pending = 0;
  const relayout = () => {
    if (!pending) pending = requestAnimationFrame(() => { pending = 0; justify(gridEl); });
  };
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
    // Load eagerly: rows are laid out from every image's shape, so late loads would reflow them.
    if (image.tagName === "IMG") image.loading = "eager";
    image.addEventListener("load", () => {
      const w = image.naturalWidth || image.width, h = image.naturalHeight || image.height;
      if (w && h) { card.style.setProperty("--ratio", (w / h).toFixed(4)); relayout(); }
    }, { once: true });
    const label = document.createElement("span");
    label.className = "card-label";
    label.textContent = displayLabel(panel);
    open.append(image, label);
    cardPanels.set(open, { panel, src });
    open.addEventListener("click", () => openLightbox(open));
    card.append(open);
    gridEl.append(card);
  }
  gridResize.observe(gridEl);
  return gridEl;
}

function loadPanels() {
  const request = ++panelsRequest;
  state.loading = true;
  render();
  return VP.panels(DISEASE, new URLSearchParams()).then(d => {
      // A previous request must not replace the current view.
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
      if (state.hasPediatric === null) state.hasPediatric = d.panels.some(isPediatric);
      render();
    }).catch(() => {
      if (request !== panelsRequest) return;
      state.loading = false;
      state.panels = [];
      render();
      $("content").innerHTML = "<p class='empty'>Images could not be loaded. Try switching sections.</p>";
    });
}

Promise.all([
  VP.diseases(),
  VP.tabs(DISEASE),
  VP.eyeEvidence(DISEASE),
]).then(([diseases, tabs, eyeEvidence]) => {
  const disease = diseases.find(x => x.key === DISEASE) || {};
  $("title").textContent = disease.name || humanize(DISEASE);
  document.title = `${disease.name || humanize(DISEASE)} · Visual library`;
  state.tabs = tabs;
  state.eyeEvidence = eyeEvidence;
  loadPanels();
});
