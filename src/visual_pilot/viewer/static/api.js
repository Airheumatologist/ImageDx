/* Data access shared by every page. The local viewer reads the live API;
   the static export (window.VP_STATIC, see site_export.py) reads JSON files
   and applies panel filters in the browser. */
const VP = (() => {
  const STATIC = Boolean(window.VP_STATIC);
  const cache = new Map();
  const norm = (value) => String(value ?? "").trim().toLowerCase();

  function json(url) {
    if (STATIC && cache.has(url)) return cache.get(url);
    const request = fetch(url).then(r => {
      if (!r.ok) throw new Error(`${r.status} ${url}`);
      return r.json();
    });
    if (STATIC) cache.set(url, request);
    return request;
  }

  // Mirrors api_panels: the finding filter keeps that finding's panels,
  // the rest compare normalized values.
  function filterPanels(data, params) {
    let panels = data.panels;
    const finding = params.get("finding");
    if (finding) panels = panels.filter(p => (p.findings || []).some(f => f.key === finding));
    for (const key of ["modality", "subtype", "skin_tone", "typicality"]) {
      const value = params.get(key);
      if (value) panels = panels.filter(p => norm(p[key]) === norm(value));
    }
    return { ...data, panels, count: panels.length };
  }

  return {
    STATIC,
    diseaseKey: () => new URLSearchParams(location.search).get("d") || location.pathname.split("/").pop(),
    homeUrl: STATIC ? "index.html" : "/",
    compareUrl: STATIC ? "compare.html" : "/compare/sle-dm-skin",
    diseaseUrl: (key) => STATIC ? `disease.html?d=${encodeURIComponent(key)}` : `/disease/${key}`,
    diseases: () => json(STATIC ? "data/diseases.json" : "/api/diseases"),
    articles: () => json(STATIC ? "data/articles.json" : "/api/articles"),
    tabs: (key) => json(STATIC ? `data/${key}/tabs.json` : `/api/diseases/${key}/tabs`),
    vocab: (key) => json(STATIC ? `data/${key}/vocab.json` : `/api/vocab?disease=${key}`),
    eyeEvidence: (key) => json(STATIC ? `data/${key}/eye-evidence.json` : `/api/diseases/${key}/eye-evidence`),
    panels: (key, params) => STATIC
      ? json(`data/${key}/panels.json`).then(d => filterPanels(d, params))
      : json(`/api/diseases/${key}/panels?${params}`),
    compare: () => json(STATIC ? "data/compare-sle-dm-skin.json" : "/api/compare/sle-dm-skin"),

    /* An element showing the panel: an <img>, or for a cropped panel of a
       whole figure (panel.crop = normalized [x0, y0, x1, y1]) a <canvas>
       holding that region. Drawing a cross-origin image to a canvas only
       taints it, which does not affect display. */
    panelImage(panel, className = "", src = panel.thumb || panel.image) {
      const img = document.createElement("img");
      img.className = className;
      img.alt = "";
      img.decoding = "async";
      if (!panel.crop) {
        img.loading = "lazy";
        img.src = src;
        return img;
      }
      const canvas = document.createElement("canvas");
      canvas.className = className;
      canvas.setAttribute("role", "img");
      img.addEventListener("load", () => {
        const [x0, y0, x1, y1] = panel.crop;
        const sx = x0 * img.naturalWidth, sy = y0 * img.naturalHeight;
        canvas.width = Math.max(1, Math.round((x1 - x0) * img.naturalWidth));
        canvas.height = Math.max(1, Math.round((y1 - y0) * img.naturalHeight));
        canvas.getContext("2d").drawImage(img, sx, sy, canvas.width, canvas.height, 0, 0, canvas.width, canvas.height);
      }, { once: true });
      img.src = src;
      return canvas;
    },
  };
})();
