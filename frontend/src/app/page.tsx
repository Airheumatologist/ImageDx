"use client";

import React, { useState, useRef, useEffect, useCallback } from "react";
import type { CSSProperties } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";

/* ───────────────────────── Types ──────────────────────────── */

interface Source {
  pmcid: string;
  title: string;
  drug_name?: string;
  journal: string;
  authors: string[] | { name: string }[];
  year: number;
  doi?: string;
  pdf_url?: string;
  article_type?: string;
  dailymed_url?: string;
  source?: string;
  evidence_grade?: "A" | "B" | "C" | "D";
  evidence_level?: 1 | 2 | 3 | 4;
  evidence_term?: string;
  evidence_source?: string;
  citation_index?: number | string | null;
}

interface EvidenceLevel {
  grade: "A" | "B" | "C" | "D";
  level: 1 | 2 | 3 | 4;
  label: string;
  terms: string[];
}

interface EvidenceHierarchy {
  levels: EvidenceLevel[];
}

interface Message {
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
  followUpQuestions?: string[];
  evidenceHierarchy?: EvidenceHierarchy;
  steps?: { title: string; status: "pending" | "loading" | "complete" }[];
  activeTab?: "answer" | "drugs" | "references";
  showAdditionalReferences?: boolean;
  finalized?: boolean;
}

interface CitationHoverState {
  messageIndex: number;
  citations: CitationEntry[];
  anchorRect: DOMRect;
}

interface CitationEntry {
  citationNumber: number;
  source: Source;
}

interface PendingCitationScrollTarget {
  messageIndex: number;
  tab: "drugs" | "references";
  citationNumber: number;
}

/* ───────────────────────── Constants ─────────────────────── */

const CITATION_COLORS = [
  "#f97316", "#3b82f6", "#10b981", "#8b5cf6", "#f59e0b", "#ec4899",
  "#06b6d4", "#ef4444",
];

const EVIDENCE_COLORS: Record<string, string> = {
  A: "#10b981", B: "#3b82f6", C: "#f59e0b", D: "#ef4444",
};

const ARTICLE_TYPE_COLORS: Record<
  string,
  { border: string; background: string; text: string }
> = {
  systematic_review: {
    border: "rgba(20, 184, 166, 0.65)",
    background: "rgba(20, 184, 166, 0.12)",
    text: "#2dd4bf",
  },
  meta_analysis: {
    border: "rgba(59, 130, 246, 0.7)",
    background: "rgba(59, 130, 246, 0.12)",
    text: "#60a5fa",
  },
  clinical_trial: {
    border: "rgba(16, 185, 129, 0.75)",
    background: "rgba(16, 185, 129, 0.12)",
    text: "#34d399",
  },
  guideline: {
    border: "rgba(245, 158, 11, 0.75)",
    background: "rgba(245, 158, 11, 0.12)",
    text: "#fbbf24",
  },
  review_article: {
    border: "rgba(168, 85, 247, 0.75)",
    background: "rgba(168, 85, 247, 0.12)",
    text: "#c084fc",
  },
  drug_label: {
    border: "rgba(249, 115, 22, 0.75)",
    background: "rgba(249, 115, 22, 0.12)",
    text: "#fb923c",
  },
};

const SUGGESTED_QUERIES = [
  "Best treatments for rheumatoid arthritis?",
  "SGLT2 inhibitors cardiovascular benefits",
  "Management of IgG4-related disease",
  "Neurobrucellosis diagnosis and treatment",
];
const API_BEARER_TOKEN = process.env.NEXT_PUBLIC_API_BEARER_TOKEN?.trim();
const MAX_FOLLOW_UPS_PER_THREAD = 3;
const CITATION_MARKER_REGEX =
  /(\[(?:\d+(?:\s*-\s*\d+)?)(?:\s*,\s*\d+(?:\s*-\s*\d+)?)*\])/g;
const CITATION_OPEN_BRACKET_REGEX = /[【［〖]/g;
const CITATION_CLOSE_BRACKET_REGEX = /[】］〗]/g;
const CITATION_PREFIX_REGEX = /\bCitations?:\s*(?=\[(?:\d|\s|,|-|–)+\])/gi;
const CITATION_LINK_PREFIX = "/__citation__/";

/* ───────────────────────── Helpers ───────────────────────── */

function getArticleUrl(s: Source): string {
  if (s.dailymed_url) return s.dailymed_url;
  if (s.doi) {
    const clean = s.doi.replace(/^https?:\/\/doi\.org\//, "");
    return `https://doi.org/${clean}`;
  }
  if (s.pmcid && !s.pmcid.toLowerCase().includes("dailymed")) {
    const id = s.pmcid.toUpperCase().startsWith("PMC")
      ? s.pmcid
      : `PMC${s.pmcid}`;
    return `https://www.ncbi.nlm.nih.gov/pmc/articles/${id}/`;
  }
  return "#";
}

function isDailyMedSource(s: Source): boolean {
  return s.source === "dailymed" || s.pmcid?.startsWith("dailymed_") || false;
}

function coerceCitationIndex(value: Source["citation_index"]): number | null {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }
  if (typeof value === "string") {
    const parsed = Number.parseInt(value.trim(), 10);
    if (Number.isFinite(parsed)) return parsed;
  }
  return null;
}

function hasCitationIndex(s: Source): boolean {
  return coerceCitationIndex(s.citation_index) !== null;
}

function getCitationIndex(s: Source, fallbackIndex: number): number {
  const citationIndex = coerceCitationIndex(s.citation_index);
  if (citationIndex !== null) return citationIndex;
  return fallbackIndex + 1;
}

function getCitationColor(citationNumber: number): string {
  return CITATION_COLORS[(citationNumber - 1) % CITATION_COLORS.length];
}

function parseCitationLabel(label: string): number[] {
  const inner = label.replace(/^\[|\]$/g, "");
  const parsed = new Set<number>();

  inner.split(",").forEach((rawPart) => {
    const token = rawPart.trim().replace(/–/g, "-");
    if (!token) return;

    if (token.includes("-")) {
      const [startRaw, endRaw] = token.split("-", 2).map((part) => part.trim());
      const start = Number.parseInt(startRaw, 10);
      const end = Number.parseInt(endRaw, 10);
      if (!Number.isFinite(start) || !Number.isFinite(end) || start > end) return;
      for (let idx = start; idx <= end; idx += 1) {
        parsed.add(idx);
      }
      return;
    }

    const value = Number.parseInt(token, 10);
    if (Number.isFinite(value)) parsed.add(value);
  });

  return [...parsed].sort((a, b) => a - b);
}

function formatCitationRangeLabel(citations: number[]): string {
  if (citations.length === 0) return "";

  const ranges: string[] = [];
  let rangeStart = citations[0];
  let prev = citations[0];

  for (let i = 1; i < citations.length; i += 1) {
    const current = citations[i];
    if (current === prev + 1) {
      prev = current;
      continue;
    }

    ranges.push(rangeStart === prev ? String(rangeStart) : `${rangeStart}-${prev}`);
    rangeStart = current;
    prev = current;
  }

  ranges.push(rangeStart === prev ? String(rangeStart) : `${rangeStart}-${prev}`);
  return ranges.join(",");
}

function normalizeCitationMarkers(text: string): string {
  return text
    .replace(CITATION_OPEN_BRACKET_REGEX, "[")
    .replace(CITATION_CLOSE_BRACKET_REGEX, "]");
}

function collapseAdjacentCitationMarkers(text: string): string {
  const adjacentCitationGroupRegex =
    /(?:\[(?:\d+(?:\s*-\s*\d+)?)(?:\s*,\s*\d+(?:\s*-\s*\d+)?)*\][ \t]*){2,}/g;

  return text.replace(adjacentCitationGroupRegex, (group) => {
    const trailingWhitespaceMatch = group.match(/[ \t]+$/);
    const trailingWhitespace = trailingWhitespaceMatch ? trailingWhitespaceMatch[0] : "";
    const citationGroup = trailingWhitespace
      ? group.slice(0, -trailingWhitespace.length)
      : group;

    const allCitations = Array.from(citationGroup.matchAll(CITATION_MARKER_REGEX))
      .flatMap((match) => parseCitationLabel(match[0]));
    const collapsedLabel = formatCitationRangeLabel(
      [...new Set(allCitations)].sort((a, b) => a - b),
    );
    return collapsedLabel
      ? `[${collapsedLabel}]${trailingWhitespace}`
      : group;
  });
}

function encodeCitationLinks(text: string): string {
  return collapseAdjacentCitationMarkers(
    normalizeCitationMarkers(text).replace(CITATION_PREFIX_REGEX, ""),
  )
    .replace(CITATION_MARKER_REGEX, (label) => {
      const value = label.slice(1, -1);
      return `[${value}](${CITATION_LINK_PREFIX}${encodeURIComponent(value)})`;
    });
}

function getCitedReferenceSources(sourceList: Source[]): Source[] {
  return sourceList
    .filter((s) => !isDailyMedSource(s) && hasCitationIndex(s))
    .map((s, i) => ({ source: s, citation: getCitationIndex(s, i) }))
    .sort((a, b) => a.citation - b.citation)
    .map((x) => x.source);
}

function getAdditionalReferenceSources(sourceList: Source[]): Source[] {
  return sourceList.filter((s) => !isDailyMedSource(s) && !hasCitationIndex(s));
}

function getInteractiveCitationSources(sourceList: Source[]): Source[] {
  return sourceList
    .filter((s) => hasCitationIndex(s))
    .map((s, i) => ({ source: s, citation: getCitationIndex(s, i) }))
    .sort((a, b) => a.citation - b.citation)
    .map((x) => x.source);
}

function getSourceMetaLine(source: Source): string {
  return [
    source.journal?.trim(),
    source.year ? String(source.year) : "",
    source.doi ? `doi:${source.doi}` : "",
  ]
    .filter(Boolean)
    .join(" | ");
}

function getCitationHoverCardStyle(anchorRect: DOMRect): CSSProperties {
  if (typeof window === "undefined") {
    return { left: 12, top: anchorRect.bottom + 10 };
  }

  const cardWidth = 340;
  const estimatedCardHeight = 210;
  const clampedLeft = Math.min(
    Math.max(anchorRect.left + anchorRect.width / 2 - cardWidth / 2, 12),
    window.innerWidth - cardWidth - 12,
  );
  const unclampedTop = anchorRect.bottom + 10;
  const maxTop = Math.max(12, window.innerHeight - estimatedCardHeight - 12);

  return {
    left: clampedLeft,
    top: Math.min(unclampedTop, maxTop),
  };
}

function updateStepStatuses(
  steps: Message["steps"],
  {
    loadingStep,
    completeUpTo,
    completeAll,
  }: {
    loadingStep?: number;
    completeUpTo?: number;
    completeAll?: boolean;
  },
) {
  return steps?.map((step, index) => {
    if (completeAll) return { ...step, status: "complete" as const };
    if (completeUpTo !== undefined && index <= completeUpTo) {
      return { ...step, status: "complete" as const };
    }
    if (loadingStep !== undefined) {
      if (index < loadingStep) return { ...step, status: "complete" as const };
      if (index === loadingStep) return { ...step, status: "loading" as const };
    }
    return step;
  });
}

function normalizeArticleType(v?: string): string {
  return (v || "")
    .trim()
    .toLowerCase()
    .replace(/[^\w\s-]/g, "")
    .replace(/[-\s]+/g, "_");
}

function toTitleCase(v: string): string {
  return v
    .replace(/_/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

function getArticleTypeBadge(source: Source): {
  label: string;
  style: CSSProperties;
} | null {
  const rawType = source.article_type || source.evidence_term;
  if (!rawType) return null;

  const normalized = normalizeArticleType(rawType);
  const palette = ARTICLE_TYPE_COLORS[normalized] || {
    border: "rgba(100, 116, 139, 0.75)",
    background: "rgba(100, 116, 139, 0.12)",
    text: "#94a3b8",
  };

  return {
    label: toTitleCase(rawType),
    style: {
      borderColor: palette.border,
      backgroundColor: palette.background,
      color: palette.text,
    },
  };
}

function getActiveStepTitle(steps?: Message["steps"]): string {
  if (!steps?.length) return "Analyzing Query";
  return (
    steps.find((step) => step.status === "loading")?.title ||
    steps.find((step) => step.status === "pending")?.title ||
    steps[steps.length - 1].title
  );
}

/* ───────────────────── Markdown Renderer ─────────────────── */

/**
 * Converts citations to inline markdown links so ReactMarkdown can
 * parse the whole answer at once, including tables, then renders
 * those links back into the colorful citation badges.
 */
function MarkdownWithCitations({
  content,
  sources,
  onCitationClick,
  onCitationHover,
  onCitationHoverEnd,
}: {
  content: string;
  sources: Source[];
  onCitationClick?: (citationNumber: number, source: Source) => void;
  onCitationHover?: (citations: CitationEntry[], anchorRect: DOMRect) => void;
  onCitationHoverEnd?: () => void;
}) {
  const sourcesByCitation = React.useMemo(() => {
    const m = new Map<number, Source>();
    sources.forEach((s, i) => {
      const n = getCitationIndex(s, i);
      if (!m.has(n)) m.set(n, s);
    });
    return m;
  }, [sources]);

  const renderCitationBadge = React.useCallback(
    (label: string, key: React.Key) => {
      const nums = parseCitationLabel(label);
      const anchorNum = nums[0] || 1;
      const color = getCitationColor(anchorNum);
      const displayLabel = formatCitationRangeLabel(nums) || label.slice(1, -1);
      const citedEntries = nums
        .map((citationNumber) => {
          const source = sourcesByCitation.get(citationNumber);
          return source ? { citationNumber, source } : null;
        })
        .filter((entry): entry is CitationEntry => entry !== null);
      const clickTarget = citedEntries[0];

      return (
        <span
          key={key}
          className={`citation-chip citation-chip-inline ${citedEntries.length > 0 ? "" : "citation-chip-disabled"}`.trim()}
          style={{ backgroundColor: color }}
          onClick={clickTarget ? () => onCitationClick?.(clickTarget.citationNumber, clickTarget.source) : undefined}
          onMouseEnter={(event) => {
            if (citedEntries.length === 0) return;
            onCitationHover?.(citedEntries, event.currentTarget.getBoundingClientRect());
          }}
          onMouseLeave={() => onCitationHoverEnd?.()}
        >
          {displayLabel}
        </span>
      );
    },
    [onCitationClick, onCitationHover, onCitationHoverEnd, sourcesByCitation],
  );

  const markdownContent = React.useMemo(
    () => encodeCitationLinks(content),
    [content],
  );

  return (
    <div className="md-content">
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[rehypeKatex]}
        components={{
          table: ({ children }) => (
            <div className="md-table-wrap">
              <table>{children}</table>
            </div>
          ),
          a: ({ href, children, ...props }) => {
            if (href?.startsWith(CITATION_LINK_PREFIX)) {
              const value = decodeURIComponent(href.slice(CITATION_LINK_PREFIX.length));
              return renderCitationBadge(`[${value}]`, href);
            }

            return (
              <a href={href} {...props}>
                {children}
              </a>
            );
          },
        }}
      >
        {markdownContent}
      </ReactMarkdown>
    </div>
  );
}

function ReferenceCard({
  source,
  citationNumber,
  onOpenPdf,
  cardRef,
  isHighlighted = false,
}: {
  source: Source;
  citationNumber?: number;
  onOpenPdf: (url: string) => void;
  cardRef?: (node: HTMLDivElement | null) => void;
  isHighlighted?: boolean;
}) {
  const articleTypeBadge = getArticleTypeBadge(source);
  const isDailyMed = isDailyMedSource(source);
  const refTitle = isDailyMed ? source.drug_name || source.title : source.title;
  const hasNumber = typeof citationNumber === "number";
  const accentColor = hasNumber
    ? getCitationColor(citationNumber)
    : "#64748b";

  return (
    <div
      ref={cardRef}
      className={`ref-card ${hasNumber ? "" : "ref-card-uncited"} ${isHighlighted ? "citation-target-card" : ""}`.trim()}
    >
      {hasNumber ? (
        <div
          className="citation-chip ref-number"
          style={{ backgroundColor: accentColor }}
        >
          {citationNumber}
        </div>
      ) : (
        <div className="ref-supplemental-marker" aria-hidden="true" />
      )}
      <div style={{ flex: 1, minWidth: 0 }}>
        <a
          href={getArticleUrl(source)}
          target="_blank"
          rel="noreferrer"
          className="ref-title"
          style={{ color: accentColor }}
        >
          {refTitle}
        </a>
        <div className="ref-meta">
          {source.journal && <span>{source.journal}. </span>}
          {source.year && <span>{source.year}; </span>}
          {source.doi && (
            <a
              href={`https://doi.org/${source.doi.replace(/^https?:\/\/doi\.org\//, "")}`}
              target="_blank"
              rel="noreferrer"
            >
              doi:{source.doi}
            </a>
          )}
        </div>

        <div className="ref-badges">
          {source.evidence_grade && (
            <span
              className="evidence-badge"
              style={{
                color: EVIDENCE_COLORS[source.evidence_grade] || "#64748b",
                borderColor: EVIDENCE_COLORS[source.evidence_grade] || "#64748b",
              }}
              title={
                source.evidence_source
                  ? `From ${source.evidence_source}`
                  : "Evidence grade"
              }
            >
              {source.evidence_grade}
              {source.evidence_level ? ` · L${source.evidence_level}` : ""}
            </span>
          )}
          {articleTypeBadge && (
            <span className="badge-article-type" style={articleTypeBadge.style}>
              📋 {articleTypeBadge.label}
            </span>
          )}
          {source.pdf_url && (
            <button
              className="pdf-badge"
              onClick={() => onOpenPdf(source.pdf_url!)}
            >
              📕 PDF
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

/* ════════════════════════ MAIN PAGE ════════════════════════ */

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [threadInitialQuery, setThreadInitialQuery] = useState<string | null>(null);
  const [followUpCount, setFollowUpCount] = useState(0);
  const [latestUserQuery, setLatestUserQuery] = useState("");
  const [latestAssistantAnswer, setLatestAssistantAnswer] = useState("");
  const [pdfUrl, setPdfUrl] = useState<string | null>(null);
  const [activeCitationHover, setActiveCitationHover] = useState<CitationHoverState | null>(null);
  const [pendingCitationScrollTarget, setPendingCitationScrollTarget] =
    useState<PendingCitationScrollTarget | null>(null);
  const [highlightedCitationKey, setHighlightedCitationKey] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const referenceCardRefs = useRef<Record<string, HTMLDivElement | null>>({});
  const drugCardRefs = useRef<Record<string, HTMLDivElement | null>>({});
  const citationHoverCardRef = useRef<HTMLDivElement | null>(null);
  const citationHoverHideTimeoutRef = useRef<number | null>(null);
  const userMessages = messages.filter((message) => message.role === "user");
  const latestUserMessage = userMessages[userMessages.length - 1];
  const followUpLimitReached =
    Boolean(threadInitialQuery) && followUpCount >= MAX_FOLLOW_UPS_PER_THREAD;

  /* Auto-scroll on new content */
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  useEffect(() => {
    if (!pendingCitationScrollTarget) return;

    const { messageIndex, tab, citationNumber } = pendingCitationScrollTarget;
    const key = `${messageIndex}:${citationNumber}`;
    const targetNode =
      tab === "references"
        ? referenceCardRefs.current[key]
        : drugCardRefs.current[key];

    if (!targetNode) return;

    targetNode.scrollIntoView({ behavior: "smooth", block: "center" });
    setHighlightedCitationKey(key);
    setPendingCitationScrollTarget(null);
  }, [messages, pendingCitationScrollTarget]);

  useEffect(() => {
    if (!highlightedCitationKey) return;
    const timer = window.setTimeout(() => {
      setHighlightedCitationKey(null);
    }, 1800);
    return () => window.clearTimeout(timer);
  }, [highlightedCitationKey]);

  useEffect(() => {
    if (!activeCitationHover) return;

    const dismissHoverOnExternalScroll = (event: Event) => {
      const target = event.target;
      if (
        citationHoverCardRef.current &&
        target instanceof Node &&
        citationHoverCardRef.current.contains(target)
      ) {
        return;
      }
      setActiveCitationHover(null);
    };
    const dismissHover = () => setActiveCitationHover(null);
    window.addEventListener("scroll", dismissHoverOnExternalScroll, true);
    window.addEventListener("resize", dismissHover);

    return () => {
      window.removeEventListener("scroll", dismissHoverOnExternalScroll, true);
      window.removeEventListener("resize", dismissHover);
    };
  }, [activeCitationHover]);

  useEffect(() => {
    return () => {
      if (citationHoverHideTimeoutRef.current !== null) {
        window.clearTimeout(citationHoverHideTimeoutRef.current);
      }
    };
  }, []);

  const resetConversation = useCallback(() => {
    setMessages([]);
    setInput("");
    setPdfUrl(null);
    setActiveCitationHover(null);
    setPendingCitationScrollTarget(null);
    setHighlightedCitationKey(null);
    setThreadInitialQuery(null);
    setFollowUpCount(0);
    setLatestUserQuery("");
    setLatestAssistantAnswer("");
    referenceCardRefs.current = {};
    drugCardRefs.current = {};
  }, []);

  /* ── Set active tab for a specific message ── */
  const setActiveTab = useCallback(
    (idx: number, tab: "answer" | "drugs" | "references") => {
      setMessages((prev) => {
        const next = [...prev];
        next[idx] = { ...next[idx], activeTab: tab };
        return next;
      });
      setActiveCitationHover((prev) => (prev?.messageIndex === idx ? null : prev));
    },
    [],
  );

  const setShowAdditionalReferences = useCallback((idx: number, show: boolean) => {
    setMessages((prev) => {
      const next = [...prev];
      next[idx] = { ...next[idx], showAdditionalReferences: show };
      return next;
    });
  }, []);

  const registerReferenceCard = useCallback(
    (messageIndex: number, citationNumber: number) =>
      (node: HTMLDivElement | null) => {
        referenceCardRefs.current[`${messageIndex}:${citationNumber}`] = node;
      },
    [],
  );

  const registerDrugCard = useCallback(
    (messageIndex: number, citationNumber: number) =>
      (node: HTMLDivElement | null) => {
        drugCardRefs.current[`${messageIndex}:${citationNumber}`] = node;
      },
    [],
  );

  const handleCitationClick = useCallback(
    (messageIndex: number, citationNumber: number, source: Source) => {
      const nextTab = isDailyMedSource(source) ? "drugs" : "references";
      setActiveCitationHover(null);
      setActiveTab(messageIndex, nextTab);
      setPendingCitationScrollTarget({ messageIndex, tab: nextTab, citationNumber });
    },
    [setActiveTab],
  );

  const handleCitationHover = useCallback(
    (
      messageIndex: number,
      citations: CitationEntry[],
      anchorRect: DOMRect,
    ) => {
      if (citationHoverHideTimeoutRef.current !== null) {
        window.clearTimeout(citationHoverHideTimeoutRef.current);
        citationHoverHideTimeoutRef.current = null;
      }
      setActiveCitationHover({ messageIndex, citations, anchorRect });
    },
    [],
  );

  const hideCitationHover = useCallback(() => {
    if (citationHoverHideTimeoutRef.current !== null) {
      window.clearTimeout(citationHoverHideTimeoutRef.current);
    }
    citationHoverHideTimeoutRef.current = window.setTimeout(() => {
      setActiveCitationHover(null);
      citationHoverHideTimeoutRef.current = null;
    }, 90);
  }, []);

  const keepCitationHoverOpen = useCallback(() => {
    if (citationHoverHideTimeoutRef.current !== null) {
      window.clearTimeout(citationHoverHideTimeoutRef.current);
      citationHoverHideTimeoutRef.current = null;
    }
  }, []);

  const sendQuery = useCallback(async (rawQuery: string) => {
    const q = rawQuery.trim();
    if (!q || isLoading) return;

    const isFollowUp = Boolean(threadInitialQuery);
    if (isFollowUp && followUpCount >= MAX_FOLLOW_UPS_PER_THREAD) {
      return;
    }
    const startedFreshThread = !threadInitialQuery;
    const initialQuery = threadInitialQuery || q;

    setMessages((prev) => [
      ...prev,
      { role: "user", content: q },
      {
        role: "assistant",
        content: "",
        followUpQuestions: [],
        showAdditionalReferences: false,
        finalized: false,
        steps: [
          { title: "Analyzing Query", status: "loading" },
          { title: "Retrieving Articles", status: "pending" },
          { title: "Reranking Evidence", status: "pending" },
          { title: "Checking Source PDFs", status: "pending" },
          { title: "Synthesizing Answer", status: "pending" },
        ],
      },
    ]);
    setInput("");
    setIsLoading(true);
    setPdfUrl(null);
    if (startedFreshThread) {
      setThreadInitialQuery(q);
    }

    try {
      const headers: Record<string, string> = {
        "Content-Type": "application/json",
      };
      if (API_BEARER_TOKEN) {
        headers.Authorization = `Bearer ${API_BEARER_TOKEN}`;
      }

      const res = await fetch("/api/chat/stream", {
        method: "POST",
        headers,
        body: JSON.stringify({
          query: q,
          stream: true,
          thread_context: {
            initial_query: initialQuery,
            is_follow_up: isFollowUp,
            follow_up_count: followUpCount,
            latest_user_query: latestUserQuery,
            latest_assistant_answer: latestAssistantAnswer,
          },
        }),
      });
      if (!res.ok) {
        let errorMessage = "API request failed";
        try {
          const errorBody = await res.json();
          if (typeof errorBody?.detail === "string") {
            errorMessage = errorBody.detail;
          } else if (typeof errorBody?.detail?.message === "string") {
            errorMessage = errorBody.detail.message;
          }
        } catch {
          // Ignore response parsing errors and use generic message.
        }
        throw new Error(errorMessage);
      }

      const reader = res.body?.getReader();
      const decoder = new TextDecoder();
      let answer = "";
      let sources: Source[] = [];
      let followUpQuestions: string[] = [];
      let hierarchy: EvidenceHierarchy | undefined;
      let buf = "";

      const updateMsg = (fn: (m: Message) => Message) =>
        setMessages((prev) => {
          const n = [...prev];
          n[n.length - 1] = fn(n[n.length - 1]);
          return n;
        });

      const syncMetadata = (event: Record<string, unknown>) => {
        if (Array.isArray(event.sources) && event.sources.length) {
          sources = event.sources as Source[];
        }
        if (Array.isArray(event.follow_up_questions)) {
          followUpQuestions = event.follow_up_questions
            .filter((item): item is string => typeof item === "string")
            .map((item) => item.trim())
            .filter(Boolean);
        }
        if (
          event.evidence_hierarchy &&
          typeof event.evidence_hierarchy === "object" &&
          Array.isArray((event.evidence_hierarchy as EvidenceHierarchy).levels)
        ) {
          hierarchy = event.evidence_hierarchy as EvidenceHierarchy;
        }
      };

      const updateAssistant = ({
        content,
        finalized,
        sync = false,
        loadingStep,
        completeUpTo,
        completeAll = false,
      }: {
        content?: string;
        finalized?: boolean;
        sync?: boolean;
        loadingStep?: number;
        completeUpTo?: number;
        completeAll?: boolean;
      }) =>
        updateMsg((m) => ({
          ...m,
          ...(content !== undefined ? { content } : {}),
          ...(finalized !== undefined ? { finalized } : {}),
          ...(sync
            ? {
              sources,
              followUpQuestions,
              evidenceHierarchy: hierarchy || m.evidenceHierarchy,
            }
            : {}),
          steps: updateStepStatuses(m.steps, { loadingStep, completeUpTo, completeAll }),
        }));

      while (true) {
        const { done, value } = (await reader?.read()) || {
          done: true,
          value: undefined,
        };
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        const lines = buf.split("\n");
        buf = lines.pop() || "";

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          const raw = line.slice(6).trim();
          if (raw === "[DONE]") break;
          if (!raw) continue;
          try {
            const d = JSON.parse(raw);
            syncMetadata(d);

            if (d.step === "query_expansion" && d.status === "running") {
              updateAssistant({ loadingStep: 0 });
            } else if (d.step === "query_expansion" && d.status === "complete") {
              updateAssistant({ completeUpTo: 0 });
            } else if (d.step === "retrieval" && d.status === "running") {
              updateAssistant({ loadingStep: 1 });
            } else if (d.step === "retrieval" && d.status === "complete") {
              updateAssistant({ completeUpTo: 1 });
            } else if (d.step === "reranking" && d.status === "running") {
              updateAssistant({ loadingStep: 2 });
            } else if (d.step === "reranking" && d.status === "complete") {
              updateAssistant({ sync: true, finalized: false, completeUpTo: 2 });
            } else if (d.step === "pdf_check" && d.status === "running") {
              updateAssistant({ loadingStep: 3 });
            } else if (d.step === "pdf_check" && d.status === "complete") {
              updateAssistant({ sync: true, finalized: false, completeUpTo: 3 });
            } else if (d.step === "generation" && d.status === "running") {
              if (d.token) {
                answer += d.token;
                updateAssistant({ content: answer, finalized: false, loadingStep: 4 });
              } else {
                updateAssistant({ loadingStep: 4 });
              }
            } else if (d.step === "generation" && d.status === "complete") {
              updateAssistant({ completeUpTo: 4 });
            } else if (d.step === "complete") {
              if (d.answer) answer = d.answer;
              updateAssistant({
                content: answer,
                sync: true,
                finalized: true,
                completeAll: true,
              });
              setLatestUserQuery(q);
              setLatestAssistantAnswer(answer);
              if (isFollowUp) {
                setFollowUpCount((prev) => Math.min(prev + 1, MAX_FOLLOW_UPS_PER_THREAD));
              }
            }
          } catch {
            /* skip bad SSE lines */
          }
        }
      }
    } catch (err) {
      console.error("Chat error:", err);
      const errorMessage = err instanceof Error ? err.message : "Failed to get a response.";
      if (errorMessage.toLowerCase().includes("follow-up limit")) {
        setFollowUpCount(MAX_FOLLOW_UPS_PER_THREAD);
      }
      if (startedFreshThread) {
        setThreadInitialQuery(null);
      }
      setMessages((prev) => {
        const n = [...prev];
        n[n.length - 1].content = `⚠️ ${errorMessage}`;
        return n;
      });
    } finally {
      setIsLoading(false);
    }
  }, [
    followUpCount,
    isLoading,
    latestAssistantAnswer,
    latestUserQuery,
    threadInitialQuery,
  ]);

  /* ── Submit handler (SSE streaming) ── */
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    await sendQuery(input);
  };

  /* ═══════════════════════ RENDER ═══════════════════════════ */

  return (
    <div className="app-container">
      {/* ── Sidebar ── */}
      <aside className="sidebar">
        <div className="sidebar-logo">Elixir AI</div>
        <div className="sidebar-subtitle">Medical Research</div>
        <button
          type="button"
          className="new-thread-btn"
          onClick={resetConversation}
          disabled={isLoading}
        >
          + New Thread
        </button>

        <div className="sidebar-section-title">Suggested</div>
        <div style={{ display: "flex", flexDirection: "column", gap: "0.4rem" }}>
          {SUGGESTED_QUERIES.map((q, i) => (
            <button
              key={i}
              className="suggestion-chip"
              style={{ textAlign: "left", fontSize: "0.78rem" }}
              onClick={() => {
                setInput(q);
              }}
            >
              {q}
            </button>
          ))}
        </div>

        <div className="sidebar-footer">
          Grounded in 1.2M+ peer-reviewed articles from PMC, PubMed &amp; DailyMed
        </div>
      </aside>

      {/* ── Main content area ── */}
      <div className="content-shell">
        <main
          className="chat-main"
          style={{
            flex: pdfUrl ? "0 0 50%" : "1",
            transition: "flex 0.3s ease",
          }}
        >
          {latestUserMessage && (
            <header className="chat-header">
              <div className="chat-header-copy">
                <div className="chat-header-title" title={latestUserMessage.content}>
                  {latestUserMessage.content}
                </div>
              </div>
            </header>
          )}

          {/* Scrollable chat */}
          <div className="chat-scroll">
            <div className="chat-inner">
              {messages.length === 0 ? (
                <div className="empty-state">
                  <h1>How can I assist your research?</h1>
                  <p>
                    Ask complex medical questions backed by peer-reviewed
                    evidence from PMC, PubMed, and DailyMed.
                  </p>
                  <div className="suggestions">
                    {SUGGESTED_QUERIES.map((q, i) => (
                      <button
                        key={i}
                        className="suggestion-chip"
                        onClick={() => setInput(q)}
                      >
                        {q}
                      </button>
                    ))}
                  </div>
                </div>
              ) : (
                messages.map((msg, idx) => {
                  const sourceList = msg.sources || [];
                  const followUpQuestions = msg.followUpQuestions || [];
                  const interactiveCitationSources = getInteractiveCitationSources(sourceList);
                  const citedReferenceSources = getCitedReferenceSources(sourceList);
                  const additionalReferenceSources = getAdditionalReferenceSources(sourceList);
                  const drugSources = sourceList.filter((s) => isDailyMedSource(s));
                  const isFinalized = Boolean(msg.finalized);
                  const referencesReady = isFinalized && citedReferenceSources.length > 0;
                  const hierarchy = msg.evidenceHierarchy;
                  const tab = msg.activeTab || "answer";
                  const showAdditionalReferences = Boolean(msg.showAdditionalReferences);
                  const loadingStatusText = getActiveStepTitle(msg.steps);

                  return (
                    <div
                      key={idx}
                      className={`msg-block ${msg.role === "user" ? "msg-block-user" : "msg-block-assistant"}`}
                    >
                      {msg.role === "user" && (
                        <div className="msg-user-row">
                          <div className="msg-user-content">{msg.content}</div>
                        </div>
                      )}

                      {msg.role === "assistant" && (
                        <section className="msg-assistant-shell">
                          {(msg.content || referencesReady || drugSources.length > 0) && (
                            <div className="tab-bar">
                              <button
                                className={tab === "answer" ? "active" : ""}
                                onClick={() => setActiveTab(idx, "answer")}
                              >
                                Answer
                              </button>
                              {drugSources.length > 0 && (
                                <button
                                  className={tab === "drugs" ? "active" : ""}
                                  onClick={() => setActiveTab(idx, "drugs")}
                                >
                                  Drugs
                                  <span className="tab-count">
                                    {drugSources.length}
                                  </span>
                                </button>
                              )}
                              {referencesReady && (
                                <button
                                  className={tab === "references" ? "active" : ""}
                                  onClick={() => setActiveTab(idx, "references")}
                                >
                                  References
                                  <span className="tab-count">
                                    {citedReferenceSources.length}
                                  </span>
                                </button>
                              )}
                            </div>
                          )}

                          <div className="msg-assistant-card">
                            {tab === "answer" && (
                              <>
                                {msg.steps && !msg.finalized && (
                                  <div
                                    className="loading-status-text"
                                    role="status"
                                    aria-live="polite"
                                    aria-busy="true"
                                    aria-label={loadingStatusText}
                                    data-text={loadingStatusText}
                                  >
                                    {loadingStatusText}
                                  </div>
                                )}

                                {msg.content && (
                                  <>
                                    <MarkdownWithCitations
                                      content={msg.content}
                                      sources={isFinalized ? interactiveCitationSources : []}
                                      onCitationClick={
                                        isFinalized
                                          ? (citationNumber, source) =>
                                            handleCitationClick(idx, citationNumber, source)
                                          : undefined
                                      }
                                      onCitationHover={
                                        isFinalized
                                          ? (citations, anchorRect) =>
                                            handleCitationHover(idx, citations, anchorRect)
                                          : undefined
                                      }
                                      onCitationHoverEnd={hideCitationHover}
                                    />

                                    {isFinalized && followUpQuestions.length > 0 && (
                                      <div className="followup-section">
                                        <div className="followup-label">
                                          Suggested follow-up questions
                                        </div>
                                        <div className="followup-chips">
                                          {followUpQuestions.map((followUp, followUpIndex) => (
                                            <button
                                              key={`${followUp}-${followUpIndex}`}
                                              type="button"
                                              className="followup-chip"
                                              disabled={isLoading || followUpLimitReached}
                                              onClick={() => {
                                                void sendQuery(followUp);
                                              }}
                                            >
                                              {followUp}
                                            </button>
                                          ))}
                                        </div>
                                      </div>
                                    )}
                                  </>
                                )}
                              </>
                            )}

                            {tab === "drugs" && drugSources.length > 0 && (
                              <div className="panel-stack">
                                {drugSources.map((s, si) => {
                                  const dmCitation = getCitationIndex(s, si);
                                  const showCitation = isFinalized && hasCitationIndex(s);
                                  const displayName = s.drug_name || s.title;
                                  const citationKey = `${idx}:${dmCitation}`;
                                  return (
                                    <div
                                      key={si}
                                      ref={showCitation ? registerDrugCard(idx, dmCitation) : undefined}
                                      className={`drug-card ${highlightedCitationKey === citationKey ? "citation-target-card" : ""}`.trim()}
                                    >
                                      <div className="drug-card-header">
                                        <div className="drug-card-title">
                                          {showCitation ? (
                                            <span
                                              className="citation-chip drug-card-citation"
                                              style={{ backgroundColor: getCitationColor(dmCitation) }}
                                            >
                                              {dmCitation}
                                            </span>
                                          ) : null}
                                          <span className="drug-card-name">
                                            {displayName}
                                          </span>
                                        </div>
                                        <a
                                          href={s.dailymed_url || getArticleUrl(s)}
                                          target="_blank"
                                          rel="noreferrer"
                                        >
                                          View on DailyMed
                                        </a>
                                      </div>
                                    </div>
                                  );
                                })}
                              </div>
                            )}

                            {tab === "references" && referencesReady && (
                              <div className="panel-stack">
                                {hierarchy?.levels?.length ? (
                                  <div className="evidence-hierarchy-box">
                                    <div className="evidence-hierarchy-label">
                                      Evidence hierarchy
                                    </div>
                                    <div className="evidence-pills">
                                      {hierarchy.levels.map((lvl) => (
                                        <span
                                          key={lvl.grade}
                                          className="evidence-badge"
                                          style={{
                                            color:
                                              EVIDENCE_COLORS[lvl.grade] ||
                                              "#64748b",
                                            borderColor:
                                              EVIDENCE_COLORS[lvl.grade] ||
                                              "#64748b",
                                          }}
                                          title={lvl.terms.join(", ")}
                                        >
                                          {lvl.grade} · L{lvl.level}
                                        </span>
                                      ))}
                                    </div>
                                  </div>
                                ) : null}

                                {citedReferenceSources.map((s, si) => (
                                  <ReferenceCard
                                    key={`${s.pmcid || s.doi || s.title || "ref"}-${si}`}
                                    source={s}
                                    citationNumber={getCitationIndex(s, si)}
                                    cardRef={registerReferenceCard(idx, getCitationIndex(s, si))}
                                    isHighlighted={
                                      highlightedCitationKey === `${idx}:${getCitationIndex(s, si)}`
                                    }
                                    onOpenPdf={(url) => setPdfUrl(url)}
                                  />
                                ))}

                                {additionalReferenceSources.length > 0 && (
                                  <div className="additional-references-section">
                                    <button
                                      className="additional-references-toggle"
                                      onClick={() =>
                                        setShowAdditionalReferences(
                                          idx,
                                          !showAdditionalReferences,
                                        )
                                      }
                                      aria-expanded={showAdditionalReferences}
                                    >
                                      <span>
                                        {showAdditionalReferences
                                          ? "Hide additional references"
                                          : "Additional references"}
                                      </span>
                                      <span className="tab-count">
                                        {additionalReferenceSources.length}
                                      </span>
                                    </button>

                                    {showAdditionalReferences && (
                                      <div className="additional-references-list">
                                        {additionalReferenceSources.map((s, si) => (
                                          <ReferenceCard
                                            key={`${s.pmcid || s.doi || s.title || "extra-ref"}-${si}`}
                                            source={s}
                                            onOpenPdf={(url) => setPdfUrl(url)}
                                          />
                                        ))}
                                      </div>
                                    )}
                                  </div>
                                )}
                              </div>
                            )}
                          </div>
                        </section>
                      )}
                    </div>
                  );
                })
              )}
              <div ref={messagesEndRef} />
            </div>
          </div>

          {activeCitationHover ? (
            <div
              ref={citationHoverCardRef}
              className="citation-hover-card"
              style={getCitationHoverCardStyle(activeCitationHover.anchorRect)}
              onMouseEnter={keepCitationHoverOpen}
              onMouseLeave={hideCitationHover}
            >
              <div className="citation-hover-card-arrow" />
              <div className="citation-hover-card-list">
                {activeCitationHover.citations.map(({ citationNumber, source }) => {
                  const articleTypeBadge = getArticleTypeBadge(source);
                  const displayTitle = isDailyMedSource(source)
                    ? source.drug_name || source.title
                    : source.title;
                  const articleUrl = getArticleUrl(source);
                  const hasArticleUrl = articleUrl !== "#";

                  return (
                    <div className="citation-hover-card-item" key={`${citationNumber}:${source.pmcid || source.title}`}>
                      <div className="citation-hover-card-header">
                        <span
                          className="citation-chip citation-hover-card-number"
                          style={{ backgroundColor: getCitationColor(citationNumber) }}
                        >
                          {citationNumber}
                        </span>
                        {hasArticleUrl ? (
                          <a
                            className="citation-hover-card-title citation-hover-card-link"
                            href={articleUrl}
                            target="_blank"
                            rel="noopener noreferrer"
                          >
                            {displayTitle}
                          </a>
                        ) : (
                          <div className="citation-hover-card-title">{displayTitle}</div>
                        )}
                      </div>
                      <div className="citation-hover-card-meta">{getSourceMetaLine(source)}</div>
                      <div className="ref-badges">
                        {source.evidence_grade ? (
                          <span
                            className="evidence-badge"
                            style={{
                              color: EVIDENCE_COLORS[source.evidence_grade] || "#64748b",
                              borderColor: EVIDENCE_COLORS[source.evidence_grade] || "#64748b",
                            }}
                          >
                            {source.evidence_grade}
                            {source.evidence_level
                              ? ` · L${source.evidence_level}`
                              : ""}
                          </span>
                        ) : null}
                        {articleTypeBadge ? (
                          <span
                            className="badge-article-type"
                            style={articleTypeBadge.style}
                          >
                            📋 {articleTypeBadge.label}
                          </span>
                        ) : null}
                        {source.pdf_url ? (
                          <button
                            className="pdf-badge"
                            onClick={() => setPdfUrl(source.pdf_url!)}
                          >
                            📕 PDF
                          </button>
                        ) : null}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          ) : null}

          {/* ── Input ── */}
          <div className="input-area">
            <form onSubmit={handleSubmit} className="input-form">
              <input
                type="text"
                className="input-field"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder={
                  followUpLimitReached
                    ? "Follow-up limit reached. Start a new thread to continue."
                    : "Ask a medical research question…"
                }
                disabled={isLoading || followUpLimitReached}
              />
              <button
                type="submit"
                disabled={isLoading || !input.trim() || followUpLimitReached}
                className="submit-btn"
              >
                {isLoading ? "…" : "Ask"}
              </button>
            </form>
            {followUpLimitReached ? (
              <div className="thread-limit-notice">
                You have reached the 3 follow-up limit for this thread. Start a new thread to continue.
              </div>
            ) : null}
            <div className="input-disclaimer">
              Elixir AI can make mistakes. Always verify with original sources.
            </div>
          </div>
        </main>

        {/* ── PDF Viewer ── */}
        {pdfUrl && (
          <div className="pdf-panel">
            <div className="pdf-header">
              <span>📄 PDF Viewer</span>
              <div className="pdf-actions">
                <a
                  href={pdfUrl}
                  target="_blank"
                  rel="noreferrer"
                  className="pdf-open-btn"
                >
                  Open in Tab
                </a>
                <button
                  className="pdf-close-btn"
                  onClick={() => setPdfUrl(null)}
                >
                  ✕ Close
                </button>
              </div>
            </div>
            <iframe
              src={`/api/pdf/proxy?url=${encodeURIComponent(pdfUrl)}`}
              style={{ flex: 1, width: "100%", border: "none" }}
              title="PDF Viewer"
            />
          </div>
        )}
      </div>
    </div>
  );
}
