"""JATS article parsing for stage 3 (workstream W5a).

Pure parser: ``parse_article(xml_text)`` returns a ``ParsedArticle`` with
figure metadata (caption, graphic href, fig-level permissions, third-party
detection, in-text mentions), the body section texts (kept in memory for
stage 7), author surnames, the article-level copyright holder, journal and
publisher names, and the corresponding-author / first-affiliation countries.

Everything is namespace-agnostic: elements are matched by local name so the
parser works whether or not the document declares the xlink/JATS namespaces.
Nothing here touches the network or the filesystem.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from lxml import etree

XLINK = "http://www.w3.org/1999/xlink"

MENTION_MAX_CHARS = 600
MAX_MENTIONS = 3

_THIRD_PARTY_RE = re.compile(
    r"(reproduced|adapted|reprinted|modified|used)\s+(with|by)\s+(kind\s+)?permission"
    r"|courtesy of",
    re.IGNORECASE,
)


@dataclass
class FigureInfo:
    fig_id: str
    label: str
    caption: str
    graphic_href: str | None
    permissions_text: str | None
    fig_license_raw: str | None
    third_party: bool
    third_party_reason: str | None
    in_text_mentions: list[str] = field(default_factory=list)


@dataclass
class ParsedArticle:
    figures: list[FigureInfo]
    body_sections: list[tuple[str, str]]
    authors: list[str]  # first 3 surnames
    author_count: int
    article_copyright_holder: str | None
    journal_name: str | None
    publisher_name: str | None
    corresp_country: str | None
    first_aff_country: str | None


# ---------------------------------------------------------------------------
# Namespace-agnostic helpers
# ---------------------------------------------------------------------------
def _local(el) -> str:
    tag = getattr(el, "tag", "")
    if not isinstance(tag, str):
        return ""
    return etree.QName(tag).localname


def _descendants(el, name: str):
    return [e for e in el.iter() if _local(e) == name]


def _first_desc(el, name: str):
    return next((e for e in el.iter() if _local(e) == name), None)


def _children(el, name: str) -> list:
    return [e for e in el if _local(e) == name]


def _first_child(el, name: str):
    return next((e for e in el if _local(e) == name), None)


def _norm(text: str | None) -> str:
    return " ".join((text or "").split())


def _all_text(el) -> str:
    return _norm("".join(el.itertext()))


def _href(el) -> str | None:
    return el.get(f"{{{XLINK}}}href") or el.get("href")


def _id_map(root) -> dict[str, etree._Element]:
    return {el.get("id"): el for el in root.iter() if el.get("id")}


# ---------------------------------------------------------------------------
# Fields
# ---------------------------------------------------------------------------
def _journal_name(root) -> str | None:
    journal_meta = _first_desc(root, "journal-meta")
    if journal_meta is not None:
        title = _first_desc(journal_meta, "journal-title")
        if title is not None and _all_text(title):
            return _all_text(title)
    title = _first_desc(root, "journal-title")
    return _all_text(title) if title is not None else None


def _publisher_name(root) -> str | None:
    publisher = _first_desc(root, "publisher-name")
    return _all_text(publisher) if publisher is not None else None


def _article_copyright_holder(article_meta) -> str | None:
    if article_meta is None:
        return None
    for permissions in _descendants(article_meta, "permissions"):
        holder = _first_desc(permissions, "copyright-holder")
        if holder is not None and _all_text(holder):
            return _all_text(holder)
    return None


def _authors(root) -> tuple[list[str], int]:
    surnames: list[str] = []
    for contrib in _descendants(root, "contrib"):
        if (contrib.get("contrib-type") or "author") != "author":
            continue
        name = _first_child(contrib, "name")
        surname = _first_desc(name, "surname") if name is not None else None
        if surname is not None and _all_text(surname):
            surnames.append(_all_text(surname))
    return surnames[:3], len(surnames)


def _aff_country_for_contrib(contrib, ids: dict[str, etree._Element]) -> str | None:
    rids: list[str] = []
    for xref in _descendants(contrib, "xref"):
        if (xref.get("ref-type") or "") == "aff":
            rids.extend((xref.get("rid") or "").split())
    for rid in rids:
        aff = ids.get(rid)
        if aff is not None:
            country = _first_desc(aff, "country")
            if country is not None and _all_text(country):
                return _all_text(country)
    for aff in _children(contrib, "aff"):
        country = _first_desc(aff, "country")
        if country is not None and _all_text(country):
            return _all_text(country)
    return None


def _corresp_country(root, ids: dict[str, etree._Element]) -> str | None:
    for contrib in _descendants(root, "contrib"):
        if (contrib.get("corresp") or "").lower() == "yes":
            country = _aff_country_for_contrib(contrib, ids)
            if country:
                return country
    for contrib in _descendants(root, "contrib"):
        if any(
            (x.get("ref-type") or "") == "corresp" for x in _descendants(contrib, "xref")
        ):
            country = _aff_country_for_contrib(contrib, ids)
            if country:
                return country
    for corresp in _descendants(root, "corresp"):
        country = _first_desc(corresp, "country")
        if country is not None and _all_text(country):
            return _all_text(country)
    return None


def _first_aff_country(root) -> str | None:
    for aff in _descendants(root, "aff"):
        country = _first_desc(aff, "country")
        if country is not None and _all_text(country):
            return _all_text(country)
    return None


def _body_sections(root) -> list[tuple[str, str]]:
    body = _first_desc(root, "body")
    if body is None:
        return []
    sections: list[tuple[str, str]] = []
    for sec in _children(body, "sec"):
        title_el = _first_child(sec, "title")
        title = _all_text(title_el) if title_el is not None else ""
        sections.append((title, _all_text(sec)))
    loose = [
        _all_text(p) for p in _children(body, "p") if _all_text(p)
    ]
    if loose:
        sections.insert(0, ("", " ".join(loose)))
    return sections


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _caption_text(caption) -> str:
    parts: list[str] = []
    title = _first_child(caption, "title")
    if title is not None and _all_text(title):
        parts.append(_all_text(title))
    for p in _children(caption, "p"):
        text = _all_text(p)
        if text:
            parts.append(text)
    if parts:
        return " ".join(parts)
    return _all_text(caption)


def _fig_permissions(fig) -> tuple[str | None, str | None, str | None]:
    """(permissions_text, license_raw, copyright_holder) from fig-level perms."""
    permissions = _first_child(fig, "permissions")
    if permissions is None:
        return None, None, None
    text = _all_text(permissions) or None
    license_raw = None
    holder = None
    for lic in _descendants(permissions, "license"):
        license_raw = _href(lic) or lic.get("license-type")
        license_p = _first_desc(lic, "license-p")
        if license_raw is None and license_p is not None:
            license_raw = _all_text(license_p)
        if license_raw:
            break
    if license_raw is None:
        license_p = _first_desc(permissions, "license-p")
        if license_p is not None:
            license_raw = _all_text(license_p)
    holder_el = _first_desc(permissions, "copyright-holder")
    if holder_el is not None and _all_text(holder_el):
        holder = _all_text(holder_el)
    return text, license_raw, holder


def _fuzzy_contains(needle: str, haystack: str) -> bool:
    """Case-insensitive substring match in either direction."""
    a, b = needle.strip().lower(), haystack.strip().lower()
    return bool(a and b) and (a in b or b in a)


def _third_party(
    permissions_text: str | None,
    fig_holder: str | None,
    references: list[str],
) -> tuple[bool, str | None]:
    if permissions_text:
        match = _THIRD_PARTY_RE.search(permissions_text)
        if match:
            return True, match.group(0)
    if fig_holder:
        refs = [r for r in references if r and r.strip()]
        if refs and not any(_fuzzy_contains(fig_holder, r) for r in refs):
            return True, f"copyright-holder: {fig_holder}"
    return False, None


def _paragraph_text_and_mark(p, targets: set[int]) -> tuple[str, int | None]:
    """Flatten paragraph text; mark the char offset of the first target xref."""
    parts: list[str] = []
    mark: int | None = None
    cursor = 0

    def walk(el):
        nonlocal mark, cursor
        if el.text:
            parts.append(el.text)
            cursor += len(el.text)
        for child in el:
            if id(child) in targets and mark is None:
                mark = cursor
            walk(child)
            if child.tail:
                parts.append(child.tail)
                cursor += len(child.tail)

    walk(p)
    return "".join(parts), mark


def _window(text: str, mark: int | None) -> str:
    normalized = _norm(text)
    if len(normalized) <= MENTION_MAX_CHARS:
        return normalized
    if mark is None:
        return normalized[:MENTION_MAX_CHARS]
    start = max(0, min(mark - 200, len(normalized) - MENTION_MAX_CHARS))
    return normalized[start : start + MENTION_MAX_CHARS].strip()


def _fig_mentions(body, fig_id: str) -> list[str]:
    mentions: list[str] = []
    seen: set[str] = set()
    if body is None:
        return mentions
    for p in _descendants(body, "p"):
        if len(mentions) >= MAX_MENTIONS:
            break
        targets = {
            id(x)
            for x in _descendants(p, "xref")
            if (x.get("ref-type") or "") == "fig"
            and fig_id in (x.get("rid") or "").split()
        }
        if not targets:
            continue
        text, mark = _paragraph_text_and_mark(p, targets)
        mention = _window(text, mark)
        if mention and mention not in seen:
            seen.add(mention)
            mentions.append(mention)
    return mentions


def _fig_ids(figs: list) -> list[str]:
    """@id where present, else fig{n} using the figure's 1-based position."""
    used = {f.get("id") for f in figs if f.get("id")}
    ids: list[str] = []
    for index, fig in enumerate(figs, start=1):
        fid = fig.get("id")
        if not fid:
            n = index
            while f"fig{n}" in used:
                n += 1
            fid = f"fig{n}"
            used.add(fid)
        ids.append(fid)
    return ids


def parse_article(xml_text: str) -> ParsedArticle:
    root = etree.fromstring(xml_text.encode("utf-8"))
    ids = _id_map(root)
    article_meta = _first_desc(root, "article-meta")
    body = _first_desc(root, "body")

    article_holder = _article_copyright_holder(article_meta)
    journal = _journal_name(root)
    publisher = _publisher_name(root)
    authors, author_count = _authors(root)
    references = [article_holder or "", journal or "", publisher or "", *authors]

    figs = _descendants(root, "fig")
    fig_ids = _fig_ids(figs)
    figures: list[FigureInfo] = []
    for fig, fig_id in zip(figs, fig_ids):
        label_el = _first_child(fig, "label")
        caption_el = _first_child(fig, "caption")
        graphic = _first_desc(fig, "graphic")
        permissions_text, license_raw, fig_holder = _fig_permissions(fig)
        third_party, reason = _third_party(permissions_text, fig_holder, references)
        figures.append(
            FigureInfo(
                fig_id=fig_id,
                label=_all_text(label_el) if label_el is not None else "",
                caption=_caption_text(caption_el) if caption_el is not None else "",
                graphic_href=_href(graphic) if graphic is not None else None,
                permissions_text=permissions_text,
                fig_license_raw=license_raw,
                third_party=third_party,
                third_party_reason=reason,
                in_text_mentions=_fig_mentions(body, fig_id),
            )
        )

    return ParsedArticle(
        figures=figures,
        body_sections=_body_sections(root),
        authors=authors,
        author_count=author_count,
        article_copyright_holder=article_holder,
        journal_name=journal,
        publisher_name=publisher,
        corresp_country=_corresp_country(root, ids),
        first_aff_country=_first_aff_country(root),
    )
