"""Faithfulness check of VLM-written report text against the detection-derived findings.

A statement is counted as hallucinated when it
  (a) cites a finding id that does not exist,
  (b) mentions a class (lexicon in prompts.CLASS_TERMS) for which the report has NO finding and the
      sentence is not a negation ("no", "not", "none", "without", "absent", "did not"), or
  (c) cites finding id(s) but mentions only classes different from those findings' classes, or
  (d) cites a finding the VLM verification rejected/left unverified (rejected_ids) without hedging it as
      unconfirmed - counted separately as `cites_rejected` and included in n_hallucinated.
In (b) only classes of NON-rejected findings count as present.
Numbers and locations quoted in free text are NOT checked (limit; tables are code-generated).
"""
from __future__ import annotations

import re

from prompts import CLASS_TERMS

ID_RE = re.compile(r"\bF\d{2,3}\b")
NEG_RE = re.compile(r"\b(no|not|none|without|absent|neither|nor|n't)\b", re.I)
HEDGE_RE = re.compile(r"unconfirmed|unverified|rejected|not confirmed|could not be confirmed|candidate", re.I)


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]


def classes_in(sentence: str) -> set[str]:
    s = sentence.lower()
    return {c for c, terms in CLASS_TERMS.items() if any(re.search(r"\b" + re.escape(t), s) for t in terms)}


def check_text(text: str, findings: list[dict], rejected_ids: set[str] | frozenset[str] = frozenset()) -> dict:
    ids = {f["id"]: f["cls"] for f in findings}
    present = {c for i, c in ids.items() if i not in rejected_ids}
    unknown_ids, absent_mentions, mismatches, cites_rejected = [], [], [], []
    cited: set[str] = set()
    n_sent = 0
    for s in split_sentences(text):
        n_sent += 1
        sid = ID_RE.findall(s)
        cited.update(i for i in sid if i in ids)
        for i in sid:
            if i not in ids:
                unknown_ids.append({"id": i, "sentence": s})
        if not HEDGE_RE.search(s):
            cites_rejected += [{"id": i, "sentence": s} for i in dict.fromkeys(sid) if i in rejected_ids]
        cl = classes_in(s)
        if not NEG_RE.search(s):
            for c in cl - present:
                absent_mentions.append({"cls": c, "sentence": s})
        known = [ids[i] for i in sid if i in ids]
        if (known and cl and not (cl & set(known)) and not NEG_RE.search(s)
                and not any(m["sentence"] == s for m in absent_mentions)):  # one flag per sentence for class issues
            mismatches.append({"ids": sid, "sentence_classes": sorted(cl), "sentence": s})
    return {
        "n_sentences": n_sent,
        "n_ids_cited": len(cited),
        "unknown_ids": unknown_ids,
        "absent_class_mentions": absent_mentions,
        "id_class_mismatches": mismatches,
        "cites_rejected": cites_rejected,
        "n_hallucinated": len(unknown_ids) + len(cites_rejected) + len(absent_mentions) + len(mismatches),
        "findings_not_mentioned": sorted(set(ids) - cited),
    }


COUNT_RE = re.compile(r"\b(\d+)\s+(?:confirmed\s+)?(?:key\s+)?findings?\b", re.I)


def check_report(summary: str, findings: list[dict], per_finding: dict[str, dict]) -> dict:
    """Faithfulness of the whole VLM-written text.

    Scope: the summary gets every check (plus a check that a stated count of "N confirmed findings"
    equals the real number of confirmed findings). The per-finding assessments of CONFIRMED findings are
    checked on their own text (no id prefix: that is table structure, not a citation). Assessments of
    unconfirmed findings are the VLM's rejection rationale shown under 'unconfirmed' and are not checked.
    """
    confirmed = {f["id"] for f in findings if per_finding[f["id"]].get("shows_class") is True}
    rejected = {f["id"] for f in findings} - confirmed
    res = check_text(summary, findings, rejected)
    n_summary = res["n_hallucinated"]
    count_errors = []
    for m in COUNT_RE.finditer(summary):
        if "confirmed" in m.group(0).lower() and int(m.group(1)) != len(confirmed):
            count_errors.append({"claimed": int(m.group(1)), "actual": len(confirmed)})
    res["count_claim_errors"] = count_errors
    res["n_hallucinated"] += len(count_errors)
    n_assess = 0
    for f in findings:
        if f["id"] not in confirmed:
            continue
        r = check_text(per_finding[f["id"]].get("assessment", ""), findings, rejected)
        n_assess += r["n_hallucinated"]
        for k in ("unknown_ids", "absent_class_mentions", "id_class_mismatches", "cites_rejected"):
            res[k] = res[k] + [dict(x, finding=f["id"]) for x in r[k]]
        res["n_sentences"] += r["n_sentences"]
    res["n_hallucinated"] += n_assess
    res["summary_only_hallucinated"] = n_summary + len(count_errors)
    res["assessment_hallucinated"] = n_assess
    res["n_confirmed"] = len(confirmed)
    res["findings_not_mentioned"] = sorted(confirmed & set(res["findings_not_mentioned"]))  # confirmed, absent from summary
    return res

