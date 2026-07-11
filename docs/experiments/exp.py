#!/usr/bin/env python3
"""Experiment-log helper for PhysioFM.

Deterministic mechanics for the lab notebook under docs/experiments/:
assign IDs, stamp dates, scaffold from the template, and regenerate the
index table from each entry's frontmatter (the single source of truth).

Agent-agnostic: any agent (or you) can run it directly.

    python exp.py new --title "Un-smoothed DE on SEED-V" --phase phase2-followup
    python exp.py result EXP-0007        # show entry + recent commits/results to fill in
    python exp.py index                   # regenerate the index table in README.md
    python exp.py list                    # print the index table to stdout
    python exp.py check                    # warn about entries missing results/verification

No third-party dependencies (frontmatter is parsed by hand).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve()
# repo root = first parent containing docs/experiments (walk up from skill dir)
for _p in ROOT.parents:
    if (_p / "docs" / "experiments").is_dir() or (_p / ".git").is_dir():
        REPO = _p
        break
else:
    REPO = Path.cwd()
EXP_DIR = REPO / "docs" / "experiments"
README = EXP_DIR / "README.md"

FIELDS = [
    "id", "title", "status", "created", "run_date",
    "agent", "phase", "verified", "tags", "commits", "verdict",
]
STATUSES = ["planned", "running", "done", "blocked"]


def today() -> str:
    return _dt.date.today().isoformat()


# ---------------------------------------------------------------- frontmatter
def parse(path: Path) -> dict:
    """Parse leading --- frontmatter into a dict (string values)."""
    text = path.read_text(encoding="utf-8")
    m = re.match(r"^---\n(.*?)\n---\n", text, re.DOTALL)
    fm: dict = {"_path": path, "_body": text}
    if not m:
        return fm
    for line in m.group(1).splitlines():
        if ":" not in line:
            continue
        k, _, v = line.partition(":")
        fm[k.strip()] = v.strip()
    return fm


def fm_block(fm: dict) -> str:
    lines = ["---"]
    for k in FIELDS:
        lines.append(f"{k}: {fm.get(k, '')}")
    lines.append("---")
    return "\n".join(lines)


def all_entries() -> list[dict]:
    entries = [parse(p) for p in EXP_DIR.glob("EXP-*.md")]
    return sorted(entries, key=lambda e: e.get("id", ""))


def next_id() -> str:
    nums = []
    for p in EXP_DIR.glob("EXP-*.md"):
        m = re.match(r"EXP-(\d+)", p.name)
        if m:
            nums.append(int(m.group(1)))
    return f"EXP-{(max(nums) + 1) if nums else 1:04d}"


def slugify(title: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return s[:50] or "experiment"


# -------------------------------------------------------------------- git aux
def git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO), *args], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return ""


def recent_commits(n: int = 12) -> str:
    return git("log", "--oneline", "-n", str(n)) or "(no git history)"


def recent_results(n: int = 12) -> str:
    rdir = REPO / "results"
    if not rdir.is_dir():
        return "(no results/ dir)"
    paths = sorted(rdir.rglob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
    out = [str(p.relative_to(REPO)) for p in paths if p.is_file()][:n]
    return "\n".join(out) or "(empty)"


# ------------------------------------------------------------------- template
def scaffold_body(fm: dict) -> str:
    return f"""{fm_block(fm)}

# {fm['id']} — {fm['title']}

> **Status:** {fm['status']} · **Created:** {fm['created']} · **Run:** {fm['run_date'] or '—'} · **Agent:** {fm['agent']} · **Phase:** {fm['phase']}

---

## 1. Why — hypothesis & motivation

<!-- What question does this answer? What prior result/null prompted it? What do
     you expect to see and why? Link related entries as [[EXP-XXXX]]. -->


---

## 2. Setup — exactly what was run

<!-- Exact command(s), script, dataset/keys, model variant, key hyperparams,
     and the interpreter. Enough that anyone can reproduce it verbatim. -->

```bash

```

- **Data:**
- **Variant / config:**
- **Output dir:**

---

## 3. Status & run log

<!-- planned -> running -> done | blocked. Note start/finish, machine, anything
     that went sideways. Keep run_date in the frontmatter in sync. -->

- {fm['created']} — created ({fm['agent']})

---

## 4. Results  *(run date: {fm['run_date'] or 'TBD'})*

<!-- Numbers, tables, results path. Mirror into the relevant docs/PHASE*.md
     table if this belongs to a phase. State plainly if it failed. -->


---

## 5. Interpretation — agent's reading

<!-- The agent that ran it explains what the numbers mean and the takeaway.
     This is a CLAIM, not yet verified by the user. -->


---

## 6. ✅ Your verification — *(reserved for Mahdiar)*

> Leave the agent's interpretation above untouched. Confirm or correct it here.

- [ ] **Verified** (set `verified: yes` in frontmatter when ticked)
- **Notes / corrections:**


---

## 7. Commits

<!-- SHAs tied to this experiment (code + this write-up). The helper can pre-fill
     recent commits; prune to the relevant ones. -->


---

## 8. Links

- Related entries:
- Docs / results:
"""


# -------------------------------------------------------------------- index
INDEX_START = "<!-- INDEX:START (auto-generated by exp.py; do not edit by hand) -->"
INDEX_END = "<!-- INDEX:END -->"


def render_index() -> str:
    rows = ["| ID | Date | Title | Status | Verified | Verdict |",
            "| --- | --- | --- | --- | :---: | --- |"]
    for e in sorted(all_entries(), key=lambda e: e.get("id", ""), reverse=True):
        ver = "✅" if e.get("verified", "no").lower() in ("yes", "true", "y") else "—"
        date = e.get("run_date") or e.get("created", "")
        title = e.get("title", "")
        link = e["_path"].name
        verdict = (e.get("verdict", "") or "").strip()
        rows.append(
            f"| [{e.get('id','')}]({link}) | {date} | {title} | "
            f"{e.get('status','')} | {ver} | {verdict} |"
        )
    return "\n".join(rows)


def write_index() -> None:
    table = render_index()
    text = README.read_text(encoding="utf-8")
    block = f"{INDEX_START}\n\n{table}\n\n{INDEX_END}"
    if INDEX_START in text and INDEX_END in text:
        text = re.sub(
            re.escape(INDEX_START) + r".*?" + re.escape(INDEX_END),
            block, text, flags=re.DOTALL,
        )
    else:
        text = text.rstrip() + "\n\n## Index\n\n" + block + "\n"
    README.write_text(text, encoding="utf-8")


# --------------------------------------------------------------------- cmds
def cmd_new(a: argparse.Namespace) -> None:
    eid = next_id()
    fm = {
        "id": eid,
        "title": a.title,
        "status": a.status,
        "created": today(),
        "run_date": "",
        "agent": a.agent,
        "phase": a.phase,
        "verified": "no",
        "tags": a.tags or "",
        "commits": "",
        "verdict": "",
    }
    path = EXP_DIR / f"{eid}-{slugify(a.title)}.md"
    if path.exists():
        sys.exit(f"refusing to overwrite {path}")
    path.write_text(scaffold_body(fm), encoding="utf-8")
    write_index()
    print(f"Created {path.relative_to(REPO)}\n")
    print("--- recent commits (pick the relevant ones for §7) ---")
    print(recent_commits())
    print("\n--- recent result files (for §2/§4) ---")
    print(recent_results())


def _find(eid: str) -> Path:
    eid = eid.upper()
    hits = list(EXP_DIR.glob(f"{eid}-*.md")) + list(EXP_DIR.glob(f"{eid}.md"))
    if not hits:
        sys.exit(f"no entry matching {eid}")
    return hits[0]


def cmd_result(a: argparse.Namespace) -> None:
    path = _find(a.id)
    print(f"Entry: {path.relative_to(REPO)}")
    print("Fill §3 status, §4 results (set run_date), §5 interpretation, §7 commits.")
    print("Then: python exp.py index\n")
    print("--- recent commits ---")
    print(recent_commits())
    print("\n--- recent result files ---")
    print(recent_results())


def cmd_index(_a: argparse.Namespace) -> None:
    write_index()
    print(f"Index regenerated in {README.relative_to(REPO)}")


def cmd_list(_a: argparse.Namespace) -> None:
    print(render_index())


def cmd_check(_a: argparse.Namespace) -> None:
    issues = []
    for e in all_entries():
        eid = e.get("id", e["_path"].name)
        st = e.get("status", "")
        if st == "done" and not e.get("run_date"):
            issues.append(f"{eid}: status=done but no run_date")
        if st == "done" and e.get("verified", "no").lower() not in ("yes", "true", "y"):
            issues.append(f"{eid}: done & awaiting your verification (§6)")
        if st == "running":
            issues.append(f"{eid}: still marked running")
    print("\n".join(issues) if issues else "All entries look complete. ✅")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("new", help="scaffold a new experiment entry")
    p.add_argument("--title", required=True)
    p.add_argument("--phase", default="phase2-followup")
    p.add_argument("--agent", default="claude-code",
                   help="claude-code | cursor | codex | <name>")
    p.add_argument("--status", default="planned", choices=STATUSES)
    p.add_argument("--tags", default="")
    p.set_defaults(func=cmd_new)

    p = sub.add_parser("result", help="show an entry + recent commits/results to fill in")
    p.add_argument("id")
    p.set_defaults(func=cmd_result)

    sub.add_parser("index", help="regenerate the index table").set_defaults(func=cmd_index)
    sub.add_parser("list", help="print the index table").set_defaults(func=cmd_list)
    sub.add_parser("check", help="list entries missing results/verification").set_defaults(func=cmd_check)

    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
