"""Download (or guide the manual download of) the EUR-Lex source PDFs.

The ingestion pipeline expects the AI Act and GDPR official PDFs under
``data/raw/``. EUR-Lex sits behind a CloudFront WAF that frequently
challenges automated clients (curl, Python httpx, even browser-UA fakes).
This script does its best:

1. Tries every known direct URL with retries and a polite User-Agent.
2. Falls back to **clear manual-download instructions** with the exact
   browser URLs and target filenames you should drop into ``data/raw/``.

You can also accept the official EUR-Lex filename verbatim — the ingestion
pipeline (``src/agentic_rag/ingest/run.py``) recognizes both the
human-friendly short name and the OJ-style name.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import httpx
from rich.console import Console

console = Console()


@dataclass
class Source:
    label: str
    short_name: str        # what we'd ideally save as
    accepted_names: list[str]  # what the ingester also accepts
    urls: list[str]
    eli_url: str           # the human-friendly EUR-Lex landing page


SOURCES: list[Source] = [
    Source(
        label="EU AI Act (Regulation (EU) 2024/1689)",
        short_name="ai_act.pdf",
        accepted_names=["ai_act.pdf", "OJ_L_202401689_EN_TXT.pdf"],
        urls=[
            "https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=OJ:L_202401689",
            "https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32024R1689",
        ],
        eli_url="https://eur-lex.europa.eu/eli/reg/2024/1689/oj/eng",
    ),
    Source(
        label="GDPR (Regulation (EU) 2016/679)",
        short_name="gdpr.pdf",
        accepted_names=["gdpr.pdf", "OJ_L_2016_119_EN_TXT.pdf"],
        urls=[
            "https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:02016R0679-20160504",
            "https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32016R0679",
        ],
        eli_url="https://eur-lex.europa.eu/eli/reg/2016/679/oj/eng",
    ),
]

UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)


def _is_present(raw_dir: Path, names: list[str]) -> Path | None:
    for n in names:
        p = raw_dir / n
        if p.exists() and p.stat().st_size > 100_000:
            return p
    return None


def _try_download(client: httpx.Client, url: str, dest: Path) -> bool:
    try:
        with client.stream("GET", url, follow_redirects=True, timeout=60) as r:
            ct = r.headers.get("content-type", "")
            if r.status_code != 200 or "application/pdf" not in ct:
                console.print(
                    f"  [yellow]·[/] {url}\n    → status={r.status_code}, "
                    f"content-type={ct or '?'}"
                )
                return False
            dest.parent.mkdir(parents=True, exist_ok=True)
            with dest.open("wb") as f:
                for chunk in r.iter_bytes(chunk_size=64 * 1024):
                    f.write(chunk)
            return True
    except httpx.HTTPError as e:
        console.print(f"  [yellow]·[/] {url}\n    → {e}")
        return False


def _print_manual_instructions(missing: list[Source], raw_dir: Path) -> None:
    console.print(
        "\n[bold red]Automated download did not succeed for "
        f"{len(missing)} file(s).[/]"
    )
    console.print(
        "EUR-Lex sometimes serves a CloudFront WAF challenge to non-browser\n"
        "clients. Please open the URL(s) below in your browser and save the\n"
        "PDF(s) into the path shown.\n"
    )
    for s in missing:
        console.print(f"[bold]{s.label}[/]")
        console.print(f"  Open in browser: [cyan]{s.eli_url}[/]")
        console.print( "  Click the [bold]'PDF'[/] link on the EUR-Lex page.")
        console.print(f"  Save as: [green]{raw_dir / s.short_name}[/]")
        console.print(
            f"  (the official OJ filename "
            f"[dim]{s.accepted_names[-1]}[/] is also accepted)\n"
        )
    console.print(
        "After saving, re-run [bold]python scripts/build_indices.py[/] to "
        "ingest the PDFs and rebuild the indices."
    )


def main() -> int:
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    headers = {
        "User-Agent": UA,
        "Accept": "application/pdf,text/html;q=0.9,*/*;q=0.1",
        "Accept-Language": "en-US,en;q=0.9",
    }

    missing: list[Source] = []
    with httpx.Client(headers=headers) as client:
        for src in SOURCES:
            existing = _is_present(raw_dir, src.accepted_names)
            if existing is not None:
                console.print(
                    f"[green]✓[/] {src.label} already present at "
                    f"[bold]{existing}[/]"
                )
                continue
            console.print(f"[bold]→[/] {src.label}")
            dest = raw_dir / src.short_name
            success = False
            for url in src.urls:
                if _try_download(client, url, dest):
                    console.print(
                        f"  [green]✓[/] saved to [bold]{dest}[/] "
                        f"({dest.stat().st_size / 1024:.0f} KB)"
                    )
                    success = True
                    break
            if not success:
                missing.append(src)

    if missing:
        _print_manual_instructions(missing, raw_dir)
        return 1
    console.print("\n[bold green]All source PDFs present.[/]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
