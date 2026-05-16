"""마크다운 → PDF 변환 스크립트 (Chrome headless 사용).

교수님 제출본(`정보통신대학원_황태욱_석사학위논문설계서.pdf`)과 유사한
양식의 PDF를 생성한다. Python markdown 라이브러리로 HTML 변환 후,
Chrome headless 모드로 PDF 출력.

사용법:
    # 기본 (v0.5 마크다운 → 황태욱_석사학위논문설계서_v1.1_2026-05-16.pdf)
    python scripts/build_pdf.py

    # 다른 입력/출력
    python scripts/build_pdf.py \\
        --input docs/THESIS_PROPOSAL_FINAL_v0.6.md \\
        --version v1.2 \\
        --date 2026-06-01

요구사항:
    - Python markdown library (uv pip install markdown)
    - Google Chrome (Windows 기본 경로 자동 탐색)
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from datetime import date as date_type
from pathlib import Path

import markdown

logger = logging.getLogger(__name__)


# 교수 제출본 양식과 유사한 CSS (한글 시스템 폰트 사용, 표/리스트 스타일)
_CSS = """
@page {
    size: A4;
    margin: 25mm 20mm;
    @bottom-center {
        content: counter(page);
        font-size: 10pt;
        color: #666;
    }
}

body {
    font-family: 'Malgun Gothic', 'Apple SD Gothic Neo', 'Noto Sans KR', sans-serif;
    font-size: 10.5pt;
    line-height: 1.55;
    color: #222;
    max-width: 170mm;
    margin: 0 auto;
}

h1 {
    font-size: 22pt;
    text-align: center;
    margin-top: 0;
    margin-bottom: 8mm;
    padding-bottom: 4mm;
    border-bottom: 2px solid #333;
}

h2 {
    font-size: 15pt;
    margin-top: 10mm;
    margin-bottom: 4mm;
    padding-bottom: 2mm;
    border-bottom: 1px solid #ccc;
}

h3 {
    font-size: 12pt;
    margin-top: 6mm;
    margin-bottom: 3mm;
    color: #2a4d7f;
}

h4 {
    font-size: 11pt;
    margin-top: 4mm;
    margin-bottom: 2mm;
    color: #444;
}

p, li {
    text-align: justify;
}

ul, ol {
    margin: 2mm 0;
    padding-left: 6mm;
}

li {
    margin-bottom: 1mm;
}

table {
    border-collapse: collapse;
    margin: 3mm 0;
    font-size: 9.5pt;
    width: 100%;
    page-break-inside: avoid;
}

th {
    background-color: #f0f0f0;
    border: 1px solid #888;
    padding: 1.5mm 2mm;
    font-weight: 600;
    text-align: left;
}

td {
    border: 1px solid #aaa;
    padding: 1.5mm 2mm;
}

code {
    background-color: #f6f8fa;
    padding: 1px 5px;
    border-radius: 3px;
    font-family: 'Consolas', 'D2Coding', monospace;
    font-size: 9.5pt;
}

pre {
    background-color: #f6f8fa;
    padding: 3mm;
    border-radius: 4px;
    overflow-x: auto;
    font-size: 9pt;
    line-height: 1.4;
    page-break-inside: avoid;
}

pre code {
    background-color: transparent;
    padding: 0;
}

blockquote {
    border-left: 3px solid #5a9;
    padding-left: 4mm;
    color: #555;
    margin: 3mm 0;
    background-color: #f7fcfa;
    padding-top: 1mm;
    padding-bottom: 1mm;
}

hr {
    border: 0;
    border-top: 1px solid #ccc;
    margin: 6mm 0;
}

strong {
    color: #1a1a1a;
}

/* 페이지 나누기 방지 */
h1, h2, h3, h4 {
    page-break-after: avoid;
}
"""


def _find_chrome() -> Path | None:
    """Windows Chrome 실행 파일 경로 자동 탐색."""
    candidates = [
        Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
        Path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
        Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
        Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def markdown_to_html(md_path: Path, version_external: str, submission_date: str) -> str:
    """마크다운 파일을 스타일이 적용된 HTML 문자열로 변환한다.

    Args:
        md_path: 마크다운 파일 경로.
        version_external: 외부 버전 라벨 (예: v1.1).
        submission_date: 제출일자 문자열 (예: 2026-05-16).

    Returns:
        완성된 HTML 문서 문자열.
    """
    md_text = md_path.read_text(encoding="utf-8")

    md = markdown.Markdown(extensions=[
        "tables",
        "fenced_code",
        "toc",
        "sane_lists",
        "attr_list",
    ])
    body_html = md.convert(md_text)

    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<title>석사학위 논문 설계서 {version_external}</title>
<style>
{_CSS}
</style>
</head>
<body>
{body_html}
</body>
</html>
"""


def html_to_pdf(html_path: Path, pdf_path: Path, chrome_path: Path) -> None:
    """Chrome headless 모드로 HTML 파일을 PDF로 변환한다.

    Args:
        html_path: 입력 HTML 절대 경로.
        pdf_path: 출력 PDF 절대 경로.
        chrome_path: Chrome 또는 Edge 실행 파일 절대 경로.
    """
    file_url = html_path.resolve().as_uri()
    cmd = [
        str(chrome_path),
        "--headless=new",
        "--disable-gpu",
        "--no-sandbox",
        "--no-pdf-header-footer",
        f"--print-to-pdf={pdf_path.resolve()}",
        file_url,
    ]
    logger.info(f"Chrome 실행: {chrome_path.name}")
    # Windows 콘솔 출력이 cp949 등으로 깨질 수 있어 errors="replace" 사용
    result = subprocess.run(
        cmd, capture_output=True, text=True,
        encoding="utf-8", errors="replace",
        timeout=120,
    )
    if result.returncode != 0:
        logger.error(f"Chrome stderr:\n{result.stderr}")
        raise RuntimeError(f"Chrome PDF 변환 실패 (exit {result.returncode})")


def build_pdf(
    md_path: Path,
    out_dir: Path,
    version_external: str,
    submission_date: str,
    author_name: str = "황태욱",
) -> Path:
    """전체 파이프라인 실행: Markdown → HTML → PDF.

    Args:
        md_path: 입력 마크다운 경로.
        out_dir: 출력 디렉토리 (PDF 저장 위치).
        version_external: 외부 버전 (예: v1.1).
        submission_date: 제출일 (YYYY-MM-DD).
        author_name: PDF 파일명에 사용할 저자명.

    Returns:
        생성된 PDF 파일 경로.
    """
    chrome = _find_chrome()
    if chrome is None:
        raise RuntimeError(
            "Chrome 또는 Edge를 찾을 수 없습니다. "
            "Windows 기본 경로에 설치되어 있는지 확인하세요."
        )

    out_dir.mkdir(parents=True, exist_ok=True)

    # HTML 중간 산출물
    html_path = out_dir / f"{author_name}_석사학위논문설계서_{version_external}_{submission_date}.html"
    pdf_path = out_dir / f"{author_name}_석사학위논문설계서_{version_external}_{submission_date}.pdf"

    logger.info(f"Markdown → HTML: {md_path.name}")
    html_content = markdown_to_html(md_path, version_external, submission_date)
    html_path.write_text(html_content, encoding="utf-8")
    logger.info(f"HTML 저장: {html_path}")

    logger.info(f"HTML → PDF: {pdf_path.name}")
    html_to_pdf(html_path, pdf_path, chrome)
    logger.info(f"PDF 저장 완료: {pdf_path}")

    return pdf_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Markdown 논문 설계서를 PDF로 변환")
    parser.add_argument(
        "--input", "-i",
        default="docs/THESIS_PROPOSAL_FINAL_v0.5.md",
        help="입력 마크다운 파일 경로",
    )
    parser.add_argument(
        "--out_dir", "-o",
        default="docs/submitted",
        help="출력 디렉토리",
    )
    parser.add_argument(
        "--version", "-v",
        default="v1.1",
        help="외부 버전 라벨 (예: v1.1)",
    )
    parser.add_argument(
        "--date", "-d",
        default=str(date_type.today()),
        help="제출일자 (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--author",
        default="황태욱",
        help="저자명 (PDF 파일명에 사용)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    md_path = Path(args.input)
    if not md_path.exists():
        logger.error(f"입력 파일이 없습니다: {md_path}")
        return 1

    out_dir = Path(args.out_dir)
    pdf_path = build_pdf(
        md_path=md_path,
        out_dir=out_dir,
        version_external=args.version,
        submission_date=args.date,
        author_name=args.author,
    )

    size_kb = pdf_path.stat().st_size / 1024
    logger.info(f"완료: {pdf_path} ({size_kb:.1f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
