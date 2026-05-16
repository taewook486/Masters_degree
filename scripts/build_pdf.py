"""마크다운 → PDF 변환 스크립트 (Chrome headless 사용).

교수님 제출본(`정보통신대학원_황태욱_석사학위논문설계서.pdf`) 양식과 일치하는
PDF를 생성한다. Python markdown 라이브러리로 HTML 변환 후,
Chrome headless 모드로 PDF 출력.

본문 내 변경 이력 표/v0.X 인용 박스/메타 마커는 자동으로 제거하여
교수 제출본 양식과 동일한 깔끔한 본문만 PDF에 포함한다.

사용법:
    # 기본 (v0.5 마크다운 → 황태욱_석사학위논문설계서_v1.1_<today>.pdf)
    python scripts/build_pdf.py

    # 버전/날짜/저자 지정
    python scripts/build_pdf.py \\
        --input docs/THESIS_PROPOSAL_FINAL_v0.6.md \\
        --version v1.2 \\
        --date 2026-06-01

    # 변경 이력/인용 박스 그대로 두기 (디버그)
    python scripts/build_pdf.py --keep-changelog

요구사항:
    - Python markdown library (uv pip install markdown)
    - Google Chrome 또는 Microsoft Edge (Windows 기본 경로 자동 탐색)
"""

from __future__ import annotations

import argparse
import logging
import re
import subprocess
import sys
from datetime import date as date_type
from pathlib import Path

import markdown

logger = logging.getLogger(__name__)


# 교수 제출본(v1.0) 양식 — 미니멀, 본문 중심
_CSS = """
@page {
    size: A4;
    margin: 25mm 22mm;
    @bottom-center {
        content: counter(page);
        font-size: 9pt;
        color: #888;
        font-family: 'Malgun Gothic', sans-serif;
    }
}

body {
    font-family: 'Malgun Gothic', 'Apple SD Gothic Neo', 'Noto Sans KR', sans-serif;
    font-size: 10.5pt;
    line-height: 1.6;
    color: #1a1a1a;
    margin: 0;
}

/* 제목: 가운데 정렬, 큰 글자, 두꺼운 가로선으로 구분 */
h1 {
    font-size: 24pt;
    font-weight: 700;
    text-align: center;
    margin: 0 0 12mm 0;
    padding-bottom: 6mm;
    border-bottom: 2px solid #1a1a1a;
    letter-spacing: -0.5pt;
}

/* 절 제목: 좌측 정렬, 굵게, 하단 얇은 선 */
h2 {
    font-size: 15pt;
    font-weight: 700;
    margin: 9mm 0 4mm 0;
    padding-bottom: 2mm;
    border-bottom: 1px solid #d0d0d0;
    color: #1a1a1a;
}

/* 소절 제목 */
h3 {
    font-size: 12pt;
    font-weight: 700;
    margin: 6mm 0 2.5mm 0;
    color: #1a1a1a;
}

h4 {
    font-size: 11pt;
    font-weight: 600;
    margin: 4mm 0 2mm 0;
    color: #2a2a2a;
}

/* 본문 */
p {
    margin: 2mm 0;
    text-align: justify;
}

/* 리스트 */
ul, ol {
    margin: 2mm 0;
    padding-left: 6mm;
}

li {
    margin-bottom: 1mm;
    line-height: 1.55;
}

/* 표 — 교수 제출본 양식: 얇은 회색 테두리, 헤더 약한 음영 */
table {
    border-collapse: collapse;
    margin: 3mm 0;
    font-size: 9.5pt;
    width: 100%;
    page-break-inside: avoid;
}

th {
    background-color: #ebebeb;
    border: 1px solid #888;
    padding: 1.8mm 2.5mm;
    font-weight: 600;
    text-align: left;
    color: #1a1a1a;
}

td {
    border: 1px solid #b0b0b0;
    padding: 1.5mm 2.5mm;
    line-height: 1.5;
}

/* 강조 */
strong, b {
    color: #1a1a1a;
    font-weight: 700;
}

/* 인라인 코드 */
code {
    background-color: #f5f5f5;
    padding: 0.5px 4px;
    border-radius: 2px;
    font-family: 'Consolas', 'D2Coding', monospace;
    font-size: 9.5pt;
    color: #c7254e;
}

/* 코드 블록 */
pre {
    background-color: #f7f7f7;
    padding: 3mm 4mm;
    border-radius: 3px;
    border-left: 3px solid #888;
    overflow-x: auto;
    font-size: 9pt;
    line-height: 1.5;
    page-break-inside: avoid;
    margin: 3mm 0;
}

pre code {
    background-color: transparent;
    padding: 0;
    color: #1a1a1a;
}

/* 인용 (남은 것이 있다면 부드럽게) */
blockquote {
    margin: 3mm 0;
    padding: 2mm 5mm;
    color: #555;
    border-left: 2px solid #b0b0b0;
    background-color: #fafafa;
    font-style: normal;
}

/* 가로선 */
hr {
    border: 0;
    border-top: 1px solid #cccccc;
    margin: 8mm 0;
}

/* 페이지 나누기 방지 */
h1, h2, h3, h4 {
    page-break-after: avoid;
    page-break-inside: avoid;
}

table, pre {
    page-break-inside: avoid;
}
"""


# 교수 제출본 양식 정리에 사용할 정규식 패턴
#
# PDF는 외부 버전(v1.X)만 표기하고, 내부 작업 버전(v0.X) 흔적은 모두 제거한다.
# 마크다운 원본은 그대로 두고 PDF 생성 시에만 적용한다.

# 1. <!-- pdf:strip-meta --> ... <!-- /pdf:strip-meta --> 블록 (버전 메타 정보)
_RE_META_BLOCK = re.compile(
    r"<!--\s*pdf:strip-meta\s*-->.*?<!--\s*/pdf:strip-meta\s*-->",
    re.DOTALL,
)

# 2. 인용(>) 줄 중 vX.Y 또는 Internal 텍스트 포함 — 변경 사유 박스, 메타 줄
_RE_VERSION_QUOTE = re.compile(
    r"^>\s*[^\n]*(?:v\d+\.\d+|Internal|External)[^\n]*(?:\n>[^\n]*)*\n?",
    re.MULTILINE,
)

# 3. 괄호 안 "v0.X 신설/강화/변경/추가" — 헤딩 또는 강조 옆 메타 마커
#    예: "(v0.5 신설)", "(간접 비교, v0.5 신설)", "통계 검증 (v0.5 강화)"
_RE_VERSION_PAREN = re.compile(
    r"\s*\([^()]*?v\d+\.\d+\s*(?:신설|강화|변경|추가)\s*\)",
)

# 4. dash/hyphen 뒤 "— v0.X 신설" 형식 — 리스트 항목 끝
#    예: "- 3.8.3 임상적 의미 분석 (WCA + ECE) — v0.5 신설"
_RE_VERSION_DASH = re.compile(
    r"\s*[—\-]+\s*v\d+\.\d+\s*(?:신설|강화|변경|추가)\s*$",
    re.MULTILINE,
)

# 5. 일반 본문 내 단순 vX.Y 언급 (단독, 콤마 또는 괄호로 둘러싸인) — 안전망
#    예: ", v0.5 변경)" 또는 부분 매칭이 남은 경우
_RE_VERSION_INLINE = re.compile(
    r"\s*,?\s*v\d+\.\d+\s*(?:신설|강화|변경|추가)",
)


def _find_chrome() -> Path | None:
    """Windows Chrome/Edge 실행 파일 경로 자동 탐색."""
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


def clean_for_pdf(md_text: str) -> str:
    """교수 제출본 양식에 맞춰 본문을 정리한다.

    PDF에는 외부 버전(v1.X)만 노출되며 내부 작업 버전(v0.X) 흔적은 모두 제거된다.
    마크다운 원본은 변경되지 않는다.

    제거 항목:
        - <!-- pdf:strip-meta --> ~ <!-- /pdf:strip-meta --> 블록
        - "> ... v0.X ... 변경" 형식의 인용 박스
        - "(v0.X 신설)" "(간접 비교, v0.5 신설)" 등 괄호 메타 마커
        - "— v0.X 신설" 같은 dash 뒤 메타 마커
        - "Internal v0.5 / External v1.1" 등 버전 라인
        - 연속된 빈 줄 정리
    """
    text = _RE_META_BLOCK.sub("", md_text)
    text = _RE_VERSION_QUOTE.sub("", text)
    text = _RE_VERSION_PAREN.sub("", text)
    text = _RE_VERSION_DASH.sub("", text)
    text = _RE_VERSION_INLINE.sub("", text)
    # 3줄 이상 연속된 빈 줄 → 2줄로
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


def markdown_to_html(md_path: Path, *, clean: bool = True) -> str:
    """마크다운 파일을 스타일이 적용된 HTML 문자열로 변환한다.

    Args:
        md_path: 마크다운 파일 경로.
        clean: True이면 본문에서 변경 이력/메타 마커 제거 (PDF 출력용).

    Returns:
        완성된 HTML 문서 문자열.
    """
    md_text = md_path.read_text(encoding="utf-8")
    if clean:
        md_text = clean_for_pdf(md_text)

    md = markdown.Markdown(extensions=[
        "tables",
        "fenced_code",
        "sane_lists",
    ])
    body_html = md.convert(md_text)

    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<title>석사학위 논문 설계서</title>
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
    """Chrome headless 모드로 HTML 파일을 PDF로 변환한다."""
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
    doc_name: str = "석사학위논문설계서",
    clean: bool = True,
) -> Path:
    """전체 파이프라인 실행: Markdown → 정리 → HTML → PDF.

    Args:
        md_path: 입력 마크다운 경로.
        out_dir: 출력 디렉토리.
        version_external: 외부 버전 (예: v1.1).
        submission_date: 제출일 (YYYY-MM-DD).
        author_name: 저자명 (파일명 앞부분).
        doc_name: 문서명 (파일명 중간 부분).
        clean: True이면 본문에서 변경 이력/메타 마커 제거.
    """
    chrome = _find_chrome()
    if chrome is None:
        raise RuntimeError(
            "Chrome 또는 Edge를 찾을 수 없습니다. "
            "Windows 기본 경로에 설치되어 있는지 확인하세요."
        )

    out_dir.mkdir(parents=True, exist_ok=True)

    base_name = f"{author_name}_{doc_name}_{version_external}_{submission_date}"
    html_path = out_dir / f"{base_name}.html"
    pdf_path = out_dir / f"{base_name}.pdf"

    logger.info(f"Markdown → HTML: {md_path.name} (clean={clean})")
    html_content = markdown_to_html(md_path, clean=clean)
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
        help="저자명 (PDF 파일명 앞부분)",
    )
    parser.add_argument(
        "--doc-name",
        default="석사학위논문설계서",
        help="문서명 (PDF 파일명 중간 부분, 예: 논문설계서변경사항보고)",
    )
    parser.add_argument(
        "--keep-changelog",
        action="store_true",
        help="변경 이력 및 v0.X 인용 박스를 그대로 유지 (디버그용)",
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
        doc_name=args.doc_name,
        clean=not args.keep_changelog,
    )

    size_kb = pdf_path.stat().st_size / 1024
    logger.info(f"완료: {pdf_path} ({size_kb:.1f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
