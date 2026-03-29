"""Markdown to PDF converter using Python markdown + Chrome headless."""
import markdown
import subprocess
import sys
import tempfile
from pathlib import Path

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<style>
  @page {{ size: A4; margin: 2cm; }}
  body {{
    font-family: 'Malgun Gothic', 'Apple SD Gothic Neo', sans-serif;
    font-size: 11pt;
    line-height: 1.7;
    color: #222;
    max-width: 800px;
    margin: 0 auto;
    padding: 20px;
  }}
  h1 {{ font-size: 20pt; border-bottom: 2px solid #333; padding-bottom: 8px; margin-top: 30px; }}
  h2 {{ font-size: 16pt; border-bottom: 1px solid #999; padding-bottom: 5px; margin-top: 25px; }}
  h3 {{ font-size: 13pt; margin-top: 20px; }}
  h4 {{ font-size: 11pt; margin-top: 15px; }}
  table {{
    border-collapse: collapse;
    width: 100%;
    margin: 15px 0;
    font-size: 10pt;
  }}
  th, td {{
    border: 1px solid #999;
    padding: 6px 10px;
    text-align: left;
  }}
  th {{ background-color: #f0f0f0; font-weight: bold; }}
  code {{
    background-color: #f5f5f5;
    padding: 2px 5px;
    border-radius: 3px;
    font-family: 'Consolas', 'D2Coding', monospace;
    font-size: 10pt;
  }}
  pre {{
    background-color: #f5f5f5;
    padding: 12px;
    border-radius: 5px;
    overflow-x: auto;
    font-size: 9pt;
  }}
  pre code {{ background: none; padding: 0; }}
  blockquote {{
    border-left: 3px solid #999;
    padding-left: 15px;
    color: #555;
    margin: 15px 0;
  }}
  ul, ol {{ padding-left: 25px; }}
  li {{ margin-bottom: 3px; }}
  hr {{ border: none; border-top: 1px solid #ccc; margin: 20px 0; }}
  strong {{ color: #111; }}
</style>
</head>
<body>
{content}
</body>
</html>"""

CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"


def convert_md_to_pdf(md_path: Path, pdf_path: Path) -> bool:
    """마크다운 파일을 PDF로 변환"""
    md_text = md_path.read_text(encoding="utf-8")
    html_content = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "toc", "nl2br"],
    )
    full_html = HTML_TEMPLATE.format(content=html_content)

    # 임시 HTML 파일을 PDF와 같은 디렉토리에 생성 (경로 이슈 방지)
    html_path = pdf_path.with_suffix(".html")
    html_path.write_text(full_html, encoding="utf-8")

    try:
        # Windows에서 Chrome headless 호출 시 절대경로 사용
        pdf_abs = str(pdf_path.resolve())
        html_url = f"file:///{html_path.resolve().as_posix()}"
        result = subprocess.run(
            [
                CHROME_PATH,
                "--headless",
                "--disable-gpu",
                "--no-sandbox",
                f"--print-to-pdf={pdf_abs}",
                html_url,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=30,
        )
        return pdf_path.exists()
    except Exception as e:
        print(f"  ERROR: {e}")
        return False
    finally:
        html_path.unlink(missing_ok=True)


def main():
    docs_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("docs")
    md_files = sorted(docs_dir.glob("*.md"))

    if not md_files:
        print("No markdown files found.")
        return

    print(f"Converting {len(md_files)} markdown files to PDF...")
    for md_file in md_files:
        pdf_file = md_file.with_suffix(".pdf")
        print(f"  {md_file.name} -> {pdf_file.name} ...", end=" ")
        if convert_md_to_pdf(md_file, pdf_file):
            size_kb = pdf_file.stat().st_size / 1024
            print(f"OK ({size_kb:.0f} KB)")
        else:
            print("FAILED")

    print("Done.")


if __name__ == "__main__":
    main()
