"""THESIS_FINAL_v2.0.md → 건국대학교 학위논문 공식 양식(.docx) 변환기.

학교 배포 양식(붙임4)의 편집용지·여백·폰트를 그대로 쓰고, 본문만 채워 넣는다.
양식 자체를 복사해 쓰므로 B5(182x257) / 여백 25mm / 휴먼명조 / 장평 97%가 보존된다.

사용법:
    python scripts/build_thesis_docx.py --lang ko
    python scripts/build_thesis_docx.py --lang en

산출물: 프로젝트 루트의 석사학위논문_국문.docx / 석사학위논문_영문.docx
"""

from __future__ import annotations

import argparse
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import docx
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Pt

# --- 규정값 (학위논문 작성 매뉴얼 2025.09.23, pp.2) --------------------------
FONT_KO = "휴먼명조"
CHAR_SCALE = 97  # 장평 97%
LINE_SPACING = 1.6  # Word 1.5~2배 규정 내
SZ_CHAPTER = 16  # 제1장
SZ_SECTION = 14  # 제1절
SZ_BODY = 11
SZ_KEYWORD = 9
SZ_NOTE = 10  # 표 아래 주석/인용

TEMPLATE_DIR = Path(r"C:\Users\taewo\Downloads\붙임4_학위별학위논문작성양석(hwp,word)")
TEMPLATES = {
    "ko": TEMPLATE_DIR / "4-5_Degree Paper Writing Form(Korean)_Master & Doctor.docx",
    "en": TEMPLATE_DIR / "4-6_Degree Paper Writing Form (English_Word)_Master.docx",
}
OUT_NAMES = {"ko": "석사학위논문_국문.docx", "en": "석사학위논문_영문.docx"}

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


# --- 마크다운 파싱 ----------------------------------------------------------
@dataclass
class Block:
    kind: str  # h | p | table | quote | code | caption | list
    text: str = ""
    level: int = 0
    rows: list[list[str]] = field(default_factory=list)
    items: list[str] = field(default_factory=list)


def parse_meta(md_path: Path) -> dict[str, str]:
    """논문 머리말의 '논문 정보'에서 제목·소속을 읽는다.

    확정되지 않은 항목(지도교수·학위명·년월)은 대괄호 플레이스홀더로 둔다.
    """
    text = md_path.read_text(encoding="utf-8")

    def grab(label: str, default: str = "") -> str:
        m = re.search(rf"\*\*{re.escape(label)}\*\*\s*:\s*(.+)", text)
        return m.group(1).strip() if m else default

    return {
        "title_ko": grab("제목(한국어)"),
        "title_en": grab("제목(영문)"),
        "grad_school": "건국대학교 정보통신대학원",
        "dept": "융합정보기술학과 인공지능전공",
        "author": "황태욱",
        "author_en": "Hwang, Taewook",
        "dept_en": "Department of Convergence Information Technology",
        "major_en": "Major in Artificial Intelligence",
        "grad_school_en": (
            "Graduate School of Information & Telecommunications,\n"
            "Konkuk University"
        ),
        "advisor": "[지도교수 성명]",
        "degree": "[학위명]",  # 예: 공학
        "date_award": "[학위수여 년월]",   # 전기 2월 / 후기 8월
        "date_submit": "[청구 년월]",      # 전기 10~11월 / 후기 4~5월
        "date_approve": "[인준 년월]",     # 전기 11~12월 / 후기 5~6월
    }


def parse_markdown(md_path: Path) -> list[Block]:
    """논문 마크다운을 블록 목록으로 변환한다.

    앞부속(제목/논문 정보/작성용 목차)은 양식이 따로 갖고 있으므로 건너뛴다.
    """
    lines = md_path.read_text(encoding="utf-8").splitlines()
    blocks: list[Block] = []
    i = 0
    started = False
    in_code = False
    code_buf: list[str] = []

    while i < len(lines):
        raw = lines[i]
        line = raw.rstrip()

        # 앞부속 건너뛰기: 국문초록부터 수록
        if not started:
            if re.match(r"^## (국문초록|제1장)", line):
                started = True
            else:
                i += 1
                continue

        if line.startswith("```"):
            if in_code:
                blocks.append(Block("code", "\n".join(code_buf)))
                code_buf, in_code = [], False
            else:
                in_code = True
            i += 1
            continue
        if in_code:
            code_buf.append(raw)
            i += 1
            continue

        if not line.strip() or line.strip() == "---":
            i += 1
            continue

        # 작업용 편집 메모(*(...)*)는 논문 본문이 아니므로 제외한다.
        if line.startswith("*(") and line.rstrip().endswith(")*"):
            i += 1
            continue

        m = re.match(r"^(#{2,4})\s+(.*)$", line)
        if m:
            blocks.append(Block("h", m.group(2).strip(), level=len(m.group(1)) - 1))
            i += 1
            continue

        # 표 캡션: **Table 4.1. ...**
        if re.match(r"^\*\*(Table|표)\s", line) and line.endswith("**"):
            blocks.append(Block("caption", line.strip("*").strip()))
            i += 1
            continue

        # 마크다운 표
        if line.lstrip().startswith("|"):
            rows: list[list[str]] = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not re.match(r"^[:\-\s]+$", "".join(cells)):  # 정렬 구분행 제외
                    rows.append(cells)
                i += 1
            if rows:
                blocks.append(Block("table", rows=rows))
            continue

        if line.startswith(">"):
            blocks.append(Block("quote", line.lstrip("> ").strip()))
            i += 1
            continue

        if re.match(r"^\s*[-*]\s+", line):
            items = []
            while i < len(lines) and re.match(r"^\s*[-*]\s+", lines[i]):
                items.append(re.sub(r"^\s*[-*]\s+", "", lines[i]).strip())
                i += 1
            blocks.append(Block("list", items=items))
            continue

        blocks.append(Block("p", line.strip()))
        i += 1

    return blocks


# --- docx 조립 도우미 -------------------------------------------------------
def _style_run(run, size: float, bold: bool = False, font: str = FONT_KO) -> None:
    """규정 서식(폰트·크기·장평 97%)을 run에 적용한다."""
    run.font.size = Pt(size)
    run.bold = bold
    rpr = run._element.get_or_add_rPr()
    fonts = rpr.find(qn("w:rFonts"))
    if fonts is None:
        fonts = rpr.makeelement(qn("w:rFonts"), {})
        rpr.insert(0, fonts)
    for attr in ("w:ascii", "w:eastAsia", "w:hAnsi", "w:cs"):
        fonts.set(qn(attr), font)
    w = rpr.find(qn("w:w"))
    if w is None:
        w = rpr.makeelement(qn("w:w"), {})
        rpr.append(w)
    w.set(qn("w:val"), str(CHAR_SCALE))


def _add_para(doc, anchor, text: str, size: float = SZ_BODY, bold: bool = False,
              align=WD_ALIGN_PARAGRAPH.JUSTIFY, outline: int | None = None,
              style_name: str | None = None, font: str = FONT_KO, indent: bool = True):
    """anchor 앞에 문단을 삽입한다. **굵게** 구간은 굵은 run으로 분리한다."""
    p = doc.add_paragraph()
    anchor.addprevious(p._p)
    if style_name:
        try:
            p.style = doc.styles[style_name]
        except KeyError:
            pass
    p.alignment = align
    pf = p.paragraph_format
    pf.line_spacing = LINE_SPACING
    pf.space_after = Pt(0)
    if indent and align == WD_ALIGN_PARAGRAPH.JUSTIFY:
        pf.first_line_indent = Pt(size)

    if outline is not None:
        ppr = p._p.get_or_add_pPr()
        lvl = ppr.makeelement(qn("w:outlineLvl"), {})
        lvl.set(qn("w:val"), str(outline))
        ppr.append(lvl)

    for chunk, is_bold in _split_bold(text):
        if not chunk:
            continue
        r = p.add_run(chunk)
        _style_run(r, size, bold or is_bold, font)
    if not p.runs:  # 빈 문단도 서식은 유지
        _style_run(p.add_run(""), size, bold, font)
    return p


def _split_bold(text: str) -> list[tuple[str, bool]]:
    """`**굵게**`와 백틱 코드를 정리해 (텍스트, 굵음) 목록으로 만든다."""
    text = text.replace("`", "")
    out: list[tuple[str, bool]] = []
    for part in re.split(r"(\*\*[^*]+\*\*)", text):
        if part.startswith("**") and part.endswith("**") and len(part) > 4:
            out.append((part[2:-2], True))
        elif part:
            out.append((part, False))
    return out


def _add_table(doc, anchor, rows: list[list[str]]) -> None:
    """마크다운 표를 Word 표로 삽입한다."""
    ncols = max(len(r) for r in rows)
    t = doc.add_table(rows=len(rows), cols=ncols)
    try:
        t.style = doc.styles["Table Grid"]
    except KeyError:
        pass
    for ri, row in enumerate(rows):
        for ci in range(ncols):
            cell = t.cell(ri, ci)
            cell.text = ""
            para = cell.paragraphs[0]
            para.alignment = WD_ALIGN_PARAGRAPH.CENTER
            para.paragraph_format.line_spacing = 1.0
            val = row[ci] if ci < len(row) else ""
            for chunk, is_bold in _split_bold(val):
                r = para.add_run(chunk)
                _style_run(r, 9, is_bold or ri == 0)
    anchor.addprevious(t._tbl)


def _set_para_text(p, text: str, size: float | None = None,
                   bold: bool | None = None) -> None:
    """문단의 서식은 유지한 채 글자만 바꾼다."""
    runs = p.runs
    if not runs:
        r = p.add_run(text)
        _style_run(r, size or SZ_BODY, bool(bold))
        return
    runs[0].text = text
    for r in runs[1:]:
        r._element.getparent().remove(r._element)
    if size is not None:
        runs[0].font.size = Pt(size)
    if bold is not None:
        runs[0].bold = bold


def _fill_toc_paragraph(p, instr: str) -> None:
    """빈 문단을 목차 필드로 채운다."""
    p.paragraph_format.line_spacing = LINE_SPACING
    r = p.add_run()
    _style_run(r, SZ_BODY)
    fld_begin = r._element.makeelement(qn("w:fldChar"), {})
    fld_begin.set(qn("w:fldCharType"), "begin")
    r._element.append(fld_begin)

    r2 = p.add_run()
    _style_run(r2, SZ_BODY)
    itxt = r2._element.makeelement(qn("w:instrText"), {})
    itxt.set(qn("xml:space"), "preserve")
    itxt.text = instr
    r2._element.append(itxt)

    r3 = p.add_run()
    _style_run(r3, SZ_BODY)
    sep = r3._element.makeelement(qn("w:fldChar"), {})
    sep.set(qn("w:fldCharType"), "separate")
    r3._element.append(sep)

    r4 = p.add_run("[여기서 F9를 눌러 목차를 갱신하세요]")
    _style_run(r4, SZ_BODY)

    r5 = p.add_run()
    _style_run(r5, SZ_BODY)
    end = r5._element.makeelement(qn("w:fldChar"), {})
    end.set(qn("w:fldCharType"), "end")
    r5._element.append(end)


def _add_toc_field(doc, anchor, instr: str) -> None:
    """anchor 앞에 목차 필드 문단을 만든다. Word/한글에서 F9로 갱신된다."""
    p = doc.add_paragraph()
    anchor.addprevious(p._p)
    _fill_toc_paragraph(p, instr)


def fill_front_matter(doc, meta: dict[str, str]) -> None:
    """속표지·청구지(T000), 인준지(T001), 목차(T002)를 채운다."""
    cover, approval, toc = doc.tables[0], doc.tables[1], doc.tables[2]

    # 속표지 (r0 머리 / r1 제목·수여년월 / r2 소속)
    c = cover.cell(0, 0)
    _set_para_text(c.paragraphs[0], "석사학위 청구논문", 14)
    _set_para_text(c.paragraphs[1], f"지도교수 {meta['advisor']}", 14)
    c = cover.cell(1, 0)
    _set_para_text(c.paragraphs[0], meta["title_ko"], 20, True)
    _set_para_text(c.paragraphs[1], "", 16)
    _set_para_text(c.paragraphs[9], meta["date_award"], 14)
    c = cover.cell(2, 0)
    _set_para_text(c.paragraphs[0], meta["grad_school"], 18)
    _set_para_text(c.paragraphs[1], meta["dept"], 16)
    _set_para_text(c.paragraphs[2], meta["author"], 18)

    # 청구지 (r3 제목 / r5 제출문 / r6 청구년월 / r7 대학원 / r8 학과·성명)
    c = cover.cell(3, 0)
    _set_para_text(c.paragraphs[0], meta["title_ko"], 20, True)
    _set_para_text(c.paragraphs[1], "", 16)
    _set_para_text(c.paragraphs[2], meta["title_en"], 16)
    _set_para_text(c.paragraphs[3], "", 14)
    _set_para_text(
        cover.cell(5, 0).paragraphs[0],
        f"이 논문을 {meta['degree']}석사학위 청구논문으로 제출합니다.", 16)
    _set_para_text(cover.cell(6, 0).paragraphs[0], meta["date_submit"], 16)
    _set_para_text(cover.cell(7, 0).paragraphs[0], meta["grad_school"], 18)
    c = cover.cell(8, 0)
    _set_para_text(c.paragraphs[0], meta["dept"], 16)
    _set_para_text(c.paragraphs[1], meta["author"], 18)

    # 인준지 — 석사는 심사위원장 1인 + 심사위원 2인(양식의 여분 2줄은 비움)
    _set_para_text(
        approval.cell(0, 0).paragraphs[1],
        f"{meta['author']}의 {meta['degree']}석사학위 청구논문을 인준함.", 16)
    _set_para_text(approval.cell(7, 0).paragraphs[0], meta["date_approve"], 16)
    _set_para_text(approval.cell(8, 0).paragraphs[0], meta["grad_school"], 18)
    for row in (4, 5):
        for p in approval.cell(row, 0).paragraphs:
            _set_para_text(p, "")

    # 목차 → 자동 목차 필드
    cell = toc.cell(0, 0)
    for p in list(cell.paragraphs)[1:]:
        p._element.getparent().remove(p._element)
    _set_para_text(cell.paragraphs[0], "목    차", SZ_CHAPTER, True)
    cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
    for instr in (r' TOC \o "1-3" \h \z \u ',):
        p = cell.add_paragraph()
        _fill_toc_paragraph(p, instr)


def _ensure_caption_style(doc):
    """표 캡션 전용 스타일(표목차 수집용)을 만든다."""
    name = "표캡션"
    if name in [s.name for s in doc.styles]:
        return name
    from docx.enum.style import WD_STYLE_TYPE

    st = doc.styles.add_style(name, WD_STYLE_TYPE.PARAGRAPH)
    st.base_style = doc.styles["Normal"]
    st.font.size = Pt(SZ_BODY)
    st.font.bold = True
    st.font.name = FONT_KO
    return name


def _clear_between(body, start_idx: int, end_idx: int) -> None:
    """양식의 예시 문단을 제거한다 (start_idx 이상 end_idx 미만)."""
    children = list(body.iterchildren())
    for el in children[start_idx:end_idx]:
        body.remove(el)


def _find_index(body, predicate) -> int:
    for idx, el in enumerate(body.iterchildren()):
        if predicate(el):
            return idx
    return -1


def _has_sectpr(el) -> bool:
    if el.tag != f"{W}p":
        return False
    ppr = el.find(qn("w:pPr"))
    return ppr is not None and ppr.find(qn("w:sectPr")) is not None


def build(lang: str, md_path: Path, out_path: Path) -> None:
    tpl = TEMPLATES[lang]
    shutil.copy(tpl, out_path)
    doc = docx.Document(str(out_path))
    body = doc.element.body
    _ensure_caption_style(doc)

    blocks = parse_markdown(md_path)
    if lang == "ko":
        fill_front_matter(doc, parse_meta(md_path))

    # 섹션 경계 문단 3개(속표지·청구지·목차 뒤)는 보존한다.
    sect_paras = [i for i, el in enumerate(body.iterchildren()) if _has_sectpr(el)]
    if len(sect_paras) < 4:
        raise SystemExit(f"양식 섹션 구조가 예상과 다릅니다 (경계 {len(sect_paras)}개)")

    front_start = sect_paras[2] + 1   # 표목차 시작
    front_end = sect_paras[3]         # 국문초록 끝(섹션 경계 직전)
    _clear_between(body, front_start, front_end)

    children = list(body.iterchildren())
    anchor_front = children[front_start]  # 섹션 경계 문단

    # --- 앞부속: 표목차 + 초록 -------------------------------------------
    _add_para(doc, anchor_front, "표 목차", SZ_CHAPTER, True,
              WD_ALIGN_PARAGRAPH.CENTER, indent=False)
    _add_para(doc, anchor_front, "", SZ_BODY, indent=False)
    _add_toc_field(doc, anchor_front, r' TOC \h \z \t "표캡션,1" ')
    _add_para(doc, anchor_front, "", SZ_BODY, indent=False)

    abstract_blocks = _slice(blocks, "국문초록")
    if abstract_blocks:
        _add_para(doc, anchor_front, "국문초록", SZ_SECTION, True,
                  WD_ALIGN_PARAGRAPH.LEFT, indent=False)
        _add_para(doc, anchor_front, "", SZ_BODY, indent=False)
        _emit(doc, anchor_front, abstract_blocks, skip_heading=True)

    # --- 본문 이후 전부 교체 ---------------------------------------------
    # 마지막 섹션 경계 문단 다음부터 문서 끝(final sectPr 제외)까지 제거
    boundaries = [i for i, el in enumerate(body.iterchildren()) if _has_sectpr(el)]
    last_boundary = boundaries[-1]
    final_sectpr_idx = len(list(body.iterchildren())) - 1
    _clear_between(body, last_boundary + 1, final_sectpr_idx)

    anchor_tail = list(body.iterchildren())[-1]  # final sectPr

    main_blocks = [b for b in blocks if b not in abstract_blocks]
    _emit(doc, anchor_tail, main_blocks)

    doc.save(str(out_path))


def _slice(blocks: list[Block], heading: str) -> list[Block]:
    """지정 제목 아래 블록만 잘라낸다."""
    out, on = [], False
    for b in blocks:
        if b.kind == "h" and b.level == 1:
            on = b.text.strip() == heading
            if on:
                out.append(b)
            continue
        if on:
            out.append(b)
    return out


def _emit(doc, anchor, blocks: list[Block], skip_heading: bool = False) -> None:
    for b in blocks:
        if b.kind == "h":
            if skip_heading and b.level == 1:
                continue
            if b.level == 1:  # 제1장 / 참고문헌 / 부록 / ABSTRACT
                _add_para(doc, anchor, "", SZ_BODY, indent=False)
                _add_para(doc, anchor, b.text, SZ_CHAPTER, True,
                          WD_ALIGN_PARAGRAPH.CENTER, outline=0, indent=False)
                _add_para(doc, anchor, "", SZ_BODY, indent=False)
            elif b.level == 2:
                _add_para(doc, anchor, b.text, SZ_SECTION, True,
                          WD_ALIGN_PARAGRAPH.LEFT, outline=1, indent=False)
            else:
                _add_para(doc, anchor, b.text, SZ_BODY, True,
                          WD_ALIGN_PARAGRAPH.LEFT, outline=2, indent=False)
        elif b.kind == "caption":
            _add_para(doc, anchor, b.text, SZ_BODY, True,
                      WD_ALIGN_PARAGRAPH.CENTER, style_name="표캡션", indent=False)
        elif b.kind == "table":
            _add_table(doc, anchor, b.rows)
            _add_para(doc, anchor, "", SZ_BODY, indent=False)
        elif b.kind == "quote":
            _add_para(doc, anchor, b.text, SZ_NOTE, False,
                      WD_ALIGN_PARAGRAPH.LEFT, indent=False)
        elif b.kind == "code":
            for ln in b.text.splitlines():
                _add_para(doc, anchor, ln or " ", SZ_KEYWORD, False,
                          WD_ALIGN_PARAGRAPH.LEFT, font="굴림체", indent=False)
        elif b.kind == "list":
            for it in b.items:
                _add_para(doc, anchor, f"· {it}", SZ_BODY, False,
                          WD_ALIGN_PARAGRAPH.JUSTIFY, indent=False)
        else:
            _add_para(doc, anchor, b.text, SZ_BODY)


def main() -> None:
    ap = argparse.ArgumentParser(description="학위논문 마크다운 → 공식 양식 docx")
    ap.add_argument("--lang", choices=["ko", "en"], default="ko")
    ap.add_argument("--md", default="docs/THESIS_FINAL_v2.0.md")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    md_path = Path(args.md)
    out_path = Path(args.out) if args.out else Path(OUT_NAMES[args.lang])
    build(args.lang, md_path, out_path)
    print(f"생성 완료: {out_path}")


if __name__ == "__main__":
    main()
