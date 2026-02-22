#!/usr/bin/env python3
"""
Extract product title lines from long scanned receipts using Tesseract.

Designed for low-resource environments (e.g., Raspberry Pi):
- Processes long pages in vertical strips
- Reconstructs OCR words into lines
- Deduplicates overlap artifacts
- Filters non-item lines (totals/payment/footer/etc.)

Examples:
  python tesseract_receipt_titles.py --input pdf/target-sale.pdf
  python tesseract_receipt_titles.py --input pdf/target-sale.pdf --dpi 170 --strip-height 1200 --strip-overlap 100
  python tesseract_receipt_titles.py --input receipt.jpg --output titles.txt --print-debug

Recommended settings (Target-style long receipt layout):
  Best quality:
    python tesseract_receipt_titles.py --input pdf/target-sale.pdf --dpi 300 --preprocess light --strip-height 1200 --strip-overlap 100 --min-conf 45 --output titles.txt

  Faster Raspberry Pi profile (slightly lower recall):
    python tesseract_receipt_titles.py --input pdf/target-sale.pdf --dpi 240 --preprocess light --strip-height 1200 --strip-overlap 100 --min-conf 45 --output titles.txt

Notes:
  - Keep --preprocess light for this layout; --binary can clip weak characters.
  - Keep striping enabled for long pages to limit memory usage.
  - If debugging missed text, use --raw-only or --print-raw-lines.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import statistics
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from PIL import Image, ImageFilter, ImageOps


@dataclass
class OcrLine:
    page: int
    text: str
    conf: float
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def yc(self) -> float:
        return (self.y0 + self.y1) / 2.0

    @property
    def h(self) -> float:
        return max(1.0, self.y1 - self.y0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract product title lines from scanned receipts with Tesseract."
    )
    parser.add_argument("--input", required=True, help="Path to PDF or image.")
    parser.add_argument(
        "--start-page", type=int, default=0, help="Start page index for PDFs (0-based)."
    )
    parser.add_argument(
        "--max-pages", type=int, default=1, help="Maximum number of pages to process."
    )
    parser.add_argument("--dpi", type=int, default=170, help="Render DPI for PDF pages.")
    parser.add_argument(
        "--strip-height",
        type=int,
        default=1200,
        help="Vertical strip height in pixels for OCR chunking.",
    )
    parser.add_argument(
        "--strip-overlap",
        type=int,
        default=100,
        help="Vertical overlap between adjacent strips in pixels.",
    )
    parser.add_argument(
        "--psm", type=int, default=6, help="Tesseract page segmentation mode."
    )
    parser.add_argument("--lang", default="eng", help="Tesseract language code.")
    parser.add_argument(
        "--min-conf",
        type=float,
        default=45.0,
        help="Minimum word confidence (0-100) to keep.",
    )
    parser.add_argument(
        "--min-conf-tail",
        type=float,
        default=30.0,
        help="Lower confidence floor for size/unit tail tokens (default: 30).",
    )
    parser.add_argument(
        "--output", default=None, help="Optional output path for plain-text titles."
    )
    parser.add_argument(
        "--print-debug",
        action="store_true",
        help="Print debug stats while processing.",
    )
    parser.add_argument(
        "--preprocess",
        choices=["none", "light", "binary"],
        default="none",
        help="Image preprocessing mode before OCR (default: none).",
    )
    parser.add_argument(
        "--print-raw-lines",
        action="store_true",
        help="Print raw OCR lines (post-strip merge, pre-filter).",
    )
    parser.add_argument(
        "--raw-output",
        default=None,
        help="Optional path to write raw OCR lines (pre-filter).",
    )
    parser.add_argument(
        "--raw-only",
        action="store_true",
        help="Output only raw OCR lines and skip title filtering.",
    )
    parser.add_argument(
        "--title-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Return only product title text (drop size/brand suffixes). Default: off.",
    )
    return parser.parse_args()


def require_tesseract() -> None:
    if shutil.which("tesseract"):
        return
    raise SystemExit(
        "tesseract binary not found on PATH.\n"
        "Install it first (Raspberry Pi OS): sudo apt-get install tesseract-ocr"
    )


def require_pytesseract() -> None:
    try:
        import pytesseract  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "pytesseract is not installed in this Python environment.\n"
            "Install it with: pip install pytesseract"
        ) from exc


def load_pdf_page(path: str, page_index: int, dpi: int) -> Image.Image:
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(path)
    page_count = len(pdf)
    if page_index < 0 or page_index >= page_count:
        raise ValueError(
            f"Invalid page index {page_index}. PDF has {page_count} page(s)."
        )
    # PDFium scale is relative to 72 DPI.
    scale = max(1.0, dpi / 72.0)
    page = pdf[page_index]
    return page.render(scale=scale).to_pil().convert("RGB")


def get_pdf_page_count(path: str) -> int:
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(path)
    return len(pdf)


def split_vertical_strips(
    image_np: np.ndarray, strip_height: int, overlap: int
) -> list[tuple[int, np.ndarray]]:
    h = int(image_np.shape[0])
    if strip_height <= 0 or strip_height >= h:
        return [(0, image_np)]

    overlap = max(0, min(overlap, strip_height - 1))
    step = max(1, strip_height - overlap)

    strips = []
    y = 0
    while y < h:
        y_end = min(h, y + strip_height)
        strips.append((y, image_np[y:y_end, :, :]))
        if y_end == h:
            break
        y += step
    return strips


def otsu_threshold(gray: np.ndarray) -> int:
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    total = gray.size
    if total == 0:
        return 127

    sum_total = float(np.dot(np.arange(256), hist))
    sum_bg = 0.0
    w_bg = 0.0
    max_var = -1.0
    thresh = 127

    for t in range(256):
        w_bg += hist[t]
        if w_bg == 0:
            continue
        w_fg = total - w_bg
        if w_fg == 0:
            break
        sum_bg += t * hist[t]
        m_bg = sum_bg / w_bg
        m_fg = (sum_total - sum_bg) / w_fg
        between = w_bg * w_fg * (m_bg - m_fg) ** 2
        if between > max_var:
            max_var = between
            thresh = t
    return int(thresh)


def preprocess_strip(strip_np: np.ndarray, mode: str) -> np.ndarray:
    image = Image.fromarray(strip_np).convert("L")
    if mode == "none":
        return np.array(image)

    image = ImageOps.autocontrast(image, cutoff=0)
    if mode == "light":
        return np.array(image)

    # "binary" mode is strongest and can improve some low-contrast scans,
    # but may clip weak characters.
    image = image.filter(ImageFilter.MedianFilter(size=3))

    arr = np.array(image)
    t = otsu_threshold(arr)
    bw = np.where(arr > t, 255, 0).astype(np.uint8)
    return bw


def safe_float(x: str, default: float = -1.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def clean_common_ocr_noise(text: str) -> str:
    t = normalize_ws(text)
    # Remove common trailing noise artifacts seen on receipts.
    t = re.sub(r"\b(?:qty|oty)\b.*$", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\bamount\b.*$", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\bdiscounts?\b.*$", "", t, flags=re.IGNORECASE)
    t = re.sub(r"[|]+", " ", t)
    # Common OCR confusion on receipts: "oz" read as "0z" (e.g. 20oz -> 200z).
    t = re.sub(r"(?i)\b(\d+(?:\.\d+)?)0z\b", r"\1oz", t)
    # Another common OCR confusion: "oz" read as "2/" before count (e.g. 13.40z/10ct -> 13.402/10ct).
    t = re.sub(r"(?i)\b(\d+(?:\.\d+)?)2/(\d+\s*ct)\b", r"\1oz/\2", t)
    t = normalize_ws(t)
    return t


def join_tokens_raw(tokens: list[str]) -> str:
    text = " ".join(tokens)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"([$])\s+(\d)", r"\1\2", text)
    return normalize_ws(text)


def looks_like_size_tail_token(text: str) -> bool:
    t = normalize_ws(text).lower()
    if not t or not any(ch.isdigit() for ch in t):
        return False
    compact = t.replace(" ", "")
    if re.search(r"(oz|0z|ct|lb|lbs|gal|kg|g)", compact):
        return True
    # Handle OCR variants like "...2/10ct" for "...oz/10ct"
    if re.search(r"\d2/\d+ct\b", compact):
        return True
    return False


def ocr_strip_to_lines(
    strip_bw: np.ndarray,
    y_offset: int,
    page_index: int,
    min_conf: float,
    min_conf_tail: float,
    psm: int,
    lang: str,
) -> list[OcrLine]:
    import pytesseract

    config = f"--oem 1 --psm {psm}"
    data = pytesseract.image_to_data(
        strip_bw, lang=lang, config=config, output_type=pytesseract.Output.DICT
    )

    rows: dict[tuple[int, int, int, int], list[dict[str, float | str]]] = defaultdict(list)
    n = len(data.get("text", []))
    for i in range(n):
        txt = normalize_ws(data["text"][i])
        if not txt:
            continue
        conf = safe_float(data["conf"][i], -1.0)
        if conf < min_conf:
            if conf < min_conf_tail or not looks_like_size_tail_token(txt):
                continue

        x = float(data["left"][i])
        y = float(data["top"][i]) + y_offset
        w = float(data["width"][i])
        h = float(data["height"][i])
        if w <= 0 or h <= 0:
            continue

        key = (
            int(data["block_num"][i]),
            int(data["par_num"][i]),
            int(data["line_num"][i]),
            int(data["page_num"][i]),
        )
        rows[key].append(
            {"text": txt, "conf": conf, "x0": x, "y0": y, "x1": x + w, "y1": y + h}
        )

    out: list[OcrLine] = []
    for items in rows.values():
        items.sort(key=lambda it: it["x0"])
        tokens = [str(it["text"]) for it in items]
        confs = [float(it["conf"]) for it in items]
        line = OcrLine(
            page=page_index,
            text=join_tokens_raw(tokens),
            conf=sum(confs) / len(confs),
            x0=min(float(it["x0"]) for it in items),
            y0=min(float(it["y0"]) for it in items),
            x1=max(float(it["x1"]) for it in items),
            y1=max(float(it["y1"]) for it in items),
        )
        if line.text:
            out.append(line)

    out.sort(key=lambda l: (l.yc, l.x0))
    return out


def normalize_for_dedupe(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return normalize_ws(text)


def dedupe_overlap_lines(lines: Iterable[OcrLine], y_tol: float = 18.0) -> list[OcrLine]:
    grouped: dict[str, list[OcrLine]] = defaultdict(list)
    kept: list[OcrLine] = []

    for line in sorted(lines, key=lambda l: (l.page, l.yc, l.x0)):
        key = normalize_for_dedupe(line.text)
        if not key:
            continue
        dup = False
        for prev in grouped[key]:
            if prev.page != line.page:
                continue
            if abs(prev.yc - line.yc) <= y_tol:
                dup = True
                break
        if not dup:
            kept.append(line)
            grouped[key].append(line)
    return kept


NON_ITEM_PATTERNS = [
    r"\bpicked up\b",
    r"\bhide details\b",
    r"\bsubtotal\b",
    r"\btotal\b",
    r"\bamount\b",
    r"\btax\b",
    r"\bfees?\b",
    r"\bpayment\b",
    r"\bbalance\s+due\b",
    r"\bchange\b",
    r"\bcash\b",
    r"\bdebit\b",
    r"\bcredit\b",
    r"\bvisa\b",
    r"\bmastercard\b",
    r"\bamex\b",
    r"\bunit\s*price\b",
    r"\bprice\b",
    r"\bqty\b",
    r"\bquantity\b",
    r"\bdiscount\b",
    r"\bsavings?\b",
    r"\bthank you\b",
    r"\bstore\b",
    r"\btarget\s*circle\b",
    r"\border\b",
    r"\breceipt\b",
    r"\binvoice\b",
    r"\bcard\b",
    r"\b(?:mon|tue|wed|thu|fri|sat|sun),\s",
    r"\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
]


def looks_like_non_item(text: str) -> bool:
    t = text.lower()
    for pat in NON_ITEM_PATTERNS:
        if re.search(pat, t):
            return True
    if re.search(r"\b\d{1,2}[:/]\d{1,2}([:/]\d{2,4})?\b", t):
        return True
    if re.search(r"\b\d{5}(-\d{4})?\b", t):  # zip-like
        return True
    return False


def likely_item_title(line: OcrLine) -> bool:
    text = clean_common_ocr_noise(line.text)
    if len(text) < 2:
        return False
    if not re.search(r"[A-Za-z]", text):
        return False
    if looks_like_non_item(text):
        return False
    if len(text.split()) < 2:
        return False
    if re.search(r"^[^A-Za-z]*[&|]+[^A-Za-z]*$", text):
        return False
    # Filter short low-confidence garbage lines (e.g. "larget Ulrcie").
    if line.conf < 78:
        words = re.findall(r"[A-Za-z]+", text)
        if len(words) <= 2 and "-" not in text and not re.search(r"\d", text):
            return False
    alpha_tokens = re.findall(r"[A-Za-z]+", text)
    if alpha_tokens:
        long_alpha_tokens = [tok for tok in alpha_tokens if len(tok) >= 3]
        if len(long_alpha_tokens) / len(alpha_tokens) < 0.5:
            return False

    letters = len(re.findall(r"[A-Za-z]", text))
    digits = len(re.findall(r"\d", text))
    if letters == 0:
        return False
    if digits > letters * 1.2:
        return False

    if re.fullmatch(r"[-$0-9., ]+", text):
        return False
    if re.search(r"\$\s*\d", text):
        # Product title lines usually don't include explicit currency values.
        return False

    return True


def to_product_title(text: str) -> str:
    t = clean_common_ocr_noise(text)
    # Most receipt lines are "title - size - brand"; keep only the title prefix.
    parts = [p.strip(" -") for p in re.split(r"\s+-\s+", t) if p.strip(" -")]
    if parts:
        t = parts[0]
    t = re.sub(r"\s{2,}", " ", t).strip(" -")
    return t


def is_orphan_fragment(text: str) -> bool:
    t = normalize_ws(text)
    if not t:
        return True
    if t.startswith("&"):
        return True
    alpha_words = re.findall(r"[A-Za-z]{2,}", t.lower())
    if len(alpha_words) < 2:
        return True
    fragment_vocab = {
        "gather",
        "good",
        "tees",
        "detail",
        "details",
        "amount",
        "discounts",
    }
    if len(alpha_words) <= 2 and any(w in fragment_vocab for w in alpha_words):
        return True
    return False


def stitch_ampersand_continuations(lines: list[str]) -> list[str]:
    if not lines:
        return lines
    out: list[str] = []
    for line in lines:
        t = normalize_ws(line)
        if t.startswith("&") and out:
            prev = out[-1]
            prev_norm = prev.lower().strip()
            # Generic structural continuation: only stitch when prior line
            # clearly indicates a broken "& ..." tail.
            if prev_norm.endswith("&"):
                merged = normalize_ws(prev.rstrip(" -") + " " + t)
                merged = re.sub(r"&\s*&\s*", "& ", merged)
                out[-1] = normalize_ws(merged)
                continue
            out.append(t)
        else:
            out.append(t)
    return out


def should_merge(prev: OcrLine, curr: OcrLine) -> bool:
    y_gap = curr.y0 - prev.y1
    if y_gap < 0 or y_gap > max(28.0, prev.h * 1.4):
        return False
    if abs(curr.x0 - prev.x0) > 28:
        return False
    if re.search(r"\$\s*\d", curr.text):
        return False

    prev_text = clean_common_ocr_noise(prev.text)
    curr_text = clean_common_ocr_noise(curr.text)

    # If current line clearly contains item-size tail, prefer merging.
    has_size_tail = bool(
        re.search(r"(?i)\b\d+(?:\.\d+)?\s*(?:oz|ct|lb|lbs|gal)\b", curr_text)
    )
    prev_has_size = bool(
        re.search(r"(?i)\b\d+(?:\.\d+)?\s*(?:oz|ct|lb|lbs|gal)\b", prev_text)
    )
    if has_size_tail and not prev_has_size:
        return True

    # Default heuristic: continuation tends to be shorter than a full title line.
    if len(curr_text) > 26 and len(prev_text) > 20:
        return False
    return True


def merge_wrapped_lines(lines: list[OcrLine]) -> list[OcrLine]:
    if not lines:
        return lines

    merged: list[OcrLine] = []
    for line in sorted(lines, key=lambda l: (l.page, l.y0, l.x0)):
        if merged and merged[-1].page == line.page and should_merge(merged[-1], line):
            prev = merged[-1]
            merged[-1] = OcrLine(
                page=prev.page,
                text=normalize_ws(prev.text + " " + line.text),
                conf=(prev.conf + line.conf) / 2.0,
                x0=min(prev.x0, line.x0),
                y0=min(prev.y0, line.y0),
                x1=max(prev.x1, line.x1),
                y1=max(prev.y1, line.y1),
            )
        else:
            merged.append(line)
    return merged


def extract_lines_from_image(
    image: Image.Image,
    page_index: int,
    strip_height: int,
    strip_overlap: int,
    min_conf: float,
    min_conf_tail: float,
    psm: int,
    lang: str,
    preprocess: str,
    print_debug: bool,
) -> tuple[list[OcrLine], list[OcrLine]]:
    page_np = np.array(image.convert("RGB"))
    strips = split_vertical_strips(page_np, strip_height, strip_overlap)
    all_lines: list[OcrLine] = []

    for i, (y_offset, strip_np) in enumerate(strips):
        bw = preprocess_strip(strip_np, mode=preprocess)
        lines = ocr_strip_to_lines(
            bw,
            y_offset=y_offset,
            page_index=page_index,
            min_conf=min_conf,
            min_conf_tail=min_conf_tail,
            psm=psm,
            lang=lang,
        )
        all_lines.extend(lines)
        if print_debug:
            print(
                f"[debug] page={page_index} strip={i+1}/{len(strips)} y={y_offset} "
                f"lines={len(lines)}"
            )

    deduped = dedupe_overlap_lines(all_lines, y_tol=18.0)
    # Merge first so split continuation lines (e.g. trailing "120z/8ct")
    # can be joined before title filtering.
    merged_all = merge_wrapped_lines(deduped)
    raw_merged = merged_all
    filtered_merged = [l for l in merged_all if likely_item_title(l)]
    return raw_merged, filtered_merged


def run(args: argparse.Namespace) -> list[str]:
    require_tesseract()
    require_pytesseract()

    ext = os.path.splitext(args.input.lower())[1]
    is_pdf = ext == ".pdf"
    titles: list[OcrLine] = []
    raw_lines: list[OcrLine] = []

    if is_pdf:
        total_pages = get_pdf_page_count(args.input)
        start = max(0, args.start_page)
        end = min(total_pages, start + max(0, args.max_pages))
        if start >= end:
            raise SystemExit(
                f"No pages selected. total_pages={total_pages}, start={start}, max_pages={args.max_pages}"
            )
        for page_index in range(start, end):
            image = load_pdf_page(args.input, page_index, args.dpi)
            page_raw, page_titles = extract_lines_from_image(
                image=image,
                page_index=page_index,
                strip_height=args.strip_height,
                strip_overlap=args.strip_overlap,
                min_conf=args.min_conf,
                min_conf_tail=args.min_conf_tail,
                psm=args.psm,
                lang=args.lang,
                preprocess=args.preprocess,
                print_debug=args.print_debug,
            )
            raw_lines.extend(page_raw)
            if args.print_debug:
                confs = [t.conf for t in page_titles]
                median_conf = statistics.median(confs) if confs else 0.0
                print(
                    f"[debug] page={page_index} title_candidates={len(page_titles)} "
                    f"median_conf={median_conf:.1f}"
                )
            titles.extend(page_titles)
    else:
        image = Image.open(args.input).convert("RGB")
        raw_lines, titles = extract_lines_from_image(
            image=image,
            page_index=0,
            strip_height=args.strip_height,
            strip_overlap=args.strip_overlap,
            min_conf=args.min_conf,
            min_conf_tail=args.min_conf_tail,
            psm=args.psm,
            lang=args.lang,
            preprocess=args.preprocess,
            print_debug=args.print_debug,
        )

    raw_out_lines = [normalize_ws(t.text) for t in raw_lines if normalize_ws(t.text)]
    if args.print_raw_lines:
        for line in raw_out_lines:
            print(f"[raw] {line}")
    if args.raw_output:
        with open(args.raw_output, "w", encoding="utf-8") as f:
            for line in raw_out_lines:
                f.write(line + "\n")
    if args.raw_only:
        return raw_out_lines

    if args.title_only:
        out_lines = [to_product_title(t.text) for t in titles if normalize_ws(t.text)]
    else:
        out_lines = [clean_common_ocr_noise(t.text) for t in titles if normalize_ws(t.text)]
        out_lines = stitch_ampersand_continuations(out_lines)

    # Drop residual non-item lines after title extraction.
    out_lines = [
        line
        for line in out_lines
        if line and not looks_like_non_item(line) and not is_orphan_fragment(line)
    ]
    # Final dedupe after merge/filter pipeline.
    seen = set()
    final: list[str] = []
    for line in out_lines:
        key = normalize_for_dedupe(line)
        if key and key not in seen:
            seen.add(key)
            final.append(line)
    return final


def main() -> None:
    args = parse_args()
    results = run(args)

    for line in results:
        print(line)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            for line in results:
                f.write(line + "\n")


if __name__ == "__main__":
    main()
