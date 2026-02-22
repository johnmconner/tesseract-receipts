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
  python tesseract_receipt_titles.py --input receipt.jpg --output titles.txt

Recommended settings (Target-style long receipt layout):
  Best quality:
    python tesseract_receipt_titles.py --input pdf/target-sale.pdf --dpi 300 --preprocess light --strip-height 1200 --strip-overlap 100 --min-conf 45 --output titles.txt

Notes:
  - Keep --preprocess light for this layout.
  - Keep striping enabled for long pages to limit memory usage.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from PIL import Image, ImageOps

try:
    import tomllib
except ImportError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]


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


@dataclass
class ReceiptItem:
    page: int
    title: str
    qty: str | None
    conf: float


@dataclass
class ThresholdConfig:
    dedupe_y_tol: float
    merge_y_gap_abs: float
    merge_y_gap_mult: float
    merge_x_tol: float
    continuation_word_conf_min: float
    short_noise_conf_max: float
    mixed_gibberish_conf_max: float


@dataclass
class RetailerConfig:
    non_item_patterns: list[str]
    fragment_vocab: set[str]
    thresholds: ThresholdConfig


DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parent / "configs" / "target.toml"
)
ACTIVE_CONFIG: RetailerConfig | None = None


def get_active_config() -> RetailerConfig:
    if ACTIVE_CONFIG is None:
        raise RuntimeError("Config not initialized.")
    return ACTIVE_CONFIG


def _require_list_str(data: object, key: str) -> list[str]:
    if not isinstance(data, list) or not all(isinstance(x, str) for x in data):
        raise ValueError(f"{key} must be a list of strings")
    return list(data)


def load_retailer_config(path: str) -> RetailerConfig:
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise SystemExit(f"Config file not found: {cfg_path}")

    with cfg_path.open("rb") as f:
        raw = tomllib.load(f)

    filters = raw.get("filters")
    thresholds = raw.get("thresholds")
    if not isinstance(filters, dict) or not isinstance(thresholds, dict):
        raise SystemExit(
            f"Invalid config format in {cfg_path}: expected [filters] and [thresholds] tables"
        )

    non_item_patterns = _require_list_str(
        filters.get("non_item_patterns"), "filters.non_item_patterns"
    )
    fragment_vocab = set(
        _require_list_str(filters.get("fragment_vocab"), "filters.fragment_vocab")
    )

    def _num(name: str, default: float) -> float:
        val = thresholds.get(name, default)
        if not isinstance(val, (int, float)):
            raise ValueError(f"thresholds.{name} must be numeric")
        return float(val)

    return RetailerConfig(
        non_item_patterns=non_item_patterns,
        fragment_vocab=fragment_vocab,
        thresholds=ThresholdConfig(
            dedupe_y_tol=_num("dedupe_y_tol", 18.0),
            merge_y_gap_abs=_num("merge_y_gap_abs", 28.0),
            merge_y_gap_mult=_num("merge_y_gap_mult", 1.4),
            merge_x_tol=_num("merge_x_tol", 28.0),
            continuation_word_conf_min=_num("continuation_word_conf_min", 20.0),
            short_noise_conf_max=_num("short_noise_conf_max", 78.0),
            mixed_gibberish_conf_max=_num("mixed_gibberish_conf_max", 80.0),
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract product title lines from scanned receipts with Tesseract."
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to TOML config for layout/retailer-specific filters and thresholds.",
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
        "--include-qty",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include quantity in output (default: on). Use --no-include-qty for title-only lines.",
    )
    parser.add_argument(
        "--preprocess",
        choices=["none", "light"],
        default="light",
        help="Image preprocessing mode before OCR (default: light).",
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


def preprocess_strip(strip_np: np.ndarray, mode: str) -> np.ndarray:
    image = Image.fromarray(strip_np).convert("L")
    if mode == "none":
        return np.array(image)

    image = ImageOps.autocontrast(image, cutoff=0)
    return np.array(image)


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


def looks_like_continuation_word(text: str) -> bool:
    t = normalize_ws(text)
    return bool(re.fullmatch(r"[A-Za-z][A-Za-z~]{1,20}", t))


def extract_qty_value(text: str) -> str | None:
    t = normalize_ws(text)
    m = re.search(r"(?i)\bqty\b\s*([0-9]+(?:\.[0-9]+)?)\b", t)
    if not m:
        return None
    return m.group(1)


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
        if conf < 0:
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
        has_amp = any(str(it["text"]).strip() == "&" for it in items)
        cfg = get_active_config()
        kept_items = []
        for idx, it in enumerate(items):
            txt = str(it["text"])
            conf = float(it["conf"])
            keep = conf >= min_conf
            if not keep and conf >= min_conf_tail and looks_like_size_tail_token(txt):
                keep = True

            # Rescue low-confidence continuation words on wrapped brand tails:
            # e.g., next line contains "& Gather" and "Gather" may score very low.
            if (
                not keep
                and has_amp
                and conf >= cfg.thresholds.continuation_word_conf_min
                and looks_like_continuation_word(txt)
            ):
                left_neighbor = items[idx - 1] if idx > 0 else None
                right_neighbor = items[idx + 1] if idx + 1 < len(items) else None
                neighbor_amp = False
                if left_neighbor is not None and str(left_neighbor["text"]).strip() == "&":
                    neighbor_amp = True
                if right_neighbor is not None and str(right_neighbor["text"]).strip() == "&":
                    neighbor_amp = True
                if neighbor_amp:
                    keep = True

            if keep:
                kept_items.append(it)

        if not kept_items:
            continue

        tokens = [str(it["text"]) for it in kept_items]
        confs = [float(it["conf"]) for it in kept_items]
        line = OcrLine(
            page=page_index,
            text=join_tokens_raw(tokens),
            conf=sum(confs) / len(confs),
            x0=min(float(it["x0"]) for it in kept_items),
            y0=min(float(it["y0"]) for it in kept_items),
            x1=max(float(it["x1"]) for it in kept_items),
            y1=max(float(it["y1"]) for it in kept_items),
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


def looks_like_non_item(text: str) -> bool:
    t = text.lower()
    for pat in get_active_config().non_item_patterns:
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
    cfg = get_active_config()
    if line.conf < cfg.thresholds.short_noise_conf_max:
        words = re.findall(r"[A-Za-z]+", text)
        if len(words) <= 2 and "-" not in text and not re.search(r"\d", text):
            return False
    # Filter low-confidence mixed alnum gibberish that lacks item structure.
    if line.conf < cfg.thresholds.mixed_gibberish_conf_max:
        if (
            re.search(r"\d", text)
            and "-" not in text
            and not re.search(r"(?i)\b(?:oz|ct|lb|lbs|gal)\b", text)
            and len(text) < 28
            and not re.search(r"(?i)\b(?:qty|amount|item|tax)\b", text)
        ):
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


def is_orphan_fragment(text: str) -> bool:
    t = normalize_ws(text)
    if not t:
        return True
    if t.startswith("&"):
        return True
    alpha_words = re.findall(r"[A-Za-z]{2,}", t.lower())
    if len(alpha_words) < 2:
        return True
    fragment_vocab = get_active_config().fragment_vocab
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


def stitch_ampersand_continuations_items(items: list[ReceiptItem]) -> list[ReceiptItem]:
    if not items:
        return items
    out: list[ReceiptItem] = []
    for item in items:
        t = normalize_ws(item.title)
        if t.startswith("&") and out:
            prev = out[-1]
            prev_norm = prev.title.lower().strip()
            if prev_norm.endswith("&"):
                merged = normalize_ws(prev.title.rstrip(" -") + " " + t)
                merged = re.sub(r"&\s*&\s*", "& ", merged)
                out[-1] = ReceiptItem(
                    page=prev.page,
                    title=normalize_ws(merged),
                    qty=prev.qty if prev.qty is not None else item.qty,
                    conf=(prev.conf + item.conf) / 2.0,
                )
                continue
        out.append(ReceiptItem(item.page, t, item.qty, item.conf))
    return out


def should_merge(prev: OcrLine, curr: OcrLine) -> bool:
    cfg = get_active_config()
    y_gap = curr.y0 - prev.y1
    if y_gap < 0 or y_gap > max(cfg.thresholds.merge_y_gap_abs, prev.h * cfg.thresholds.merge_y_gap_mult):
        return False
    if abs(curr.x0 - prev.x0) > cfg.thresholds.merge_x_tol:
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


def build_receipt_items(merged_all: list[OcrLine]) -> list[ReceiptItem]:
    title_idxs = [i for i, line in enumerate(merged_all) if likely_item_title(line)]
    if not title_idxs:
        return []

    items: list[ReceiptItem] = []
    for pos, idx in enumerate(title_idxs):
        line = merged_all[idx]
        next_idx = title_idxs[pos + 1] if pos + 1 < len(title_idxs) else len(merged_all)
        qty: str | None = None

        for j in range(idx + 1, min(next_idx, idx + 10)):
            q = extract_qty_value(merged_all[j].text)
            if q is not None:
                qty = q
                break

        items.append(
            ReceiptItem(page=line.page, title=line.text, qty=qty, conf=line.conf)
        )
    return items


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
) -> list[ReceiptItem]:
    page_np = np.array(image.convert("RGB"))
    strips = split_vertical_strips(page_np, strip_height, strip_overlap)
    all_lines: list[OcrLine] = []

    for y_offset, strip_np in strips:
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

    deduped = dedupe_overlap_lines(all_lines, y_tol=get_active_config().thresholds.dedupe_y_tol)
    # Merge first so split continuation lines (e.g. trailing "120z/8ct")
    # can be joined before title filtering.
    merged_all = merge_wrapped_lines(deduped)
    return build_receipt_items(merged_all)


def run(args: argparse.Namespace) -> list[str]:
    global ACTIVE_CONFIG
    ACTIVE_CONFIG = load_retailer_config(args.config)

    require_tesseract()
    require_pytesseract()

    ext = os.path.splitext(args.input.lower())[1]
    is_pdf = ext == ".pdf"
    items: list[ReceiptItem] = []

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
            page_items = extract_lines_from_image(
                image=image,
                page_index=page_index,
                strip_height=args.strip_height,
                strip_overlap=args.strip_overlap,
                min_conf=args.min_conf,
                min_conf_tail=args.min_conf_tail,
                psm=args.psm,
                lang=args.lang,
                preprocess=args.preprocess,
            )
            items.extend(page_items)
    else:
        image = Image.open(args.input).convert("RGB")
        items = extract_lines_from_image(
            image=image,
            page_index=0,
            strip_height=args.strip_height,
            strip_overlap=args.strip_overlap,
            min_conf=args.min_conf,
            min_conf_tail=args.min_conf_tail,
            psm=args.psm,
            lang=args.lang,
            preprocess=args.preprocess,
        )
    cleaned_items = [
        ReceiptItem(
            page=item.page,
            title=clean_common_ocr_noise(item.title),
            qty=item.qty,
            conf=item.conf,
        )
        for item in items
        if normalize_ws(item.title)
    ]
    cleaned_items = stitch_ampersand_continuations_items(cleaned_items)

    # Drop residual non-item lines after title extraction.
    cleaned_items = [
        item
        for item in cleaned_items
        if item.title
        and not looks_like_non_item(item.title)
        and not is_orphan_fragment(item.title)
    ]

    final: list[str] = []
    for item in cleaned_items:
        if args.include_qty:
            qty = item.qty if item.qty is not None else ""
            final.append(f"{qty}\t{item.title}")
        else:
            final.append(item.title)
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
