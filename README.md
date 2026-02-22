# Receipt Title OCR (Tesseract)

Extracts item title lines (and quantity) from long scanned sales receipts (PDF/image) using Tesseract.

Current setup is tuned for a Target app receipt layout and works with both:
- "Show details" version
- simpler/no-details version

Output format:
- `qty<TAB>title line`

Example:
```text
1	Kidfresh Frozen Chicken Sticks Value Pack - 16.4oz
1	Mission Carb Balance Taco Size Soft flour Tortillas - 12oz/8ct
```

## Install

macOS:
```bash
brew install tesseract
python -m venv .venv
source .venv/bin/activate
pip install pytesseract pypdfium2 pillow numpy
```

Raspberry Pi (Debian/Raspberry Pi OS):
```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr
python3 -m venv .venv
source .venv/bin/activate
pip install pytesseract pypdfium2 pillow numpy
```

## Recommended Run (Target Layout)

Best quality:
```bash
python tesseract_receipt_titles.py \
  --input ./pdf/target-sale.pdf \
  --dpi 300 \
  --preprocess light \
  --strip-height 1200 \
  --strip-overlap 100 \
  --min-conf 45 \
  --output titles.txt
```

## Configs

Retailer/layout-specific rules live in TOML files.

Default config:
- `configs/target.toml`

Use a custom config:
```bash
python tesseract_receipt_titles.py --input receipt.pdf --config configs/target.toml
```

## Notes

- The script keeps strip-based OCR for long receipts (memory-friendly, Pi-friendly).
- It preserves full title lines (size/brand suffixes) and extracts `Qty`.
- OCR variance on symbols like `™` is normal; core title/size/brand text is the primary target.
