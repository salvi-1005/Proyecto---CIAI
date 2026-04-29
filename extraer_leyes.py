"""
Extrae texto por OCR de los PDFs en una carpeta
y guarda el resultado en un archivo JSON estructurado.

Requisitos:
    pip install pymupdf pytesseract pillow

Uso:
    python extraer_leyes.py                        # carpeta actual
    python extraer_leyes.py --input ./pdfs --output leyes.json
    python extraer_leyes.py --workers 4            # paralelo (más rápido)
"""
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

import argparse
import json
import re
import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import fitz          # PyMuPDF
import pytesseract
from PIL import Image
import io

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ── Configuración 
DEFAULT_INPUT  = "."          # carpeta con los PDFs
DEFAULT_OUTPUT = "leyes.json" # archivo de salida
DPI            = 200          # resolución de rasterización (150=rápido, 200=balance, 300=máx calidad)
LANG           = "spa"        # idioma Tesseract (español)
WORKERS        = 4          # procesos paralelos (1 = serie; 4 = más rápido en multicore)

# ── Parseo del nombre de archivo 

FILENAME_RE = re.compile(
    r"^(?P<year>\d{4})[\s_\-]*"           # año: 1974
    r"(?P<tipo>Ley|Decreto|DNU|Let)"       # tipo de norma
    r"[\s_\-]*"
    r"(?P<numero>[\w\.]+)"                 # número: 20744 ó 1684 ó 357
    r"(?:_.*)?\.pdf$",
    re.IGNORECASE,
)

TIPO_NORM = {
    "ley":     "Ley",
    "decreto": "Decreto",
    "dnu":     "DNU",
    "let":     "Ley",   # typo en algunos archivos ("Let" en vez de "Ley")
}

def parse_filename(name: str) -> dict:
    """Extrae año, tipo y número del nombre del archivo."""
    m = FILENAME_RE.match(name)
    if not m:
        return {"year": None, "tipo": None, "numero": None}
    tipo_raw = m.group("tipo").lower()
    return {
        "year":   int(m.group("year")),
        "tipo":   TIPO_NORM.get(tipo_raw, m.group("tipo")),
        "numero": m.group("numero"),
    }

# ── OCR de un PDF ──────────────────────────────────────────────────────────────

def ocr_pdf(pdf_path: Path, dpi: int = DPI, lang: str = LANG) -> dict:
    """
    Rasteriza cada página del PDF y aplica OCR.
    Devuelve un dict con metadatos + texto completo.
    """
    meta = parse_filename(pdf_path.name)
    result = {
        "id":       pdf_path.stem,
        "filename": pdf_path.name,
        "year":     meta["year"],
        "tipo":     meta["tipo"],
        "numero":   meta["numero"],
        "pages":    0,
        "text":     "",
        "error":    None,
    }

    try:
        doc = fitz.open(str(pdf_path))
        result["pages"] = len(doc)
        pages_text = []

        zoom = dpi / 72          # factor: 72 DPI es el default de PyMuPDF
        mat  = fitz.Matrix(zoom, zoom)

        for page in doc:
            pix  = page.get_pixmap(matrix=mat)
            img  = Image.open(io.BytesIO(pix.tobytes("png")))
            text = pytesseract.image_to_string(img, lang=lang)
            pages_text.append(text)

        doc.close()
        result["text"] = "\n\n--- PÁGINA ---\n\n".join(pages_text).strip()

    except Exception as e:
        result["error"] = str(e)

    return result


# ── Worker para multiproceso (debe estar en top-level) ────────────────────────

def _worker(args):
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    path, dpi, lang = args
    return ocr_pdf(path, dpi=dpi, lang=lang)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="OCR de PDFs InfoLEG → JSON")
    parser.add_argument("--input",   default=DEFAULT_INPUT,  help="Carpeta con los PDFs")
    parser.add_argument("--output",  default=DEFAULT_OUTPUT, help="Archivo JSON de salida")
    parser.add_argument("--dpi",     type=int, default=DPI,  help="DPI de rasterización (default 200)")
    parser.add_argument("--lang",    default=LANG,           help="Idioma Tesseract (default: spa)")
    parser.add_argument("--workers", type=int, default=WORKERS, help="Procesos paralelos (default 1)")
    parser.add_argument("--resume",  action="store_true",    help="Saltar PDFs ya presentes en el JSON de salida")
    args = parser.parse_args()

    input_dir   = Path(args.input)
    output_file = Path(args.output)

    pdfs = sorted(input_dir.glob("*.pdf"))
    if not pdfs:
        print(f"No se encontraron PDFs en: {input_dir.resolve()}")
        sys.exit(1)

    print(f"PDFs encontrados : {len(pdfs)}")
    print(f"DPI              : {args.dpi}")
    print(f"Idioma OCR       : {args.lang}")
    print(f"Workers          : {args.workers}")
    print(f"Salida           : {output_file.resolve()}")
    print()

    # Resume: cargar resultados previos si existen
    done_ids = set()
    existing = []
    if args.resume and output_file.exists():
        with open(output_file, encoding="utf-8") as f:
            existing = json.load(f)
        done_ids = {r["id"] for r in existing}
        print(f"Resume: {len(done_ids)} documentos ya procesados, se saltean.\n")

    pending = [p for p in pdfs if p.stem not in done_ids]
    print(f"A procesar: {len(pending)} PDFs\n")

    results  = list(existing)
    errors   = []
    t_start  = time.time()

    if args.workers == 1:
        # Serie — más simple, mejor para depurar
        for i, pdf in enumerate(pending, 1):
            t0  = time.time()
            res = ocr_pdf(pdf, dpi=args.dpi, lang=args.lang)
            elapsed = time.time() - t0
            status  = f"ERROR: {res['error']}" if res["error"] else f"{res['pages']}pp, {len(res['text'])} chars"
            print(f"[{i:>3}/{len(pending)}] {pdf.name:<45} {status}  ({elapsed:.1f}s)")
            results.append(res)
            if res["error"]:
                errors.append(res["filename"])

            # Guardado incremental cada 10 documentos
            if i % 10 == 0:
                _save(results, output_file)
    else:
        # Paralelo
        task_args = [(p, args.dpi, args.lang) for p in pending]
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(_worker, a): a[0] for a in task_args}
            done = 0
            for future in as_completed(futures):
                done += 1
                res = future.result()
                status = f"ERROR: {res['error']}" if res["error"] else f"{res['pages']}pp"
                print(f"[{done:>3}/{len(pending)}] {res['filename']:<45} {status}")
                results.append(res)
                if res["error"]:
                    errors.append(res["filename"])
                if done % 10 == 0:
                    _save(results, output_file)

    # Guardado final
    _save(results, output_file)

    elapsed_total = time.time() - t_start
    print()
    print("=" * 60)
    print(f"Procesados : {len(pending)}")
    print(f"Errores    : {len(errors)}")
    print(f"Total JSON : {len(results)} documentos")
    print(f"Tiempo     : {elapsed_total/60:.1f} min")
    print(f"Salida     : {output_file.resolve()}")
    if errors:
        print(f"\nArchivos con error:")
        for e in errors:
            print(f"  - {e}")


def _save(results: list, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
