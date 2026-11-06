"""
Convert VERITAS_MM_FINAL_PROJECT_DOCUMENTATION.docx to PDF using Microsoft Word COM.
Preserves exact typography, page breaks, embedded high-resolution figures, and margins.
"""

import sys
from pathlib import Path
import pythoncom
import win32com.client

WORKSPACE = Path(r"C:\Users\94in\.gemini\antigravity\scratch\multimodal-misinfo-submission")
DOCX_PATH = WORKSPACE / "VERITAS_MM_FINAL_PROJECT_DOCUMENTATION.docx"
PDF_PATH = WORKSPACE / "VERITAS_MM_FINAL_PROJECT_DOCUMENTATION.pdf"

def convert_to_pdf():
    print(f"Opening DOCX: {DOCX_PATH}")
    if not DOCX_PATH.exists():
        print(f"ERROR: DOCX file not found at {DOCX_PATH}")
        sys.exit(1)

    pythoncom.CoInitialize()
    word = win32com.client.DispatchEx("Word.Application")
    word.Visible = False
    word.DisplayAlerts = 0

    try:
        doc = word.Documents.Open(str(DOCX_PATH.resolve()))
        print(f"Exporting to PDF: {PDF_PATH}")
        doc.SaveAs(str(PDF_PATH.resolve()), FileFormat=17) # 17 = wdFormatPDF
        doc.Close(SaveChanges=0)
        print("Export complete.")
    except Exception as e:
        print(f"Conversion failed: {e}")
        raise
    finally:
        word.Quit()
        pythoncom.CoUninitialize()

    if PDF_PATH.exists():
        print(f"SUCCESS: PDF created at {PDF_PATH} ({PDF_PATH.stat().st_size / (1024*1024):.2f} MB)")
    else:
        print("ERROR: PDF was not generated.")

if __name__ == "__main__":
    convert_to_pdf()
