"""Smoke test: renders both pages headlessly and fails on any exception.
Run from anywhere:  python tests/smoke_test.py
"""
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(BASE)
sys.path.insert(0, BASE)
from streamlit.testing.v1 import AppTest  # noqa: E402


def run_page(rel_path):
    at = AppTest.from_file(os.path.join(BASE, rel_path), default_timeout=300)
    at.run()
    return at


if __name__ == "__main__":
    ok = True
    for name, path in [("Extraction page", "views/extraction_page.py"), ("Evaluation page", "views/evaluation_page.py")]:
        at = run_page(path)
        ok &= not at.exception
        print(f"{name}: {'OK' if not at.exception else 'EXCEPTION'}")
        for e in at.exception:
            print("   ", e.value)
        text = " ".join(m.value for m in at.markdown)
        print("   markdown blocks:", len(at.markdown), "| buttons:", len(at.button), "| tabs:", len(at.tabs))
    sys.exit(0 if ok else 1)
