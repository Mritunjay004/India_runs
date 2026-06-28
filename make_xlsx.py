#!/usr/bin/env python3
"""
Convert the canonical submission.csv into submission.xlsx.

submission_spec.md (Section 2-3) and validate_submission.py require the *CSV* to be
exactly: header + 100 rows, columns candidate_id,rank,score,reasoning. The hackathon
PORTAL form, however, asks for the ranked output as an .xlsx. So we keep the CSV as the
canonical, validator-passing artifact and produce a byte-identical-content XLSX for the
form from it — same rows, same order, correct cell types.

    python make_xlsx.py --in submission.csv --out submission.xlsx
"""

import argparse
import csv

from openpyxl import Workbook
from openpyxl.styles import Font


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="submission.csv")
    ap.add_argument("--out", dest="outp", default="submission.xlsx")
    args = ap.parse_args()

    wb = Workbook()
    ws = wb.active
    ws.title = "ranking"

    with open(args.inp, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        ws.append(header)  # candidate_id, rank, score, reasoning
        rows = 0
        for cid, rank, score, reasoning in reader:
            # Preserve correct types: id/reasoning as text, rank as int, score as float.
            ws.append([cid, int(rank), float(score), reasoning])
            rows += 1

    # light formatting: bold header, sensible widths
    for cell in ws[1]:
        cell.font = Font(bold=True)
    ws.column_dimensions["A"].width = 16
    ws.column_dimensions["B"].width = 6
    ws.column_dimensions["C"].width = 10
    ws.column_dimensions["D"].width = 100
    ws.freeze_panes = "A2"

    wb.save(args.outp)
    print(f"wrote {args.outp} with {rows} data rows (+ header)")


if __name__ == "__main__":
    main()
