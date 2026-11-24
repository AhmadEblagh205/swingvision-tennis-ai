from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd


def inspect_workbook(path: str | Path) -> dict:
    path=Path(path)
    xls=pd.ExcelFile(path)
    out={"file": path.name, "sheets": {}}
    for sheet in xls.sheet_names:
        frame=pd.read_excel(path,sheet_name=sheet,nrows=5)
        out["sheets"][sheet]={"columns":[str(c) for c in frame.columns], "sample_rows_read": int(len(frame))}
    return out


def main():
    p=argparse.ArgumentParser(description="Inspect a private SwingVision workbook schema without printing row values")
    p.add_argument("workbook"); p.add_argument("--out",default=None)
    args=p.parse_args(); result=inspect_workbook(args.workbook)
    text=json.dumps(result,indent=2)
    if args.out: Path(args.out).write_text(text,encoding="utf-8")
    else: print(text)

if __name__=="__main__": main()
