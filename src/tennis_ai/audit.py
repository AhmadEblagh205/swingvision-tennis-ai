from __future__ import annotations
from pathlib import Path
from .simulate import make_demo_sessions
from .report import build_report, make_figures


def main():
    root=Path(__file__).resolve().parents[2]
    df=make_demo_sessions()
    demo=root/"data"/"demo"/"sessions.csv"; demo.parent.mkdir(parents=True,exist_ok=True)
    df.to_csv(demo,index=False)
    report=build_report(df,root/"reports")
    make_figures(df,root/"figures")
    print(report)

if __name__ == "__main__": main()
