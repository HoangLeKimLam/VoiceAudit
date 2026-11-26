"""So T1.x trước / sau C4 từ tệp kết quả đầy đủ (TOM chỉ in 80 ca sai đầu). Chỉ đọc.

  .venv/Scripts/python.exe experts/P1/coverage/so_c4.py [tag_truoc] [tag_sau]
"""
from __future__ import annotations

import collections
import json
import os
import sys

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".tmp_v6", "results")


def doc(tag: str) -> dict:
    r = json.load(open(os.path.join(RES, f"k1_axes_{tag}.json"), encoding="utf-8"))
    r = r["rows"] if isinstance(r, dict) else r
    return {(x["ca"], x["base"]): x for x in r}


def main(a: str = "p1_v1", b: str = "p1_v1_c4") -> None:
    A, B = doc(a), doc(b)
    assert set(A) == set(B), (len(A), len(B))
    tot = collections.Counter()
    sua, hong = collections.Counter(), []
    for k in A:
        tot[(A[k]["ok"], B[k]["ok"])] += 1
        if not A[k]["ok"] and B[k]["ok"]:
            sua[k[0]] += 1
        if A[k]["ok"] and not B[k]["ok"]:
            hong.append((k, B[k]["loi"]))
    print(f"{len(A)} ca · sai trước {sum(not x['ok'] for x in A.values())} · sai sau {sum(not x['ok'] for x in B.values())}")
    print("đúng→đúng / đúng→sai / sai→đúng / sai→sai:", tot[(True, True)], tot[(True, False)], tot[(False, True)], tot[(False, False)])
    print("sửa được:", dict(sua))
    print("hỏng mới:", len(hong))
    for k, l in hong:
        print("  ", k, str(l)[:160])
    doi = [(k, A[k]["loi"], B[k]["loi"]) for k in A if not A[k]["ok"] and not B[k]["ok"] and A[k]["loi"] != B[k]["loi"]]
    print("vẫn sai nhưng đổi lý do:", len(doi))
    for k, x, y in doi[:10]:
        print("  ", k, str(x)[:80], "→", str(y)[:80])


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main(*sys.argv[1:3])
