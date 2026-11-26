"""Bài kiểm hàng Constraint «Số tệp: đúng 1» (thay phần unit test của vaudit.k1 không chạy được vì bộ gửi thiếu vaudit/test_k1.py).
Không cần mô hình, không cần dữ liệu: mọi đầu vào không phải MỘT tệp có thật phải ra `tu_choi` với đúng cụm lý do hợp đồng,
và phải dừng trước khi giải mã (không có số đo nào ngoài quyết định).

  .venv/Scripts/python.exe experts/P1/coverage/test_so_tep.py
"""
from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from vaudit import k1  # noqa: E402

LY_DO = "từ chối nhận · đúng 1 tệp mỗi lần"


def main() -> None:
    assert k1.REJECT["so_tep"] == LY_DO, k1.REJECT["so_tep"]
    with tempfile.TemporaryDirectory() as d:
        a, b = os.path.join(d, "a.wav"), os.path.join(d, "b.wav")
        for p in (a, b):
            open(p, "wb").write(b"RIFF")
        ca = {"thư mục": d, "danh sách hai tệp": [a, b], "bộ hai tệp": (a, b), "chuỗi rỗng": "", "None": None,
              "tệp không tồn tại": os.path.join(d, "khong_co.wav"), "hai đường dẫn nối bằng dấu phẩy": f"{a},{b}", "số": 3}
        for ten, x in ca.items():
            r = k1.check(x)
            assert r["decision"] == "tu_choi" and r["reasons"] == [LY_DO], (ten, r["decision"], r["reasons"])
            assert not r.get("measures"), (ten, r.get("measures"))          # dừng trước khi đọc tệp
        r = k1.check(a)                                                      # MỘT tệp có thật (nội dung hỏng): không được rơi vào «số tệp»
        assert LY_DO not in r["reasons"], r["reasons"]
    print(f"test_so_tep dat · {len(ca)} đầu vào sai đều ra tu_choi đúng lý do · tệp đơn (hỏng) ra {r['decision']}: {r['reasons']}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
