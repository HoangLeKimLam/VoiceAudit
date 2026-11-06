# -*- coding: utf-8 -*-
"""Kiem ham thuan cua goi P3 v2 (khong can mo hinh / du lieu): python experts/P3/p3_v2/tests/test_p3v2.py"""
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402


def test_all() -> None:
    import compose, ctxfeat, evalp3, feats, feats_pause, post, score, signs  # noqa: E401
    from train_frames import frame_labels, turn_plan
    for m in (feats, ctxfeat, post, signs, evalp3, compose, feats_pause, score):
        m.selftest()
    yd, yv, ign, ky = frame_labels({"edits": [{"level2": "02_giong_cuc_bo", "kind": "segment", "start": 1.0, "end": 1.6}],
                                    "edits_khong_tinh": [{"kind": "segment", "start": 2.0, "end": 2.5}]}, 200)
    assert yv.sum() == 30 and not yd.any() and ky.sum() == 25
    items = {f"r{k}": {"real": True, "cfg": "goc", "set": "v6naten", "fold": k % 3, "meta": {"person": f"p{k}"}, "_T": 400} for k in range(30)}
    plan = turn_plan(items, list(items), 30)
    assert plan and all(len({items[i]["fold"] for i, _, _ in p}) == 1 for _, _, p in plan)      # khong tron fold
    s = np.zeros(300, np.float32)
    s[100:140] = 0.9
    s[200:203] = 1.0                                                                             # gai ngan
    r = post.post_raw_mx(s, 1, 1)
    assert r.max() == s.max() and r[120] >= 0.9                                                  # nguong tep that khong doi
    print("test_p3v2 dat")


if __name__ == "__main__":
    test_all()
