# -*- coding: utf-8 -*-
"""TAI TRUOC mo hinh Hugging Face ma san pham (voiceaudit_v9) can — san pham chay OFFLINE (HF_HUB_OFFLINE=1), nen may moi phai tai MOT lan.

Ghim dung phien ban (commit) may goc dang dung; sau khi tai ghi refs/main -> commit do (san pham goi from_pretrained khong ghi phien ban).
Thu muc bo dem = dung cho san pham doc: dat moi truong y nhu san pham (v6.common.setup_env: HF_HOME = <kho>/.hf_cache, tru khi may
da dat TRANSFORMERS_CACHE / HUGGINGFACE_HUB_CACHE rieng).
  XLS-R deepfake (khung, dau ca tep)  ~1,2 GB · whisper-small (K1 mo rong: ngon ngu) ~0,9 GB · AST (loai am thanh) ~0,3 GB
  · wav2vec2 ASR tieng Viet (mat na tieng noi) ~0,7 GB
CLI: python -m v6.product.tai_mo_hinh            (tai cai con thieu, roi kiem nap offline)
     python -m v6.product.tai_mo_hinh --kiem     (chi kiem: nap offline tung mo hinh nhu san pham; ma thoat 1 neu thieu)
     python -m v6.product.tai_mo_hinh selftest   (ham thuan, khong mang)"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from v6 import common  # noqa: E402,F401  (setup_env() khi import: HF_HOME giong san pham)

MODELS = {  # repo -> commit dang dung tren may goc (20/10/2026)
    "Gustking/wav2vec2-large-xlsr-deepfake-audio-classification": "f7050b586236dc910d1157f430def2d0647b02b4",
    "openai/whisper-small": "973afd24965f72e36ca33b3055d56a652f456b4d",
    "MIT/ast-finetuned-audioset-10-10-0.4593": "f826b80d28226b62986cc218e5cec390b1096902",
    "nguyenvulebinh/wav2vec2-base-vietnamese-250h": "69e9000591623e5a4fc2f502407860bcdc0de0b2",
}


def cache_dir() -> str:
    """Thu muc 'hub' ma transformers cua san pham se doc."""
    if os.environ.get("TRANSFORMERS_CACHE"):
        return os.environ["TRANSFORMERS_CACHE"]
    from huggingface_hub import constants
    return constants.HF_HUB_CACHE


def ref_path(cache: str, repo: str) -> str:
    """<cache>/models--<to>--<ten>/refs/main (ham thuan)."""
    return os.path.join(cache, "models--" + repo.replace("/", "--"), "refs", "main")


def pin_main(cache: str, repo: str, sha: str) -> None:
    p = ref_path(cache, repo)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        fh.write(sha)


def kiem(cache: str) -> dict:
    """{repo: 'ok' | loi} — nap cau hinh OFFLINE theo refs/main, nhu san pham."""
    os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from transformers import AutoConfig
    out = {}
    for repo, sha in MODELS.items():
        try:
            AutoConfig.from_pretrained(repo, cache_dir=cache)
            with open(ref_path(cache, repo), encoding="utf-8") as fh:
                got = fh.read().strip()
            out[repo] = "ok" if got == sha else f"phien ban khac: {got[:10]} (can {sha[:10]})"
        except Exception as e:  # noqa: BLE001  (bao tung mo hinh, khong dung o mo hinh dau)
            out[repo] = f"thieu: {type(e).__name__}"
    return out


def tai(cache: str) -> None:
    os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "0"
    from huggingface_hub import snapshot_download
    for repo, sha in MODELS.items():
        print(f"tai {repo} @ {sha[:10]} -> {cache}", flush=True)
        snapshot_download(repo, revision=sha, cache_dir=cache)
        pin_main(cache, repo, sha)


def selftest() -> None:
    import tempfile
    with tempfile.TemporaryDirectory() as t:
        pin_main(t, "a/b", "123")
        assert open(ref_path(t, "a/b"), encoding="utf-8").read() == "123"
        assert ref_path(t, "a/b").endswith(os.path.join("models--a--b", "refs", "main"))
    assert len(MODELS) == 4 and all(len(s) == 40 for s in MODELS.values())
    print("tai_mo_hinh selftest dat")


def main(argv: list[str] | None = None) -> int:
    a = list(sys.argv[1:] if argv is None else argv)
    if a[:1] == ["selftest"]:
        selftest(); return 0
    cache = cache_dir()
    if "--kiem" not in a:
        tai(cache)
    res = kiem(cache)
    for repo, v in res.items():
        print(f"  {v:40s} {repo}")
    print(f"bo dem: {cache}")
    return 0 if all(v == "ok" for v in res.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
