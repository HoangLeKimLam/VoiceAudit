# -*- coding: utf-8 -*-
"""VoiceAudit -- mo phong du lieu chinh sua thuc te (dung noi dung goc) va pipeline danh gia."""
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_HOME", os.path.join(BASE, ".hf_cache"))
SR = 16000
