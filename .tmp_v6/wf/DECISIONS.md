# Quyet dinh cua nguoi dung (08/10/2026) - rang buoc cho moi cong viec tiep theo

1. CAO DU LIEU: YouTube chi video gan nhan Creative Commons (CC BY) - kiem giay phep TUNG VIDEO tu metadata (truong license),
   cong them nguon mo ngoai YouTube (archive.org CC BY/BY-SA/CC0, FOSDEM CC BY 2.0, media.ccc.de CC BY). Chi dung NOI BO nghien cuu,
   khong phat tan lai am thanh, ghi nguon (URL, kenh, giay phep) cho tung tep. Nguoi dung chap nhan rui ro dieu khoan YouTube.
   Khong dung cookie tai khoan. Khong tai noi dung co tre em / ton giao nhay cam neu co the loc bang metadata (uu tien lop hoc, hoi thao, hop).
2. NHAI GIONG (TTS clone, VC): CHI nguoi noi trong bo du lieu nghien cuu cong khai: VIVOS, VoxVietnam, Vietnam-Celeb, EdAcc, AMI,
   VoxPopuli, LibriSpeech. KHONG nhai nguoi trong am thanh cao, KHONG nhai VSS / VietSuperSpeech, VSASV, viVoice, nguoi noi tieng.
   Mau giong lay tu BAN GHI KHAC cua cung nguoi (khong lay tu chinh tep se chen). Khong phat tan am thanh sinh ra.
   Tranh mo hinh co dieu khoan cam nhai giong khi chua dong y (vd hynt/F5-TTS-Vietnamese-ViVoice).
3. BO NHO: KHONG doi pagefile. Moi viec nang (nap torch / mo hinh / GPU / > 1 GB RAM) chay TUAN TU, mot viec mot luc,
   qua khoa .tmp_v6/wf/run_serial.sh. Dat TMP, TEMP, TORCH_HOME, TTS_HOME, PIP_CACHE_DIR, UV_CACHE_DIR, XDG_CACHE_HOME ve D:.
4. KHU ON MAC DINH CUA UNG DUNG (Zoom, Meet, Zalo, WebRTC NS): nhan 01 (dieu kien ghi binh thuong), khong phai 02.
   Ghi lai diem lech voi cau chu hop dong (khu on ca tep = 02 vo hai) trong so lech; KHONG sua VoiceAudit_6Muc.html.
5. Tam bo qua rang buoc thoi gian (Req 15) - ket hop moi dac trung (do tay, xu huong, anh pho, embedding) truoc, toi uu toc do sau.
6. Mo hinh hoc sau: chi dung mo hinh DA HUAN LUYEN SAN (dong bang / LoRA), khong hoc tu dau.

# Quyet dinh bo sung (12/10/2026, sau dot viet ma wf_76ac5e88)
7. HANG GPU: HOAN moi viec GPU (sinh giong may, VC, trich dac trung GPU, huan luyen fx_stack/fx_tcn, ft2x). Chi chay viec CPU va mang.
   Khong doi pagefile. Khi nao chay GPU se hoi lai nguoi dung.
8. CAO: chinh sach CHAT nhu muc 1 (chi CC BY / BY-SA / CC0; loai NC, ND, ban tai lai archive.org; loai kenh ton giao, noi dung
   tre em / truong hoc). LID: vi 0,9, en 0,5. Ngan sach tai ~17 GB tren D:. Tep cao gan nhan '01 co kha nang' (bo loc vet cat
   khong chung nhan duoc 'chua sua'); bao rieng ti le bao nham tren tep cao.
9. GIONG MAY TIENG VIET: KHONG dung Vietnam-Celeb (nguoi noi tieng). Nang tran moi nguoi len 6 % (--person-cap-frac 0.06)
   cho ttsmulti_vi va vcmulti.
10. Hang doi chay qua .tmp_v6/wf/queue_runner.py (lan A: CPU; lan B: cao), dung khoa run_serial.sh chung.
    Con cho quyet dinh: A4 (nguoi noi moi, tap khoa), A5, A6, B1-B6, C1-C7 trong QUYET_DINH_CAN_CHOT_12-10.md.

# Quyet dinh bo sung (17/10 va 19/10/2026)
11. YOUTUBE CHI KENH KHONG CHUYEN: ghi xong la dang, khong hau ky chuyen nghiep, thuong it nguoi dang ky (quy tac lam viec <= ~10k);
    khong kenh cong ty / chinh thuc / TV / content creator / vlog. Chan tai moi qua v6/scrape/fetch.py NO_NEW_FETCH; du lieu cu tu kenh lon duoc giu.
12. DU LIEU SAI: CACH LY (_quarantine / _pruned kem nhat ky), KHONG XOA. Da cach ly 17/10: Nguoi Phuong Ngoai (ton giao), Giap bui dinh (giong tre em).
    Ngoai le da duyet: xoa am goc cua video bi bo loc loai (10,65 GB, danh sach data/scrape/meta/deleted_raw_17-10.jsonl).
13. LID tieng Anh giu nguong 0,5 (luat cap video thu 17/10: bo vi tieng Viet qua ASR tieng Anh chong len tieng Anh that).
14. 19/10: GPU VAN HOAN (nhac lai sau khi xem ly do bo nho). Cac lane lam viec CPU truoc.
15. 19/10: Cay phan ra chot theo VoiceAudit_TomTat.html §02 (goc la van de ban dau: 7 van de -> 9 cau hoi ky thuat -> 36 van de con);
    cac tai lieu khac (CAY_PHAN_RA_V2.html, VoiceAudit.html, lanes/) theo cay do. Tai lieu cho thay theo dung sau muc da nop.
16. 19/10: NGAN SACH VU OAN THEO KENH (nguoi dung chot): tong <= 3 % (Req 4) chia: ghep 0,5 % · xoa 0,5 % · khu on vung 0,5 % · doi giong vung 0,5 %
    · dau khu on ca tep 0,25 % · dau doi giong ca tep 0,25 % · giong may mot doan 0,25 % · giong may ca tep 0,25 %.
    P0 dat nguong moi kenh theo phan nay; moi lane toi da phat hien TRONG phan cua minh (lanes/CHUNG.md §3.1). Giai mau thuan Scope (<= 10 % moi thao tac)
    voi Req 4 (<= 3 % ca tep) theo Req 4 — so lech dong 7. Khi co o 2 (sang loc / cong 01) co the xet noi lai.
17. 20/10: TAI LIEU PHAN VIEC CHI DINH HUONG MUC TIEU (nguoi dung: "file phan viec chi cho nguoi ta huong muc tieu, bao quat bai toan, hieu sau
    muc tieu va nhiem vu, khong rao huong di"; "du lieu may hien tai phan phoi cho ho cung chi la example"; "Sua het di").
    lanes/<Px>/CLAUDE.md chi giu: cho cam theo hop dong · dich den · Hieu bai toan · baseline (so de vuot) · so huu · luat · nop · phoi hop.
    Moi y tuong / huong di / da thu / gia thuyet / so do kien truc hien tai -> lanes/<Px>/Y_TUONG.md, lanes/Y_TUONG_CHUNG.md, doc SAU khi
    thanh vien tu de xuat huong. Du lieu gui di = du lieu mau; co dinh chi la truong hop phai phu + luat them du lieu (lanes/DU_LIEU.md).
    Cau chu luat cung duoc lam ro (nguoi dung dong y 20/10):
    a) Ngan sach thoi gian chay (RTF lane) ap KHI NOP de ghep vao san pham; luc kham pha duoc cham hon (theo muc 5).
    b) Luat 1: mo hinh nen lon phai la mo hinh da huan luyen san; dau nho / bo phan loai nho tren dac trung duoc hoc tu dau.
    c) Luat 4 cu "khong dung som theo loss CE" la bai hoc phuong phap -> chuyen sang Y_TUONG_CHUNG.md; luat moi: chon diem dung / so luot /
       sieu tham so tren du lieu hoc / dev theo fold nguoi noi, khong nhin tap cham.
18. 20/10: QUYET DINH DU LIEU (nguoi dung chon):
    a) 554 tep giong may cu nhan ban celeb / VSS (430 natvi: celeb 221 + VSS 209; 124 ban tang cuong trong aug) — trai muc 2, 7:
       DANH DAU meta.chinh_sach, khong dung de hoc tiep, khong gui trong goi; am thanh va nhan giu nguyen (khong xoa, khong cach ly).
       load_items (v6.boundary.detect) bo ca hoc lan cham. P5 cham lai baseline khi co mo hinh moi.
    b) CHOT: tieu chi la LOAI UNG DUNG / CONG CU, khong phai ca tep hay mot doan. Khu on don so co san cua ung dung ghi am / goi pho thong
       (ban chat van la ghi am binh thuong) = 01; khu on chuyen nghiep / manh (bat luc ghi hoac sau khi ghi) = 02.
       WebRTC NS muc 0-2 = 01, muc 3 = 02 (do: muc 2 giam on ~14 dB, muc 3 ~17 dB; kenh gia lap ung dung ~11-15 dB).
       Thang hai bac TAM (cho P2 / P3 do tren ung dung that de chot so): 01 = mac dinh / don so (WebRTC 0-2, Zoom / Teams Auto-Low-Medium,
       may ghi am dien thoai); 02 = nguoi dung bat manh hoac AI (WebRTC 3, Zoom / Teams High, iPhone Voice Isolation / Enhance Recording,
       Krisp, noisereduce, afftdn, RNNoise).
       Da ap (scratchpad doi_nhan_n7.py, ban luu .tmp_v6/docbak/manifest_truoc_n7_2010/): 971 tep N7 muc <= 2 -> 01 (thao tac chuyen sang
       meta.edits_khong_tinh), 106 tep bot mot thao tac; 618 doan N7 khong ghi muc -> meta.tam_loai (sinh lai co ns_level); 6 id trung o
       chain -> tam_loai. kiem_bo bat WebRTC phai ghi ns_level, muc <= 2 khong duoc ghi la thao tac.
       Cham lai baseline (mo hinh con hoc nhan cu): OOF ngan sach 3 % Req 4 vi 7,0 % / en 3,8 % (tep 01 goc 5,3 / 2,6 %; 212 tep vua chuyen
       01: 16,7 / 13,5 %); khu on mot doan vi 69 -> 9 % vi nguong den bi day len. Can hoc lai baseline tren nhan moi (FT-2 can GPU -> hoi).
    c) Tap khoa tieng Anh nguoi moi = VoxPopuli (17 nguoi chua dung) + kenh cao en crc32 % 5 == 0. Da curate --add voxp va freeze lai.
       Khong dung LibriSpeech (sach noi — Constraint: khong phai ghi am loi noi).
    d) vcmulti (chuyen giong AI, nhan 02): P3 dung va do; P5 giu ha tang sinh chung (v6/tts_multi) va sinh theo yeu cau cua P3.
19. 20/10: BO GUI THANH VIEN (nguoi dung): chi gui de dung muc tieu, dung huong, de ghep, Claude cua thanh vien hieu TOAN he thong,
    biet nhiem vu, du lieu dung pham vi. Gui: tai lieu he thong (6 muc, tom tat, danh gia, VoiceAudit.html, CAY PHAN RA — cay la co dinh,
    dinh hinh bai toan va cach he thong van hanh; cach giai tung van de co the khac), lanes/, DECISIONS; ma do / kiem / ghep thu (dung tep
    THUC SU nap khi chay lenh tai lieu bat chay); ma bat buoc cua lane (P1 + P2 san pham, P1 k1_axes / eval_signal, P2 chuoi cao + trang thai,
    P5 ha tang sinh). KHONG gui ma / bao cao thu nghiem cu (thien lech) — xin P0 sau buoc 3 (Y_TUONG §0). Thuc hien: v6.data.goi_lane
    (tai_lieu, ma_chung, ma_san_pham, ma_k1, ma_cao, ma_sinh; liet-ke, --chi); bo duong dan cung D:\TuDuyTTNT (216 tep py + 15 sh, gia tri
    giu nguyen tren may goc); requirements.txt; v6.product.tai_mo_hinh (4 mo hinh HF ghim phien ban); trang thai cao loc 102 kenh tap khoa;
    hosts natural_v6 ghi san grow/session. Thu bo gui o thu muc khac: ghep_thu giong het, san pham ma thoat 0, test P1/P2/P5 dat.
20. 21/10: LAY DU LIEU TU DO (nguoi dung): thanh vien tu tim / tu thu du lieu thoai mai, KHONG can xin duyet nguon, kenh, kho nghien cuu
    (ke ca kho NC / ND, chi dung noi bo) — mien dung pham vi du lieu cua lane (DU_LIEU.md, <Px>/DU_LIEU.md) va dung chinh sach du lieu
    (CC BY / BY-SA / CC0 khi cao; YouTube kenh khong chuyen; khong ton giao / tre em / truong pho thong; khong nhai giong nguoi cao duoc /
    nguoi noi tieng). Ghi nguon + giay phep vao the bo; chay kiem_bo truoc khi dung (tu bat tap khoa, ranh gioi nhan, chinh sach nhai giong).
    Bo khoi danh sach phai hoi: them nguon / kenh cao. Van phai hoi: GPU, cham tap khoa, xoa / chuyen du lieu lon, viec > 6 gio, doi chinh sach.
    Da sua: BAT_DAU muc 6 + muc 11 (cau dau tien) + muc 12, CHUNG luat 8 + muc 7, DU_LIEU 6.1 + 6.6, P2/CLAUDE muc 6, P2/DU_LIEU muc 6-7.
