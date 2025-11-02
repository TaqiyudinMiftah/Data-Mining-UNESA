# README — Pemahaman Paper ViT-GZSL & Diskusi “Feature Extraction vs Fine-Tuning”

Dokumen ini merangkum apa yang saya pelajari pada sesi ini tentang paper **“Vision Transformer-based Feature Extraction for Generalized Zero-Shot Learning (ViT-GZSL)”** dan implikasinya pada praktik **GZSL** (Generalized Zero-Shot Learning). Fokusnya: **mengapa memanfaatkan *feature extraction* dari ViT + modul ringan di atasnya dapat unggul dibanding langsung *fine-tuning* backbone Transformer end-to-end** dalam konteks GZSL.

---

## 1) Latar Belakang: GZSL Singkat

* **Tujuan GZSL**: Model harus mengenali **kelas seen** (pernah dilatih) **dan** **kelas unseen** (tidak pernah dilihat) pada saat uji.
* **Bantuan semantik**: Tiap kelas punya **atribut** (mis. “berparuh”, “berkaki empat”). Atribut inilah “jembatan” dari kelas ke citra.
* **Bias umum**: Model cenderung **berat ke kelas seen** bila tidak ada mekanisme penyeimbang; metrik penting: **Harmonic Mean (H)** dari akurasi seen & unseen.

---

## 2) Gagasan Utama Paper

**Gunakan Vision Transformer (ViT) sebagai ekstraktor fitur** dan **gabungkan dua jenis fitur**:

1. **Fitur global [CLS]** → ringkasan seluruh citra.
2. **Fitur lokal patch** → detail spasial (atribut lokal seperti “tanduk”, “ekor”, dll).

Agar patch benar-benar “terarah ke atribut”, paper memperkenalkan dua modul:

* **F2A (Feature-to-Attribute)**: MLP 2-layer yang **mensintesis vektor atribut** dari **CLS**. Pada inferensi, atribut gambar asli tidak ada; F2A mengisi peran itu.
* **AAM (Attribute Attention Module)**: Modul **cross-attention** dengan **Q=atribut sintetis (F2A)**, **K/V=patch features** → menghasilkan **fitur AAM** (agregasi patch *aware* atribut).

**Fitur akhir** untuk generasi/klasifikasi: **concat([CLS] || AAM])**. ViT **dibekukan** (tidak di-fine-tune).

---

## 3) Alur Pelatihan (End-to-End Pipeline)

1. **Ekstraksi Fitur (ViT beku)**

   * Ambil **CLS** & **patch features** dari layer ViT yang dipilih.
2. **F2A (CLS → â)**

   * Latih **MSE(â, a)** pada data **seen** (hanya saat train atribut asli tersedia).
3. **AAM (Q=â, K/V=patch)**

   * Hasilkan **x_AAM**; latih **auxiliary classifier** dengan **Cross-Entropy**.
   * **Loss total = CE(AAM) + MSE(F2A)**.
4. **Bentuk x_ViT = concat([CLS] || x_AAM)**.
5. **Generatif untuk Kelas Unseen**

   * Latih **CVAE (atau generator lain)** untuk memodelkan ( p(x_{\text{ViT}} \mid a) ).
   * Saat uji, **sintesis fitur unseen** dari atribut kelas-unseen → **latih classifier** di atas gabungan fitur seen (nyata) + unseen (sintetis).
6. **Evaluasi**

   * Laporkan **Acc_seen**, **Acc_unseen**, dan **Harmonic Mean (H)**.

> Intuisi: **AAM** menimbang patch yang relevan dengan atribut; **F2A** memastikan *query* atensi tetap bermakna meskipun atribut gambar tidak tersedia saat inferensi.

---

## 4) Mengapa Pendekatan Ini Bekerja

* **ViT mempertahankan detail spasial** (patch) tanpa pooling agresif → lebih baik untuk **atribut lokal**.
* **AAM > rata-rata patch**: rata-rata meredam sinyal atribut yang jarang muncul; AAM **memfokuskan** bobot ke patch relevan.
* **F2A** mengatasi **ketiadaan atribut per-gambar pada inferensi**, sehingga atensi tidak “buta”.

---

## 5) “Feature Extraction vs Fine-Tuning” — Kapan Memilih yang Mana

### Mengapa **feature extraction (ViT beku)** efektif di GZSL:

* **Kurangi bias ke seen**: Representasi tetap umum/serbaguna; penyeimbang seen-unseen dipindahkan ke modul atas (AAM/F2A/generator/classifier).
* **Stabil & hemat komputasi**: Latih modul ringan, bukan seluruh backbone.
* **H sering naik**: Mudah menjaga keseimbangan performa seen–unseen.

### Kapan **fine-tuning (end-to-end)** bisa unggul:

* **Domain shift besar** (dataset sangat berbeda dari pra-latih) → butuh penyesuaian.
* **Kontrol ketat terhadap bias**:

  * **Unfreeze parsial** (1–3 layer atas), **LoRA/Adapter**.
  * **Regularisasi** (weight decay, dropout, L2-SP/EWC).
  * **Kalibrasi logit** & **loss khusus GZSL** (temperature, margin, bias correction).
  * **Early stopping** pakai metrik **H**, bukan akurasi seen saja.

**Resep praktis**:

1. Mulai dari **ViT beku + F2A + AAM + generator + classifier**.
2. Kalau ingin dorong lebih jauh, **unfreeze tipis** (atau LoRA) dengan **monitor H**; jika H turun → kembali beku/atur regulasi.

---

## 6) Rekomendasi Implementasi (Checklist)

* [ ] Backbone ViT (mis. DeiT-B) **dibekukan**.
* [ ] Ambil **CLS** & **patch features** dari layer (uji 9–12 untuk *sweet spot*).
* [ ] **F2A**: MLP( CLS → atribut sintetis â ), **MSE** vs atribut asli (train-seen).
* [ ] **AAM**: Cross-attention (Q=â, K/V=patch), multi-head, **residual pada V**; **CE** via aux-classifier.
* [ ] **x_ViT** = concat([CLS] || AAM).
* [ ] **Generator**: CVAE / (opsional) TF-VAEGAN, f-CLSWGAN, dll.
* [ ] **Classifier**: softmax/SVM/KNN di atas fitur seen (nyata) + unseen (sintetis).
* [ ] **Evaluasi**: Acc_seen, Acc_unseen, **H**; plot **trade-off** seen vs unseen.

---

## 7) Catatan Praktis & Tips Eksperimen

* **Validasi layer ViT**: uji beberapa layer untuk ekstraksi CLS/patch; performa bisa berbeda.
* **Ablation penting**:

  * Tanpa **MSE** pada F2A → AAM kehilangan “arah”.
  * Rata-rata patch vs **AAM** → cek perbedaan H.
* **Kalibrasi saat inference**: *logit scaling* atau *bias rectification* dapat membantu mengimbangi kecenderungan kelas seen.
* **Visualisasi**: peta atensi dari AAM membantu memverifikasi bahwa patch relevan atribut memang disorot.

---

## 8) Rumus Ringkas (Intuisi)

* **AAM (single-head)**:
  ( Q = W_Q \hat a,; K = W_K X_{\text{patch}},; V = X_{\text{patch}} + W_V X_{\text{patch}} )
  ( A = \mathrm{softmax}\big(QK^\top/\sqrt d\big),\quad x_{\text{AAM}} = AV )
* **Loss total**:
  ( \mathcal{L} = \underbrace{\mathrm{CE}(\text{aux_clf}(x_{\text{AAM}}), y)}_{\text{latih AAM}} + \underbrace{| \hat a - a |*2^2}*{\text{latih F2A}} )
* **CVAE (gagasan)**:
  Maksimalkan ELBO untuk ( \log p(x_{\text{ViT}} \mid a) ); saat uji, sampling fitur **unseen** dari ( p_\theta(x\mid a, z) ).

---

## 9) Glosarium Mini

* **CLS**: Token ringkasan global pada ViT.
* **Patch features**: Representasi lokal per potongan citra.
* **F2A**: MLP yang memetakan fitur ke ruang atribut.
* **AAM**: Cross-attention untuk *attribute-guided pooling* patch.
* **CVAE**: Generator fitur kondisional (di-condition oleh atribut).
* **Harmonic Mean (H)**: Metrik kunci GZSL untuk keseimbangan seen–unseen.

---

## 10) Referensi

* Paper: *Vision Transformer-based Feature Extraction for Generalized Zero-Shot Learning (ViT-GZSL)*, arXiv: **2302.00875**.
* Kata kunci: *GZSL, Vision Transformer, Cross-Attention, Attribute-Guided Pooling, CVAE*.

---

## 11) Rencana Eksperimen Sederhana (Template)

```text
1) Dataset: CUB / AWA2 / SUN (atribut tersedia).
2) Ekstraksi:
   - ViT-B beku → CLS (L=11), patch (L=11). Simpan.
3) Modul:
   - F2A: MLP(d_CLS → d_attr), loss = MSE.
   - AAM: MH cross-attn (Q=â, K/V=patch), aux-CE.
   - x_ViT = concat([CLS] || x_AAM).
4) Generator & Klasifikasi:
   - Latih CVAE pada (x_ViT, a). Sintesis fitur unseen.
   - Latih softmax pada fitur seen (nyata) + unseen (sintetis).
5) Evaluasi:
   - Acc_seen, Acc_unseen, H. Simpan kurva kalibrasi.
6) Opsional fine-tune:
   - Unfreeze 1–2 layer atas / LoRA. Monitor H. Rollback bila H turun.
```

---

**Kesimpulan praktik**: Untuk GZSL, **mulai dari *feature extraction* ViT + F2A + AAM + generator** adalah baseline yang **kuat, stabil, dan hemat komputasi**. *Fine-tuning* bisa dicoba terbatas (parsial/LoRA) **bila** ada domain shift besar atau peluang kenaikan H yang terukur—dengan kontrol ketat terhadap bias ke kelas seen.
