# README — Catatan Sesi: Linear Probe vs Fine-Tuning

Dokumen ini merangkum apa yang dipelajari pada sesi ini: konsep **linear probe (linear evaluation)**, perbedaannya dengan **fine-tuning**, cara implementasi ringkas (PyTorch), kapan masing-masing pendekatan dipakai, serta referensi paper kunci.

---

## 1) Intuisi Singkat

* **Linear Probe**: Bekukan backbone (CNN/ViT), latih **satu lapis linear** di atas fitur. Tujuan utamanya **menilai kualitas representasi** backbone—seberapa “linearly separable” fitur tanpa adaptasi non-linear tambahan.
* **Fine-Tuning**: Buka sebagian/seluruh layer backbone dan latih ulang pada data target. Tujuannya **maksimalkan performa tugas** (bukan sekadar evaluasi fitur).

**Ringkas:** Linear probe = *tes kualitas fitur*; Fine-tuning = *kejar akurasi puncak/adaptasi domain*.

---

## 2) Arsitektur & Alur Data

```
Input → Backbone (frozen) → [CLS]/Pooling → Linear Head (trainable) → Softmax
```

Untuk ViT:

* Ambil token **[CLS]** (atau global average pooling) sebagai representasi gambar.
* Umpankan ke lapis **Linear** (logit kelas).
* Loss: **Cross-Entropy**; yang diupdate **hanya** bobot linear head.

---

## 3) Rumus (Klasifikasi Multi-kelas)

Dari fitur backbone beku ( h = f_\theta(x) ), klasifier linear ( W, b ):
[
z = Wh + b, \quad
p(y=k|x) = \frac{e^{z_k}}{\sum_j e^{z_j}}, \quad
\mathcal{L} = -\frac{1}{N} \sum_i \log p(y_i|x_i)
]
Parameter yang diupdate: **( W, b )** saja.

---

## 4) Implementasi Ringkas (PyTorch)

### 4.1 Linear Probe (bekukan backbone, latih 1 lapis linear)

```python
import torch, torch.nn as nn

model = torchvision.models.vit_b_16(weights="IMAGENET1K_SWAG_E2E_V1")
# Bekukan semua layer
for p in model.parameters():
    p.requires_grad = False

# Ganti head menjadi linear trainable
num_features = model.heads.head.in_features
model.heads.head = nn.Linear(num_features, num_classes)

optimizer = torch.optim.SGD(model.heads.head.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()

# Loop training standar (optimizer hanya head)
```

### 4.2 Fine-Tuning (buka sebagian/semua layer)

```python
# Contoh: unfreeze blok atas (layer-wise unfreeze)
for p in model.parameters():
    p.requires_grad = False
for p in model.encoder.layers[-2:].parameters():  # dua layer terakhir ViT
    p.requires_grad = True
for p in model.heads.head.parameters():
    p.requires_grad = True

optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                              lr=3e-4, weight_decay=0.05)
# (Opsional) layer-wise lr decay
```

---

## 5) Kapan Memilih Linear Probe vs Fine-Tuning?

**Pilih Linear Probe bila:**

* Ingin **benchmark cepat** kualitas fitur backbone.
* Dataset terbatas; ingin hindari overfitting & hemat komputasi.
* Perbandingan adil antar backbone tanpa efek non-linear adaptasi.

**Pilih Fine-Tuning bila:**

* Mengejar **performa puncak** pada tugas target.
* Ada **perbedaan domain** besar dari data pretraining (medical, satelit, dsb.).
* Butuh adaptasi fitur tingkat tinggi (augmentasi kuat, prompt-tuning, LoRA, dsb.).

---

## 6) Pola Temuan Umum dari Literatur

* Linear probe pada backbone SSL modern (MAE/DINOv2) sering **sudah tinggi**; **fine-tuning memberi kenaikan moderat** di beberapa set (fitur sangat kuat “out-of-the-box”).
* Di skenario **transfer** luas, **fine-tuning** umumnya unggul untuk akurasi puncak—terutama bila domain bergeser—namun **tidak selalu**; pada kondisi **out-of-distribution**, linear probe bisa **lebih stabil**.
* Kualitas pretraining (skala data/label) berkorelasi kuat dengan hasil **baik untuk linear probe maupun fine-tuning**.

> Catatan: angka spesifik bergantung dataset/backbone; gunakan protokol evaluasi yang konsisten.

---

## 7) Protokol Eksperimen yang Disarankan

1. **Baseline Linear Probe**

   * Bekukan backbone; latih head linear.
   * Tuning sederhana: LR {0.01, 0.05, 0.1}, weight decay {1e-4, 5e-4}.

2. **Partial Fine-Tuning**

   * Unfreeze bagian atas (mis. 1–3 blok terakhir ViT).
   * Coba **layer-wise lr decay** (mis. 0.75) dan **AdamW**.

3. **Full Fine-Tuning** (jika compute cukup)

   * Early stopping, cosine LR, mixup/cutmix (opsional).

4. **Pelaporan Konsisten**

   * Split tetap, tiga seed (avg ± std), log **Top-1/Top-5**, confusion matrix.

**Template Tabel Hasil**

| Metode       |          Unfreeze |   LR |   WD | Top-1 (%) | Catatan |
| ------------ | ----------------: | ---: | ---: | --------: | ------- |
| Linear Probe |              Head | 0.10 | 1e-4 |           |         |
| Partial FT   | Last-2 blk + Head | 3e-4 | 0.05 |           |         |
| Full FT      |               All | 5e-5 | 0.05 |           |         |

---

## 8) Tips Praktis

* **Imbalance**: pakai `class_weight` atau **weighted sampler**.
* **Normalisasi fitur** (opsional) sebelum logistic regression/linear SVM.
* **Augmentasi**: jaga konsistensi train/val; untuk linear probe, gunakan augmentasi “aman”.
* **Regularisasi**: weight decay / dropout pada head linear (jika perlu).
* **Early Stopping**: berdasarkan val loss/accuracy—hindari overfitting cepat pada dataset kecil.

---

## 9) Referensi Utama (untuk dibaca lanjut)

* Kornblith et al., **Do Better ImageNet Models Transfer Better?** (CVPR 2019).
* Kolesnikov et al., **Big Transfer (BiT): General Visual Representation Learning** (ECCV 2020).
* He et al., **Masked Autoencoders are Scalable Vision Learners (MAE)** (2021).
* Oquab et al., **DINOv2: Learning Robust Visual Features without Supervision** (2023).
* Radford et al., **CLIP: Learning Transferable Visual Models from Natural Language Supervision** (ICML 2021).
* Pégeot et al., **A Comprehensive Study of Transfer Learning Under Constraints** (ICCV 2023 W).

> Gunakan paper-paper ini untuk melihat **angka** linear probe vs fine-tuning pada berbagai backbone (termasuk ViT-B/16) dan beragam dataset.

---

## 10) Next Steps (Opsional)

* Jalankan **baseline linear probe** pada dataset Anda (ViT-B/16).
* Bandingkan dengan **partial & full fine-tuning** (laporkan rata-rata 3 seed).
* Coba **LoRA**/**bias-tuning** sebagai alternatif hemat memori untuk model besar.
* Dokumentasikan hasil dalam tabel di atas dan simpan konfigurasi (yaml/args) untuk reprodusibilitas.

---

**Kesimpulan:**
Linear probe adalah **alat diagnostik cepat** untuk memeriksa seberapa bagus representasi backbone. Fine-tuning tetap **andalan untuk performa puncak**, terutama saat domain bergeser. Kombinasinya memberi gambaran utuh: mulai **linear probe** untuk mengukur kualitas fitur, lanjut **fine-tuning** bila masih ada headroom akurasi yang bermakna.
