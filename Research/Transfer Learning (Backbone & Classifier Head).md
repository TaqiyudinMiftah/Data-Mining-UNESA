# README — Transfer Learning (Backbone & Classifier Head)

*Updated: 2 Nov 2025 (Asia/Jakarta)*

Dokumen ini merangkum yang saya pelajari pada sesi ini tentang **feature extraction**, **freeze backbone**, dan **fine-tuning** pada model visi (CNN/ViT) menggunakan PyTorch.

---

## 1) Inti Pembelajaran

* **Backbone** = ekstraktor fitur umum (mis. ResNet/EfficientNet, ViT encoder).
* **Classifier head** = mapper ringan dari fitur → logits kelas (Linear/MLP).
* **Feature extraction (freeze backbone)** membuat backbone **tidak ikut di-update**, hanya head yang dilatih.
* `requires_grad` mengontrol apakah parameter dihitung gradiennya dan di-update oleh optimizer.
* Strategi praktik yang efektif: **linear probe (freeze)** → **unfreeze bertahap** → **full fine-tune** bila validasi masih naik.

---

## 2) Kenapa Freeze Backbone?

**Tujuan:**

* Hemat **VRAM & waktu** (tak ada backward di backbone).
* **Stabil** pada dataset kecil (kurangi overfitting & catastrophic forgetting).
* Dapat **baseline cepat**: cek apakah fitur pretrained sudah informatif untuk task baru.

**Kapan cocok?**

* Data **kecil/sedang** atau domain mirip pretraining (mis. ImageNet → objek umum).
* Saat sumber daya terbatas dan butuh iterasi eksperimen cepat.

**Kapan tidak cukup?**

* Ada **domain shift besar** (medis/satelit/sketsa) atau task **fine-grained** → perlu unfreeze sebagian/penuh.

---

## 3) Arsitektur Singkat

```
[ Image ] →  Backbone (CNN/ViT)  →  vektor fitur  →  Head (Linear/MLP)  →  Logits
```

* **CNN**: (B,3,H,W) → feature map (B,C,H’,W’) → GAP → (B,C) → Linear → logits
* **ViT**: patchify → token (B,N,D) → encoder → pool (CLS/mean) → (B,D) → Linear → logits

---

## 4) PyTorch: Potongan Kode Penting

### 4.1. Mengaktifkan/menonaktifkan gradient

```python
# Aktifkan grad semua parameter (full fine-tune)
for p in model.parameters():
    p.requires_grad = True

# Freeze backbone saja (feature extraction)
for p in model.backbone.parameters():
    p.requires_grad = False
```

> **Catatan:** Setelah mengubah `requires_grad`, **buat ulang optimizer** agar hanya menangkap parameter yang trainable.

### 4.2. Optimizer untuk head saja

```python
optimizer = torch.optim.Adam(
    (p for p in model.parameters() if p.requires_grad),
    lr=1e-3, weight_decay=1e-4
)
```

### 4.3. Unfreeze bertahap + discriminative LR

```python
# Contoh: unfreeze blok terakhir backbone (sesuaikan nama layer model Anda)
for name, p in model.backbone.named_parameters():
    if "layer4" in name or "blocks.-1" in name:  # ResNet/ViT (sesuaikan)
        p.requires_grad = True

optimizer = torch.optim.Adam([
    {"params": (p for n,p in model.backbone.named_parameters() if p.requires_grad), "lr": 1e-4},
    {"params": model.head.parameters(), "lr": 5e-4},
], weight_decay=1e-4)
```

### 4.4. Ganti head pada ViT torchvision

```python
import torch.nn as nn
import torchvision.models as models

num_classes = 10
vit = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)

feat_dim = vit.heads.head.in_features
vit.heads.head = nn.Linear(feat_dim, num_classes)  # head baru

# Freeze encoder, latih head
for n, p in vit.named_parameters():
    p.requires_grad = n.startswith("heads")
```

---

## 5) Freeze vs Langsung Fine-Tune — Apa Kata Riset?

* **Tidak ada aturan universal menang**.
* **Linear probe / freeze** berguna sebagai langkah awal (cepat, stabil); di banyak dataset, **fine-tune penuh** akhirnya memberi akurasi lebih tinggi.
* **Selective/gradual unfreezing** sering lebih **stabil** dan kadang **lebih baik** pada domain tertentu (terutama saat data terbatas atau ada perbedaan domain).
* Jika data cukup + resep fine-tuning matang, **full fine-tune** biasanya unggul.

> Praktik yang direkomendasikan: **mulai dari freeze (baseline)** → **unfreeze bertahap** → **full fine-tune** bila metrik validasi masih naik.

---

## 6) Checklist Eksperimen

* [ ] Mulai dari **feature extraction** (freeze backbone), latih **linear/MLP head**.
* [ ] Pantau **val loss/acc**; gunakan **early stopping**.
* [ ] Coba **unfreeze blok terakhir** + **discriminative LR**.
* [ ] Jika masih naik, **full fine-tune** dengan LR kecil, **weight decay**, **label smoothing**, dan **augmentasi** yang memadai.
* [ ] Untuk CNN dengan BatchNorm yang di-freeze, pertimbangkan `backbone.eval()` (hindari running stats berubah). ViT pakai LayerNorm (bukan isu).
* [ ] Setelah mengubah `requires_grad`, **re-init optimizer**.

---

## 7) Istilah Singkat

* **Backbone**: bagian model untuk belajar representasi visual umum.
* **Classifier head**: lapisan akhir untuk memetakan fitur → kelas.
* **Feature extraction (freeze)**: latih head saja, backbone tetap.
* **Fine-tuning**: menyesuaikan (sebagian/seluruh) backbone untuk task baru.
* **Discriminative LR**: LR berbeda per bagian model (head > backbone).
* **Gradual unfreezing**: membuka trainability layer demi layer untuk stabilitas.

---

## 8) Referensi singkat (untuk ditinjau lebih lanjut)

* Transfer learning & linear probing pada visi (umum).
* Praktik **BiT (Big Transfer)** untuk fine-tuning skala besar.
* Strategi **gradual unfreezing** (terkenal di NLP, relevan untuk stabilitas).
* Studi domain-spesifik (mis. medis/satelit) yang menunjukkan pentingnya strategi selektif.

---

## 9) Ringkasan Eksekutif

* Mulai **freeze backbone** untuk baseline yang cepat & stabil.
* Lanjutkan **unfreeze bertahap** + **discriminative LR** bila metrik masih membaik.
* Akhiri dengan **full fine-tune** jika dataset dan augmentasi mendukung—biasanya memberi performa terbaik.

> Kunci sukses: **ukur tiap tahap**, simpan model terbaik (early stopping), dan kelola LR/regularisasi dengan hati-hati.
