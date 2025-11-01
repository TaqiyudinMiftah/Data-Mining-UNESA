# Data-Mining-UNESA
# Klasifikasi Makanan Indonesia 

Proyek ini adalah model *deep learning* untuk mengklasifikasikan 16 jenis makanan khas Indonesia. Model ini menggunakan arsitektur **Vision Transformer (ViT)**, khususnya `vit_b_16` yang sudah dilatih (pre-trained) pada dataset ImageNet.

Teknik *transfer learning* (fine-tuning) diterapkan untuk menyesuaikan model dengan dataset makanan Indonesia. Proyek ini dibangun menggunakan **PyTorch** dan **Torchvision**.

## 📋 Dataset

Dataset yang digunakan adalah kumpulan gambar pribadi yang dibagi menjadi 16 kelas. Struktur folder mengikuti format `ImageFolder` standar dari PyTorch, seperti yang terlihat pada data `train_with_label`.

**Kelas-kelas yang ada dalam dataset:**

* `Ayam Bakar`
* `Ayam Betutu`
* `Ayam Goreng`
* `Ayam Pop`
* `Bakso`
* `Coto Makassar`
* `Gado Gado`
* `Gudeg`
* `Nasi Goreng`
* `Pempek`
* `Rawon`
* `Rendang`
* `Sate Madura`
* `Sate Padang`
* `Soto`
