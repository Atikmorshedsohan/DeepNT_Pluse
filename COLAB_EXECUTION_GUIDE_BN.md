# Google Colab Execution Guide: DeepNT+ (বাংলা নির্দেশিকা)

এই নির্দেশিকাটি আপনাকে Google Colab-এ **DeepNT+**-এর সমস্ত কোড ও এক্সপেরিমেন্ট সহজে ও ক্রমানুসারে রান করতে সাহায্য করবে।

---

## ১. Colab Notebook-গুলোর তালিকা ও ভূমিকা

আপনার `colab/` ফোল্ডারে ৩টি সম্পূর্ণ Colab ফরম্যাটের নোটবুক প্রস্তুত রয়েছে:

| নোটবুকের নাম | ভূমিকা ও বৈশিষ্ট্য |
| :--- | :--- |
| **`DeepNT_Plus_Master_Benchmark.ipynb`** 🌟 *(সর্বাধিক প্রস্তাবিত)* | **অল-ইন-ওয়ান মাস্টার নোটবুক**। এতে ড্রপডাউন মেনু দিয়ে `run_03` (IPv4 compressed), `run_02` (IPv4 15-graph), এবং `run_ipv6` (Global IPv6) সিলেক্ট করা যায়। ৫টি সিডের বেঞ্চমার্ক, Leave-Monitor-Out (LMO), BALD Active Learning কার্ভ এবং প্রকাশনাযোগ্য সব চার্ট একসাথে জেনারেট করে। |
| **`DeepNT_Plus.ipynb`** | IPv4 ডেটাসেটের জন্য ডেডিকেটেড নোটবুক (৫ সিড এবং ল্যাপলাস লস সহ আপডেটেড)। |
| **`DeepNT_Plus_IPv6.ipynb`** | গ্লোবাল IPv6 ডেটাসেটের জন্য ডেডিকেটেড নোটবুক (৫ সিড এবং ওয়ার্ম-স্টার্ট প্রাইয়র সহ আপডেটেড)। |

---

## ২. Google Drive-এ ফাইল প্রস্তুতকরণ (Upload Structure)

Google Colab রান করার পূর্বে আপনার Google Drive-এর মূল ডিরেক্টরিতে (`My Drive`) নিম্নরূপ ফোল্ডার কাঠামো থাকতে হবে:

```
My Drive/
└── Thesis/
    ├── colab/
    │   ├── DeepNT_Plus_Master_Benchmark.ipynb
    │   ├── DeepNT_Plus.ipynb
    │   ├── DeepNT_Plus_IPv6.ipynb
    │   └── deepnt_plus/                 <-- (সম্পূর্ণ deepnt_plus প্যাকেজ ফোল্ডার)
    │       ├── __init__.py
    │       ├── data.py
    │       ├── model.py
    │       ├── train.py
    │       ├── baselines.py
    │       ├── layers.py
    │       ├── tgn.py
    │       ├── active.py
    │       ├── metrics.py
    │       ├── report.py
    │       └── run.py
    └── DeepNT_Output/
        ├── data/
        │   ├── processed/               <-- (IPv4 run_03 compressed ডেটা)
        │   └── processed_15graphs_backup <-- (IPv4 run_02 15-graph ডেটা)
        └── reconstructed_dataset/
            └── ipv6/                    <-- (IPv6 5-graph balanced ডেটা)
```

> **টিপ:** আপনি আপনার লোকাল পিসির `C:\Users\KINGDOM\Desktop\Thesis` ফোল্ডারটিকে সরাসরি Google Drive-এ আপলোড করলেই এই স্ট্রাকচার তৈরি হয়ে যাবে।

---

## ৩. `DeepNT_Plus_Master_Benchmark.ipynb` রান করার ধারাবাহিক ধাপসমূহ

Google Colab-এ নোটবুকটি ওপেন করার পর নিচের সিরিয়ালে সেলগুলো রান করুন:

### ধাপ ১: রানটাইম নির্বাচন (GPU Accelerated)
1. Colab মেনুবারে যান: **Runtime** $\to$ **Change runtime type**
2. **Hardware accelerator** সিলেক্ট করুন: **T4 GPU** (বা CPU দিলেও চলবে, কোড অত্যন্ত অপ্টিমাইজড)
3. **Save** চাপুন।

### ধাপ ২: Cell-by-Cell এক্সেকিউশন সিরিয়াল

* **Cell 1 (Environment Setup & Google Drive Mount):**
  * এটি রান করলে Google Drive পারমিশন চাইবে। `Connect to Google Drive`-এ ক্লিক করে পারমিশন দিন। ড্রাইভ স্বয়ংক্রিয়ভাবে `/content/drive` পাথে মাউন্ট হবে।
* **Cell 2 (Benchmark Configuration & Dataset Selector):**
  * `DATASET_CHOICE` ড্রপডাউন থেকে আপনার পছন্দমতো ডেটাসেট বাছাই করুন:
    * `run_03` (IPv4 5-graph, দ্রুততম ও সেরা পারফরম্যান্স)
    * `run_02` (IPv4 15-graph)
    * `run_ipv6` (Global IPv6 Ark)
  * সিড (`0, 1, 2, 3, 4`), Epochs (`40`), এবং লস (`laplace`) কনফিগার হবে।
* **Cell 3 (Dataset Loading & Topology Inspection):**
  * নোড, এজ, পাথ এবং অবজার্ভড/হিডেন পাথের সংখ্যা প্রিন্ট করবে।
* **Cell 4 (Model & Baseline Architecture Definition):**
  * Split-Weight GNN, Additive Tomography, Residual Fusion Gate ($b_g = -2.0$), CosineAnnealingLR, এবং আনসার্টেইনটি হেড লোড হবে।
* **Cell 5 (5-Seed Benchmark Runner):**
  * ৫টি সিডের জন্য সব মডেল (DeepNT+, DeepNT static, GAT, GraphSAGE, Convex, Median, Hop-count, RF, MLP) ক্রমান্বয়ে ট্রেইন ও টেস্ট করবে।
  * *সময়:* `run_03`-তে প্রায় ৩–৪ মিনিট; `run_ipv6`-তে প্রায় ১৫–২০ মিনিট।
* **Cell 6 (Results Summary Table Generation):**
  * Test MAE, RMSE, $R^2$, এবং Uncertainty Coverage (Cov 90%, ECE)-এর মিন ± স্ট্যান্ডার্ড ডেভিয়েশনের সুশৃঙ্খল টেবিল প্রদর্শন ও CSV-তে সেভ করবে।
* **Cell 7 (Leave-Monitor-Out Generalization Test):**
  * বাংলাদেশের মনিটর `dac-bd` বাদ দিয়ে ট্রেইন করে অদেখা মনিটরে জেনারেলিজেশনের ফলাফল দেখাবে।
* **Cell 8 (Empirical Active Learning Probing Curves):**
  * ১৫%, ৩০%, এবং ৪৫% পরিমাপ বাজেটে BALD বনাম Random বনাম Edge Coverage-এর অ্যাক্টিভ লার্নিং সিমুলেশন চালাবে।
* **Cell 9 (Publication-Quality Figures):**
  * তিনটি হাই-রেজোলিউশন গ্রাফ তৈরি করবে: (১) মডেলের MAE বার চার্ট, (২) True RTT বনাম Predicted RTT স্ক্যাটার প্লট, এবং (৩) Active Learning প্রোবিং কার্ভ।
* **Cell 10 (Export Summary & Drive Persistence):**
  * সব প্রেডিকশন এবং মেট্রিক্স ফাইল আপনার Google Drive-এর নির্দিষ্ট ফোল্ডারে স্থায়ীভাবে সেভ হয়েছে কি না তা নিশ্চিত করবে।

---

## ৪. এক্সপেক্টেড আউটপুট রেফারেন্স

আপনি সফলভাবে রান করলে নিচের মতো মূল ফলাফলগুলো দেখতে পাবেন:

1. **IPv4 (`run_03`):**
   * **DeepNT (static, causal):** Test MAE **43.53 ± 1.04 ms** ($R^2 = 0.612$) 🥇
   * **Convex Tomography:** Test MAE **46.28 ms**
   * **DeepNT+ (full, improved):** Test MAE **46.65 ± 3.28 ms**
2. **IPv4 (`run_02`):**
   * **DeepNT+ (full, improved):** Test MAE **48.71 ± 1.91 ms** (আগের 62.62 ms থেকে সমাধানকৃত)
   * **Convex Tomography:** Test MAE **48.37 ms**
3. **Global IPv6 (`run_ipv6`):**
   * **Median Baseline:** Test MAE **242.67 ms**
   * **DeepNT (static, causal):** Test MAE **246.22 ± 0.58 ms**
   * **DeepNT+ (full, improved):** Test MAE **246.33 ± 0.06 ms**
   * **Convex Tomography (warm-start):** Test MAE **265.95 ms** (আগের 289.51 ms থেকে উন্নত)
