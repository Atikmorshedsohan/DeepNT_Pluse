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

1. **IPv4 (`run_03`, ৫টি ঘনীভূত গ্রাফ):**
   * **DeepNT (Static + Convex Gate):** Test MAE **43.53 ± 1.04 ms** ($R^2 = 0.612$) 🥇
   * **Convex Tomography (Tuned Ridge):** Test MAE **46.28 ms** ($R^2 = 0.606$)
   * **DeepNT+ (Full Dynamic: Temporal + Convex):** Test MAE **46.65 ± 3.28 ms** ($R^2 = 0.595$)
   * **GAT (Pure Spatial GNN):** Test MAE **62.30 ± 5.60 ms**
   * **DeepNT (Neural Tomo, No Convex Gate):** Test MAE **65.17 ± 4.12 ms** (অ্যাবলেশন প্রমাণ)
   * **Hop-count (Linear w/ Intercept):** Test MAE **78.95 ms** ($R^2 \approx -0.018$, ইন্টারসেপ্ট সহ ফিক্সড)
   * **Median Baseline:** Test MAE **79.43 ms**
2. **IPv4 (`run_02`, ১৫টি গ্রাফ):**
   * **Convex Tomography (Tuned Ridge):** Test MAE **48.37 ms** ($R^2 = 0.572$)
   * **DeepNT+ (Full Dynamic: Temporal + Convex):** Test MAE **48.71 ± 1.91 ms** (কনভেক্সের সমকক্ষ, ব্যবধান std-এর ভেতরে)
   * **DeepNT (Static + Convex Gate):** Test MAE **48.74 ± 1.89 ms**
   * **GAT (Pure Spatial GNN):** Test MAE **54.33 ± 6.15 ms**
   * **DeepNT (Neural Tomo, No Convex Gate):** Test MAE **64.80 ± 3.50 ms**
   * **Median Baseline:** Test MAE **78.16 ms**
3. **Global IPv6 (`run_ipv6`, ৫টি গ্লোবাল গ্রাফ):**
   * **Median Baseline:** Test MAE **242.67 ms**
   * **Hop-count (Linear w/ Intercept):** Test MAE **242.67 ms**
   * **DeepNT (Static + Convex Gate):** Test MAE **246.22 ± 0.58 ms**
   * **Convex Tomography (Tuned Ridge, Warm-Start):** Test MAE **265.95 ms** (আগের 289.51 ms থেকে সমাধানকৃত)

---

## ৫. থিসিস মেথডোলজি ও সংশোধিত ৯টি মূল পয়েন্ট (Technical Reference)

1. **আসল DeepNT+ ও Temporal Branch:** Kaggle স্ক্রিপ্টে পূর্বে সরলীকৃত ক্লাস ছিল ও সময়ের মান শূন্য ছিল (`t=0`)। এখন লাইব্রেরির আসল `DeepNTPlus` ও `TemporalBranch` সংযুক্ত করা হয়েছে এবং `measurement_time` থেকে প্রকৃত `t_hours` পার্স করা হচ্ছে।
2. **ন্যায্য Convex Baseline:** পূর্বে Kaggle স্ক্রিপ্টে ফিক্সড $\alpha=0.01$ এবং প্রাইয়রহীন সূত্র থাকায় ফল খারাপ হয়ে 52.03 ms দেখাচ্ছিল। এখন লাইব্রেরির মতো Warm-Start Ridge ও ভ্যালিডেশন টিউনিং সহ রান_০৩-তে 46.28 ms এবং রান_০২-তে 48.37 ms ন্যায্যভাবে রিপোর্ট করা হচ্ছে।
3. **Hop-count Intercept:** ইন্টারসেপ্ট ($\beta_0 + \beta_1 \cdot \text{hops}$) যুক্ত করায় $R^2 = -15\%$ অ্যানোমালি দূর হয়ে Median-এর স্বাভাবিক সমতুল্য হয়েছে।
4. **হার্ডকোডেড ভ্যালু মুক্ত:** কোনো 14.0 বা 1.5 হার্ডকোড নেই; ট্রেনিং ডেটা থেকে গতিশীলভাবে `prior_edge_ms` ও রেট গণনা করা হচ্ছে।
5. **নামকরণের স্বচ্ছতা:** `DeepNT (Neural Tomo, No Convex)` (আগে কাগলে যাকে static বলা হচ্ছিল, MAE ~65 ms) এবং `DeepNT (Static + Convex Gate)` (লাইব্রেরির static, MAE 43.5 ms)-কে স্পষ্টভাবে আলাদা করা হয়েছে।
6. **Geo-দূরত্ব ও লিকেজ বন্ধ (`use_geo=False`):** অপূর্ণাঙ্গ lat/lon এবং সার্কুলার লিকেজ এড়াতে ডিফল্টভাবে `use_geo=False` কার্যকর। আগের 34.38/32.56 ফলকে থিসিসে আলাদা "Auxiliary Side-Information Track" হিসেবে উপস্থাপন করা হবে।
7. **ক্যালিব্রেশন মেট্রিক সামঞ্জস্য:** `score()` এবং `reliability_curve()` উভয় স্থানে ল্যাপলাস ডিস্ট্রিবিউশন ($-b \ln(1 - L)$) সুসংগত করা হয়েছে।
8. **GPU Graph Caching ও ৫টি সিড:** ট্রেনিং শুরুর আগে গ্রাফগুলো GPU-তে ক্যাশ করায় গতি বহুগুণ বৃদ্ধি পেয়েছে এবং সিড সংখ্যা ৫টি (`0, 1, 2, 3, 4`) করা হয়েছে।
9. **IPv6 টাইমস্ট্যাম্প সীমাবদ্ধতা:** CAIDA Ark IPv6 ডেটাসেটের বিভিন্ন ভ্যান্টেজ পয়েন্টে সুসংগত সময়কাল না থাকায় এটি সৎভাবে স্ট্যাটিক/নন-কজাল মেথডোলজিতে মূল্যায়ন করা হয়েছে।
