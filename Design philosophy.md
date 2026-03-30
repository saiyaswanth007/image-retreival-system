# 🧠 CORE DESIGN RULES (ENFORCED, NOT SUGGESTED)

### RULE 1 — Stage isolation

Each stage:

- reads ONLY from previous stage output
- writes ONLY to its own directory

---

### RULE 2 — Output reset

Before running a stage:

- its output directory is **deleted completely**

No partial overwrites. No silent reuse.

---

### RULE 3 — Method separation (NO COLLISIONS)

Every method gets its own namespace:

outputs/{dataset}/{method_name}/...

Example:

outputs/cifar10/cnn/  
outputs/cifar10/dsfm/  
outputs/cifar10/hybrid_osag/

---

### RULE 4 — Relative paths ONLY

No absolute paths anywhere.

Everything rooted at project root:

ROOT = Path(__file__).resolve().parents[2]

---

### RULE 5 — Deterministic execution

Each stage:

- has fixed inputs
- produces fixed outputs
- no hidden state

---

# 🏗️ REPO STRUCTURE (UPDATED FOR YOUR RULES)

image_retrieval/  
│  
├── src/  
│   ├── stages/  
│   │   ├── extract_features.py  
│   │   ├── train_model.py  
│   │   ├── build_index.py  
│   │   ├── retrieve.py  
│   │   └── evaluate.py  
│   │  
│   ├── models/  
│   ├── features/  
│   ├── fusion/  
│   ├── losses/  
│   └── utils/  
│  
├── outputs/  
│   └── (auto-created, never committed)  
│  
├── data/  
│  
├── configs/  
│   └── experiment.yaml  
│  
└── run.py

---

# 🔄 PIPELINE (STRICT STAGE EXECUTION)

[RAW DATA]  
   ↓  
(1) feature extraction  
   ↓  
(2) training  
   ↓  
(3) embedding generation  
   ↓  
(4) retrieval  
   ↓  
(5) evaluation

Each stage = **independent script**

---

# 📁 OUTPUT DIRECTORY DESIGN (CRITICAL)

outputs/  
 └── cifar10/  
      └── hybrid_osag/  
           ├── features/  
           │    ├── train.npy  
           │    └── test.npy  
           │  
           ├── model/  
           │    └── model.pt  
           │  
           ├── embeddings/  
           │    ├── train.npy  
           │    └── test.npy  
           │  
           ├── retrieval/  
           │    └── topk.json  
           │  
           └── metrics/  
                └── results.json

---

# 🔥 STAGE IMPLEMENTATION PATTERN

Every stage follows SAME pattern.

---

## 🔹 Example: Feature Extraction

def run_feature_extraction(cfg):  
    out_dir = ROOT / "outputs" / cfg.dataset / cfg.method / "features"  
  
    # 🔴 RULE: DELETE BEFORE RUN  
    if out_dir.exists():  
        shutil.rmtree(out_dir)  
    out_dir.mkdir(parents=True)  
  
    dataset = load_dataset(cfg.dataset)  
    extractor = build_feature(cfg.method)  
  
    features = []  
    for x in dataset:  
        features.append(extractor(x))  
  
    np.save(out_dir / "features.npy", features)

---

## 🔹 Example: Retrieval Stage

def run_retrieval(cfg):  
    base = ROOT / "outputs" / cfg.dataset / cfg.method  
  
    emb_path = base / "embeddings" / "test.npy"  
    out_dir = base / "retrieval"  
  
    if out_dir.exists():  
        shutil.rmtree(out_dir)  
    out_dir.mkdir()  
  
    embeddings = np.load(emb_path)  
  
    results = compute_topk(embeddings, k=cfg.top_k)  
  
    save_json(out_dir / "topk.json", results)

---

# 🧩 METHOD NAMING (NO CONFUSION EVER)

Method name = **explicit composition**

Examples:

cnn  
dsfm  
cnn+lbp  
cnn+dsfm+osag

---

### Enforced rule:

method_name = f"{cfg.backbone}+{cfg.feature}+{cfg.fusion}"

---

# ⚙️ CONFIG (NO MAGIC, JUST INPUT)

dataset: cifar10  
  
backbone: resnet18  
feature: dsfm  
fusion: osag  
  
top_k: 5  
batch_size: 128  
epochs: 50

---

# 🧠 EXECUTION CONTROL (run.py)

if __name__ == "__main__":  
    cfg = load_config()  
  
    run_feature_extraction(cfg)  
    run_training(cfg)  
    run_embedding(cfg)  
    run_retrieval(cfg)  
    run_evaluation(cfg)

---

# 🔒 STRICT VALIDATION RULES

Before each stage runs:

### ✅ Check inputs exist

assert input_path.exists(), "Previous stage output missing"

---

### ❌ If missing → FAIL HARD

No fallback. No auto-recompute.

# ⚠️ IMPORTANT EDGE RULES

### 1. Never mix datasets

Each dataset fully isolated:

outputs/mnist/  
outputs/cifar10/

---

### 2. Never overwrite another method

Each method isolated:

outputs/cifar10/cnn/  
outputs/cifar10/dsfm/

---

### 3. No shared temp files

Everything lives inside its method folder.

---

### 4. No implicit caching

If you want reuse → explicitly skip deletion.

---

# 🚀 OPTIONAL (HIGH VALUE ADDITIONS)

### ✔ Stage toggle

if cfg.run.feature:  
    run_feature_extraction(cfg)

---

### ✔ Dry run mode

Check paths without executing

---

### ✔ Logging per stage

logs/feature.log  
logs/train.log

---

# 🧨 Final Truth

This design gives you:

- **zero ambiguity**
- **zero silent bugs**
- **full traceability**
