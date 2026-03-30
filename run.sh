#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
#  run.sh — Master Pipeline Script
#  ═══════════════════════════════════════════════════════════════════════════════
#  Executes the COMPLETE image retrieval pipeline end-to-end for ALL datasets
#  and ALL methods as required by the assignment:
#
#    Assignment Tasks Covered:
#      Task 1: Classical Retrieval using LBP
#      Task 2: Retrieval using NN and Deep NN
#      Task 3: CNN-based Retrieval
#      Task 4: Novel Feature Extraction (DSFM)
#      Task 5: Hybrid Retrieval Model (OSAG)
#      Task 6: Inter- and Intra-Color Feature Analysis
#      + Comparative Analysis Table (Precision, Recall, mAP)
#      + Robustness to Transformations (CNN & OSAG)
#      + Cross-Dataset Generalization (Zero-Shot)
#
#    Datasets:
#      - CIFAR-10   (mandatory)
#      - MNIST      (mandatory)
#      - Flowers102 (additional dataset — chosen for domain diversity)
#
#    Pipeline order per method:
#      download → sanity_check → preprocess → train → extract → retrieve → evaluate
#
#  Usage:
#      chmod +x run.sh
#      ./run.sh              # Full run
#      ./run.sh 2>&1 | tee run.log   # Full run with log capture
# ═══════════════════════════════════════════════════════════════════════════════

set -e  # Exit on first error — fail hard, zero silent bugs

PYTHON="./dlcv-env/bin/python"
K=10
EPOCHS=20

DATASETS=("cifar10" "mnist" "flowers102")
NEURAL_METHODS=("nn" "dnn" "cnn" "dsfm" "osag")
CLASSICAL_METHODS=("lbp" "color")
ALL_METHODS=("lbp" "color" "nn" "dnn" "cnn" "dsfm" "osag")

echo "═══════════════════════════════════════════════════════════════════════"
echo " Image Retrieval Pipeline — Full Execution"
echo " Datasets:  ${DATASETS[*]}"
echo " Methods:   ${ALL_METHODS[*]}"
echo " Top-K:     $K"
echo " Epochs:    $EPOCHS"
echo " Python:    $PYTHON"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""

START_TIME=$SECONDS

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 0: DATA DOWNLOAD (Optional)
# ─────────────────────────────────────────────────────────────────────────────
echo "╔═══════════════════════════════════════════════════════════════════╗"
echo "║  PHASE 0: DATA DOWNLOAD                                         ║"
echo "╚═══════════════════════════════════════════════════════════════════╝"

if [ ! -d "data" ] || [ -z "$(ls -A data)" ]; then
    echo "▶ Data directory missing or empty. Downloading datasets..."
    $PYTHON src/stages/download_datasets.py --datasets "${DATASETS[@]}"
else
    echo "▶ Data already present in data/. Skipping download."
    echo "  (Run 'rm -rf data' first if you want a fresh download)"
fi

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1: SANITY CHECK
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "╔═══════════════════════════════════════════════════════════════════╗"
echo "║  PHASE 1: DATA SANITY CHECK                                     ║"
echo "╚═══════════════════════════════════════════════════════════════════╝"

for ds in "${DATASETS[@]}"; do
    echo ""
    echo "▶ Sanity check: $ds"
    $PYTHON src/stages/sanity_check.py --dataset "$ds"
done

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2: PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "╔═══════════════════════════════════════════════════════════════════╗"
echo "║  PHASE 2: PREPROCESSING                                         ║"
echo "╚═══════════════════════════════════════════════════════════════════╝"

for ds in "${DATASETS[@]}"; do
    echo ""
    echo "▶ Preprocessing: $ds"
    $PYTHON src/stages/preprocess.py --dataset "$ds"
done

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 3: TRAINING (Neural methods only)
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "╔═══════════════════════════════════════════════════════════════════╗"
echo "║  PHASE 3: TRAINING (Tasks 2, 3, 4, 5)                           ║"
echo "╚═══════════════════════════════════════════════════════════════════╝"

for ds in "${DATASETS[@]}"; do
    for method in "${NEURAL_METHODS[@]}"; do
        echo ""
        echo "▶ Training: $ds / $method"
        $PYTHON src/stages/train_model.py --dataset "$ds" --method "$method" --epochs "$EPOCHS"
    done
done

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 4: FEATURE EXTRACTION (All methods)
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "╔═══════════════════════════════════════════════════════════════════╗"
echo "║  PHASE 4: FEATURE EXTRACTION (Tasks 1, 2, 3, 4, 5, 6)           ║"
echo "╚═══════════════════════════════════════════════════════════════════╝"

for ds in "${DATASETS[@]}"; do
    for method in "${ALL_METHODS[@]}"; do
        echo ""
        echo "▶ Extracting features: $ds / $method"
        $PYTHON src/stages/extract_features.py --dataset "$ds" --method "$method"
    done
done

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 5: RETRIEVAL (All methods)
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "╔═══════════════════════════════════════════════════════════════════╗"
echo "║  PHASE 5: TOP-K RETRIEVAL                                       ║"
echo "╚═══════════════════════════════════════════════════════════════════╝"

for ds in "${DATASETS[@]}"; do
    for method in "${ALL_METHODS[@]}"; do
        echo ""
        echo "▶ Retrieving top-$K: $ds / $method"
        $PYTHON src/stages/retrieve.py --dataset "$ds" --method "$method" -k "$K"
    done
done

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 6: EVALUATION (All methods)
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "╔═══════════════════════════════════════════════════════════════════╗"
echo "║  PHASE 6: EVALUATION — Precision, Recall, mAP                   ║"
echo "╚═══════════════════════════════════════════════════════════════════╝"

for ds in "${DATASETS[@]}"; do
    for method in "${ALL_METHODS[@]}"; do
        echo ""
        echo "▶ Evaluating: $ds / $method"
        $PYTHON src/stages/evaluate.py --dataset "$ds" --method "$method"
    done
done

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 7: COMPARATIVE ANALYSIS TABLE
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "╔═══════════════════════════════════════════════════════════════════╗"
echo "║  PHASE 7: COMPARATIVE ANALYSIS TABLE                             ║"
echo "╚═══════════════════════════════════════════════════════════════════╝"
echo " Generating side-by-side Precision / Recall / mAP tables..."

for ds in "${DATASETS[@]}"; do
    echo ""
    echo "▶ Comparison table: $ds"
    $PYTHON src/experiments/run_all.py --dataset "$ds" --methods "${ALL_METHODS[@]}" -k "$K"
done

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 8: ROBUSTNESS ANALYSIS (Task 3 requirement)
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "╔═══════════════════════════════════════════════════════════════════╗"
echo "║  PHASE 8: ROBUSTNESS TO TRANSFORMATIONS (CNN & OSAG)             ║"
echo "╚═══════════════════════════════════════════════════════════════════╝"

for ds in "${DATASETS[@]}"; do
    for method in "cnn" "osag"; do
        echo ""
        echo "▶ Robustness analysis: $ds / $method"
        $PYTHON src/experiments/robustness.py --dataset "$ds" --method "$method" -k "$K"
    done
done

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 9: CROSS-DATASET GENERALIZATION
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "╔═══════════════════════════════════════════════════════════════════╗"
echo "║  PHASE 9: CROSS-DATASET ZERO-SHOT EVALUATION                    ║"
echo "╚═══════════════════════════════════════════════════════════════════╝"
echo " Training on CIFAR-10, evaluating on MNIST (and vice versa)..."

for method in "cnn" "osag"; do
    echo ""
    echo "▶ Cross-dataset: cifar10 → mnist / $method"
    $PYTHON src/experiments/cross_dataset.py --source cifar10 --target mnist --method "$method" -k "$K"
    
    echo ""
    echo "▶ Cross-dataset: mnist → cifar10 / $method"
    $PYTHON src/experiments/cross_dataset.py --source mnist --target cifar10 --method "$method" -k "$K"
done

# ─────────────────────────────────────────────────────────────────────────────
# DONE
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo " PIPELINE COMPLETE"
echo " Total wall time: $(( SECONDS - START_TIME )) seconds"
echo ""
echo " Results location:"
echo "   outputs/{dataset}/{method}/evaluate/metrics.json   — per-method scores"
echo "   outputs/{dataset}/comparison.json                  — side-by-side table"
echo "   outputs/{dataset}/{method}/robustness/results.json — augmentation analysis"
echo "   outputs/cross_dataset/{src}_to_{tgt}/{method}/     — zero-shot scores"
echo "═══════════════════════════════════════════════════════════════════════"
