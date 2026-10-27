# XGBoost Pathogenicity Predictor

A Python project for exploring ClinVar variant data and comparing baseline models for classifying variants as benign or pathogenic. The current pipeline builds a small set of allele and genomic-position features, then evaluates logistic regression and XGBoost using stratified cross-validation.

## Requirements

- Python 3.9 or newer
- The Python packages listed in `requirements.txt`
- A tab-separated ClinVar variant summary file

## Setup

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

## Data

The loader expects a tab-separated ClinVar variant summary with columns including `ClinicalSignificance`, `GeneSymbol`, `ReferenceAllele`, `AlternateAllele`, and either `PositionVCF` or `StartPosition`.

Download the ClinVar summary into the repository's `data/` directory:

```bash
mkdir -p data
curl -L https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz \
  -o data/variant_summary.txt.gz
gzip -dk data/variant_summary.txt.gz
```

The default input path is `data/variant_summary.txt`. The repository may also include smaller sample files for quick inspection; edit the path in `src/XGBPP/main.py` to use one of those instead.

The loader retains variants with these clinical significance labels:

- Pathogenic, Likely pathogenic, Pathogenic/Likely pathogenic → `1`
- Benign, Likely benign, Benign/Likely benign → `0`

All other labels are filtered out.

## Run

From the repository root:

```bash
python src/XGBPP/main.py
```

The program prints a basic dataset and missing-value summary, creates a gene-grouped train/test split, and runs five-fold stratified cross-validation on the training partition. It reports mean ROC-AUC and average precision (PR-AUC) for logistic regression and XGBoost.

## Features and evaluation notes

The current features are reference and alternate allele lengths, their length difference, a transition/transversion indicator (with indels and complex changes marked separately), and genomic position when available.

Variants are grouped by `GeneSymbol` during the train/test split to reduce gene overlap between partitions. The current script does not yet evaluate or report metrics on the held-out test partition; the reported scores are cross-validation scores on the training partition.

Five-fold stratified cross-validation needs enough examples of each label in the training data to distribute them across five folds. If one class has fewer than five examples, reduce the number of folds or, preferably, add more labeled data. Scores based on very few examples can be unstable.
