import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

def run_eda(df: pd.DataFrame):
    print("Running Exploratory Data Analysis...")

    ## Check the datashape and the ratio between benign and pathogenic variants
    print("=" * 60)
    print("DATASET SHAPE & LABEL BALANCE")
    print("=" * 60)
    print(f"Labeled data shape: {df.shape}")

    counts = df['target'].value_counts()
    percentages = df['target'].value_counts(normalize=True) * 100
    print("\nClass Distribution:")
    print(f"  - Benign (0):     {counts.get(0, 0):,} ({percentages.get(0, 0):.2f}%)")
    print(f"  - Pathogenic (1): {counts.get(1, 0):,} ({percentages.get(1, 0):.2f}%)")

    imbalance_ratio = counts.get(0, 1) / counts.get(1, 1) if counts.get(1, 0) > 0 else 0
    print(f"  - Imbalance Ratio (Benign:Pathogenic): {imbalance_ratio:.2f}:1")

    ## Check the distribution of genes and the diversity of variants per gene
    print("\n" + "=" * 60)
    print("GENE DISTRIBUTION & DIVERSITY")
    print("=" * 60)
    unique_gene = df['GeneSymbol'].nunique()
    print(f"Number of unique genes: {unique_gene}")
    print("\nTop 5 Genes with Most Variants:")
    print(df['GeneSymbol'].value_counts().head(5))

    ## Check for missing values in the dataset
    print("\n" + "=" * 60)
    print("MISSING VALUE AUDIT (Top Columns)")
    print("=" * 60)
    missing = df.isna().sum()
    missing_percent = (missing / len(df)) * 100
    missing_df = pd.DataFrame({'Missing Values': missing, 'Percentage': missing_percent})
    if missing_df['Missing Values'].sum() == 0:
        print("No missing values found in the dataset.")
    else:
        print("Columns with Missing Values:")
        print(missing_df[missing_df['Missing Values'] > 0].sort_values(by='Missing Values', ascending=False).head(10))

def build_base_features(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)

    # 1. Reference & Alternate Allele Lengths
    ref_str = df['ReferenceAllele'].astype(str)
    alt_str = df['AlternateAllele'].astype(str)

    # Checking for single nucleotide variants (SNVs)
    X['ref_len'] = ref_str.str.len()
    X['alt_len'] = alt_str.str.len()
    X['len_diff'] = X['alt_len'] - X['ref_len']

    # 2. Single Nucleotide Variant (SNV) Transition vs Transversion Indicator
    purines = {'A', 'G'}
    pyrimidines = {'C', 'T'}

    def classify_mutation(row):
        ref, alt = str(row['ReferenceAllele']).upper(), str(row['AlternateAllele']).upper()

        if len(ref) == 1 and len(alt) == 1 and ref in (purines | pyrimidines) and alt in (purines | pyrimidines):
            if (ref in purines and alt in purines) or (ref in pyrimidines and alt in pyrimidines):
                return 1  # Transition (A<->G or C<->T)
            return 0      # Transversion (Purine <-> Pyrimidine)
        return -1         # Indel or complex change

    X['is_transition'] = df.apply(classify_mutation, axis=1)

    # 3. Genomic Position (as float for matrix operations)
    if 'PositionVCF' in df.columns:
        X['position'] = pd.to_numeric(df['PositionVCF'], errors='coerce').fillna(0)
    elif 'StartPosition' in df.columns:
        X['position'] = pd.to_numeric(df['StartPosition'], errors='coerce').fillna(0)
    
    return X

def perform_gene_grouped_split(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    """
    Splits dataset into train and test sets grouped by GeneSymbol.
    Ensures all variants from any single gene stay strictly in train OR test.
    """
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    genes = df['GeneSymbol'].fillna('UNKNOWN_GENE')

    train_idx, test_idx = next(gss.split(df, groups=genes))
    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()

    print("\n" + "=" * 60)
    print("GROUPED TRAIN / TEST SPLIT COMPLETED")
    print("=" * 60)
    print(f"Train Set: {len(train_df):,} variants ({train_df['GeneSymbol'].nunique():,} genes)")
    print(f"Test Set:  {len(test_df):,} variants ({test_df['GeneSymbol'].nunique():,} genes)")

    overlap = set(train_df['GeneSymbol']).intersection(set(test_df['GeneSymbol'])) - {'UNKNOWN_GENE'}
    print(f"Gene Leakage Count: {len(overlap)} genes in common (Target: 0)")

    return train_df, test_df