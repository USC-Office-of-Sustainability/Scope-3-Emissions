"""
prepare_data.py — NAICS training data preparation pipeline.

Edit the settings at the top of this file, then run:
  python prepare_data.py
"""

from pathlib import Path
import pandas as pd

# ── File paths ─────────────────────────────────────────────────────────────────

INPUT_FILE  = "data/YOUR_INPUT_FILE.csv"
OUTPUT_FILE = "data/output.csv"

# ── Category columns & rules ───────────────────────────────────────────────────

CATEGORY_COL       = "ALT.SPEND.CATEGORY.LEVEL.2"  # raw spend-category input column
SPEND_CATEGORY_COL = "spend_category"               # standardized category output

# [substring_to_match (case-insensitive), label_to_assign]
# First match wins; unmatched rows get "UNCATEGORIZED"
CATEGORY_RULES = [
    ["IT HARDWARE",      "IT HARDWARE"],
    ["IT EQUIPMENT",     "IT HARDWARE"],
    ["LAB AND RESEARCH", "LAB"],
    ["OFFICE",           "OFFICE"],
    ["FURNITURE",        "FURNITURE"],
]

# ── Item grouping & NAICS code columns ────────────────────────────────────────

NAICS_CODE_COL = "NAICS.2022.CODE.FINAL"
NAICS_DESC_COL = "NAICS.2022.DESC.FINAL"

# Items sharing the same values in these columns are grouped together;
# the most common NAICS code within each group is selected as the result.
NAICS_KEY_COLS = [
    "LINE.ITEM.DESCRIPTION",
    "spend_category",
    "SUPPLIER",
]

# ── Steps to run ───────────────────────────────────────────────────────────────

RUN_SAMPLE               = False   # randomly sample N rows from input
SAMPLE_SIZE              = 10_000

RUN_ADD_CATEGORY         = True    # map CATEGORY_COL → SPEND_CATEGORY_COL via CATEGORY_RULES
RUN_DEDUP                = False   # drop exact-duplicate rows
RUN_MOST_COMMON_NAICS = False   # resolve conflicting NAICS codes per key group by majority


# ══════════════════════════════════════════════════════════════════════════════
# Step implementations
# ══════════════════════════════════════════════════════════════════════════════

def step_sample(df):
    total = len(df)
    if total <= SAMPLE_SIZE:
        print(f"  [sample] Only {total:,} rows — keeping all (requested {SAMPLE_SIZE:,}).")
        return df
    result = df.sample(n=SAMPLE_SIZE, random_state=42)
    print(f"  [sample] {total:,} → {len(result):,} rows")
    return result


def step_add_category(df):
    if CATEGORY_COL not in df.columns:
        raise ValueError(f"Column '{CATEGORY_COL}' not found. Available: {df.columns.tolist()}")

    rules = [(kw.upper(), label) for kw, label in CATEGORY_RULES]

    def categorize(val):
        v = str(val).upper()
        for kw, label in rules:
            if kw in v:
                return label
        return "UNCATEGORIZED"

    df = df.copy()
    df[SPEND_CATEGORY_COL] = df[CATEGORY_COL].apply(categorize)

    print(f"  [add_category] '{CATEGORY_COL}' → '{SPEND_CATEGORY_COL}'")
    for cat, cnt in df[SPEND_CATEGORY_COL].value_counts().items():
        print(f"    {cat:<22}  {cnt:>8,}  ({cnt / len(df) * 100:.1f}%)")

    unc = (df[SPEND_CATEGORY_COL] == "UNCATEGORIZED").sum()
    if unc:
        print(f"  [add_category] WARNING: {unc:,} rows are UNCATEGORIZED — consider adding rules.")
    else:
        print(f"  [add_category] All rows matched a rule.")
    return df


def step_dedup(df):
    before = len(df)
    result = df.drop_duplicates()
    print(f"  [dedup] {before:,} → {len(result):,} rows  (dropped {before - len(result):,} duplicates)")
    return result


def step_most_common_naics(df):
    target_cols = [NAICS_CODE_COL]
    if NAICS_DESC_COL in df.columns:
        target_cols.append(NAICS_DESC_COL)

    missing = [c for c in NAICS_KEY_COLS + [NAICS_CODE_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df.copy()
    for col in NAICS_KEY_COLS + target_cols:
        df[col] = df[col].astype(str).str.strip()

    rows_before  = len(df)
    groups_total = df.groupby(NAICS_KEY_COLS, sort=False).ngroups

    counts = (
        df.groupby(NAICS_KEY_COLS + target_cols, sort=False)
          .size()
          .reset_index(name="_count")
    )
    winner = (
        counts
        .sort_values(by=["_count", NAICS_CODE_COL], ascending=[False, True])
        .drop_duplicates(subset=NAICS_KEY_COLS, keep="first")
        .drop(columns=["_count"])
        .reset_index(drop=True)
    )

    rows_after  = len(winner)
    distinct    = df.groupby(NAICS_KEY_COLS)[NAICS_CODE_COL].nunique().reset_index(name="_n")
    conflicted  = (distinct["_n"] > 1).sum()
    no_conflict = (distinct["_n"] == 1).sum()

    print(f"  [most_common_naics] {rows_before:,} → {rows_after:,} rows  "
          f"({(1 - rows_after / rows_before) * 100:.1f}% reduction)")
    print(f"    Key groups    : {groups_total:,}")
    print(f"    No conflict   : {no_conflict:,}  ({no_conflict / groups_total * 100:.1f}%)")
    print(f"    Had conflicts : {conflicted:,}  ({conflicted / groups_total * 100:.1f}%)")

    if conflicted:
        top = distinct[distinct["_n"] > 1].sort_values("_n", ascending=False).head(5)
        print("    Top conflicted groups:")
        for _, row in top.iterrows():
            print(f"      [{int(row['_n'])} codes]  "
                  + "  |  ".join(str(row[c])[:40] for c in NAICS_KEY_COLS))

    return winner


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline runner
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline():
    input_path  = Path(INPUT_FILE)
    output_path = Path(OUTPUT_FILE)

    steps = []
    if RUN_SAMPLE:               steps.append("sample")
    if RUN_ADD_CATEGORY:         steps.append("add_category")
    if RUN_DEDUP:                steps.append("dedup")
    if RUN_MOST_COMMON_NAICS: steps.append("most_common_naics")

    if not steps:
        print("No steps are enabled — nothing to do.")
        return

    print(f"\n{'='*65}")
    print(f"  NAICS Data Preparation Pipeline")
    print(f"{'='*65}")
    print(f"  Input : {input_path}")
    print(f"  Output: {output_path}")
    print(f"  Steps : {' → '.join(steps)}")
    print(f"{'='*65}\n")

    print(f"Loading {input_path.name} …", flush=True)
    try:
        df = pd.read_csv(input_path, dtype=str, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(input_path, dtype=str, encoding="latin-1")
    df.columns = [c.upper() for c in df.columns]
    print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns.\n")

    if RUN_SAMPLE:
        print("── Step: sample " + "─" * 49)
        df = step_sample(df)
        print()

    if RUN_ADD_CATEGORY:
        print("── Step: add_category " + "─" * 43)
        df = step_add_category(df)
        print()

    if RUN_DEDUP:
        print("── Step: dedup " + "─" * 50)
        df = step_dedup(df)
        print()

    if RUN_MOST_COMMON_NAICS:
        print("── Step: most_common_naics " + "─" * 35)
        df = step_most_common_naics(df)
        print()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"{'='*65}")
    print(f"  Pipeline complete.")
    print(f"  Final output: {output_path}  ({len(df):,} rows)")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    if not Path(INPUT_FILE).exists():
        print(f"Error: input file not found: {INPUT_FILE}")
        raise SystemExit(1)
    run_pipeline()
