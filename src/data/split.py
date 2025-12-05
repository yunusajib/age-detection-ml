import os
import pandas as pd
from sklearn.model_selection import train_test_split


def create_age_bins(age_series):
    bins = [0, 10, 20, 30, 40, 50, 60, 200]
    labels = ["0-10", "10-20", "20-30", "30-40", "40-50", "50-60", "60+"]
    return pd.cut(age_series, bins=bins, labels=labels, include_lowest=True)


def stratify_key(df):
    df["age_bin"] = create_age_bins(df["age"])
    df["stratify_group"] = (
        df["age_bin"].astype(str) + "_" +
        df["gender"].astype(str) + "_" +
        df["ethnicity"].astype(str)
    )
    return df


def split_dataset(df, test_size=0.2, val_size=0.1, seed=42):

    df = stratify_key(df)

    # Train + temp split
    train_df, temp_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df["stratify_group"]
    )

    # Validation vs Test
    val_ratio = val_size / (1 - test_size)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - val_ratio),
        random_state=seed,
        stratify=temp_df["stratify_group"]
    )

    # Drop helper columns
    train_df = train_df.drop(columns=["age_bin", "stratify_group"])
    val_df = val_df.drop(columns=["age_bin", "stratify_group"])
    test_df = test_df.drop(columns=["age_bin", "stratify_group"])

    return train_df, val_df, test_df


def save_splits(train_df, val_df, test_df, out_dir="data/processed"):
    os.makedirs(out_dir, exist_ok=True)
    train_df.to_csv(os.path.join(out_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(out_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(out_dir, "test.csv"), index=False)
    print("Saved train/val/test splits into:", out_dir)
