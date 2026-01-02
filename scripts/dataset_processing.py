import os
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import sys
import re

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def chk_dup(df):
    df["url"] = df["url"].astype(str).str.strip()
    df["label"] = df["label"].astype(int)
    duplicates = df[df.duplicated(subset=["url"])]
    print(f"Found duplicated rows by 'url':\n{duplicates}")
    return df.drop_duplicates(subset=["url"])


def chk_stat_col(df, col_name):
    """Check statistic of specific column: count all values and display ratio using matplotlib"""
    if col_name not in df.columns:
        print(f"Column '{col_name}' not found in dataframe.")
        return df
    value_counts = df[col_name].value_counts(dropna=False)
    ratios = value_counts / value_counts.sum()
    print(f"Value counts for column '{col_name}':\n{value_counts}")
    print(f"Ratios for column '{col_name}':\n{ratios}")
    plt.figure(figsize=(8, 6))
    ratios.plot(kind="pie", autopct="%1.1f%%", startangle=90, legend=False)
    plt.title(f"Ratio of values in '{col_name}'")
    plt.ylabel("")
    plt.show()
    return df


def chk_NaN(df):
    """Check NaN"""
    nan_counts = df.isna().sum()
    print("NaN counts per column:")
    print(nan_counts)
    rows_with_nan = df[df.isna().any(axis=1)]
    print(f"Rows with any NaN values ({len(rows_with_nan)} rows):")
    print(rows_with_nan)
    return df.dropna()


def chk_null(df):
    """Check null"""
    null_counts = df.isnull().sum()
    print("Null counts per column:")
    print(null_counts)
    rows_with_null = df[df.isnull().any(axis=1)]
    print(f"Rows with any null values ({len(rows_with_null)} rows):")
    print(rows_with_null)
    return df.drop(columns=rows_with_null.columns)


def chk_url_existence(df, compare_file):
    """Check if URLs exist in another CSV file and output missing URLs with their labels"""
    try:
        compare_df = pd.read_csv(
            compare_file, sep=";", engine="python", encoding="utf-8"
        )
        if "url" not in compare_df.columns:
            print(f"Column 'url' not found in comparison file: {compare_file}")
            return df
        df["url"] = df["url"].astype(str).str.strip()
        compare_df["url"] = compare_df["url"].astype(str).str.strip()
        missing_urls = df[~df["url"].isin(compare_df["url"])]
        if missing_urls.empty:
            print(f"All URLs from the input file exist in {compare_file}")
        else:
            print(f"URLs missing in {compare_file} ({len(missing_urls)} URLs):")
            print(missing_urls[["url", "label"]])
            missing_urls.to_csv(
                "missing.csv", index=False, sep=";", encoding="utf-8-sig"
            )
        return df
    except Exception as e:
        print(f"Error reading comparison file '{compare_file}': {e}")
        return df


def drop_col(df, col_name):
    """Drop specific columns"""
    if col_name not in df.columns:
        print(f"Column '{col_name}' not found in dataframe. Skipping drop.")
        return df
    print(f"Dropping column: {col_name}")
    return df.drop(columns=[col_name])


def drop_all_cols(df, except_cols):
    """Drop all columns except the specified columns"""
    missing_cols = [col for col in except_cols if col not in df.columns]
    if missing_cols:
        print(
            f"Warning: Columns {missing_cols} not found in dataframe. Proceeding with available columns."
        )
    valid_except_cols = [col for col in except_cols if col in df.columns]
    if not valid_except_cols:
        print("No valid columns to keep. Returning original dataframe.")
        return df
    print(f"Dropping all columns except: {valid_except_cols}")
    return df[valid_except_cols]


def drop_null_cols(df):
    """Drop all columns that contain any null values"""
    null_counts = df.isnull().sum()
    cols_with_null = null_counts[null_counts > 0].index.tolist()
    if not cols_with_null:
        print("No columns with null values found.")
        return df
    print(f"Dropping columns with null values: {cols_with_null}")
    return df.drop(columns=cols_with_null)


def add_col(df, col_name, col_val):
    """Add specific value for columns"""
    if col_name in df.columns:
        print(f"Column '{col_name}' already exists. Overwriting with value: {col_val}")
    else:
        print(f"Adding column: {col_name} with value: {col_val}")
    df[col_name] = col_val
    return df


def merge_files(df, second_file, output_file):
    """Merge two CSV files based on 'url' column, ensuring consistent labels and combining features"""
    try:
        # Read the second CSV file
        second_df = pd.read_csv(second_file, sep=";", engine="python", encoding="utf-8")

        # Verify required columns
        required_cols = ["url", "label"]
        for df_name, df_check in [("first file", df), ("second file", second_df)]:
            missing_cols = [col for col in required_cols if col not in df_check.columns]
            if missing_cols:
                print(f"Error: Columns {missing_cols} not found in {df_name}.")
                return df

        # Ensure URLs are strings and stripped
        df["url"] = df["url"].astype(str).str.strip()
        second_df["url"] = second_df["url"].astype(str).str.strip()
        df["label"] = df["label"].astype(int)
        second_df["label"] = second_df["label"].astype(int)

        # Check for label consistency
        merged_check = pd.merge(
            df[["url", "label"]],
            second_df[["url", "label"]],
            on="url",
            how="inner",
            suffixes=("_first", "_second"),
        )
        label_mismatches = merged_check[
            merged_check["label_first"] != merged_check["label_second"]
        ]
        if not label_mismatches.empty:
            print(
                f"Warning: Found {len(label_mismatches)} URLs with mismatched labels:"
            )
            print(label_mismatches[["url", "label_first", "label_second"]])

        # Perform full merge (inner) to keep only URLs present in both files
        merged_df = pd.merge(
            df, second_df, on="url", how="inner", suffixes=("_first", "_second")
        )

        # If labels are identical, keep only one label column
        if "label_first" in merged_df.columns and "label_second" in merged_df.columns:
            if (merged_df["label_first"] == merged_df["label_second"]).all():
                merged_df = merged_df.drop(columns=["label_second"])
                merged_df = merged_df.rename(columns={"label_first": "label"})
            else:
                print(
                    "Warning: Labels are not consistent across files. Keeping both label columns."
                )

        # Reorder columns to ensure label(s) are at the end
        cols = merged_df.columns.tolist()
        if "label" in cols:
            cols.remove("label")
            cols.append("label")
        elif "label_first" in cols and "label_second" in cols:
            cols.remove("label_first")
            cols.remove("label_second")
            cols.extend(["label_first", "label_second"])
        merged_df = merged_df[cols]

        # Report URLs not present in both files
        urls_first_only = df[~df["url"].isin(second_df["url"])][["url", "label"]]
        urls_second_only = second_df[~second_df["url"].isin(df["url"])][
            ["url", "label"]
        ]
        if not urls_first_only.empty:
            print(
                f"URLs in first file but not in second file ({len(urls_first_only)} URLs):"
            )
            print(urls_first_only)
        if not urls_second_only.empty:
            print(
                f"URLs in second file but not in first file ({len(urls_second_only)} URLs):"
            )
            print(urls_second_only)

        # Save merged DataFrame to output file
        if output_file:
            merged_df.to_csv(output_file, index=False, sep=";", encoding="utf-8-sig")
            print(f"Merged data saved to {output_file}")

        return merged_df
    except Exception as e:
        print(f"Error merging files with '{second_file}': {e}")
        return df


MODES = {
    "chk_dup": chk_dup,
    "chk_stat_col": chk_stat_col,
    "chk_NaN": chk_NaN,
    "chk_null": chk_null,
    "chk_url_existence": chk_url_existence,
    "drop_col": drop_col,
    "drop_all_cols": drop_all_cols,
    "drop_null_cols": drop_null_cols,
    "add_col": add_col,
    "merge_files": merge_files,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="URL Multi-Labels Dataset Processing")
    parser.add_argument(
        "--dir", type=str, required=True, help="Directory name of dataset"
    )
    parser.add_argument(
        "--modes", type=str, required=True, nargs="+", help="Processing modes"
    )
    parser.add_argument(
        "--cols", type=str, required=False, nargs="+", help="Specific columns"
    )
    parser.add_argument(
        "--except_cols", type=str, required=False, nargs="+", help="Exceptional columns"
    )
    parser.add_argument(
        "--val",
        type=str,
        required=False,
        help="Specific value for columns (add_col only)",
    )
    parser.add_argument(
        "--compare_file",
        type=str,
        required=False,
        help="Comparison CSV file for chk_url_existence mode",
    )
    parser.add_argument("--o", type=str, required=False, help="Output file name")
    parser.add_argument(
        "--files", type=str, required=True, nargs="+", help="Processing files"
    )
    args = parser.parse_args()

    dir_path = os.path.join(BASE_DIR, args.dir)
    if not os.path.isdir(dir_path):
        print(f"Directory not found: '{dir_path}'")
        sys.exit(1)

    if not args.files:
        print(f"No CSV files chosen: '{dir_path}'")
        sys.exit(1)

    if not args.modes or not all(mode in MODES for mode in args.modes):
        print(f"Invalid mode arguments chosen. Available modes: {list(MODES.keys())}")
        sys.exit(1)

    for file in args.files:
        try:
            file_path = os.path.join(dir_path, file)
            df = pd.read_csv(file_path, sep=",", engine="python", encoding="utf-8-sig")

            for mode in args.modes:
                if mode == "chk_stat_col" or mode == "drop_col":
                    if args.cols:
                        for col_name in args.cols:
                            df = MODES[mode](df, col_name)
                    else:
                        col_name = df.columns[0]
                        df = MODES[mode](df, col_name)
                elif mode == "drop_all_cols":
                    if args.except_cols:
                        df = MODES[mode](df, args.except_cols)
                    else:
                        print("drop_all_cols mode requires --except_cols argument.")
                        sys.exit(1)
                elif mode == "add_col":
                    if not args.cols or not args.val:
                        print("add_col mode requires both --cols and --val arguments.")
                        sys.exit(1)
                    if isinstance(args.val, list) and len(args.cols) == len(args.val):
                        for col_name, col_val in zip(args.cols, args.val):
                            df = MODES[mode](df, col_name, col_val)
                    elif isinstance(args.val, list) and len(args.val) == 1:
                        for col_name in args.cols:
                            df = MODES[mode](df, col_name, args.val[0])
                    elif not isinstance(args.val, list):
                        for col_name in args.cols:
                            df = MODES[mode](df, col_name, args.val)
                    else:
                        print("Number of columns and values for add_col do not match.")
                        sys.exit(1)
                elif mode == "chk_url_existence":
                    if not args.compare_file:
                        print(
                            "chk_url_existence mode requires --compare_file argument."
                        )
                        sys.exit(1)
                    compare_file_path = os.path.join(dir_path, args.compare_file)
                    df = MODES[mode](df, compare_file_path)
                elif mode == "merge_files":
                    if len(args.files) != 2:
                        print("merge_files mode requires exactly two input files.")
                        sys.exit(1)
                    if not args.o:
                        print("merge_files mode requires --o (output file) argument.")
                        sys.exit(1)
                    second_file_path = os.path.join(dir_path, args.files[1])
                    df = MODES[mode](df, second_file_path, args.o)
                    break  # Exit after merging to avoid processing the second file separately
                else:
                    df = MODES[mode](df)
        except Exception as e:
            print(f"Error processing '{file_path}': {e}")
            sys.exit(1)

    if args.o and "merge_files" not in args.modes:
        df.to_csv(args.o, index=False, sep=";", encoding="utf-8-sig")
