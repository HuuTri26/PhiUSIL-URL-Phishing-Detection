import os
import glob
import pandas as pd
import argparse
import sys

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='URL Multi-Labels Dataset Merger')
    parser.add_argument('--dir', type=str, required=True, help='Directory name of dataset')
    parser.add_argument('--o', type=str, default="final_dataset.csv", help='Output file name (default: final_dataset.csv)')
    args = parser.parse_args()
    
    dir_path = os.path.join(BASE_DIR, args.dir)
    if not os.path.isdir(dir_path):
        print(f"Directory not found: '{dir_path}'")
        sys.exit(1)
    
    all_files = sorted(glob.glob(os.path.join(dir_path, "*.csv")))
    if not all_files:
        print(f"No CSV files found in directory: '{dir_path}'")
        sys.exit(1)
    
    df_list = []
    for file_path in all_files:
        try:
            df = pd.read_csv(file_path, sep=";", engine="python", encoding="utf-8")
            df_list.append(df)
        except Exception as e:
            print(f"Error reading '{file_path}': {e}")
            sys.exit(1)
    
    merged_df = pd.concat(df_list, ignore_index=True)
    merged_df.to_csv(args.o, index=False)
    print(f"Merged {len(all_files)} files into {args.o}")
