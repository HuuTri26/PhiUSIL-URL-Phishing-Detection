import pandas as pd
from url_features_extractor_static import URL_EXTRACTOR
import os
import argparse
import sys

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='URL Multi-Labels Dataset Generator')
    parser.add_argument('--file', type=str, required=True, help='Dataset file')
    parser.add_argument('--start_idx', type=int, help='Start index (inclusive)')
    parser.add_argument('--end_idx', type=int, help='End index (exclusive)')
    args = parser.parse_args()

    if args.file:
        DATASET_PATH = os.path.join(BASE_DIR, args.file)
    else:
        print(f"File {args.file} not found!")
        sys.exit(1)

    df = pd.read_csv(DATASET_PATH, sep=";", engine="python", quotechar='"', encoding="utf-8")
    total = len(df)
    temp = []

    if args.start_idx is not None and args.end_idx is not None:
        # Slice the DataFrame according to the provided indices
        sliced_df = df.iloc[args.start_idx:args.end_idx]
        print(f"Extracting urls from {args.start_idx} to {args.end_idx} ({len(sliced_df)})")
        
        for batch_idx, item in enumerate(sliced_df.itertuples(index=True), 1):
            print("="*150)
            print(f"[{batch_idx}/{len(sliced_df)}] (global idx: {item.Index}) Extracting features for:")
            print(f"  URL  : {item.url}")
            print(f"  Label: {item.label}")
            extractor = URL_EXTRACTOR(item.url, item.label)
            data = extractor.extract_to_dataset()
            print(f"  URL '{item.url}' took '{round(extractor.exec_time, 2)}' seconds to extract")
            temp.append(data)

        print("="*150)
        final_dataset = pd.DataFrame(temp)
        final_dataset.to_csv(f"final_dataset_{args.start_idx}_{args.end_idx}.csv", index=False)
    else:
        print(f"Extracting url all {len(df)} urls")
        for idx, item in enumerate(df.itertuples(index=False), 1):
            print("="*150)
            print(f"[{idx}/{total}] Extracting features for:")
            print(f"  URL  : {item.url}")
            print(f"  Label: {item.label}")
            extractor = URL_EXTRACTOR(item.url, item.label)
            data = extractor.extract_to_dataset()
            print(f"  URL '{item.url}' took '{round(extractor.exec_time, 2)}' seconds to extract")
            temp.append(data)

        print("="*150)
        final_dataset = pd.DataFrame(temp)
        final_dataset.to_csv("final_dataset.csv", index=False)