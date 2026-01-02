#!/usr/bin/env python3
import pandas as pd
from url_features_extractor import URL_EXTRACTOR
import os
import argparse
import sys
import re
import glob
import tempfile

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_DIR = os.getcwd()  # nơi lưu checkpoint (thư mục hiện hành)

def find_latest_checkpoint(start_idx, end_idx, out_dir=OUT_DIR):
    pattern = os.path.join(out_dir, f"final_dataset_{start_idx}_*.csv")
    files = glob.glob(pattern)
    max_end = None
    max_path = None
    for f in files:
        m = re.match(rf".*final_dataset_{start_idx}_(\d+)\.csv$", f)
        if m:
            val = int(m.group(1))
            if val <= end_idx and (max_end is None or val > max_end):
                max_end = val
                max_path = f
    return max_path, max_end

def atomic_write_df(df, out_path):
    # write to temp then replace
    dirn = os.path.dirname(out_path) or "."
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", dir=dirn, text=True)
    os.close(fd)
    df.to_csv(tmp, index=False)
    os.replace(tmp, out_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='URL Multi-Labels Dataset Generator with checkpoint/resume')
    parser.add_argument('--file', type=str, required=True, help='Dataset file (relative to project base)')
    parser.add_argument('--start_idx', type=int, help='Start index (inclusive)', required=True)
    parser.add_argument('--end_idx', type=int, help='End index (exclusive)', required=True)
    parser.add_argument('--checkpoint_step', type=int, help='Checkpoint per step', default=200)
    args = parser.parse_args()

    DATASET_PATH = os.path.join(BASE_DIR, args.file)
    if not os.path.exists(DATASET_PATH):
        print(f"File {DATASET_PATH} not found!")
        sys.exit(1)

    df = pd.read_csv(DATASET_PATH, sep=";", engine="python", quotechar='"', encoding="utf-8")
    total = len(df)
    start = args.start_idx
    end = args.end_idx
    step = args.checkpoint_step

    if start < 0 or end > total or start >= end:
        print(f"Invalid start/end. dataset_len={total}, start={start}, end={end}")
        sys.exit(1)

    print(f"Extracting urls from {start} to {end} (count {end-start}), checkpoint_step={step}")
    # check existing checkpoint to resume
    latest_path, latest_end = find_latest_checkpoint(start, end)
    if latest_path:
        print(f"Found existing checkpoint: {latest_path} (checkpoint_end={latest_end}) -> resuming from {latest_end}")
        try:
            existing_df = pd.read_csv(latest_path)
            temp = existing_df.to_dict(orient="records")
        except Exception as e:
            print(f"Warning: cannot read existing checkpoint {latest_path}: {e}. Starting fresh.")
            temp = []
        resume_from = latest_end
    else:
        temp = []
        resume_from = start

    # prepare skipped log file
    skipped_path = os.path.join(OUT_DIR, f"skipped_{start}_{end}.csv")
    skipped_exists = os.path.exists(skipped_path)
    if not skipped_exists:
        # write header
        with open(skipped_path, "w", encoding="utf-8") as fh:
            fh.write("global_idx,url,label,reason\n")

    sliced_df = df.iloc[resume_from:end]
    if len(sliced_df) == 0:
        print("Nothing to process (resume_from >= end). Exiting.")
        sys.exit(0)

    try:
        for rel_idx, item in enumerate(sliced_df.itertuples(index=True), 1):
            global_idx = item.Index
            print("="*120)
            print(f"[{rel_idx}/{len(sliced_df)}] (global idx: {global_idx}) Extracting features for:")
            print(f"  URL  : {item.url}")
            print(f"  Label: {item.label}")

            try:
                extractor = URL_EXTRACTOR(item.url, item.label)
                data = extractor.extract_to_dataset()
            except Exception as e:
                reason = f"exception_init_or_extract: {e}"
                print(f"  ❌ Error extracting {item.url}: {reason}")
                with open(skipped_path, "a", encoding="utf-8") as fh:
                    fh.write(f"{global_idx},{item.url},{item.label},{reason}\n")
                # continue to next URL
                processed_global = global_idx
                # checkpoint logic still applies below
                pass
            else:
                # if crawler indicates site not alive -> skip but record reason
                if getattr(extractor, "content_features", {}).get("is_alive", 1) == 0:
                    reason = "crawler_not_alive"
                    print(f"  ⚠️ Skipping URL '{item.url}' ({reason})")
                    with open(skipped_path, "a", encoding="utf-8") as fh:
                        fh.write(f"{global_idx},{item.url},{item.label},{reason}\n")
                    processed_global = global_idx
                else:
                    # success
                    temp.append(data)
                    print(f"  URL '{item.url}' took '{round(extractor.exec_time,2)}' seconds to extract")
                    processed_global = global_idx

            # checkpoint decision
            # processed_global is the latest global index we've just finished (could be skipped)
            rel_processed_count = processed_global - start
            # If processed_global might be < start? not possible
            is_checkpoint = ((processed_global - start) % step == 0)
            is_last = (processed_global >= end - 1)  # global index is 0-based; end is exclusive

            if is_checkpoint or is_last:
                checkpoint_end = processed_global + 1  # make checkpoint_end exclusive index
                out_file = os.path.join(OUT_DIR, f"final_dataset_{start}_{checkpoint_end}.csv")
                df_checkpoint = pd.DataFrame(temp)
                try:
                    atomic_write_df(df_checkpoint, out_file)
                    print(f"  💾 Saved checkpoint: {out_file} (rows={len(df_checkpoint)})")
                except Exception as e:
                    print(f"  ❌ Failed to save checkpoint {out_file}: {e}")

        # finished loop
        print("="*120)
        final_out = os.path.join(OUT_DIR, f"final_dataset_{start}_{end}.csv")
        atomic_write_df(pd.DataFrame(temp), final_out)
        print(f"✅ Finished. Final dataset: {final_out} (rows={len(temp)})")

    except KeyboardInterrupt:
        print("Interrupted by user. Will try to save current progress.")
        # try to save a final checkpoint using last processed index if possible
        try:
            last_processed_global = resume_from + rel_idx - 1
            checkpoint_end = last_processed_global + 1
            out_file = os.path.join(OUT_DIR, f"final_dataset_{start}_{checkpoint_end}.csv")
            atomic_write_df(pd.DataFrame(temp), out_file)
            print(f"  💾 Saved checkpoint at interrupt: {out_file} (rows={len(temp)})")
        except Exception as e:
            print("  ❌ Failed to save checkpoint after interrupt:", e)
        sys.exit(1)
