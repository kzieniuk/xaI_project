import os
import shutil

base_dir = "experiments_results/exchange_rate"
summary_file = os.path.join(base_dir, "sweep_summary.csv")

dir_10pct = "experiments_results/exchange_rate_10pct"
dir_2pct = "experiments_results/exchange_rate_2pct"

os.makedirs(dir_10pct, exist_ok=True)
os.makedirs(dir_2pct, exist_ok=True)

if not os.path.exists(summary_file):
    print(f"Error: {summary_file} does not exist.")
    exit(1)

with open(summary_file, 'r') as f:
    lines = f.readlines()

header = lines[0]
# Depending on how the append happened (if it added a header again?)
# The code said: if not file_exists: writer.writerow(header)
# So valid file exists -> no new header.
# So lines[1:] should be data.
# 150 samples * 3 models = 450 rows per run.

run1_lines = lines[1:451] # rows 0 to 449
run2_lines = lines[451:]  # rows 450 to end

print(f"Total lines: {len(lines)}")
print(f"Run 1 lines: {len(run1_lines)}")
print(f"Run 2 lines: {len(run2_lines)}")

with open(os.path.join(dir_10pct, "sweep_summary.csv"), 'w') as f:
    f.write(header)
    f.writelines(run1_lines)

with open(os.path.join(dir_2pct, "sweep_summary.csv"), 'w') as f:
    f.write(header)
    f.writelines(run2_lines)

print("Split complete.")
