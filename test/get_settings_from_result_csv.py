import csv

ret = []
with open("smv_baseline_result.csv", 'r') as csv_file:
    reader = csv.DictReader(csv_file)
    for x in reader:
        ret.append((
            "SVM",
            1,
            1,
            1,
            1,
            float(x["Learning_rate"]),
            int(x["Batch_size"]),
            "bo",
            int(x["N_workers"]),
            int(x["N_intra"]),
            str(x["Optimizer"]),
            "dry_run"
        ))

print(ret)