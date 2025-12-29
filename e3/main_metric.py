import json
import numpy as np

def pass_at_k(n, c, k):
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

def get_pass_at_n(res, n_list=None): 
    if n_list is None:
        n_list = [1,2,4,8,16]
    dataset_passn = {}
    for key in res['acc']:
        pass_at_n = {}
        for n in n_list:
            pass_rate = []
            for item in res['acc'][key]:
                pass_rate.append(pass_at_k(len(item), sum(item), n))
            # store as a single-element list (kept as-is to minimize changes)
            pass_at_n[n] = [sum(pass_rate) / len(pass_rate) * 100]
        dataset_passn[key] = pass_at_n
    return dataset_passn

def compute_metric(metric_files, n_list=None):
    per_run_pass = {}
    per_run_length = {}
    for file in metric_files:
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)
        pass_at_n = get_pass_at_n(data, n_list)
        print("Pass at n:", pass_at_n)
        # unwrap the single-element lists so the output is clean per run
        per_run_pass[file] = {
            dataset: {n: vals[0] for n, vals in pass_at_n[dataset].items()}
            for dataset in pass_at_n.keys()
        }
        per_run_length[file] = data.get("length", {})
    return per_run_pass, per_run_length

if __name__ == "__main__":
    per_run_pass, per_run_length = compute_metric(
        [
            "./metrics/e3_1_5b_grpo_id_s32_1_metrics.json",
            "./metrics/e3_1_5b_grpo_2_id_s32_1_metrics.json",
            # "./metrics/e3_1_5b_grpo_3_id_s32_1_metrics.json",
            "./metrics/e3_1_5b_e3_id_s32_1_metrics.json",
            "./metrics/e3_1_5b_e3_2_id_s32_1_metrics.json",
            "./metrics/e3_1_5b_e3_3_id_s32_1_metrics.json",
            "./metrics/dGRPO_1_5b_0.00003_id_s32_1_metrics.json",
            "./metrics/dGRPO_1_5b_0.00001_id_s32_1_metrics.json",
            "./metrics/dgrpo_1_5b_0.00001_2_id_s32_1_metrics.json",
            "./metrics/dGRPO_1_5b_0.00001_3_id_s32_1_metrics.json",
            "./metrics/dGRPO_1_5b_0.00005_id_s32_1_metrics.json",
            "./metrics/dGRPO_0.5_seqq_norm_id_s32_1_metrics.json",
            "./metrics/dGRPO_1.0_seqq_norm_id_s32_1_metrics.json",
            "./metrics/dGRPO_1_5b_1.0_a_norm_id_s32_1_metrics.json",
            "./metrics/dGRPO_1_5b_1.0_a_norm_dyn_id_s32_1_metrics.json",
            "./metrics/dGRPO_1_5b_1.0_a1_id_s32_1_metrics.json",
            "./metrics/dGRPO_1_5b_1.0_a2_id_s32_1_metrics.json",
            "./metrics/dGRPO_1_5b_1.0_a3_id_s32_1_metrics.json",
            "./metrics/dGRPO_1_5b_1.0_a4_id_s32_1_metrics.json",
            "./metrics/dGRPO_1_5b_1.0_a5_id_s32_1_metrics.json",
            "./metrics/dGRPO_1_5b_1.0_a6_id_s32_1_metrics.json",
            "./metrics/dGRPO_1_5b_1.0_a7_id_s32_1_metrics.json",
            "./metrics/dGRPO_1_5b_1.0_a8_id_s32_1_metrics.json",
            "./metrics/dGRPO_1_5b_1.0_a9_id_s32_1_metrics.json",
            "./metrics/dGRPO_1_5b_1.0_a10_id_s32_1_metrics.json",
            "./metrics/dGRPO_1_5b_1.0_a11_id_s32_1_metrics.json",
            "./metrics/dGRPO_1_5b_1.0_a13_id_s32_1_metrics.json",
            "./metrics/dGRPO_1_5b_1.0_a14_id_s32_1_metrics.json",
            "./metrics/dGRPO_1_5b_1.0_a15_id_s32_1_metrics.json",
            "./metrics/dGRPO_1_5b_1.0_a16_id_s32_1_metrics.json",
            # "./metrics/e3_1_5b_e3_s16_1_metrics.json",
            # "./metrics/e3_1_5b_e3_2_s16_1_metrics.json",
            # "./metrics/e3_1_5b_e3_3_s16_1_metrics.json"
        ],
        n_list=[1,2,3,4,6,8,12,16,32]
    )

    print('--- pass@k (per run) ---')
    print(per_run_pass)
    print('--- length (per run) ---')
    print(per_run_length)

    # ---------- TSV OUTPUT (pass@1 + length) ----------
    runs = list(per_run_pass.keys())
    datasets = sorted({ds for r in per_run_pass.values() for ds in r.keys()})

    output_tsv = "metrics_summary_id_32.tsv"
    with open(output_tsv, "w") as f:
        # header
        header = ["run"]
        for ds in datasets:
            header.append(f"{ds}@1")
            header.append(f"{ds}@16")
            header.append(f"{ds}@32")
        for ds in datasets:
            header.append(f"{ds}_l")
        header += ["avg@1", "avg@16", "avg@32", "avg_len"]
        f.write("\t".join(header) + "\n")
        # rows
        for run in runs:
            row = [run[10:-4]]
            vals_1,vals_16,vals_32,lens=[],[],[],[]
            # pass@1 values
            for ds in datasets:
                v = per_run_pass[run].get(ds, {}).get(1, "")
                row.append(f"{v:.4f}" if v != "" else "")
                vals_1.append(v)
                v = per_run_pass[run].get(ds, {}).get(16, "")
                row.append(f"{v:.4f}" if v != "" else "")
                vals_16.append(v)
                v = per_run_pass[run].get(ds, {}).get(32, "")
                row.append(f"{v:.4f}" if v != "" else "")
                vals_32.append(v)
                
            # length values
            for ds in datasets:
                v = per_run_length[run].get(ds, "")
                row.append(f"{v:.2f}" if v != "" else "")
                lens.append(v)
            avg1 = np.nanmean(vals_1)
            avg16 = np.nanmean(vals_16)
            avg32 = np.nanmean(vals_32)
            avgl = np.nanmean(lens)
            row += [f"{avg1:.4f}", f"{avg16:.4f}", f"{avg32:.4f}", f"{avgl:.4f}"]
            f.write("\t".join(row) + "\n")

    print("Saved pass_length.tsv")
