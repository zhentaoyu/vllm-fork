import json
import argparse

import matplotlib.pyplot as plt
import pandas as pd

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="visualize_benchmark_results args.")
    parser.add_argument("--device",
                        type=str,
                        default="cuda",
                        help="device to run the benchmark")
    parser.add_argument("--results_dir",
                        type=str,
                        default="results",
                        help="directory to load and save the benchmark results.")
    args = parser.parse_args()

    data = []
    label = ['disagg_prefill', 'chunked_prefill']
    if args.device == "hpu":
        label = ['disagg_prefill', 'baseline_prefill']

    for name in label:
        for qps in [2, 4, 6, 8]:
            with open(f"{args.results_dir}/{name}-qps-{qps}.json", "r") as f:
                x = json.load(f)
                x['name'] = name
                x['qps'] = qps
                data.append(x)

    df = pd.DataFrame.from_dict(data)
    dis_df = df[df['name'] == label[0]]
    chu_df = df[df['name'] == label[1]]

    plt.style.use('bmh')
    plt.rcParams['font.size'] = 20

    for key in [
            'mean_ttft_ms', 'median_ttft_ms', 'p99_ttft_ms', 'mean_itl_ms',
            'median_itl_ms', 'p99_itl_ms'
    ]:

        fig, ax = plt.subplots(figsize=(11, 7))
        plt.plot(dis_df['qps'],
                 dis_df[key],
                 label=label[0],
                 marker='o',
                 linewidth=4)
        plt.plot(chu_df['qps'],
                 chu_df[key],
                 label=label[1],
                 marker='o',
                 linewidth=4)
        ax.legend()

        ax.set_xlabel('QPS')
        ax.set_ylabel(key)
        ax.set_ylim(bottom=0)
        fig.savefig(f'{args.results_dir}/{key}.png')
        plt.close(fig)
