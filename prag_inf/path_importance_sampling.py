import argparse
import os
import json
import numpy as np
from collections import defaultdict
from scipy.special import softmax


def compute_importance_sampling(input_file):
    instrid2samples = defaultdict(list)
    instrid2vlnbert_scores = defaultdict(list)
    with open(input_file) as f:
        tmp_data = json.load(f)
        for instr_sample_id, item in tmp_data.items():
            instr_id = instr_sample_id.split("_sample")[0]
            instrid2samples[instr_id].append(item)
            instrid2vlnbert_scores[instr_id].append(item["result"]["vln_match"])

    data_json = {}
    for instr_id, samples in instrid2samples.items():
        metric2res = {"score": [], "ndtw": [], "sdtw": [], "spl": []}
        vlnbert_scores = instrid2vlnbert_scores[instr_id]
        normalized_scores = softmax(vlnbert_scores)
        for i, sample in enumerate(samples):
            # vln_match = sample["result"]["vln_match"]
            prob = sample["result"]["prob"]
            for metric in metric2res.keys():
                res = path_importance_sampling(normalized_scores[i], prob, float(sample["result"][metric]))
                metric2res[metric].append(res)

        for metric in metric2res.keys():
            metric2res[metric] = np.average(metric2res[metric])

        new_item = samples[0]
        new_item["instr_id"] = instr_id
        new_item.pop("pred_path", None)
        new_item.pop("result", None)
        new_item["result"] = metric2res
        data_json[instr_id] = new_item

    output_dir = os.path.dirname(input_file)
    output_filename = "final_" + os.path.basename(input_file)
    output_path = os.path.join(output_dir, output_filename)
    json.dump(data_json, open(output_path, "w"), indent=2)
    print("saving scores: ", output_path)


def path_importance_sampling(normalized_vlnbert_score, sample_prob, sample_score):
    res = normalized_vlnbert_score / sample_prob * sample_score
    return res


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', help='input file')
    args = parser.parse_args()
    compute_importance_sampling(args.input_file)


