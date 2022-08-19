import json
import argparse
from collections import defaultdict
import numpy as np
import random
from scipy.stats import sem
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, average_precision_score


def compute_listener_score1(input_voted_json_file, input_complete_json_file, score_metric="ndtw", speaker_weight=0.0, speaker_model=None):
    print("\n\nRank instructions by: ", score_metric)
    print("Listener weight: ", 1.0-speaker_weight)
    print("Speaker weight: ", speaker_weight)
    print("Speaker score model: ", speaker_model)

    path2voted_instrs = defaultdict(list)
    with open(input_voted_json_file) as f:
        tmp_data = json.load(f)
        for instr_id, item in tmp_data.items():
            path_id = instr_id.split("_")[0]
            if score_metric not in item['overall_voting_result']:
                print("Exiting: metric {} not in voting file!".format(score_metric))
                return False, False
            path2voted_instrs[path_id].append((instr_id, item['overall_voting_result'][score_metric]))

    count_groups = 0
    avg_precision_list = []
    instr2speaker_score = {}
    unique_paths = set()
    all_pos_instrs, all_neg_instrs = [], []
    with open(input_complete_json_file) as f:
        tmp_data = json.load(f)
        for group_name, group in tmp_data.items():
            count_groups += 1
            positive_instrs, negative_instrs = [], []
            for item in group:
                instr_id = item['instr_id']
                path_id = instr_id.split("_")[0]
                unique_paths.add(path_id)
                if item['instr_label'] == "positive":
                    positive_instrs.append(instr_id)
                elif item['instr_label'] == "negative":
                    negative_instrs.append(instr_id)
                else:
                    print("Unknown instr label: ", item['instr_label'])
                if speaker_weight:
                    instr2speaker_score[instr_id] = item["speaker_result"][speaker_model]

            all_pos_instrs += positive_instrs
            all_neg_instrs += negative_instrs

            voted_instrs = list(path2voted_instrs[path_id])
            instr_labels = []
            instr_preds = []
            for instr_id, score in voted_instrs:
                if instr_id in positive_instrs+negative_instrs:
                    if speaker_weight:
                        speaker_score = instr2speaker_score[instr_id]
                        score = pow(score, 1.0-speaker_weight) * pow(speaker_score, speaker_weight)
                    if instr_id in positive_instrs:
                        instr_labels.append((instr_id, 1))
                        instr_preds.append((instr_id, score))
                    elif instr_id in negative_instrs:
                        instr_labels.append((instr_id, 0))
                        instr_preds.append((instr_id, score))
            y_true = np.array([x[1] for x in instr_labels])
            y_predict = np.array([x[1] for x in instr_preds])
            avg_precision = average_precision_score(y_true, y_predict)
            avg_precision_list.append(avg_precision)

    num_pos_instrs = len(list(set(all_pos_instrs)))
    num_neg_instrs = len(list(set(all_neg_instrs)))
    print("Number of paths counted: ", len(list(unique_paths)))
    print("Number of positive instrs counted: ", num_pos_instrs)
    print("Number of negative instrs counted: ", num_neg_instrs)
    print("Number of groups counted: ", count_groups)

    # Bootstrap confidence intervals
    n_iterations = 100
    alpha = 15.0  # 85% CI
    bootstrap_maps = []
    for i in range(n_iterations):
        bootstrap_aps = np.random.choice(avg_precision_list, len(avg_precision_list), replace=True)
        map = np.average(bootstrap_aps)
        bootstrap_maps.append(map)

    mean_avg_precision = np.average(avg_precision_list)
    # ste_ap = 1.44 * sem(avg_precision_list)
    # print("\nAvg precision_list: \n", avg_precision_list)
    lower_p = alpha / 2.0
    lower = max(0.0, np.percentile(bootstrap_maps, lower_p))
    upper_p = (100 - alpha) + (alpha / 2.0)
    upper = min(1.0, np.percentile(bootstrap_maps, upper_p))

    output_str = "Mean average precision [85% CI lower bound, upper bound] = {:.1f} [{:.1f}, {:.1f}]".format(100 * mean_avg_precision, 100 * lower, 100 * upper)
    print(output_str)
    return mean_avg_precision, output_str


def compute_listener_score(input_voted_json_file, input_complete_json_file, score_metric="ndtw", speaker_weight=0.0, speaker_model=None):
    print("\n\nRank instructions by: ", score_metric)
    print("Listener weight: ", round(1.0-speaker_weight, 1))
    print("Speaker weight: ", round(speaker_weight, 1))
    print("Speaker score model: ", speaker_model)
    listener_result_key = "result"  # 'overall_voting_result'

    path2voted_instrs = defaultdict(list)
    with open(input_voted_json_file) as f:
        tmp_data = json.load(f)
        for instr_id, item in tmp_data.items():
            path_id = instr_id.split("_")[0]
            if score_metric not in item[listener_result_key]:
                print("Exiting: metric {} not in voting file!".format(score_metric))
                return False, False
            path2voted_instrs[path_id].append((instr_id, item[listener_result_key][score_metric]))

    path2positive_instrs = defaultdict(list)
    path2negative_instrs = defaultdict(list)
    instr2speaker_score = {}
    with open(input_complete_json_file) as f:
        tmp_data = json.load(f)
        for instr_id, item in tmp_data.items():
            path_id = instr_id.split("_")[0]
            # if item['instr_label'] == "positive" or item['model'] == "speaker_ref_agent1_eval":
            if item['instr_label'] == "positive":
                path2positive_instrs[path_id].append(instr_id)
            elif item['instr_label'] == "negative":
                path2negative_instrs[path_id].append(instr_id)
            else:
                print("Unknown instr label: ", item['instr_label'])

            if speaker_weight:
                instr2speaker_score[instr_id] = item["speaker_result"][speaker_model]

    sorted_path_ids = sorted(list(path2negative_instrs.keys()))
    count_groups = 0
    n_iterations = 100  # 100
    map_values = []
    scores_variance = []

    for j in range(n_iterations):
        count_paths = 0
        num_pos_instrs, num_neg_instrs = 0, 0
        avg_precision_list = []
        for path_id in sorted_path_ids:
            positive_instrs = path2positive_instrs[path_id]
            negative_instrs = path2negative_instrs[path_id]
            voted_instrs = list(path2voted_instrs[path_id])
            count_paths += 1
            num_pos_instrs += len(positive_instrs)
            num_neg_instrs += len(negative_instrs)

            path_dataset = generate_path_dataset(positive_instrs, negative_instrs)
            path_scores = []
            for positive_instr, negative_group in path_dataset:
                count_groups += 1
                instr_labels = []
                instr_preds = []
                for instr_id, score in voted_instrs:
                    if speaker_weight:
                        speaker_score = instr2speaker_score[instr_id]
                        score = pow(score, 1.0-speaker_weight) * pow(speaker_score, speaker_weight)
                    if instr_id == positive_instr:
                        instr_labels.append((instr_id, 1))
                        instr_preds.append((instr_id, score))
                        path_scores.append(score)
                    elif instr_id in negative_group:
                        instr_labels.append((instr_id, 0))
                        instr_preds.append((instr_id, score))
                        path_scores.append(score)
                    #else:
                    #    print("WARNING: instr id not in either positive or negative category: ", instr_id)
                y_true = np.array([x[1] for x in instr_labels])
                y_predict = np.array([x[1] for x in instr_preds])
                avg_precision = average_precision_score(y_true, y_predict)
                avg_precision_list.append(avg_precision)

            scores_variance.append(np.var(path_scores))

        mean_avg_precision = np.average(avg_precision_list)
        map_values.append(mean_avg_precision)

    print("Number of paths counted: ", count_paths)
    print("Number of positive instrs counted: ", num_pos_instrs)
    print("Number of negative instrs counted: ", num_neg_instrs)
    print("Number of groups counted for 100 iterations: ", count_groups)

    avg_scores_variance = np.average(scores_variance)

    # confidence intervals
    alpha = 15.0  # 85% CI
    avg_map = np.average(map_values)
    lower_p = alpha / 2.0
    lower = max(0.0, np.percentile(map_values, lower_p))
    upper_p = (100 - alpha) + (alpha / 2.0)
    upper = min(1.0, np.percentile(map_values, upper_p))

    output_str = "Avg variance for samples in a group: {}\n".format(round(avg_scores_variance, 3))
    output_str += "Mean average precision [85% CI lower bound, upper bound] = {:.1f} [{:.1f}, {:.1f}]".format(100 * avg_map, 100 * lower, 100 * upper)
    print(output_str)

    return avg_map, output_str


def generate_path_dataset(positive_instructions, negative_instructions):
    num_negatives_per_group = 4
    num_test_cases = 5
    path_dataset = []

    for i in range(num_test_cases):
        pos_instr = np.random.choice(positive_instructions, 1, replace=False)
        neg_instrs = np.random.choice(negative_instructions, num_negatives_per_group, replace=True)
        path_dataset.append((pos_instr, neg_instrs))

    return path_dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_voted_json_file', help='input voted file')
    parser.add_argument('--input_complete_json_file', help='input original file')
    parser.add_argument('--listener_metric', default=None, help='listener metric')
    args = parser.parse_args()

    if not args.listener_metric:
        metrics = ['ndtw', 'sdtw', 'spl', 'score', 'prob']
    else:
        metrics = [args.listener_metric]
    speaker_weights = [0.1 * x for x in range(0, 11)]
    # speaker_weights = [0.0]
    speaker_models = ["clip", "finetuned_gpt"]

    for speaker_model in speaker_models:
        print("\n\n\nSpeaker model: ", speaker_model)
        metric2best_score = defaultdict(float)
        metric2best_string = defaultdict(str)
        for speaker_weight in speaker_weights:
            for metric in metrics:
                score, output_str = compute_listener_score(args.input_voted_json_file, args.input_complete_json_file, score_metric=metric, speaker_weight=speaker_weight, speaker_model=speaker_model)
                if score > metric2best_score[metric] and speaker_weight not in [0.0, 1.0]:
                    print("New best score for metric {}: {}".format(metric, score))
                    metric2best_score[metric] = score
                    metric2best_string[metric] = output_str + " (lda={})".format(round(1.0-speaker_weight, 1))

        for metric in metrics:
            print("\nSpeaker model {} final best score for metric {}: ".format(speaker_model, metric))
            print(metric2best_string[metric])


