import json
import argparse
from collections import defaultdict
import numpy as np
import random
from scipy.stats import sem
from scipy.special import softmax
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, average_precision_score


def compute_listener_score(input_voted_json_file, input_complete_json_file, score_metric="ndtw", speaker_weight=0.0,
                           speaker_model=None, normalize_listener=0, normalize_speaker=0,
                           listener_result_key="overall_voting_result", speaker_result_key="speaker_result",
                           matcher_weight=0.0, matcher_model="vln_match",
                           normalize_matcher=0,
                           matcher_result_key="result"):
    print("\n\nRank instructions by: ", score_metric)
    print("Listener weight: ", round(1.0-speaker_weight, 1))
    print("Speaker weight: ", round(speaker_weight, 1))
    print("Speaker score model: ", speaker_model)
    print("Matcher weight: ", round(matcher_weight, 1))
    print("Matcher model: ", matcher_model)

    path2voted_instrs = defaultdict(list)
    with open(input_voted_json_file) as f:
        tmp_data = json.load(f)
        for instr_id, item in tmp_data.items():
            path_id = instr_id.split("_")[0]
            if score_metric not in item[listener_result_key]:
                print("Exiting: metric {} not in voting file!".format(score_metric))
                return False, False
            path2voted_instrs[path_id].append((instr_id, item[listener_result_key][score_metric]))

    if normalize_listener:
        for path_id, instr_listener_scores in path2voted_instrs.items():
            listener_scores = np.array([x[1] for x in instr_listener_scores])
            # normalized_scores = (listener_scores - np.min(listener_scores)) / (np.max(listener_scores) - np.min(listener_scores))
            normalized_scores = softmax(listener_scores)
            path2voted_instrs[path_id] = []
            for i in range(len(instr_listener_scores)):
                instr_id, _ = instr_listener_scores[i]
                normalized_score = normalized_scores[i]
                path2voted_instrs[path_id].append((instr_id, normalized_score))

    path2positive_instrs = defaultdict(set)
    path2negative_instrs = defaultdict(set)
    instr2speaker_score = {}
    instr2speaker_model = {}
    path2speaker_scores = defaultdict(list)
    instr2matcher_score = {}
    path2matcher_scores = defaultdict(list)
    with open(input_complete_json_file) as f:
        tmp_data = json.load(f)
        for instr_id, item in tmp_data.items():
            path_id = instr_id.split("_")[0]
            # if item['instr_label'] == "positive" or item['model'] == "speaker_ref_agent1_eval":
            if item['instr_label'] == "positive":
                path2positive_instrs[path_id].add(instr_id)
            elif item['instr_label'] == "negative":
                path2negative_instrs[path_id].add(instr_id)
            else:
                print("Unknown instr label: ", item['instr_label'])

            if speaker_weight:
                instr2speaker_score[instr_id] = item[speaker_result_key][speaker_model]
                path2speaker_scores[path_id].append((instr_id, item[speaker_result_key][speaker_model]))

            if matcher_weight:
                instr2matcher_score[instr_id] = item[matcher_result_key][matcher_model]
                path2matcher_scores[path_id].append((instr_id, item[matcher_result_key][matcher_model]))

            if item['model'] == "speaker_ref_agent1_eval":
                instr2speaker_model[instr_id] = "ref"
            elif item['model'] in ["pi_vote-10ila_test-10vln", "speaker_gpt2_db7", "pi_vote-1ila_test-5vln", "speaker-gpt_pi-10ila-sample"]:
                instr2speaker_model[instr_id] = "gpt"
            elif item['model'] in ["speaker-clip_greedy", "speaker-clip_vote-10ila", "speaker-clip_vote-1ila", "speaker-clip10_pi-10ila-sample"]:
                instr2speaker_model[instr_id] = "clip"

    if normalize_speaker:
        for path_id, instr_speaker_scores in path2speaker_scores.items():
            speaker_scores = np.array([x[1] for x in instr_speaker_scores])
            # normalized_scores = (listener_scores - np.min(listener_scores)) / (np.max(listener_scores) - np.min(listener_scores))
            normalized_scores = softmax(speaker_scores)
            for i in range(len(instr_speaker_scores)):
                instr_id, _ = instr_speaker_scores[i]
                normalized_score = normalized_scores[i]
                instr2speaker_score[instr_id] = normalized_score

    if normalize_matcher:
        for path_id, instr_matcher_scores in path2matcher_scores.items():
            matcher_scores = np.array([x[1] for x in instr_matcher_scores])
            # normalized_scores = (listener_scores - np.min(listener_scores)) / (np.max(listener_scores) - np.min(listener_scores))
            normalized_scores = softmax(matcher_scores)
            for i in range(len(instr_matcher_scores)):
                instr_id, _ = instr_matcher_scores[i]
                normalized_score = normalized_scores[i]
                instr2matcher_score[instr_id] = normalized_score

    sorted_path_ids = sorted(list(path2negative_instrs.keys()))
    count_groups = 0
    n_iterations = 20  # 100
    map_values = []
    scores_variance = []
    count_top_instruction_model = defaultdict(int)
    count_intruction_model_labels = defaultdict(int)
    scale = 1e-2

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
                    if matcher_weight:
                        matcher_score = instr2matcher_score[instr_id]
                        score = pow(score, 1.0 - matcher_weight) * pow(matcher_score * scale, matcher_weight)
                    if instr_id == positive_instr:
                        instr_labels.append((instr_id, 1))
                        instr_preds.append((instr_id, score))
                        path_scores.append(score)
                        count_intruction_model_labels[instr2speaker_model[instr_id]] += 1
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

                top_instr_index = np.argmax(y_predict)
                top_instr = instr_preds[top_instr_index][0]
                count_top_instruction_model[instr2speaker_model[top_instr]] += 1

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

    total_test_cases = sum(count_top_instruction_model.values())
    print("Total test cases for {} iterations: {}".format(total_test_cases, n_iterations))
    for instruction_model, count_top in count_top_instruction_model.items():
        percentage = float(count_top) / total_test_cases
        print("Instruction model {} ranked top percentage: {}%".format(instruction_model, round(percentage * 100, 1)))
    for instruction_model, count_labels in count_intruction_model_labels.items():
        percentage = float(count_labels) / total_test_cases
        print("Instruction model {} labels percentage: {}%".format(instruction_model, round(percentage * 100, 1)))

    return avg_map, output_str


def generate_path_dataset(positive_instructions, negative_instructions):
    num_negatives_per_group = 4
    num_test_cases = 5
    path_dataset = []

    for i in range(num_test_cases):
        pos_instr = np.random.choice(list(positive_instructions), 1, replace=False)
        neg_instrs = np.random.choice(list(negative_instructions), num_negatives_per_group, replace=True)
        path_dataset.append((pos_instr, neg_instrs))

    return path_dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_voted_json_file', help='input voted file')
    parser.add_argument('--input_complete_json_file', help='input original file')
    parser.add_argument('--listener_metric', default=None, help='listener metric')
    parser.add_argument('--normalize_listener', default=0, help='listener normalize')
    parser.add_argument('--normalize_speaker', default=0, help='speaker normalize')
    parser.add_argument('--normalize_matcher', default=0, help='matcher normalize')
    parser.add_argument('--listener_result_key', default="overall_voting_result", help='listener key')
    parser.add_argument('--speaker_result_key', default="speaker_result", help='speaker key')
    parser.add_argument('--matcher_result_key', default=None, help='matching key')
    args = parser.parse_args()

    if not args.listener_metric:
        metrics = ['ndtw', 'sdtw', 'spl', 'score', 'prob']
    else:
        metrics = [args.listener_metric]
    speaker_weights = [0.1 * x for x in range(0, 11)]
    # speaker_weights = [1.0]
    speaker_models = ["clip", "finetuned_gpt"]
    # speaker_models = ["vln_match"]

    if args.matcher_result_key:
        matcher_weights = [0.1 * x for x in range(0, 11)]
        matcher_model = "vln_match"
        for speaker_model in speaker_models:
            print("\n\n\nSpeaker model: ", speaker_model)
            metric2best_score = defaultdict(float)
            metric2best_string = defaultdict(str)
            for speaker_weight in speaker_weights:
                for matcher_weight in matcher_weights:
                    for metric in metrics:
                        score, output_str = compute_listener_score(args.input_voted_json_file, args.input_complete_json_file,
                                                                   score_metric=metric,
                                                                   normalize_listener=args.normalize_listener,
                                                                   listener_result_key=args.listener_result_key,
                                                                   speaker_weight=speaker_weight, speaker_model=speaker_model,
                                                                   normalize_speaker=args.normalize_speaker,
                                                                   speaker_result_key=args.speaker_result_key,
                                                                   matcher_weight=matcher_weight, matcher_model=matcher_model,
                                                                   normalize_matcher=args.normalize_matcher,
                                                                   matcher_result_key=args.matcher_result_key)
                        if score > metric2best_score[metric] and speaker_weight not in [0.0, 1.0]:
                            print("New best score for metric {}: {}".format(metric, score))
                            metric2best_score[metric] = score
                            metric2best_string[metric] = output_str + " (lda={}, beta={})".format(round(1.0-speaker_weight, 1), round(matcher_weight, 1))

            for metric in metrics:
                print("\nSpeaker model {} final best score for metric {}: ".format(speaker_model, metric))
                print(metric2best_string[metric])

    else:
        for speaker_model in speaker_models:
            print("\n\n\nSpeaker model: ", speaker_model)
            metric2best_score = defaultdict(float)
            metric2best_string = defaultdict(str)
            for speaker_weight in speaker_weights:
                for metric in metrics:
                    score, output_str = compute_listener_score(args.input_voted_json_file, args.input_complete_json_file,
                                                               score_metric=metric,
                                                               normalize_listener=args.normalize_listener,
                                                               speaker_weight=speaker_weight, speaker_model=speaker_model,
                                                               normalize_speaker=args.normalize_speaker,
                                                               listener_result_key=args.listener_result_key,
                                                               speaker_result_key=args.speaker_result_key)
                    if score > metric2best_score[metric] and speaker_weight not in [0.0, 1.0]:
                        print("New best score for metric {}: {}".format(metric, score))
                        metric2best_score[metric] = score
                        metric2best_string[metric] = output_str + " (lda={})".format(round(1.0-speaker_weight, 1))

            for metric in metrics:
                print("\nSpeaker model {} final best score for metric {}: ".format(speaker_model, metric))
                print(metric2best_string[metric])

