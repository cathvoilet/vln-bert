# pylint: disable=no-member, not-callable
import json
import copy

import networkx as nx
import numpy as np
import torch


def load_json_data(path):
    with open(path, "r") as fid:
        data = json.load(fid)
    return data


def load_speaker_json_data(path):
    data = []
    with open(path, "r") as fid:
        tmp_data = json.load(fid)
        for instr_id, item in tmp_data.items():
            data.append(item)

    return data


def load_speaker_path_sampling_json_data(path, sample_size=10):
    data = []
    path_sample_prefix = "pred_path_sample_"
    result_sample_prefix = "result_sample_"

    with open(path, "r") as fid:
        tmp_data = json.load(fid)
        for instr_id, item in tmp_data.items():
            base_item = {x: item[x] for x in item if (not x.startswith(path_sample_prefix) and not x.startswith(result_sample_prefix))}
            for i in range(sample_size):
                new_item = copy.deepcopy(base_item)
                new_item["instr_id"] = new_item["instr_id"] + "_sample_" + str(i)
                new_item["pred_path"] = item[path_sample_prefix + str(i)]
                new_item["result"] = item[result_sample_prefix + str(i)]
                data.append(new_item)

    return data


def save_json_data(data, path):
    with open(path, "w") as fid:
        json.dump(data, fid)


def load_nav_graphs(scans):
    """ Load connectivity graph for each scan """

    def distance(pose1, pose2):
        """ Euclidean distance between two graph poses """
        return (
            (pose1["pose"][3] - pose2["pose"][3]) ** 2
            + (pose1["pose"][7] - pose2["pose"][7]) ** 2
            + (pose1["pose"][11] - pose2["pose"][11]) ** 2
        ) ** 0.5

    graphs = {}
    for scan in scans:
        with open("data/connectivity/%s_connectivity.json" % scan) as f:
            G = nx.Graph()
            positions = {}
            data = json.load(f)
            for i, item in enumerate(data):
                if item["included"]:
                    for j, conn in enumerate(item["unobstructed"]):
                        if conn and data[j]["included"]:
                            positions[item["image_id"]] = np.array(
                                [item["pose"][3], item["pose"][7], item["pose"][11]]
                            )
                            assert data[j]["unobstructed"][
                                i
                            ], "Graph should be undirected"
                            G.add_edge(
                                item["image_id"],
                                data[j]["image_id"],
                                weight=distance(item, data[j]),
                            )
            nx.set_node_attributes(G, values=positions, name="position")
            graphs[scan] = G
    return graphs


def load_distances(scans):
    distances = {}
    for scan in scans:
        with open(f"data/distances/{scan}_distances.json", "r") as fid:
            distances[scan] = json.load(fid)
    return distances


def get_headings(g, path, first_heading):
    # get xy positions for path
    pos = nx.get_node_attributes(g, "position")
    pos = {node: pos[node][:2] for node in path}

    # calculate headdings
    headings = [first_heading]
    for source, target in zip(path[:-1], path[1:]):
        dx = pos[target][0] - pos[source][0]
        dy = pos[target][1] - pos[source][1]
        # use dx/dy because heading is from north (i.e. y)
        headings.append(np.arctan2(dx, dy))
    return headings


def tokenize(data, tokenizer, max_instruction_length, key="instructions"):
    for item in data:
        item["instruction_tokens"] = []
        item["instruction_token_masks"] = []
        item["instruction_segment_ids"] = []
        if key == "instructions":
            instructions = item[key]
        else:
            instructions = [item[key]]
        for instruction in instructions:
            tokens = tokenizer.tokenize(instruction)

            # add a classification and seperator tokens
            tokens = ["[CLS]"] + tokens + ["[SEP]"]
            tokens = [tokenizer.vocab[token] for token in tokens]
            tokens = tokens[:max_instruction_length]

            masks = [1] * len(tokens)
            segment_ids = [0] * len(tokens)

            # pad lists
            pad_token = tokenizer.vocab["[PAD]"]
            pad_length = max_instruction_length - len(tokens)

            tokens = tokens + [pad_token] * pad_length
            masks = masks + [0] * pad_length
            segment_ids = segment_ids + [0] * pad_length

            # add to data
            item["instruction_tokens"].append(tokens)
            item["instruction_token_masks"].append(masks)
            item["instruction_segment_ids"].append(segment_ids)


def randomize_tokens(tokens, mask, tokenizer):
    """ Return tokens randomly masked using standard BERT probabilities. """
    targets = torch.ones_like(tokens) * -1

    # get random data
    p = torch.rand_like(tokens.float()) * mask.float()
    random_tokens = torch.randint_like(tokens, len(tokenizer.vocab))

    # set targets for masked tokens
    thresh = 0.85
    targets[p >= thresh] = tokens[p >= thresh]

    # progressively overwrite tokens while increasing the threshold

    # replace 80% with '[MASK]' token
    tokens[p >= thresh] = tokenizer.vocab["[MASK]"]

    # replace 10% with a random word
    thresh = 0.85 + 0.15 * 0.8
    tokens[p >= thresh] = random_tokens[p >= thresh]

    # keep 10% unchanged
    thresh = 0.85 + 0.15 * 0.9
    tokens[p >= thresh] = targets[p >= thresh]

    return tokens, targets


def randomize_regions(features, probs, mask):
    """ Return features after randomly masking using ViLBERT probabilities.

    Let B equal the batch size and N equal the number of regions.

    Parameters
    ----------
    features : torch.tensor, (B, N, 2048)
        The original feature vectors.
    probs : torch.tensor, (B, N, 2048)
        The target probability distribution for each region.
    mask : torch.tensor, (B, N)
        A zero-one mask where zeros represent missing regions.
    """
    targets = torch.ones_like(probs) / probs.shape[-1]
    targets_mask = torch.zeros_like(mask)

    p = torch.rand_like(mask.float()) * mask.float()

    # set targets for masked regions
    thresh = 0.85
    targets[p >= thresh] = probs[p >= thresh]
    targets_mask[p >= thresh] = 1

    # replace 90% of the masked features with zeros
    thresh = 0.85 + 0.15 * 0.1
    features[p >= thresh] = 0

    return features, targets, targets_mask


def get_viewpoints(
    data, graphs, feature_reader,
):
    """ Return a list of viewpoints that are in the graphs and feature reader. """
    scan_list = set(item["scan"] for item in data)
    viewpoints = {}
    for scan in scan_list:
        graph_viewpoints = set(graphs[scan].nodes())
        feats_viewpoints = feature_reader.viewpoints[scan]
        viewpoints[scan] = feats_viewpoints.intersection(graph_viewpoints)
    return viewpoints
