"""
contains a lot of different functions used to inspect a model. e.g. calculate FLOPs (after pruning) etc.
important caveat: to load parameters of a pruned model to a new model instance, this new model instance
must have the module.weight_orig etc. attributes, that are created when pruning a model. The new model instance
can be pruned using pytorch l1 unstructured for example to load the weights of the desired pruned model.
Afterwards prune.remove() can be used to create a 'normal' model and pass this model to the inspection functions.
"""

import torch
import json
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters.rank import entropy
import skimage.morphology as skmorph
from torch.nn.utils import prune
import os
import math
import torchvision


import configurations as c
from Models.ModelZoo import get_model_from_zoo
from Models.ModelsSegmentation import vgg11_unet_configs
from Development.DevUtils import load_model_from_json_path, get_model_parameters, shrutika_prune
from Models.ModelUtils import Conv2DChannelWeights
from Preprocessing.Datasets import SegmentationDataSetNPZ
from collections import OrderedDict


def pad_num(num, target_length):
    """
    helper for pretty printing
    """
    num_str = str(num)
    if len(num_str) < target_length:
        num_str += " "*(target_length-len(num_str))
    else:
        num_str = num_str[:target_length]
    return num_str


# ====================================================================================================================
# OLD // This stuff can be viewed as depreciated and must be used VERY carefully
# ====================================================================================================================

def weights_overview(model):
    """
    DEPRECIATED --> USE CAREFULLY!
    """
    print("\nMODEL SUMMARY " + "-" * 150)
    print(model)

    print("\nWEIGHTS SUMMARY " + "-" * 150)
    print("Conv Layers:")
    n_convs = 0
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            weights = m.weight
            print("{}\tmin: {}  \tmean: {} \tmax: {}".format(
                n_convs,
                pad_num(round(weights.min().item(), 4), 5),
                pad_num(round(weights.mean().item(), 4), 5),
                pad_num(round(weights.max().item(), 4), 5)
            ))
            n_convs += 1

    print("Conv. Transpose Layers:")
    n_convT = 0
    for m in model.modules():
        if isinstance(m, torch.nn.ConvTranspose2d):
            weights = m.weight
            print("{}\tmin: {}  \tmean: {} \tmax: {}".format(
                n_convT,
                pad_num(round(weights.min().item(), 4), 5),
                pad_num(round(weights.mean().item(), 4), 5),
                pad_num(round(weights.max().item(), 4), 5)
            ))
            n_convT += 1

    print("Parameters:")
    n_p = 0
    for m in model.modules():
        if isinstance(m, torch.nn.ParameterList):
            for param in m:
                weights = param.data
                print("{}\tmin: {}  \tmean: {} \tmax: {}".format(
                    n_p,
                    pad_num(round(weights.min().item(), 4), 5),
                    pad_num(round(weights.mean().item(), 4), 5),
                    pad_num(round(weights.max().item(), 4), 5)
                ))
                n_p += 1


def show_wcc_hist(model, per_vector=False):
    """
    DEPRECIATED --> USE CAREFULLY!
    """
    print("\nGETTING PARAMETER WEIGHTS " + "-" * 135)
    n_p = 0
    weights_per_layer = {}
    all_weights = []
    for m in model.modules():
        if isinstance(m, torch.nn.ParameterList):
            for param in m:
                weights = param.data
                print("{}\tlen: {}\tmin: {}  \tmean: {} \tmax: {}".format(
                    n_p,
                    pad_num(weights.shape[0], 5),
                    pad_num(round(weights.min().item(), 4), 5),
                    pad_num(round(weights.mean().item(), 4), 5),
                    pad_num(round(weights.max().item(), 4), 5)
                ))
                w = weights.detach().cpu().numpy().flatten().tolist()
                weights_per_layer[n_p] = w
                all_weights.extend(w)
                n_p += 1

    for layer_key in weights_per_layer:
        plt.hist(x=weights_per_layer[layer_key], alpha=0.5)

    plt.legend(weights_per_layer.keys())
    plt.show()

    plt.hist(x=all_weights)
    plt.show()

    if per_vector:
        for key in weights_per_layer.keys():
            plt.hist(x=weights_per_layer[key])
            plt.title(f"weight vector {key}")
            plt.show()


def show_Conv2DChannelWeights_hist(model, per_vector=False):
    """
    DEPRECIATED --> USE CAREFULLY!
    """
    print("\nGETTING WEIGHTS FOR EACH CONV LAYER" + "-" * 135)
    n_p = 0
    weights_per_layer = {}
    all_weights = []
    for m in model.modules():
        if isinstance(m, Conv2DChannelWeights):

            weights = m.get_weights()
            print("{}\tlen: {}\tmin: {}  \tmean: {} \tmax: {}".format(
                n_p,
                pad_num(weights.shape[0], 5),
                pad_num(round(weights.min().item(), 4), 5),
                pad_num(round(weights.mean().item(), 4), 5),
                pad_num(round(weights.max().item(), 4), 5)
            ))
            w = weights.detach().cpu().numpy().flatten().tolist()
            weights_per_layer[n_p] = w
            all_weights.extend(w)
            n_p += 1

    for layer_key in weights_per_layer:
        plt.hist(x=weights_per_layer[layer_key], alpha=0.5, bins=15, align="left")

    plt.legend(weights_per_layer.keys())
    plt.show()

    plt.hist(x=all_weights)
    plt.show()

    if per_vector:
        for key in weights_per_layer.keys():
            plt.hist(x=weights_per_layer[key])
            plt.title(f"weight vector {key}")
            plt.show()


def show_Conv2D_hist(model, per_vector=False):
    """
    DEPRECIATED --> USE CAREFULLY!
    """
    print("\nGETTING WEIGHTS FOR EACH CONV LAYER" + "-" * 135)
    n_p = 0
    weights_per_layer = {}
    all_weights = []
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):

            weights = m.weight
            print("{}\tlen: {}\tmin: {}  \tmean: {} \tmax: {}".format(
                n_p,
                pad_num(weights.shape[0], 5),
                pad_num(round(weights.min().item(), 4), 5),
                pad_num(round(weights.mean().item(), 4), 5),
                pad_num(round(weights.max().item(), 4), 5)
            ))
            w = weights.detach().cpu().numpy().flatten()
            weights_per_layer[n_p] = w
            all_weights.extend(w.tolist())
            n_p += 1

    bins = np.arange(-0.155, 0.155, 0.01)
    ctr = 0

    fig = plt.figure(figsize=(5, 5))
    for layer_key in range(4, 0, -1):
        plt.hist(x=weights_per_layer[layer_key], alpha=0.5, bins=bins, histtype="stepfilled", log=False,
                 density=False, stacked=False, edgecolor="black")
        ctr += 1
        if ctr == 4:
            break

    plt.ylim(0, 600000)
    plt.yticks(ticks=range(0, 700000, 100000), labels=[str(i)+"k" for i in range(0, 700, 100)])
    plt.legend([f"conv layer {i}" for i in range(4, 0, -1)])
    plt.show()

    if per_vector:
        for key in weights_per_layer.keys():
            plt.hist(x=weights_per_layer[key])
            plt.title(f"weight vector {key}")
            plt.show()


# ====================================================================================================================
# NOW BEGINS THE STUFF USED FOR FILTER PRUNING PAPER
# ====================================================================================================================

def parameters_after_pruning(model):
    """
    counts the number of weight and bias parameters in the Conv2d and ConvTranspose2d layers of the model
    and subtracts the number weights in Conv2d layers, that are equal to zero,
    resulting in the number of parameters that remain after pruning. This function assumes only conv layers are pruned
    and the model consists only of Conv2d and ConvTranspose2d layers.

    Args:
        model: (nn.Module) the model instance, if model is pruned, use prune.remove() since this uses module.weight and
               module.bias to get parameters

    Returns: (int) remaining parameters that are not zero
    """

    conv_layers = []
    n_weights_model = 0
    n_bias_model = 0
    n_connections_model = 0

    print("=" * 150)
    print("ALL MODEL PARAMETERS:")
    print("-"*150, pad_num("\nName", 21), pad_num("Weights", 10), pad_num("Bias", 10),
          "Connections\n" + "-"*150)
    for m in model.modules():
        n_weights_layer = 1
        n_bias_layer = 1
        n_connections_layer = 1

        if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.ConvTranspose2d):
            for sw in m.weight.shape:
                n_weights_layer *= sw

            if m.bias is not None:
                for sb in m.bias.shape:
                    n_bias_layer *= sb
            else:
                n_bias_layer = 0

            n_connections_layer *= m.weight.shape[0]
            n_connections_layer *= m.weight.shape[1]

            print(f"{pad_num(m._get_name(), 20)} {pad_num(n_weights_layer, 10)} {pad_num(n_bias_layer, 10)} " +
                  f"{pad_num(n_connections_layer, 10)}")
            n_bias_model += n_bias_layer
            n_weights_model += n_weights_layer
            n_connections_model += n_connections_layer

        if isinstance(m, torch.nn.Conv2d):
            conv_layers.append((m, n_weights_layer))

    print("-"*150)
    print("PARAMETERS:  ", n_weights_model + n_bias_model)
    print("└- Weights:  ", n_weights_model)
    print("└- Bias:     ", n_bias_model)
    # print("CONNECTIONS: ", n_connections_model)

    print("=" * 150)
    print("PRUNED WEIGHTS IN CONV2D LAYERS")
    print("-"*150, pad_num("\ni", 4), pad_num("Total Weights", 15),
          pad_num("Pruned Weights", 15), "%\n" + "-"*150)
    idx, ctr = 0, 0
    n_overall_pruned = 0
    for cl, n_weights in conv_layers:
        idx += 1
        pruned_weights = torch.eq(torch.abs(cl.weight), 0).sum().item()
        perc_pruned = round((pruned_weights/n_weights)*100, 2)
        bar = "|" + "="*int(perc_pruned)
        print(pad_num(idx, 3), pad_num(n_weights, 15), pad_num(pruned_weights, 15), pad_num(perc_pruned, 5), bar)
        n_overall_pruned += pruned_weights

    print("-" * 150)
    print("REMAINING PARAMETERS:", n_weights_model + n_bias_model - n_overall_pruned)
    print("-" * 150)

    return n_weights_model + n_bias_model - n_overall_pruned


def n_filters_after_pruning(model):
    """
    counts filters in Conv2d and ConvTranspose2d layers that are not all zeros

    Args:
        model: (nn.Module) the model instance, if model is pruned, use prune.remove() since this uses module.weight and
               module.bias to get parameters

    Returns: (int) number of unpruned filters
    """

    FILTERS = {
        "unpruned": [],
        "pruned": []
    }

    def hook(module, input, output):
        n_inputs = 0
        n_filters = 0
        pruned_inputs = 0
        pruned_filters = 0

        t = "conv" if isinstance(module, torch.nn.Conv2d) else "convT"
        f_out_dim = 0 if isinstance(module, torch.nn.Conv2d) else 1

        for i in range(0, module.weight.shape[f_out_dim]):
            n_filters += 1
            if torch.eq(torch.abs(module.weight[i]).sum(), 0):
                pruned_filters += 1

        FILTERS["unpruned"].append(n_filters)
        FILTERS["pruned"].append(pruned_filters)

        print(pad_num(t, 7), pad_num(n_filters, 15), pad_num(pruned_filters, 20))

    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.ConvTranspose2d):
            m.register_forward_hook(hook)

    print(pad_num("type", 7), pad_num("filters", 15), pad_num("pruned f.", 20))
    x = torch.ones((1, 3, 256, 256))
    _ = model(x)
    return FILTERS


def similarity_mat_of_filters(model, conv_layer_idx, similarity):
    """
    plots and saves similarity matrices of the filter weights in the specified conv. layer.
    the similarity measure can be specified as well.

    Args:
        model: (nn.Module) the model instance, if model is pruned, use prune.remove() since this uses module.weight and
               module.bias to get parameters
        conv_layer_idx: (int) index of conv. layer which should be plotted
        similarity: (str) e.g. 'cosine' for cosine similarity , 'dist_1' or 'dist_2' for l1 or l2 distance, 'dot'
                    for dot product similarity or 'cc' for correlation coefficient

    Returns: void

    """
    ctr = 0
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            ctr += 1
            if ctr == conv_layer_idx:
                print("_______________________________")
                print(m.weight[0])
                filters = torch.reshape(m.weight, (m.weight.shape[0], -1))
                print(filters[0])
                break

    print(filters.shape)

    sim_matrix = torch.zeros((filters.shape[0], filters.shape[0]))
    if similarity == "cosine":
        for i in range(0, sim_matrix.shape[0]):
            for j in range(0, sim_matrix.shape[1]):
                sim_matrix[i, j] += torch.nn.functional.cosine_similarity(filters[i], filters[j], dim=0)

    elif similarity == "dist_1":
        filters = torch.unsqueeze(filters, 1)
        for i in range(0, sim_matrix.shape[0]):
            for j in range(0, sim_matrix.shape[1]):
                # print(filters[i].shape, torch.nn.functional.pairwise_distance(filters[i], filters[j], p=2.0))
                sim_matrix[i, j] += torch.nn.functional.pairwise_distance(filters[i], filters[j], p=1.0)[0]

    elif similarity == "dist_2":
        filters = torch.unsqueeze(filters, 1)
        for i in range(0, sim_matrix.shape[0]):
            for j in range(0, sim_matrix.shape[1]):
                # print(filters[i].shape, torch.nn.functional.pairwise_distance(filters[i], filters[j], p=2.0))
                sim_matrix[i, j] += torch.nn.functional.pairwise_distance(filters[i], filters[j], p=2.0)[0]

    elif similarity == "dot":
        for i in range(0, sim_matrix.shape[0]):
            for j in range(0, sim_matrix.shape[1]):
                sim_matrix[i, j] += torch.dot(filters[i], filters[j].T)

    elif similarity == "cc":
        for i in range(0, sim_matrix.shape[0]):
            fi = filters[i]
            for j in range(0, sim_matrix.shape[1]):
                fj = filters[j]
                vi = fi - torch.mean(fi)
                vj = fj - torch.mean(fj)
                sim_matrix[i, j] += torch.sum(vi * vj) / (torch.sqrt(torch.sum(vi ** 2)) * torch.sqrt(torch.sum(vj ** 2)))

    plt.matshow(sim_matrix.detach().cpu().numpy())
    plt.colorbar()
    plt.savefig(f"layer {conv_layer_idx} {similarity}.png")


def inspect_connections_after_pruning(model):
    """
    counts the remaining connections between layers of type Conv2d and ConvTranspose2d.
    If a filter map that a layer outputs is all zeros this connection is considered not existing.
    uses pytorch forward hooks

    Args:
        model: (nn.Module) the model instance, if model is pruned, use prune.remove() since this uses module.weight and
               module.bias to get parameters

    Returns: void

    """

    CONNECTIONS = []

    def hook(module, input, output):

        n_inputs = 0
        n_filters = 0
        pruned_inputs = 0
        pruned_filters = 0

        t = "conv" if isinstance(module, torch.nn.Conv2d) else "convT"
        f_out_dim = 0 if isinstance(module, torch.nn.Conv2d) else 1

        #print(t, input[0].shape, output.shape, module.weight.shape)

        for i in range(0, input[0].shape[1]):
            n_inputs += 1
            # print(input[0][0, i, :, :])
            if torch.eq(torch.abs(input[0][0, i, :, :]).sum(), 0):
                pruned_inputs += 1
        for i in range(0, module.weight.shape[f_out_dim]):
            n_filters += 1
            if torch.eq(torch.abs(module.weight[i]).sum(), 0):
                pruned_filters += 1

        CONNECTIONS.append((n_filters - pruned_filters) * (n_inputs - pruned_inputs))
        print(pad_num(t, 7), pad_num(n_filters, 15), pad_num(pruned_filters, 20),
              pad_num(n_inputs, 15), pad_num(pruned_inputs, 15))

        return output.abs()

    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.ConvTranspose2d):
            m.register_forward_hook(hook)

    print(pad_num("type", 7), pad_num("filters", 15), pad_num("pruned f.", 20), pad_num("input f.", 15),
          pad_num("pruned input f.", 15))
    x = torch.ones((1, 3, 256, 256))
    _ = model(x)
    print(sum(CONNECTIONS))


def calculate_FLOPs(model):
    """
    calculates the FLOPs of a model following 'Liu et al: Compressing CNNs using Multi-level Filter Pruning
    for the Edge Nodes of Multimedia Internet of Things'. Input is assumed to be of size (3, 256, 256).
    linear layers are not implemented, relu and pooling are omitted for simplicity. (so only Conv2d and ConvTranspose2d
    are used acutally)

    Args:
        model: (nn.Module) the model instance, if model is pruned, use prune.remove() since this uses module.weight and
               module.bias to get parameters

    Returns: (int) number of FLOPs

    """

    FLOPs = []

    def hook(module, input, output):

        n_inputs = 0
        n_filters = 0
        pruned_inputs = 0
        pruned_filters = 0

        t = "conv" if isinstance(module, torch.nn.Conv2d) else "convT"
        f_out_dim = 0 if isinstance(module, torch.nn.Conv2d) else 1

        #print(t, input[0].shape, output.shape, module.weight.shape)

        for i in range(0, input[0].shape[1]):
            n_inputs += 1
            # print(input[0][0, i, :, :])
            if torch.eq(torch.abs(input[0][0, i, :, :]).sum(), 0):
                pruned_inputs += 1
        for i in range(0, module.weight.shape[f_out_dim]):
            n_filters += 1
            if torch.eq(torch.abs(module.weight[i]).sum(), 0):
                pruned_filters += 1

        rem_input_filters = n_inputs - pruned_inputs
        rem_layer_filters = n_filters - pruned_filters

        kernel_size = module.weight.shape[-1] * module.weight.shape[-2]
        n_steps = output.shape[-1] * output.shape[-2]

        # following: Liu et al: Compressing CNNs using Multi-level Filter Pruning
        #                       for the Edge Nodes of Multimedia Internet of Things
        f = n_steps * (kernel_size * rem_input_filters + 1) * rem_layer_filters

        FLOPs.append(f)

        print(pad_num(t, 7), pad_num(rem_layer_filters, 15), pad_num(rem_input_filters, 20), pad_num(kernel_size, 15),
              pad_num(n_steps, 15), pad_num(FLOPs[-1]/1e9, 15))

        return output.abs()

    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.ConvTranspose2d):
            m.register_forward_hook(hook)

    print(pad_num("type", 7), pad_num("filters", 15), pad_num("input f.", 20), pad_num("kernel size", 15),
          pad_num("steps", 15), pad_num("G-FLOPs", 15))
    x = torch.ones((1, 3, 256, 256))
    _ = model(x)
    print("TOTAL Giga FLOPs:", sum(FLOPs)/1e9)
    return FLOPs


def get_2d_image_entropy(filter: torch.tensor, glob_min: torch.tensor = None, glob_max: torch.tensor = None):
    """
    calculates entropy for a filter using skimage.morphology and skimage.filters.rank.entropy. See docs of skimage
    for further details. if no global min. and max. are passed the min. and max. of the filter values are used

    Args:
        filter: (torch.tensor) the filter matrix, the output, NOT the weights of the layer
        glob_min: (flaot) optional
        glob_max: (float) optional

    Returns: entropy map, filter values (both type np.array)

    """

    np_filter = filter.detach().numpy()

    if glob_min is not None and glob_max is not None:
        min_f = glob_min.detach().numpy()
        max_f = glob_max.detach().numpy()
    else:
        min_f = np_filter.flatten().min()
        max_f = np_filter.flatten().max()

    if min_f == max_f:
        return np.zeros_like(np_filter), filter.detach().numpy()

    np_filter = (np_filter - min_f) / (max_f - min_f)
    np_filter *= 255
    np_filter = np_filter.astype(np.uint16)

    struc_mat = skmorph.square(3)
    struc_mat[1][1] = 0

    entr_map = entropy(np_filter, struc_mat)
    #print(np_filter.shape, entr_map.shape)
    return entr_map, filter.detach().numpy()


def calculate_entropy_for_layer(model: torch.nn.Module,
                                conv_layer_nr: int,
                                dataloader: torch.utils.data.DataLoader,
                                n_samples: int):
    """
    calls get_2d_image_entropy(). calculates entropies for filters of specified layer in model
    for the specified number of samples in the dataloader. also shows a plot of these.

    Args:
        model: (nn.Module) the model instance, if model is pruned, use prune.remove() since this uses module.weight and
               module.bias to get parameters
        conv_layer_nr: (int) index of conv. layer to use
        dataloader: (torch.utils.data.DataLoader) data used to for calculation
        n_samples: (int) will be stopped after this much samples are processed

    Returns: mean entropy for filters over samples

    """

    def hook(module, input, output):
        glob_max = torch.max(output)
        glob_min = torch.min(output)

        for i in range(0, output.shape[1]):
            e, f = get_2d_image_entropy(output[0][i], glob_min, glob_max)
            ENTROPY_MAPS.append(e)
            FILTER_MAPS.append(f)

    idx = 0
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            idx += 1
            if idx == conv_layer_nr:
                m.register_forward_hook(hook)

    scores = []
    for i, (data, mask) in enumerate(dataloader):
        ENTROPY_MAPS = []
        FILTER_MAPS = []
        _ = model(data)

        """fig, axs = plt.subplots(4, 2, figsize=(5, 10))
        for n in range(0, 4):
            axs[n, 0].imshow(FILTER_MAPS[n])
            axs[n, 1].imshow(ENTROPY_MAPS[n])
        plt.show()"""

        n_per_axs = int(math.sqrt(len(FILTER_MAPS)))
        fig, axs = plt.subplots(n_per_axs, n_per_axs, figsize=(6.4, 6.4),
                                subplot_kw={'xticks': [], 'yticks': []})
        for i in range(0, n_per_axs):
            for j in range(0, n_per_axs):
                n = i * n_per_axs + j
                axs[i, j].imshow(FILTER_MAPS[n], cmap = "summer")
        # plt.show()
        plt.savefig(f"/data/project-gxb/johannes/innspector/img_{idx + 1}_layer_{conv_layer_nr}.png", format="png")

        scores.append([entropy_map.mean() for entropy_map in ENTROPY_MAPS])

        print("> done with sample", i)
        if i+1 == n_samples:
            break

    scores_np = np.array(scores)
    means = np.mean(scores_np, axis=0)
    stds = np.std(scores_np, axis=0)

    plt.bar(range(0, len(means)), means, yerr=stds)
    plt.show()

    return means


def plot_filter_maps(model: torch.nn.Module,
                     conv_layer_nr: int,
                     dataloader: torch.utils.data.DataLoader,
                     n_samples: int):
    """
    uses forward hook to plot and save filter maps

    Args:
        model: (nn.Module) the model instance, if model is pruned, use prune.remove() since this uses module.weight and
               module.bias to get parameters
        conv_layer_nr: (int) index of conv. layer to use
        dataloader: (torch.utils.data.DataLoader) data used for plotting
        n_samples: (int) will be stopped after this much samples are processed

    Returns: void

    """

    def hook(module, input, output):
        for i in range(0, output.shape[1]):
            f = output[0][i].detach().numpy()
            f -= f.mean()
            f /= f.std()
            FILTER_MAPS.append(f)

    idx = 0
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            idx += 1
            if idx == conv_layer_nr:
                m.register_forward_hook(hook)

    for idx, (data, mask) in enumerate(dataloader):
        FILTER_MAPS = []

        _ = model(data)

        n_per_axs = int(math.sqrt(len(FILTER_MAPS)))
        print("n per axs ", n_per_axs)
        fig, axs = plt.subplots(n_per_axs, n_per_axs, figsize=(6.4, 6.4),
                                subplot_kw={'xticks': [], 'yticks': []})
        for i in range(0, n_per_axs):
            for j in range(0, n_per_axs):
                n = i*n_per_axs + j
                axs[i, j].imshow(FILTER_MAPS[n], cmap = "summer")
        #plt.show()
        plt.savefig(f"/data/project-gxb/johannes/innspector/img_{idx+1}_layer_{conv_layer_nr}.png", format = "png")

        print("> done with sample", idx+1)
        if idx+1 == n_samples:
            break


if __name__ == "__main__":

    path = "/data/project-gxb/johannes/innspector/_results/saved_models/pruning-references-test_2021-03-29_08-29-00.json"
    path_2 = "/data/project-gxb/johannes/innspector/_results/saved_models/pruning-references-test_2021-03-29_08-29-00.pt"
    #paper_20-l1_struc_retrained_2021-04-07_13-06-13_last_epoch.pt
    #paper_20-rand_struc_retrained_2021-04-08_07-58-32.pt
    #paper_40-l1_struc_retrained_2021-04-27_07-15-49.pt
    #paper_40-rand_struc_retrained_2021-04-28_04-27-56_last_epoch.pt
    #paper_60-l1_struc_retrained_2021-05-27_09-57-25.pt
    #paper_60-rand_struc_retrained_2021-05-27_17-14-31.pt
    #paper_20-shrutika-l1-df_retrained_2021-04-09_16-49-37.pt
    #paper_40-shrutika-l1-df_retrained_2021-04-10_19-26-30.pt
    #paper_60-shrutika-l1-df_retrained_2021-04-28_02-44-39.pt

    minfo = "UNet-VGG11_l1 structured-40"

    model_spec = None
    """model_spec = ("UNetVGGbase", {
        "down_block_configs": [[[[3, 3], 64]], [[[3, 3], 128]], [[[3, 3], 256], [[3, 3], 256]],
                              [[[3, 3], 512], [[3, 3], 512]], [[[3, 3], 512], [[3, 3], 512]]],
        "up_block_configs": [{"conv": [[[3, 3], 768, 512]], "convT": [[[3, 3], 512, 256]]},
                              {"conv": [[[3, 3], 768, 512]], "convT": [[[3, 3], 512, 128]]},
                              {"conv": [[[3, 3], 384, 256]], "convT": [[[3, 3], 256, 64]]},
                              {"conv": [[[3, 3], 192, 128]], "convT": [[[3, 3], 128, 32]]},
                              {"conv": [[[3, 3], 96, 32]], "convT": None}],
        "in_channels": 3,
        "activation": "relu",
        "add_conv_channel_weights": False,
    })"""

    """model_spec = ("UNetClassic", {
        "in_channels": 3
    })"""

    #model = get_model_from_zoo(model_spec)
    model = load_model_from_json_path(model_json_path=path, load_state_dict=False)

    conv_layers = []
    for m in model.down_blocks.modules():
        if isinstance(m, torch.nn.Conv2d):
            conv_layers.append(m)
    for m in model.latent.modules():
        if isinstance(m, torch.nn.Conv2d):
            conv_layers.append(m)
    for m in model.up_blocks.modules():
        if isinstance(m, torch.nn.Conv2d):
            conv_layers.append(m)

    for layer in conv_layers:
        prune.random_structured(layer, name="weight", amount=0.6, dim=0)

    '''if path_2.__contains__("pruning-references-test_") is False and model_spec is None:
        for layer in conv_layers:
            #prune.ln_structured(layer, name="weight", amount=0.2, n=1, dim=0)
            prune.RandomStructured(amount=0.2, dim=0)'''

    state_dict = torch.load(path_2)
    new_state_dict = OrderedDict()

    for k, v in state_dict.items():
        if k[:7] == "module.":
            name = k[7:]
        else:
            name = k
        new_state_dict[name] = v
    #model.load_state_dict(new_state_dict)

    for layer in conv_layers:
        prune.remove(layer, "weight")
    '''if path_2.__contains__("pruning-references-test_") is False and model_spec is None:
        for layer in conv_layers:
            prune.remove(layer, "weight")'''

    """d_conf = c.get_data_config()["inria"]
    val_names = os.listdir("../" + d_conf["val_dir"] + d_conf["image_folder"])
    p_imgs_vl = ["../" + d_conf["val_dir"] + d_conf["image_folder"] + n for n in val_names]
    p_msk_vl = ["../" + d_conf["val_dir"] + d_conf["mask_folder"] + n for n in val_names]
    valset = SegmentationDataSetNPZ(img_paths=p_imgs_vl[255:], mask_paths=p_msk_vl[255:],
                                    p_flips=None, p_noise=None)
    trainloader = torch.utils.data.DataLoader(valset, batch_size=1, shuffle=False)


    inverse_norm1 = torchvision.transforms.Normalize(mean=[0, 0, 0],
                                                    std=[1/0.229, 1/0.224, 1/0.225])
    inverse_norm2 = torchvision.transforms.Normalize(mean=[-0.485, -0.456, -0.406],
                                                    std=[1, 1, 1])"""

    """for idx, (data, mask) in enumerate(trainloader):

        out = model(data)
        data = inverse_norm1(data[0])
        data = inverse_norm2(data)
        print(data.max(), data.min())
        data_np = data.permute(1, 2, 0).detach().numpy()
        out_np = out[0][0].detach().numpy()
        mask_np = mask[0][0].detach().numpy()
        #plt.imshow(data_np)
        #plt.savefig(f"orig_{idx+1}.png")

        fig, axs = plt.subplots(1, 3, figsize=(12, 6))
        axs[0].imshow(data_np)
        axs[1].imshow(mask_np)
        axs[2].imshow(out_np)

        plt.savefig(f"{minfo}_pred_{idx+1}.png")

        if idx+1 == 2:
            break"""

    d_conf = c.get_data_config()["inria"]
    val_names = "austin10-329.npz"
    #p_imgs_vl = ["../" + d_conf["train_dir"] + d_conf["image_folder"] + n for n in val_names]
    #p_msk_vl = ["../" + d_conf["train_dir"] + d_conf["mask_folder"] + n for n in val_names]
    p_imgs_vl = ["../" + d_conf["train_dir"] + d_conf["image_folder"] + val_names]
    p_msk_vl = ["../" + d_conf["train_dir"] + d_conf["mask_folder"] + val_names]
    valset = SegmentationDataSetNPZ(img_paths=p_imgs_vl, mask_paths=p_msk_vl,
                                    p_flips=None, p_noise=None)
    trainloader = torch.utils.data.DataLoader(valset, batch_size=1, shuffle=False)
    print(model)
    model.eval()

    #parameters_after_pruning(model)

    #filters = n_filters_after_pruning(model)
    #print(f"{sum(filters['pruned'])}/{sum(filters['unpruned'])}")

    #show_Conv2D_hist(model)
    #similarity_mat_of_filters(model, 1, "dist_1")

    #inspect_connections_after_pruning(model)
    calculate_FLOPs(model)
    #calculate_entropy_for_layer(model, 1, trainloader, 1)
    plot_filter_maps(model, 1, trainloader, 1)
    #plot_filter_maps(model, 2, trainloader, 2)
    #plot_filter_maps(model, 13, trainloader, 2)
    #plot_filter_maps(model, 14, trainloader, 2)
