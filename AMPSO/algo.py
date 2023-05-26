"""

AMPSO algorithm classes and functions for model training and pruning experiments

"""

import torch
from torch.nn.utils import prune
import time
import copy
from pymoo.algorithms.moo.nsga2 import calc_crowding_distance
import numpy as np
import json
import torch.optim as optim
from collections import OrderedDict
from kneed import KneeLocator

from Development.DevUtils import load_model_from_json_path
import Development.DevUtils as devU

def load_model(model_path, state_dict_path):
    """
        Function to load models, given that models could be saved as DataParallel or DistributedDataParallel
        modules

        Args:
            model_path: (str) path to pretrained model
            state_dict_paths: (str) path to weights ...

        Returns: loaded model

        """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model_from_json_path(model_json_path=model_path, load_state_dict=False)
    state_dict = torch.load(state_dict_path,  map_location = device)
    new_state_dict = OrderedDict()

    #replace xx.module to xx for parameters wrapped in DataParallel/DDP
    for k, v in state_dict.items():
        if k[:7] == "module.":
            name = k[7:]
        else:
            name = k
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)

    return model

def compute_metrics(preds, targets, metrics):
    """
    Wrapper to flexible compute metrics directly on gpu
    See Development.DevUtils.py for implementations
    Note that some classification metrics may require to use torch.max() or something ...

    Args:
        preds: (torch.Tensor) predictions, passed to metric funtion(s)
        targets: (torch.Tensor) targets, passed to metric funtion(s)
        metrics: (list of str) list of metric keywords e.g. 'acc.', 'bce', 'jaccard score', ...

    Returns: dictionary with metric keywords as keys and metric values as values

    """
    metrics_funcs = {
        "jaccard score": devU.JaccardScore(threshold=0.5, reduction="mean"),
        "acc. seg.": devU.AccuracySegmentation(threshold=0.5),
    }

    res_dict = {}
    for key in metrics:
        try:
            res_dict[key] = metrics_funcs[key](preds, targets)
        except:
            raise NotImplementedError(f"Error in compute_classification_metrics() for {key} " +
                                      "- make sure a metric function is implemented ... ")
    return res_dict


class Swarm(object):
    """
    initializes a swarm of n particles.
    stores all variables, that are shared over all it's particles.
    """

    def __init__(self, n, particle_dim, particle_p0, w_min, w_max, t_max, c_1, c_2):
        """
        Args:
            n: number of particles
            particle_dim: particle dimension, should match number of prunable filters
            particle_p0: probability of an item in the particle position to be initialized with a zero,
                         which means this filter will be pruned
            w_min: min. inertia weight
            w_max: max. inertia weight
            t_max: max. number of iterations
            c_1: c_1 parameter
            c_2: c_2 parameter
        """

        self.particles = [Particle(dim=particle_dim, p0=particle_p0) for _ in range(0, n)]

        self.w_min = w_min
        self.w_max = w_max
        self.t_max = t_max

        self.weight = (w_max - w_min) * 1 + w_min

        self.g_best_position = None
        self.g_best_objectives = None
        self.c_1 = c_1
        self.c_2 = c_2

        # archive that stores non dominated solution sets
        self.archive_objectives = []
        self.archive_positions = []

    def adapt_weight(self, t):
        self.weight = (self.w_max - self.w_min) * ((self.t_max - t) / self.t_max) + self.w_min
        return self.weight

    def update_archive(self, perf, flops, seg_acc, meanIoU, parameters, filters, position):
        if len(self.archive_positions) < 1:
            # if archive is empty this is trivial
            self.archive_objectives.append([perf, flops, seg_acc, meanIoU, parameters, filters])
            self.archive_positions.append(position.clone().detach().requires_grad_(False))
            # if a tensor is manipulated inplace the list item would change and you know, you never know :D
        else:
            if any(ap < perf and af < flops for (ap, af,_, _, _, _) in self.archive_objectives):
                # check if item is dominated by any item, in that case we can skip all the rest
                pass
            else:
                # item seems to be non dominated, therefore we loop through the archive an remove all positions that
                # are dominated by the new item. then we append the new item
                pop_idxs = []
                for i in range(0, len(self.archive_positions)):
                    if self.archive_objectives[i][0] > perf and self.archive_objectives[i][1] > flops:
                        pop_idxs.append(i)

                for index in sorted(pop_idxs, reverse=True):
                    del self.archive_objectives[index]
                    del self.archive_positions[index]

                self.archive_objectives.append([perf, flops, seg_acc, meanIoU, parameters, filters])
                self.archive_positions.append(position.clone().detach().requires_grad_(False))

        # print(self.archive_objectives)

    def sort_resize_archive(self, max_size):
        if len(self.archive_objectives) > max_size:
            distances = calc_crowding_distance(F=np.array(self.archive_objectives))
            # print(distances)
            idx = np.flip(np.argsort(distances)).astype(int)
            # print(idx)

            new_obj = [self.archive_objectives[i] for i in idx]
            new_pos = [self.archive_positions[i] for i in idx]
            self.archive_objectives = new_obj[:max_size]
            self.archive_positions = new_pos[:max_size]

    def update_global_best(self):
        '''#calculate best objective based on the l1 from origin
        obj_1 = [row[0] for row in self.archive_objectives]
        obj_2 = [row[1] for row in self.archive_objectives]
        distances = [a + (b/2e10) for a, b in zip(obj_1, obj_2)]
        index = np.argmin(distances)'''

        index = np.random.randint(0, len(self.archive_positions))
        self.g_best_position = self.archive_positions[index]
        self.g_best_objectives = self.archive_objectives[index]
        print("new global best obj.: {}".format(self.g_best_objectives))

        return self.g_best_objectives

    def fine_tune_best(self, model_path, state_dict_path, n_classes, trainloader, valloader,save_path ):
        """
        Fine tune best particle configuration/solution
        :param model_path: (str) path to pretrained models
        :param state_dict_path: (str) path to pretrained weights
        :param n_classes: (int) number of classes
        :param trainloader: dataloader for train dataset
        :param valloader: dataloader for validation dataset
        :param save_path: (str) path for saving the best solution
        :return: res_dict, Result dictionary with accuracy, meanIoU, FLOPS, performance score, number of filters, number of parameters
        """

        print("Starting fine tuning processes of the best solution :")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        load_path_positions = save_path
        ctr = 0

        res_dict = {
            "particle": [],
            "Performance": [],
            "FLOPS": []
        }

        for particle in self.particles:

            particle.position = torch.load(f"{load_path_positions}/particle_{ctr}.pt")

            p_metrics, p_perf, p_flops, p_parameters, p_filters = get_objective_value_for_particle(model_path=model_path,
                                                                                state_dict_path=state_dict_path,
                                                                                trainloader=trainloader,
                                                                                valloader=valloader,
                                                                                particle_position_vector=particle.position,
                                                                                n_classes=n_classes)

            seg_acc = p_metrics["vl acc. seg."]
            meanIoU = p_metrics["vl jaccard score"]
            print(
                f"  > particle {ctr} / performance = {round(p_perf, 7)} / flops = {p_flops} / seg. acc. = {seg_acc} / meanIoU = {meanIoU}"
                f"/ param = {p_parameters} / filters = {p_filters}")
            ctr += 1

            # check against p_best and non-dominated solutions in archive and update accordingly
            particle.update_p_best(perf=p_perf, flops=p_flops, seg_acc=seg_acc, meanIoU=meanIoU, parameters = p_parameters
                                       , filters = p_filters)
            res_dict["Performance"].append(p_perf)
            res_dict["FLOPS"].append(p_flops)


            self.archive_positions.append(particle.position.clone().detach().requires_grad_(False))
        #get knee solution
        kl = KneeLocator(res_dict["FLOPS"], res_dict["Performance"], curve="concave")
        knee_index = res_dict["FLOPS"].index(kl.elbow)
        self.g_best_position = self.archive_positions[knee_index]

        #prune and load models from the given knee optimal solution
        print("Pruning and fine tuning knee solution from particle {} :".format(knee_index))
        pruned_model = load_model(model_path, state_dict_path)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        pruned_model.to(device=device)
        layers, n_filters = get_layers_for_pruning(model=pruned_model, n_classes=n_classes)
        prune_layers(layers=layers, particle_position_vector=self.g_best_position)

        metrics = ["jaccard score", "acc. seg."]
        max_epochs = 15
        fine_tuned_model, results, performance_score = train_model(model=pruned_model,
                                                         trainloader=trainloader,
                                                         valloader=valloader,
                                                         metrics=metrics,
                                                         device=device,
                                                         lr=0.001,
                                                         max_epochs=max_epochs)

        #calculate pruned flops, number of filters and parameters
        flops_score = calculate_pruned_flops(pruned_model, relevant_layers=layers, shape=(1, 3, 256, 256),
                                             device=device)
        filters = n_filters_after_pruning(pruned_model)
        parameters = parameters_after_pruning(pruned_model)

        print(f"  > Fine tuned particle {knee_index}/ performance = {round(performance_score, 4)} / flops = {flops_score}")
        positions_save_path = save_path + "/particle-" + str(knee_index) + "fine-tuning.pt"

        #Prune the weights before saving the pruned model
        layers, n_filters = get_layers_for_pruning(model=fine_tuned_model, n_classes=n_classes)
        prune_layers(layers=layers, particle_position_vector=self.g_best_position)
        for layer in layers:
            prune.remove(layer, "weight")

        torch.save(fine_tuned_model.state_dict(), positions_save_path)

        res_dict = {
            "particle": [],
            "Performance": [],
            "FLOPS": [],
            "Seg. Acc." :[],
            "meanIoU":[],
            "parameters" : [],
            "filters": []
        }
        res_dict["particle"].append(knee_index)
        res_dict["Performance"].append(performance_score)
        res_dict["FLOPS"].append(kl.elbow)
        res_dict["Seg. Acc."].append(results["vl acc. seg."][-1])
        res_dict["meanIoU"].append(results["vl jaccard score"][-1])
        res_dict["parameters"].append(parameters)
        res_dict["filters"].append(filters)

        return res_dict

        #knee_index = [i for i, x in enumerate(res_dict[FLOPS]) if x == "whatever"]
        #save knee_index, train 15 epochs

    def save_best_model(self, model_path, state_dict_path, n_classes, model_save_path, positions_save_path):
        """
        Function to save the best performing model
        :param model_path: (str) path to pretrained models
        :param state_dict_path: (str) path to weights
        :param n_classes: (int) number of classes
        :param model_save_path: (str) path to the model save directory
        :param positions_save_path: (str) path to the particles' positions save directory
        :return: None
        """
        model = load_model(model_path,state_dict_path)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device=device)

        particle_position_vector = self.g_best_position

        layers, n_filters = get_layers_for_pruning(model=model, n_classes=n_classes)
        assert len(particle_position_vector) == n_filters

        # prune the model - if position is 1: filter is selected, if position is 0: filter is pruned
        prune_layers(layers=layers, particle_position_vector=particle_position_vector)

        for layer in layers:
            prune.remove(layer, "weight")

        torch.save(model.state_dict(), model_save_path)

        model_dict = model.get_model_dict()
        with open(model_save_path.replace(".pt", ".json"), "w") as mf:
            json.dump(model_dict, mf)

        for idx, archive_positions in enumerate(self.archive_positions):
            torch.save(archive_positions,f"{positions_save_path}/particle_{idx}.pt")

    def fine_tune_archives(self, model_path, state_dict_path, n_classes, trainloader, valloader, load_positions,
                           fine_tune_batch, save_path ):
        """
        Function to finetune the archive solutions
        :param model_path: (str) path to pretrained models
        :param state_dict_path: (str) path to pretrained weights
        :param n_classes: (int) number of classes
        :param trainloader: dataloader for train dataset
        :param valloader: dataloader for validation dataset
        :param load_positions: (bool) load positions from saved positions or from the best positions of the particle
        :param fine_tune_batch: (int) number of batch for finetuning
        :param save_path:(str) save path of the finetuned model
        :return: result dictionaries
        """
        print("Starting fine tuning processes of archive solutions :")
        ctr = fine_tune_batch*7
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        load_path_positions = save_path

        res_dict = {
            "particle": [],
            "Performance": [],
            "FLOPS": []
        }


        for particle in self.particles[ctr:ctr+7]:
            #load models and get layers for pruning
            pruned_model = load_model(model_path, state_dict_path)
            layers, n_filters = get_layers_for_pruning(model=pruned_model, n_classes=n_classes)
            assert len(particle.p_best_position) == n_filters

            # prune the model - if position is 1: filter is selected, if position is 0: filter is pruned
            # either load position from saved positions or from best particle position
            if load_positions:
                particle_position_vector = torch.load(f"{load_path_positions}/particle_{ctr}.pt")
            else:
                particle_position_vector = particle.p_best_position

            prune_layers(layers=layers, particle_position_vector=particle_position_vector)

            # train the pruned model
            metrics = ["jaccard score", "acc. seg."]
            max_epochs = 15
            pruned_model,results, performance_score = train_model(model=pruned_model,
                                                            trainloader=trainloader,
                                                            valloader=valloader,
                                                            metrics=metrics,
                                                            device=device,
                                                            lr=0.001,
                                                            max_epochs=max_epochs)

            # calculate pruned FLOPS
            flops_score = calculate_pruned_flops(pruned_model, relevant_layers=layers, shape=(1, 3, 256, 256), device=device)

            print(f"  > Fine tuned particle {ctr}/ performance = {round(performance_score, 4)} / flops = {flops_score}" )
            positions_save_path = save_path + "/particle-" + str(ctr) + "-first-fine-tuning.pt"

            #save the pruned model
            for layer in layers:
                prune.remove(layer, "weight")

            torch.save(pruned_model.state_dict(), positions_save_path)

            res_dict["particle"].append(ctr)
            res_dict["Performance"].append(round(performance_score, 4))
            res_dict["FLOPS"].append(flops_score)
            ctr += 1

        return res_dict

    def pareto_sort(self, load_path, model_path, state_dict_path, valloader, trainloader):
        """
        Function for Pareto sorting of the swarm
        :param load_path: (str) path for saved positions loading
        :param model_path: (str) path to pretrained models
        :param state_dict_path: (str) path to pretrained weights
        :param trainloader: dataloader for train dataset
        :param valloader: dataloader for validation dataset
        :return:
        """

        ctr = 0
        load_path_positions = load_path
        metrics = ["jaccard score", "acc. seg."]
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        costs = []

        res_dict = {
            "particle": [],
            "Performance": [],
            "FLOPS": []
        }

        for particle in self.particles:
            #load particles and model
            particle.position = torch.load(f"{load_path_positions}/particle_{ctr}.pt")
            model = load_model(model_path,state_dict_path)
            model.to(device)

            # prune layers and filters
            layers, n_filters = get_layers_for_pruning(model=model, n_classes=1)
            prune_layers(layers=layers, particle_position_vector=particle.position)


            # validate models
            run_metrics, performance_score = validate_model(model=model,
                                                            valloader=valloader,
                                                            criterion=torch.nn.BCELoss(reduction="mean"),
                                                            metrics=metrics,
                                                            device=device)


            # calculate number of FLOPS and filters
            layers = []
            n_filters = 0
            for m in model.modules():
                if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.ConvTranspose2d):
                    if m.weight.shape[0] != 1 and m.weight.shape[0] != 1:
                        layers.append(m)
                        n_filters += m.weight.shape[0]

            flops_score = calculate_pruned_flops(model, relevant_layers=layers, shape=(1, 3, 256, 256),
                                                 device=device)
            print(f"  > Pruned particle {ctr}/ performance = {round(performance_score, 4)} / flops = {flops_score}")
            ctr += 1

            res_dict["particle"].append(ctr)
            res_dict["Performance"].append(performance_score)
            res_dict["FLOPS"].append(flops_score)

        ctr = 0

        for particle in self.particles:
            #load model and particle
            particle_model_path = load_path + "/particle-" + str(ctr) + "-first-fine-tuning.pt"
            pruned_model = load_model(model_path, particle_model_path)
            pruned_model.to(device)

            layers, n_filters = get_layers_for_pruning(model=pruned_model, n_classes=1)

            # validate performance
            run_metrics, performance_score = validate_model(model=pruned_model,
                                                            valloader=valloader,
                                                            criterion=torch.nn.BCELoss(reduction="mean"),
                                                            metrics=metrics,
                                                            device=device)

            # print("  performance took", time.time() - start_t, "seconds")

            # prune the model based on the loaded particle
            particle.position = torch.load(f"{load_path_positions}/particle_{ctr}.pt")
            prune_layers(layers=layers, particle_position_vector=particle.position)

            #calculate pruned flops, filters and layers
            layers = []
            n_filters = 0
            for m in pruned_model.modules():
                if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.ConvTranspose2d):
                    if m.weight.shape[0] != 1:
                        layers.append(m)
                        n_filters += m.weight.shape[0]

            flops_score = calculate_pruned_flops(pruned_model, relevant_layers=layers, shape=(1, 3, 256, 256), device=device)

            # get results seg_acc , meanIoU and append to the archive adjectives
            seg_acc = run_metrics["vl acc. seg."]
            meanIoU = run_metrics["vl jaccard score"]
            self.archive_objectives.append([performance_score, flops_score, seg_acc, meanIoU])
            self.archive_positions.append(particle.position.clone().detach().requires_grad_(False))

            print(f"  > Fine tuned particle {ctr}/ performance = {round(performance_score, 4)} / flops = {flops_score}")
            ctr += 1
            costs.append([performance_score,flops_score])
            res_dict["particle"].append(ctr)
            res_dict["Performance"].append(performance_score)
            res_dict["FLOPS"].append(flops_score)

        # finding the efficient particles to determine the Pareto sorting
        is_efficient = np.ones(len(costs), dtype=bool)
        costs = np.array(costs)


        for i, c in enumerate(costs):
            if is_efficient[i]:
                is_efficient[is_efficient] = np.any(costs[is_efficient] < c, axis=1)  # Keep any point with a lower cost
                is_efficient[i] = True  # And keep self
            if i == 75:
                is_efficient[i] = False



        for index, c in reversed(list(enumerate(costs))):
            # delete inefficient indices and Pareto sort the efficient ones
            if is_efficient[index] == False:
                del self.archive_objectives[index]
                del self.archive_positions[index]
            else:
                print(f"  > Pareto sorted particles {index}/ performance = {round(self.archive_objectives[index][0], 4)} / flops = {self.archive_objectives[index][1]}" )
                res_dict["particle"].append(index)
                res_dict["Performance"].append(self.archive_objectives[index][0])
                res_dict["FLOPS"].append(self.archive_objectives[index][1])

        #randomly select one of the PAreto sorted solutions and finetune the model with the pruning configuration
        rand_index = np.random.randint(0, len(self.archive_positions))
        flops_score = res_dict["FLOPS"][rand_index]
        index = res_dict["particle"][rand_index]

        particle_model_path = load_path + "/particle-" + str(ctr) + "-first-fine-tuning.pt"
        pruned_model = load_model(model_path, particle_model_path)
        pruned_model.to(device)

        pruned_model.to(device)
        max_epochs = 1
        pruned_model, _, performance_score = train_model(pruned_model, trainloader, valloader, max_epochs, metrics)

        print(f"  > Fine tuned particle {index}/ performance = {round(performance_score, 4)} / flops = {flops_score}")
        positions_save_path = load_path_positions + "/particle-" + str(index) + "-second-fine-tuning.pt"


        torch.save(pruned_model.state_dict(), positions_save_path)
        res_dict["particle"].append(index)
        res_dict["Performance"].append(round(performance_score, 4))
        res_dict["FLOPS"].append(flops_score)


        return res_dict




class Particle(object):
    """
    a basic particle class used for Multi Objective Particle Swarm Optimization
    """

    def __init__(self, dim, p0):
        """
        creates and randomly initializes a particle.
        position vector is filled with either 0 or 1.
        velocity vector is filled from standard normal distribution.

        Args:
            dim: (int) dimensionality of position and velocity vectors
            p0: (float) probability of a 0 during init. at any position. this means filter is pruned
        """

        # randomly initialize position, velocity and inertia weight
        self.position = torch.greater(torch.rand(dim, requires_grad=False), p0).int()  # 1 = selected, 0 = pruned
        self.velocity = torch.randn(dim)  # TODO change if we want to

        # init. personal best with init. position
        self.p_best_position = self.position.clone().detach().requires_grad_(False)  # copies data
        self.p_best_objectives = [None, None, None, None, None ]

        self.c_1 = 2.0
        self.c_2 = 2.0

    def step(self, g_best_position, weight):
        """
        updates velocity (v) and position (p) acc. to formula:
        v_new = weight * v + c1 * random * (p_best_personal - p) + c2 * random * (p_best_global - p)
        p_new = greater((p + v_new), 0.5).int()

        Args:
            g_best_position: global best position vector from swarm
            weight: weight for curr. iteration  from swarm
        """

        r_1 = torch.rand(1)
        r_2 = torch.rand(1)

        self.velocity = weight * self.velocity \
                        + self.c_1 * r_1 * (self.p_best_position - self.position) \
                        + self.c_2 * r_2 * (g_best_position - self.position)

        self.position = torch.greater(self.position + self.velocity, 0.5).int()

    def update_p_best(self, perf, flops, seg_acc, meanIoU, parameters, filters ):
        """
        assumes perf and flops are based on the current particle position vector
        """
        if self.p_best_objectives[0] is None and self.p_best_objectives[1] is None:
            # catch initial case
            self.p_best_objectives = [perf, flops, seg_acc, meanIoU, parameters, filters]
        else:
            if perf < self.p_best_objectives[0] and flops < self.p_best_objectives[1]:
                # if new objectives dominate last best objectives replace it
                self.p_best_objectives = [perf, flops, seg_acc, meanIoU, parameters, filters]
                self.p_best_position = self.position.clone().detach().requires_grad_(False)
            elif perf > self.p_best_objectives[0] and flops > self.p_best_objectives[1]:
                # if last best objectives dominate new objectives drop them
                pass
            else:
                # this means both are non-dominated --> update with chances of 50/50:
                if torch.rand(1) > 0.5:
                    self.p_best_objectives = [perf, flops, seg_acc, meanIoU, parameters, filters]
                    self.p_best_position = self.position.clone().detach().requires_grad_(False)


def get_objective_value_for_particle(model_path, state_dict_path, trainloader, valloader, particle_position_vector, n_classes):
    """
    builds the model from model_path, loads state dict from state_dict_path, prunes acc. to particle_position_vector
    and n_classes and evaluates on valloader and counts flops.

    returns the performance score and flops for a particle position vector.
    """

    # create a new model instance everytime, I don't now why, but only re-loading the state dict is not enough ...
    # this could be optimized (a little) by e.g. not loading from json but from dict ...
    model = load_model(model_path, state_dict_path)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device=device)
    # get the layers to prune
    layers, n_filters = get_layers_for_pruning(model=model, n_classes=n_classes)
    assert len(particle_position_vector) == n_filters

    start_t = time.time()
    # prune the model - if position is 1: filter is selected, if position is 0: filter is pruned
    prune_layers(layers=layers, particle_position_vector=particle_position_vector)
    # print("  pruning took", time.time() - start_t, "seconds")

    metrics =  ["jaccard score", "acc. seg."]

    # validate performance
    run_metrics, performance_score = validate_model(model=model,
                                                    valloader=valloader,
                                                    criterion=torch.nn.BCELoss(reduction="mean"),
                                                    metrics=metrics,
                                                    device=device)
    '''
    _,run_metrics, performance_score = train_model(model=model,
                                       trainloader = trainloader,
                                       valloader=valloader,
                                       metrics = metrics,
                                       device=device,
                                       lr  = 0.05,
                                       max_epochs=5)'''
    #criterion=torch.nn.BCELoss(reduction="mean")
    # print("  performance took", time.time() - start_t, "seconds")

    # get flops estimate
    start_t = time.time()
    '''
    layers = []
    n_filters = 0
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d) or isinstance(m,torch.nn.ConvTranspose2d):
            if m.weight.shape[0] != n_classes and m.weight.shape[0] != 1:
                layers.append(m)
                n_filters += m.weight.shape[0]'''
    flops_score = calculate_pruned_flops(model, relevant_layers=layers, shape=(1, 3, 256, 256), device=device)
    parameters = parameters_after_pruning(model)
    filters = n_filters_after_pruning(model)
    # print("  flops took", time.time() - start_t, "seconds")

    # performance_score = torch.rand(1).item()
    # flops_score = torch.rand(1).item() * 1000
    return run_metrics, performance_score, flops_score, parameters, filters


def validate_model(model, valloader, criterion, metrics, device):
    """
    computation is performed on device.
    python float on cpu is returned

    Args:
        model: (torch.nn.Module) model instance
        valloader: (torch.utils.data.DataLoader) validation data, ideally batch_size=1
        criterion: (torch.nn.Module) criterion for performance assessment
        device: (torch.device) where to map the data

    Returns: mean performance, mean
    """
    run_crit = 0
    model.eval()
    run_metrics = {
        "vl loss": 0.0
    }

    for m in metrics:
        run_metrics["vl " + m] = 0.0

    for x, y in valloader:
        x, y = x.to(device), y.to(device)

        with torch.set_grad_enabled(False):
            y_hat = model(x)
            run_crit += criterion(y_hat, y).item()

            metrics_val = compute_metrics(preds=y_hat, targets=y, metrics=metrics)
            for m in metrics:
                run_metrics["vl " + m] += metrics_val[m].item()

    for m in metrics:
        run_metrics["vl " + m] = run_metrics["vl " + m] / len(valloader)

    return run_metrics, run_crit / len(valloader)


def prune_layers(layers, particle_position_vector):
    """
    prunes the list of layers according to the particle position vector.
    it is assumed, that the length of the particle_position_vector equals the
    total number of filters in the list of layers
    """
    pidx = 0
    for l in layers:
        p = particle_position_vector[pidx: pidx + l.weight.shape[0]]

        mask = torch.zeros_like(l.weight)
        mask[p == 1, :, :, :] += 1

        prune.CustomFromMask(mask).apply(l, "weight", mask)
        pidx += l.weight.shape[0]



def calculate_pruned_flops(model, relevant_layers, shape, device):
    """
    this is a very bare bone way of calculationg the flops,
    based on the assumptions, that:
    - bias parameters are not pruned, therefore inputs to layers are not reduced
    - only convolutional layers are pruned (those in relevant_layers)
    - a pruned model with module.weight_mask is passed
    """

    FLOPs = []

    def hook(module, input, output):


        if isinstance(module, torch.nn.Conv2d):

            rem_input_filters = input[0].shape[1]
            rem_layer_filters = 0
            for i in range(0, module.weight_mask.shape[0]):
                if 1.0 in module.weight_mask[i]:
                    rem_layer_filters += 1

            # since bias terms are not pruned, no input filters can be empty
            # if inputs should be checked as well, this can be done via:
            # rem_input_filters = 0
            # for i in range(0, input[0].shape[1]):
            #    if torch.eq(torch.abs(input[0][0, i, :, :]).sum(), 0) is False:
            #         rem_input_filters += 1

            kernel_size = module.weight.shape[-1] * module.weight.shape[-2]
            n_steps = output.shape[-1] * output.shape[-2]
            # following: Liu et al: Compressing CNNs using Multi-level Filter Pruning
            #                       for the Edge Nodes of Multimedia Internet of Things
            f = n_steps * (kernel_size * rem_input_filters + 1) * rem_layer_filters
            FLOPs.append(f)
        '''n_inputs = 0
        n_filters = 0
        pruned_inputs = 0
        pruned_filters = 0

        f_out_dim = 0 if isinstance(module, torch.nn.Conv2d) else 1

        # print(t, input[0].shape, output.shape, module.weight.shape)

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
        n_steps = output.shape[-1] * output.shape[-2]'''



        return output.abs()

    for l in relevant_layers:
        l.register_forward_hook(hook)

    x = torch.ones(shape, device=device, requires_grad=False)
    _ = model(x)
    return sum(FLOPs)


def get_layers_for_pruning(model, n_classes):
    """
    assumes only filters of conv. layers are pruned.
    if a conv. layer only has one output filter/channel or n_classes output filters/channels, it is ignored
    """
    layers = []
    n_filters = 0
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            if m.weight.shape[0] != n_classes and m.weight.shape[0] != 1:
                layers.append(m)
                n_filters += m.weight.shape[0]

    return layers, n_filters


def train_model(model,trainloader, valloader, max_epochs, metrics, lr, device):
    """
    the core function that handles training and validation of the model.
    By default this function saves the model parameters (also called its state dict)
    of the epoch with the lowest validation set loss at the specified model_path (XY.pt).
    It additionally saves the parameters from the last training epoch as XY_last-epoch.pt.
    This may differ a bit if a lr_scheduler is used. Additional info from model.get_model_dict() is saved as XY.json.
    Based on these files the model can be rebuilt (according to XY.json) and
    its preferred state dict can be loaded (e.g. XY.pt).

    Args:
        model: (torch.nn.Module) the model that should be trained
        model_path: (str) path to save model state dict, should end with .pt
        trainloader: (torch.utils.data.DataLoader) used to update parameters
        valloader: (torch.utils.data.DataLoader) used to validate after each epoch
        max_epochs: (int) maximum number of training epochs
        optimizer_def: (list) e.g. ['adam', { 'lr': 0.001 }], passed to get_optimizer()
        criterion: (str) e.g. 'crossentropy', passed to get_criterion()
        metrics: (list) e.g. ['acc.', 'bce'], passed as arg. to compute_metrics()
        lr_scheduler: (None or str) a keyword that specifies a lr schedule. If None no schedule is used. Please mind
                      the specifications in configurations.py
        early_stopping: (bool) if True the training is aborted if vl loss is not decreasing for as long as stated in
                        configurations.py

    Returns: model_dict (contains model info), res_dict (contains train logs)

    """

    # setup return variables
    res_dict = {
        "epoch": [],
        "tr loss": [],
        "vl loss": []
    }
    for m in metrics:
        res_dict["tr " + m] = []
        res_dict["vl " + m] = []

    # get available devices for parallelization
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model).cuda()


    # build optimizer after model is on gpu (some need it to be ... )
    optimizer = optim.Adam(model.parameters(),lr = lr)
    criterion = torch.nn.BCELoss(weight=None, reduction="mean").to(device)

    '''print("-"*150 +
          "\nTRAINING STARTED")'''
    # ============================================================================================================

    for epoch in range(max_epochs):
        # setup some more variables to save train stats
        run_loss_tr = 0.0
        run_loss_val = 0.0
        run_metrics = {}
        for m in metrics:
            run_metrics["tr " + m] = 0.0
            run_metrics["vl " + m] = 0.0

        eptime = time.time()
        # ========================================================================================================
        # train on trainloader
        model.train()
        for data, labels in trainloader:
            data, labels = data.to(device), labels.to(device)

            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                outputs = model(data).to(device)

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            metrics_tr = compute_metrics(preds=outputs, targets=labels, metrics=metrics)
            run_loss_tr += loss.item()
            for m in metrics:
                run_metrics["tr " + m] += metrics_tr[m].item()

        # ========================================================================================================
        # evaluate on valloader
        model.eval()
        for data, labels in valloader:
            data, labels = data.to(device), labels.to(device)

            optimizer.zero_grad()
            with torch.set_grad_enabled(False):
                outputs = model(data).to(device)

                loss = criterion(outputs, labels)

            metrics_val = compute_metrics(preds=outputs, targets=labels, metrics=metrics)
            run_loss_val += loss.item()
            for m in metrics:
                run_metrics["vl " + m] += metrics_val[m].item()

        res_dict["epoch"].append(epoch + 1)
        res_dict["tr loss"].append(run_loss_tr / len(trainloader))
        res_dict["vl loss"].append(run_loss_val / len(valloader))
        for m in metrics:
            res_dict["tr " + m].append(run_metrics["tr " + m] / len(trainloader))
            res_dict["vl " + m].append(run_metrics["vl " + m] / len(valloader))

        for key in res_dict.keys():
            value = str(round(res_dict[key][-1], 3))
            pad = 5 - len(value)
            value += pad*" "
            print("{}: {} | ".format(key, value), end="")
        print(f" {round(time.time()-eptime)} seconds")

    return model,res_dict, run_loss_val / len(valloader)


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


            n_bias_model += n_bias_layer
            n_weights_model += n_weights_layer
            n_connections_model += n_connections_layer

        if isinstance(m, torch.nn.Conv2d):
            conv_layers.append((m, n_weights_layer))


    idx, ctr = 0, 0
    n_overall_pruned = 0
    for cl, n_weights in conv_layers:
        idx += 1
        pruned_weights = torch.eq(torch.abs(cl.weight), 0).sum().item()
        perc_pruned = round((pruned_weights/n_weights)*100, 2)
        bar = "|" + "="*int(perc_pruned)
        n_overall_pruned += pruned_weights

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



    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.ConvTranspose2d):
            m.register_forward_hook(hook)

    x = torch.ones((1, 3, 256, 256))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = x.to(device)
    _ = model(x)
    return sum(FILTERS["pruned"])