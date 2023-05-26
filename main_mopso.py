"""Main function to run AMPSO experiments"""

import torch
from torch.utils.data import DataLoader
import os
import csv
import argparse
import copy

import AMPSO.algo as AMPSO
from Preprocessing.Datasets import SegmentationDataSetNPZ, SegmentationDataSetPNG
from Development.DevUtils import load_model_from_json_path
import configurations as c

#AMPSO Hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument('--iter', type=int, default=30, help='iteration amount')
parser.add_argument('--amount', type=float, default=0.2, help='pruning amount')
parser.add_argument('--c1', type=float, default=1.5, help='exploitation amount')
parser.add_argument('--c2', type=float, default=1.5, help='exploration amount')
parser.add_argument('--swarm', type=int, default=30, help='exploitation amount')
parser.add_argument('--fine_tune_batch', type = int, default = None, help = 'fine tuning in batches ( 5 batches of 6 ) for parallelization')
parser.add_argument('--architecture' , type = str, default = "TernausNet", help = 'Architecture' )
parser.add_argument('--use_static_weights' , type = bool, default = False, help = 'Use static weights' )
parser.add_argument('--database', type = str , default = "INRIA", help ='database')
args = parser.parse_args()

#set number of images and classes here
N_IMAGES = 100
N_CLASSES = 1

#uncomment for JSON and state_dict
# MODEL_JSON = "_results/saved_models_cluster/paper/pruning-references-test_2021-03-29_08-29-00.json" pruning-UnetClassic-reference_2021-05-05_09-24-39.json
# MODEL_STATE_DICT = "_results/saved_models_cluster/paper/pruning-references-test_2021-03-29_08-29-00.pt"pruning-reference-TernausNet-15-epochs_2021-09-20_11-42-09

#setting up constants from argparse inputs
SWARM_SIZE = args.swarm  # optimal 10 - 30
PARTICLE_P0 = args.amount  # controls pruning amount
W_MIN = 0.4
W_MAX = 0.9
T_MAX = args.iter  # usually > 100
C_1 = args.c1  # controls exploitation
C_2 = args.c2  # controls exploration, should be the same as C_1 (but could theoretically be changed)
ARCHIVE_SIZE = SWARM_SIZE
USE_STATIC_WEIGHTS = args.use_static_weights


def main():
    #set model name based on the model type/architecture
    if args.architecture == "TernausNet":
        model_name = 'pruning-references-test_2021-03-29_08-29-00'
    elif args.architecture == "UNet":
        model_name = 'pruning-UnetClassic-reference_2021-05-05_09-24-39'
    elif args.architecture == "FCN":
        model_name =  'pruning-reference-FCN8VGG16-15-epochs_2021-09-30_14-11-47'

    #loading models from saved trained models in the cluster, commented part is for local tests with local directory
    MODEL_JSON = "/data/project-gxb/johannes/innspector/_results/saved_models/"  + model_name + ".json"
    MODEL_STATE_DICT = "/data/project-gxb/johannes/innspector/_results/saved_models/"  + model_name + ".pt"
    #MODEL_JSON = "_results/saved_models/" + model_name + ".json"
    #MODEL_STATE_DICT = "_results/saved_models/" + model_name + ".pt"

    # prepare validation dataloader
    if args.database == "AIRS":
        d_conf = c.get_data_config()["AIRS"]
    else:
        d_conf = c.get_data_config()["inria"]

    #get all the image and image masks filenames from training and validation folders
    train_names = os.listdir(d_conf["train_dir"] + d_conf["image_folder"])
    val_names = os.listdir(d_conf["val_dir"] + d_conf["image_folder"])
    p_imgs_train = [d_conf["train_dir"] + d_conf["image_folder"] + n for n in train_names]
    p_msk_train = [d_conf["train_dir"] + d_conf["mask_folder"] + n for n in train_names]
    p_imgs_val = [d_conf["val_dir"] + d_conf["image_folder"] + n for n in val_names]
    p_msk_val = [d_conf["val_dir"] + d_conf["mask_folder"] + n for n in val_names]

    #setting number of images for particle search optimization and evaluation, using the whole dataset is costly, thus we set to 100
    if N_IMAGES is not None:
        img_paths = p_imgs_val[:N_IMAGES]
        mask_paths = p_msk_val[:N_IMAGES]
        eval_img_paths = p_imgs_train
        eval_mask_paths = p_msk_train
    else:
        img_paths = p_imgs_val
        mask_paths = p_msk_val

    #initialize the dataset classes based on the dataset
    if args.database == "AIRS":
        valset = SegmentationDataSetPNG(img_paths=img_paths, mask_paths=mask_paths, p_flips=None, p_noise=None)
        evalset = SegmentationDataSetPNG(img_paths=eval_img_paths, mask_paths=eval_mask_paths, p_flips=None,
                                         p_noise=None)
        trainset = SegmentationDataSetPNG(img_paths=p_imgs_train, mask_paths=p_msk_train,
                                          p_flips=0.5, p_noise=0.5)
    else:
        valset = SegmentationDataSetNPZ(img_paths=img_paths, mask_paths=mask_paths, p_flips=None, p_noise=None)
        evalset = SegmentationDataSetNPZ(img_paths=eval_img_paths, mask_paths=eval_mask_paths, p_flips=None, p_noise=None)
        trainset = SegmentationDataSetNPZ(img_paths=p_imgs_train, mask_paths=p_msk_train,
                                          p_flips=0.5, p_noise=0.5)

    #initialize corresponding data loader
    valloader = DataLoader(valset, batch_size=64, shuffle=False, num_workers=8, pin_memory = True )
    evalloader = DataLoader(evalset, batch_size=64, shuffle=False, num_workers=8, pin_memory=True)

    # initialize one or multiple swarms of particles
    _, filters = AMPSO.get_layers_for_pruning(load_model_from_json_path(model_json_path=MODEL_JSON, load_state_dict=False), N_CLASSES)

    swarm = AMPSO.Swarm(n=SWARM_SIZE, particle_dim=filters, particle_p0=PARTICLE_P0,
                        w_min=W_MIN, w_max=W_MAX, t_max=T_MAX, c_1=C_1, c_2=C_2)


    #setting up model names and particle save names
    save_path = copy.deepcopy(MODEL_STATE_DICT)
    save_path = save_path.replace( model_name + ".pt",
                                  "AMPSO-innspector-p0-" + str(PARTICLE_P0) + "-iteration-" + str(T_MAX) + args.database + ".pt")
    positions_save_path = copy.deepcopy(MODEL_STATE_DICT)
    positions_save_path = positions_save_path.replace(model_name + ".pt", "AMPSO positions/p0_" +str(PARTICLE_P0))

    if args.architecture != "TernausNet":
        positions_save_path = positions_save_path + "_"  + args.architecture

    if args.use_static_weights:
        positions_save_path = positions_save_path + "_static_weights"

    if args.database == "AIRS":
        positions_save_path = positions_save_path + "_" + args.database



    #further finetuning with best particle configurations
    if args.fine_tune_batch != None:

        trainloader = DataLoader(trainset, batch_size=64, shuffle=False, num_workers=8, pin_memory=True)


        res_dict = swarm.fine_tune_best(model_path=MODEL_JSON, state_dict_path=MODEL_STATE_DICT, n_classes = N_CLASSES,
                                 trainloader =evalloader, valloader = valloader, load_positions = True
                                 , save_path = positions_save_path)

    #normal particle optimization procedure
    else:
        csv_file = "/data/project-gxb/johannes/innspector/_results/AMPSO_pareto_eval_test_4.csv"
        # start iterating
        for t in range(T_MAX):
            print(f"\niter {t} " + "-" * 100)
            ctr = 0  # particle counter for print
            for particle in swarm.particles:
                #get objective value for the specific particle within the swarm
                p_metrics, p_perf, p_flops, p_parameters, p_filters = AMPSO.get_objective_value_for_particle(model_path=MODEL_JSON,
                                                                                    state_dict_path=MODEL_STATE_DICT,
                                                                                    trainloader=evalloader,
                                                                                    valloader=valloader,
                                                                                    particle_position_vector=particle.position,
                                                                                    n_classes=N_CLASSES)

                seg_acc = p_metrics["vl acc. seg."]
                meanIoU = p_metrics["vl jaccard score"]
                print(
                    f"  > particle {ctr} / performance = {round(p_perf, 4)} / flops = {p_flops} / seg. acc. = {seg_acc} / meanIoU = {meanIoU} "
                    f"/ param = {p_parameters} / filters = {p_filters}")
                ctr += 1

                # check against p_best and non-dominated solutions in archive and update accordingly
                particle.update_p_best(perf=p_perf, flops=p_flops, seg_acc=seg_acc, meanIoU=meanIoU, parameters = p_parameters
                                       , filters = p_filters )
                swarm.update_archive(perf=p_perf, flops=p_flops, seg_acc=seg_acc, meanIoU=meanIoU, parameters = p_parameters, filters = p_filters,
                                     position=particle.position)

            # if archive is longer than max len it is sorted by crowding distances and cropped to max len
            swarm.sort_resize_archive(max_size=ARCHIVE_SIZE)

            if t < T_MAX - 1:
                best_obj = swarm.update_global_best()  # randomly select one of archive solutions as g_best
                if not USE_STATIC_WEIGHTS:
                    swarm.adapt_weight(t=t)  # adapt inertia weight acc. iteration

                for particle in swarm.particles:
                    # update position and velocity values for every particle
                    particle.step(g_best_position=swarm.g_best_position, weight=swarm.weight)

            '''Generation plot thing'''
            '''if (t+1)%5 == 0:
                csv_columns = ['Performance', 'FLOPS', 'c_1 ' + str(C_1), 'c_2 ' + str(C_2),
                               'Swarm Size ' + str(SWARM_SIZE) + 'amount ' + str(PARTICLE_P0) + ' ' + args.architecture
                               + 'Generation ' + str(t)]
                file_exists = False
                for objectives in swarm.archive_objectives:

                    dict = [
                        {'Performance': objectives[0], 'FLOPS': objectives[1]}
                    ]

                    try:
                        with open(csv_file, 'a') as csvfile:
                            writer = csv.DictWriter(csvfile, delimiter=',', fieldnames=csv_columns)
                            if not file_exists:
                                writer.writeheader()
                                file_exists = True
                            for data in dict:
                                writer.writerow(data)
                    except IOError:
                        print("I/O error")'''

        #save best performing model
        swarm.save_best_model(model_path=MODEL_JSON, state_dict_path=MODEL_STATE_DICT, n_classes=N_CLASSES,
                              model_save_path=save_path, positions_save_path=positions_save_path)


        #writing and saving results into .csv
        csv_columns = ['Pruning amount', 'Iteration', 'Performance','FLOPS','Seg. Acc.', 'meanIoU', 'parameters', 'filters', 'amount ' + str(PARTICLE_P0) + ' ' + args.architecture]
        dict = [
            {'Pruning amount': PARTICLE_P0, 'Iteration': T_MAX, 'Performance': best_obj[0], 'FLOPS': best_obj[1], 'Seg. Acc.': best_obj[2], 'meanIoU' : best_obj[3]
             , 'parameters' : best_obj[4], 'filters' : best_obj[5]}
        ]
        if args.use_static_weights:
            csv_file = "/data/project-gxb/johannes/innspector/_results/AMPSO_static_weights_prune.csv"
        else:
            csv_file = "/data/project-gxb/johannes/innspector/_results/AMPSO_prune_amount_AIRS.csv"
        file_exists = os.path.isfile(csv_file)

        try:
            with open(csv_file, 'a') as csvfile:
                writer = csv.DictWriter(csvfile, delimiter=',', fieldnames=csv_columns)
                if not file_exists:
                    writer.writeheader()
                for data in dict:
                    writer.writerow(data)
        except IOError:
            print("I/O error")

    #csv writing procedure for further finetuned networks
    if args.fine_tune_batch != None:

        if args.use_static_weights:
            csv_file = "/data/project-gxb/johannes/innspector/_results/AMPSO_static_weights_fine_tune.csv"
        else:
            csv_file = "/data/project-gxb/johannes/innspector/_results/AMPSO_pareto_finetuning_AIRS.csv"

        csv_columns = ['particle', 'Performance', 'FLOPS', 'seg_acc', 'meanIoU', 'parameters', 'filters', 'amount ' + str(PARTICLE_P0) + ' ' + args.architecture]
        file_exists = False

        for particle,perf,flops,seg_acc,meanIoU, parameters, filters in zip(res_dict["particle"], res_dict["Performance"], res_dict["FLOPS"], res_dict["Seg. Acc."], res_dict["meanIoU"],
                                                                            res_dict["parameters"], res_dict["filters"]):
            dict = [
                {'particle' : particle, 'Performance': perf, 'FLOPS': flops, 'seg_acc': seg_acc, 'meanIoU' : meanIoU, 'parameters' : parameters, 'filters' : filters}
            ]
            try:
                with open(csv_file, 'a') as csvfile:
                    writer = csv.DictWriter(csvfile, delimiter=',', fieldnames=csv_columns)
                    if not file_exists:
                        writer.writeheader()
                        file_exists = True
                    for data in dict:
                        writer.writerow(data)
            except IOError:
                print("I/O error")

    




if __name__ == "__main__":
    main()

