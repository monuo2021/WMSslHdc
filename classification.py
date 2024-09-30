## Non-HDFF Imports
import torch
import utils.calculate_log as callog

## HDFF Imports
from HDFF.VSAs import HDFF_VSA
from HDFF.feature_monitor import FeatureMonitor

## Eval imports 
from utils.eval import inference_loop, predict_loop

## Datasets
from utils.datasets import get_ood_dataset

## Setup Imports 
from utils.setup import setup_args, setup_model, setup_data

## Plotting imports 
from utils.plotting import plot_f1, plot_angles

def main(args, config):
    device = 0 if not args.cpu else 'cpu'
    ## Setup our models
    models = setup_model(args, config, device=device)

    ## Setup the VSA & Feature Monitor instances
    VSA = HDFF_VSA()
    FM = FeatureMonitor(models, VSA, config, device=device)

    ## Hook into the layers & projections matrices; preparing for inference
    FM.hookLayers()

    ## Setup the data
    # id_calibration_set: 用于校准ID数据的特征均值。
    # id_test_set: 用于测试模型在ID数据集上的性能。
    # near_ood_test_set: 一个接近分布（nearOOD）的测试数据集。
    # ood_names: 列表，包含不同OOD数据集的名称。
    id_calibration_set, id_test_set, near_ood_test_set, ood_names = setup_data(args, config)

    FM.captureFeatureMeans(id_calibration_set)
    FM.createClassBundles(id_calibration_set)

    ## Inference
    # First on the ID test set
    predict_loop(FM, id_test_set)


if __name__ == '__main__':
    ## Retrieve our arguments
        args = setup_args()

        ## Setup defaults & config from args
        n_classes = 100 if '100' in args.modelpath else 10
        data = f'cifar{n_classes}'
        config = {
                'hyper_size': int(1e4),
                'n_classes': n_classes,
                'model': args.modelpath,
                'input_size': (32, 32),
                'pool': args.pooling,
                'name': data
                }

        main(args, config)

