import os
import os.path
import sys
import logging
import copy
import time
import torch
from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import count_parameters


def test(args):
    seed_list = copy.deepcopy(args['seed'])
    device = copy.deepcopy(args['device'])
    device = device.split(',')

    for seed in seed_list:
        args['seed'] = seed
        args['device'] = device
        _test(args)

    myseed = 42069  # set a random seed for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(myseed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(myseed)


def _test(args):
    if args['model_name'] in ['InfLoRA', 'InfLoRA_domain', 'InfLoRAb5_domain', 'InfLoRAb5', 'InfLoRA_CA', 'InfLoRA_CA1']:
        logdir = 'logs/{}/{}_{}_{}/{}/{}/{}/{}_{}-{}'.format(args['dataset'], args['init_cls'], args['increment'], args['net_type'], args['model_name'], args['optim'], args['rank'], args['lamb'], args['lame'], args['lrate'])
    else:
        logdir = 'logs/{}/{}_{}_{}/{}/{}'.format(args['dataset'], args['init_cls'], args['increment'], args['net_type'], args['model_name'], args['optim'])

    if not os.path.exists(logdir):
        os.makedirs(logdir)
    logfilename = os.path.join(logdir, '{}'.format('test'))
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(filename)s] => %(message)s',
        handlers=[
            logging.FileHandler(filename=logfilename + '.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    _set_random(args)
    _set_device(args)
    print_args(args)
    data_manager = DataManager(args['dataset'], args['shuffle'], args['seed'], args['init_cls'], args['increment'], args)
    args['class_order'] = data_manager._class_order
    model = factory.get_model(args['model_name'], args)
    model._network.load_state_dict(torch.load('./logs/ImageNet_R/10_10_sip/InfLoRA/adam/10/0.98_1.0-0.0005/0/task_19.pth'))
    model._network.eval()  # 设置模型为评估模式
    
    cnn_curve, cnn_curve_with_task, nme_curve, cnn_curve_task, cnn_curve_with_task_on_key = {'top1': []}, {'top1': []}, {'top1': []}, {'top1': []}, {'top1': []}
    cnn_curve_task_keys = [[] for _ in range(12)]
    time_start = time.time()
    cnn_accy, cnn_accy_with_task, nme_accy, cnn_accy_task, cnn_accy_task_keys, cnn_accy_with_task_on_key = model.test(data_manager.nb_tasks, data_manager)
    # cnn_accy, cnn_accy_with_task, nme_accy, cnn_accy_task = model.eval_task()
    time_end = time.time()
    logging.info('Time:{}'.format(time_end - time_start))
    # raise Exception
    logging.info('CNN: {}'.format(cnn_accy['grouped']))
    cnn_curve['top1'].append(cnn_accy['top1'])
    cnn_curve_with_task['top1'].append(cnn_accy_with_task['top1'])
    cnn_curve_task['top1'].append(cnn_accy_task)
    logging.info('CNN top1 curve: {}'.format(cnn_curve['top1']))
    logging.info('CNN top1 with task curve: {}'.format(cnn_curve_with_task['top1']))
    logging.info('CNN top1 task curve: {}'.format(cnn_curve_task['top1']))
    
    logging.info('CNN with task on key: {}'.format(cnn_accy_with_task_on_key['grouped']))
    cnn_curve_with_task_on_key['top1'].append(cnn_accy_with_task_on_key['top1'])
    logging.info('CNN top1 with task on key curve: {}'.format(cnn_curve_with_task_on_key['top1']))
    for i in range(len(cnn_accy_task_keys)):
        cnn_curve_task_keys[i].append(cnn_accy_task_keys[i])
        logging.info('CNN top1 task key in layer_{} curve: {}'.format(i, cnn_curve_task_keys[i]))

        

        # if task >= 3: break


def _set_device(args):
    device_type = args['device']
    gpus = []

    for device in device_type:
        if device_type == -1:
            device = torch.device('cpu')
        else:
            device = torch.device('cuda:{}'.format(device))

        gpus.append(device)

    args['device'] = gpus


def _set_random(args):
    torch.manual_seed(args['seed'])
    torch.cuda.manual_seed(args['seed'])
    torch.cuda.manual_seed_all(args['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_args(args):
    for key, value in args.items():
        logging.info('{}: {}'.format(key, value))

import json
import argparse

def main():
    args = setup_parser().parse_args()
    param = load_json(args.config)
    args = vars(args)  # Converting argparse Namespace to a dict.
    args.update(param)  # Add parameters from json
    test(args)


def load_json(settings_path):
    with open(settings_path) as data_file:
        param = json.load(data_file)

    return param


def setup_parser():
    parser = argparse.ArgumentParser(description='Reproduce of multiple continual learning algorthms.')
    parser.add_argument('--config', type=str, default='./exps/finetune.json',
                        help='Json file of settings.')
    parser.add_argument('--device', type=str, default='0')

    # # optim
    # parser.add_argument('--optim', type=str, default='adam', choices=['adam', 'sgd'])
    return parser


if __name__ == '__main__':
    # # 指定保存的模型参数文件路径
    # model_path = "./logs/cifar100/10_10_sip/InfLoRA/adam/10/0.95_1.0-0.0005/0/task_9.pth"
    # # 加载模型参数
    # state_dict = torch.load(model_path)
    # # 打印模型参数的键
    # print("Keys in the state_dict:")
    # for key in state_dict.keys():
    #     print(key)
    # exit()
    main()