import torch
import torch.nn as nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

import logging
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans

from methods.base import BaseLearner
from utils.toolkit import tensor2numpy, accuracy
from models.sinet_inflora_trans import SiNet
from models.vit_inflora_trans import Attention_LoRA
from copy import deepcopy
from utils.schedulers import CosineSchedule
import ipdb
import math
from torch.distributions.multivariate_normal import MultivariateNormal

from utils.attack import Attack

class InfLoRA_CA(BaseLearner):

    def __init__(self, args):
        super().__init__(args)

        if args["net_type"] == "sip":
            self._network = SiNet(args)
        else:
            raise ValueError('Unknown net: {}.'.format(args["net_type"]))
        
        for module in self._network.modules():
            if isinstance(module, Attention_LoRA):
                module.init_param()

        self.args = args
        self.optim = args["optim"]
        self.EPSILON = args["EPSILON"]
        self.init_epoch = args["init_epoch"]
        self.init_lr = args["init_lr"]
        self.init_lr_decay = args["init_lr_decay"]
        self.init_weight_decay = args["init_weight_decay"]
        self.epochs = args["epochs"]
        self.lrate = args["lrate"]
        self.lrate_decay = args["lrate_decay"]
        self.batch_size = args["batch_size"]
        self.weight_decay = args["weight_decay"]
        self.num_workers = args["num_workers"]
        self.lamb = args["lamb"]
        self.lame = args["lame"]
        self.total_sessions = args["total_sessions"]
        self.dataset = args["dataset"]
        self.__device = args['device'][1]
        if self.dataset == 'cifar100':
            self.logit_norm = None
        else:
            self.logit_norm = 0.1

        self.topk = 1  # origin is 5
        self.class_num = self._network.class_num
        self.debug = False

        self.all_keys = []
        self.feature_list = []
        self.project_type = []
        self.base_list = []
        self.base_type = ['remove' for i in range(12)]
        self.task_r = []

        self.task_sizes = []

        self.data_manager = 0

    def after_task(self):
        # self._old_network = self._network.copy().freeze()
        self._known_classes = self._total_classes
        logging.info('Exemplar size: {}'.format(self.exemplar_size))

    def incremental_train(self, data_manager):
        self.data_manager = data_manager
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self.task_sizes.append(data_manager.get_task_size(self._cur_task))
        self._network.update_fc(self._total_classes)

        logging.info('Learning on {}-{}'.format(self._known_classes, self._total_classes))

        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source='train', mode='train')
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,
                                       num_workers=self.num_workers)
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source='test', mode='test')
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False,
                                      num_workers=self.num_workers)

        # if len(self._multiple_gpus) > 1:
        #     self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)
        self.clustering(self.train_loader)
        # if len(self._multiple_gpus) > 1:
        #     self._network = self._network.module

        # CA
        # self._network.classifier_backup(self._cur_task)
        # if self.save_before_ca:
        #     self.save_checkpoint(self.log_path+'/'+self.model_prefix+'_seed{}_before_ca'.format(self.seed), head_only=self.fix_bcb)
        
        # self._compute_class_mean(data_manager, check_diff=False, oracle=False)
        if self._cur_task>0:
            self._stage2_compact_classifier(self.task_sizes[-1])
            if len(self._multiple_gpus) > 1:
                self._network = self._network.module

        # mean_errors, cov_errors = self._get_real_mean_cov()
        # logging.info("Mean errors: {}".format(mean_errors))
        # logging.info("Cov errors: {}".format(cov_errors))

    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                if isinstance(model, nn.DataParallel):
                    outputs = model.module.interface(inputs)
                else:
                    outputs = model.interface(inputs)
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)

        return np.around(tensor2numpy(correct)*100 / total, decimals=2)

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        # if self._old_network is not None:
        #     self._old_network.to(self._device)

        for name, param in self._network.named_parameters():
            param.requires_grad_(False)
            try:
                if "classifier_pool" + "." + str(self._network.module.numtask - 1) in name:
                    param.requires_grad_(True)
                if "lora_B_k" + "." + str(self._network.module.numtask - 1) in name:
                    param.requires_grad_(True)
                if "lora_B_v" + "." + str(self._network.module.numtask - 1) in name:
                    param.requires_grad_(True)
                # if "lora_B_trans_k" + "." + str(self._network.module.numtask - 1) in name:
                #     param.requires_grad_(True)
                # if "lora_B_trans_v" + "." + str(self._network.module.numtask - 1) in name:
                #     param.requires_grad_(True)
                if "lora_S_trans_k" + "." + str(self._network.module.numtask - 1) in name:
                    param.requires_grad_(True)
                if "lora_S_trans_v" + "." + str(self._network.module.numtask - 1) in name:
                    param.requires_grad_(True)

            except:
                if "classifier_pool" + "." + str(self._network.numtask - 1) in name:
                    param.requires_grad_(True)
                if "lora_B_k" + "." + str(self._network.numtask - 1) in name:
                    param.requires_grad_(True)
                if "lora_B_v" + "." + str(self._network.numtask - 1) in name:
                    param.requires_grad_(True)
                # if "lora_B_trans_k" + "." + str(self._network.numtask - 1) in name:
                #     param.requires_grad_(True)
                # if "lora_B_trans_v" + "." + str(self._network.numtask - 1) in name:
                #     param.requires_grad_(True)
                if "lora_S_trans_k" + "." + str(self._network.numtask - 1) in name:
                    param.requires_grad_(True)
                if "lora_S_trans_v" + "." + str(self._network.numtask - 1) in name:
                    param.requires_grad_(True)

        # Double check
        enabled = set()
        for name, param in self._network.named_parameters():
            if param.requires_grad:
                enabled.add(name)

        with torch.no_grad():
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                self._network(inputs, get_cur_feat=True)
                # if i > 3: break

            if self._cur_task == 0:
                for module in self._network.modules():
                    if isinstance(module, Attention_LoRA):
                        cur_matrix = module.cur_matrix
                        U, S, V = torch.svd(cur_matrix)
                        module.lora_A_k[self._cur_task].weight.data.copy_(U[:,:module.rank].T/math.sqrt(3))
                        module.lora_A_v[self._cur_task].weight.data.copy_(U[:,:module.rank].T/math.sqrt(3))
                        # module.lora_A_trans_k[self._cur_task].weight.data.copy_(U[:,:module.rank].T/math.sqrt(3))
                        # module.lora_A_trans_v[self._cur_task].weight.data.copy_(U[:,:module.rank].T/math.sqrt(3))
                        module.lora_A_trans_k[self._cur_task].weight.data.zero_()
                        module.lora_A_trans_v[self._cur_task].weight.data.zero_()
                        module.cur_matrix.zero_()
                        module.n_cur_matrix = 0
            else:
                kk = 0
                for module in self._network.modules():
                    if isinstance(module, Attention_LoRA):
                        cur_matrix = module.cur_matrix
                        cur_matrix_new = torch.mm(self.feature_mat[kk],cur_matrix)
                        cur_matrix = cur_matrix - torch.mm(self.feature_mat[kk],cur_matrix)
                        cU, cS, cV = torch.svd(cur_matrix)
                        module.lora_A_k[self._cur_task].weight.data.copy_(cU[:,:module.rank].T/math.sqrt(3))
                        module.lora_A_v[self._cur_task].weight.data.copy_(cU[:,:module.rank].T/math.sqrt(3))
                        cU_n, cS_n, cV_n = torch.svd(cur_matrix_new)
                        # A_trans_pinv = torch.pinverse(cU_n[:,:module.rank].T/math.sqrt(3))  # 结果形状 (k, m) = (50, 100)
                        module.lora_A_trans_k[self._cur_task].weight.data.copy_(cU_n[:,:module.rank].T/math.sqrt(3))
                        # B_trans_k = weight_k @ A_trans_pinv
                        B_trans_k = torch.zeros_like(cU_n[:,:module.rank])
                        B_trans_k[:module.rank, :module.rank] = torch.eye(module.rank)
                        module.lora_B_trans_k[self._cur_task].weight.data.copy_(B_trans_k)
                        module.lora_A_trans_v[self._cur_task].weight.data.copy_(cU_n[:,:module.rank].T/math.sqrt(3))
                        # B_trans_v = weight_v @ A_trans_pinv
                        B_trans_v = torch.zeros_like(cU_n[:,:module.rank])
                        B_trans_v[:module.rank, :module.rank] = torch.eye(module.rank)
                        module.lora_B_trans_v[self._cur_task].weight.data.copy_(B_trans_v)
                        module.cur_matrix.zero_()
                        module.n_cur_matrix = 0
                        kk += 1

                # kk = 0
                # for module in self._network.modules():
                #     if isinstance(module, Attention_LoRA):
                #         cur_matrix = module.cur_matrix
                #         cU_n, cS_n, cV_n = torch.svd(cur_matrix)
                #         if self.project_type[kk] == 'remove':
                #             cur_matrix_new = torch.mm(self.feature_mat[kk],cur_matrix)
                #             cur_matrix = cur_matrix - torch.mm(self.feature_mat[kk],cur_matrix)
                #         else:
                #             assert self.project_type[kk] == 'retain'
                #             cur_matrix_new = cur_matrix - torch.mm(self.feature_mat[kk],cur_matrix)
                #             cur_matrix = torch.mm(self.feature_mat[kk],cur_matrix)
                #         cU, cS, cV = torch.svd(cur_matrix)
                #         module.lora_A_k[self._cur_task].weight.data.copy_(cU[:,:module.rank].T/math.sqrt(3))
                #         module.lora_A_v[self._cur_task].weight.data.copy_(cU[:,:module.rank].T/math.sqrt(3))
                #         cU_n, cS_n, cV_n = torch.svd(cur_matrix_new)
                #         # A_trans_pinv = torch.pinverse(cU_n[:,:module.rank].T/math.sqrt(3))  # 结果形状 (k, m) = (50, 100)
                #         module.lora_A_trans_k[self._cur_task].weight.data.copy_(cU_n[:,:module.rank].T/math.sqrt(3))
                #         # B_trans_k = weight_k @ A_trans_pinv
                #         B_trans_k = torch.zeros_like(cU_n[:,:module.rank])
                #         B_trans_k[:module.rank, :module.rank] = torch.eye(module.rank)
                #         module.lora_B_trans_k[self._cur_task].weight.data.copy_(B_trans_k)
                #         module.lora_A_trans_v[self._cur_task].weight.data.copy_(cU_n[:,:module.rank].T/math.sqrt(3))
                #         # B_trans_v = weight_v @ A_trans_pinv
                #         B_trans_v = torch.zeros_like(cU_n[:,:module.rank])
                #         B_trans_v[:module.rank, :module.rank] = torch.eye(module.rank)
                #         module.lora_B_trans_v[self._cur_task].weight.data.copy_(B_trans_v)

                #         module.cur_matrix.zero_()
                #         module.n_cur_matrix = 0
                #         kk += 1
        
        print(f"Parameters to be updated: {enabled}")
        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)

        if self._cur_task==0:
            if self.optim == 'sgd':
                optimizer = optim.SGD(self._network.parameters(), momentum=0.9,lr=self.init_lr,weight_decay=self.init_weight_decay)
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,T_max=self.init_epoch)
            elif self.optim == 'adam':
                optimizer = optim.Adam(self._network.parameters(),lr=self.init_lr,weight_decay=self.init_weight_decay, betas=(0.9,0.999))
                scheduler = CosineSchedule(optimizer=optimizer,K=self.init_epoch)
            else:
                raise Exception
            self.run_epoch = self.init_epoch
            self.train_function(train_loader,test_loader,optimizer,scheduler)
        else:
            if self.optim == 'sgd':
                optimizer = optim.SGD(self._network.parameters(), momentum=0.9,lr=self.lrate,weight_decay=self.weight_decay)
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,T_max=self.epochs)
            elif self.optim == 'adam':
                optimizer = optim.Adam(self._network.parameters(),lr=self.lrate,weight_decay=self.weight_decay, betas=(0.9,0.999))
                scheduler = CosineSchedule(optimizer=optimizer,K=self.epochs)
            else:
                raise Exception
            self.run_epoch = self.epochs
            self.train_function(train_loader, test_loader, optimizer, scheduler)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

        with torch.no_grad():
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                self._network(inputs, get_cur_feat=True)

            mat_list = []
            layer = 0
            for module in self._network.modules():
                if isinstance(module, Attention_LoRA):
                    layer += 1
                    mat_list.append(deepcopy(module.cur_matrix))
                    module.cur_matrix.zero_()
                    module.n_cur_matrix = 0
                    logging.info('Layer {} - lora_S_trans_k {}: {}, lora_S_trans_v {}: {}'
                                 .format(layer, torch.sum(torch.abs(module.lora_S_trans_k[self._cur_task].weight)), module.lora_S_trans_k[self._cur_task].weight.cpu().numpy(),
                                        torch.sum(torch.abs(module.lora_S_trans_v[self._cur_task].weight)), module.lora_S_trans_v[self._cur_task].weight.cpu().numpy()))
            self.update_GPM(mat_list)
            # self.update_DualGPM(mat_list)

            # Projection Matrix Precomputation
            self.feature_mat = []
            for p in range(len(self.feature_list)):
                Uf=torch.Tensor(np.dot(self.feature_list[p],self.feature_list[p].transpose()))
                print('Layer {} - Projection Matrix shape: {}'.format(p+1,Uf.shape))
                self.feature_mat.append(Uf)

        return

    def train_function(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.run_epoch))
        for _, epoch in enumerate(prog_bar):
            self._network.eval()
            losses = 0.
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):

                inputs, targets = inputs.to(self._device), targets.to(self._device)
                mask = (targets >= self._known_classes).nonzero().view(-1)
                inputs = torch.index_select(inputs, 0, mask)
                targets = torch.index_select(targets, 0, mask)-self._known_classes

                logits = self._network(inputs)['logits']
                loss = F.cross_entropy(logits[:, self._known_classes:], targets)

                optimizer.zero_grad()
                loss.backward()

                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)
                if self.debug and i > 10: break

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            # test_acc = self._compute_accuracy_domain(self._network, test_loader)
            info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}'.format(
                self._cur_task, epoch + 1, self.run_epoch, losses / len(train_loader), train_acc)
            prog_bar.set_description(info)

        logging.info(info)

    def clustering(self, dataloader):
        features = []
        for i, (_, inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(self._device), targets.to(self._device)
            mask = (targets >= self._known_classes).nonzero().view(-1)
            inputs = torch.index_select(inputs, 0, mask)
            with torch.no_grad():
                if isinstance(self._network, nn.DataParallel):
                    feature = self._network.module.extract_vector(inputs)
                else:
                    feature = self._network.extract_vector(inputs)
            feature = feature / feature.norm(dim=-1, keepdim=True)
            features.append(feature)
        features = torch.cat(features, 0).cpu().detach().numpy()
        clustering = KMeans(n_clusters=5, random_state=0).fit(features)
        self.all_keys.append(torch.tensor(clustering.cluster_centers_).to(feature.device))

    def eval_task(self):
        y_pred, y_pred_with_task, y_true, y_pred_task, y_true_task = self._eval_cnn(self.test_loader)
        y_pred1, y_pred_with_task1, y_true1, y_pred_task1, y_true_task1 = self._eval_cnn1(self.test_loader)
        cnn_accy = self._evaluate(y_pred, y_true)
        cnn_accy1 = self._evaluate(y_pred1, y_true1)
        cnn_accy_with_task = self._evaluate(y_pred_with_task, y_true)
        cnn_accy_with_task1 = self._evaluate(y_pred_with_task1, y_true1)
        cnn_accy_task = (y_pred_task == y_true_task).sum().item()/len(y_pred_task)
        cnn_accy_task1 = (y_pred_task1 == y_true_task1).sum().item()/len(y_pred_task1)

        if hasattr(self, '_class_means'):
            y_pred, y_true = self._eval_nme(self.test_loader, self._class_means)
            nme_accy = self._evaluate(y_pred, y_true)
        else:
            nme_accy = None

        return cnn_accy, cnn_accy_with_task, nme_accy, cnn_accy_task

    def _evaluate(self, y_pred, y_true):
        ret = {}
        grouped = accuracy(y_pred, y_true, self._known_classes, self.class_num)
        ret['grouped'] = grouped
        ret['top1'] = grouped['total']
        return ret

    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        y_pred_with_task = []
        y_pred_task, y_true_task = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            targets = targets.to(self._device)

            with torch.no_grad():
                y_true_task.append((targets//self.class_num).cpu())

                if isinstance(self._network, nn.DataParallel):
                    outputs = self._network.module.interface(inputs)
                else:
                    outputs = self._network.interface(inputs)

            predicts = torch.topk(outputs, k=self.topk, dim=1, largest=True, sorted=True)[1].view(-1)  # [bs, topk]
            y_pred_task.append((predicts//self.class_num).cpu())

            outputs_with_task = torch.zeros_like(outputs)[:,:self.class_num]
            for idx, i in enumerate(targets//self.class_num):
                en, be = self.class_num*i, self.class_num*(i+1)
                outputs_with_task[idx] = outputs[idx, en:be]
            predicts_with_task = outputs_with_task.argmax(dim=1)
            predicts_with_task = predicts_with_task + (targets//self.class_num)*self.class_num

            # print(predicts.shape)
            y_pred.append(predicts.cpu().numpy())
            y_pred_with_task.append(predicts_with_task.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(y_pred_with_task), np.concatenate(y_true), torch.cat(y_pred_task), torch.cat(y_true_task)  # [N, topk]

    def _eval_cnn1(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        y_pred_with_task = []
        y_pred_task, y_true_task = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            targets = targets.to(self._device)

            with torch.no_grad():
                y_true_task.append((targets//self.class_num).cpu())
                if isinstance(self._network, nn.DataParallel):
                    outputs = self._network.module.interface(inputs)
                else:
                    outputs = self._network.interface(inputs)

            predicts = torch.topk(outputs, k=self.topk, dim=1, largest=True, sorted=True)[1].view(-1)  # [bs, topk]
            y_pred_task.append((predicts//self.class_num).cpu())

            with torch.no_grad():
                if isinstance(self._network, nn.DataParallel):
                    outputs_ = self._network.module.interface1(inputs, predicts//self.class_num)
                else:
                    outputs_ = self._network.interface1(inputs, predicts//self.class_num)
            # if (predicts != outputs_.argmax(dim=1)).sum() != 0:
            #     import ipdb
            #     ipdb.set_trace()
            predicts = outputs_.argmax(dim=1) + (predicts//self.class_num)*self.class_num

            with torch.no_grad():
                if isinstance(self._network, nn.DataParallel):
                    outputs_with_task = self._network.module.interface1(inputs, targets//self.class_num)
                else:
                    outputs_with_task = self._network.interface1(inputs, targets//self.class_num)
            predicts_with_task = outputs_with_task.argmax(dim=1)
            predicts_with_task = predicts_with_task + (targets//self.class_num)*self.class_num

            # print(predicts.shape)
            y_pred.append(predicts.cpu().numpy())
            y_pred_with_task.append(predicts_with_task.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(y_pred_with_task), np.concatenate(y_true), torch.cat(y_pred_task), torch.cat(y_true_task)  # [N, topk]


    def _compute_accuracy_domain(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = model(inputs)['logits']

            predicts = torch.max(outputs, dim=1)[1]
            correct += ((predicts % self.class_num).cpu() == (targets % self.class_num)).sum()
            total += len(targets)

        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)

    def update_GPM (self, mat_list):
        threshold = (self.lame - self.lamb)*self._cur_task/self.total_sessions + self.lamb
        print ('Threshold: ', threshold) 
        if len(self.feature_list) == 0:
            # After First Task 
            for i in range(len(mat_list)):
                activation = mat_list[i]
                U,S,Vh = np.linalg.svd(activation, full_matrices=False)
                # U=self.pq(torch.from_numpy(U), 0.25)
                # U = U.numpy()
                # criteria (Eq-5)
                sval_total = (S**2).sum()
                sval_ratio = (S**2)/sval_total
                r = np.sum(np.cumsum(sval_ratio)<threshold) #+1  
                self.feature_list.append(U[:,0:max(r,1)])
                self.task_r.append(max(r,1))
        else:
            for i in range(len(mat_list)):
                activation = mat_list[i]
                U1,S1,Vh1=np.linalg.svd(activation, full_matrices=False)
                sval_total = (S1**2).sum()
                # Projected Representation (Eq-8)
                act_hat = activation - np.dot(np.dot(self.feature_list[i],self.feature_list[i].transpose()),activation)
                U,S,Vh = np.linalg.svd(act_hat, full_matrices=False)
                # U=self.pq(torch.from_numpy(U), 0.25)
                # U = U.numpy()
                # criteria (Eq-9)
                sval_hat = (S**2).sum()
                sval_ratio = (S**2)/sval_total               
                accumulated_sval = (sval_total-sval_hat)/sval_total
            
                r = 0
                for ii in range (sval_ratio.shape[0]):
                    if accumulated_sval < threshold:
                        accumulated_sval += sval_ratio[ii]
                        r += 1
                    else:
                        break
                self.task_r.append(self.task_r[-1]+r)
                if r == 0:
                    print ('Skip Updating GPM for layer: {}'.format(i+1)) 
                    continue
                # update GPM
                Ui=np.hstack((self.feature_list[i],U[:,0:r]))  
                if Ui.shape[1] > Ui.shape[0] :
                    self.feature_list[i]=Ui[:,0:Ui.shape[0]]
                else:
                    self.feature_list[i]=Ui
    
        print('-'*40)
        print('Gradient Constraints Summary')
        print('-'*40)
        for i in range(len(self.feature_list)):
            print ('Layer {} : {}/{}'.format(i+1,self.feature_list[i].shape[1], self.feature_list[i].shape[0]))
        print('-'*40)  

    def update_DualGPM (self, mat_list):
        threshold = (self.lame - self.lamb)*self._cur_task/self.total_sessions + self.lamb
        print ('Threshold: ', threshold) 
        if len(self.feature_list) == 0:
            # After First Task 
            for i in range(len(mat_list)):
                activation = mat_list[i]
                U,S,Vh = np.linalg.svd(activation, full_matrices=False)
                # criteria (Eq-5)
                sval_total = (S**2).sum()
                sval_ratio = (S**2)/sval_total
                r = np.sum(np.cumsum(sval_ratio)<threshold) #+1  
                if r < (activation.shape[0]/2):
                    self.feature_list.append(U[:,0:max(r,1)])
                    self.project_type.append('remove')
                else:
                    self.feature_list.append(U[:,0:max(r,1)])
                    self.project_type.append('retain')
        else:
            for i in range(len(mat_list)):
                if self.project_type[i] == 'remove':
                    activation = mat_list[i]
                    U1,S1,Vh1=np.linalg.svd(activation, full_matrices=False)
                    sval_total = (S1**2).sum()
                    # Projected Representation (Eq-8)
                    act_hat = activation - np.dot(np.dot(self.feature_list[i],self.feature_list[i].transpose()),activation)
                    U,S,Vh = np.linalg.svd(act_hat, full_matrices=False)
                    # criteria (Eq-9)
                    sval_hat = (S**2).sum()
                    sval_ratio = (S**2)/sval_total               
                    accumulated_sval = (sval_total-sval_hat)/sval_total
            
                    r = 0
                    for ii in range (sval_ratio.shape[0]):
                        if accumulated_sval < threshold:
                            accumulated_sval += sval_ratio[ii]
                            r += 1
                        else:
                            break
                    if r == 0:
                        print ('Skip Updating DualGPM for layer: {}'.format(i+1)) 
                        continue
                    # update GPM
                    Ui=np.hstack((self.feature_list[i],U[:,0:r]))
                    if Ui.shape[1] > Ui.shape[0] :
                        self.feature_list[i]=Ui[:,0:Ui.shape[0]]
                    else:
                        self.feature_list[i]=Ui
                else:
                    assert self.project_type[i] == 'retain'
                    activation = mat_list[i]
                    U1,S1,Vh1=np.linalg.svd(activation, full_matrices=False)
                    sval_total = (S1**2).sum()
                    # Projected Representation (Eq-8)
                    act_hat = np.dot(np.dot(self.feature_list[i],self.feature_list[i].transpose()),activation)
                    U,S,Vh = np.linalg.svd(act_hat, full_matrices=False)
                    # criteria (Eq-9)
                    sval_hat = (S**2).sum()
                    sval_ratio = (S**2)/sval_total               
                    accumulated_sval = sval_hat/sval_total

                    r = 0
                    for ii in range (sval_ratio.shape[0]):
                        if accumulated_sval >= (1-threshold):
                            accumulated_sval -= sval_ratio[ii]
                            r += 1
                        else:
                            break
                    if r == 0:
                        print ('Skip Updating DualGPM for layer: {}'.format(i+1)) 
                        continue

                    # update GPM by Projected Representation (Eq-8)
                    act_feature = self.feature_list[i] - np.dot(np.dot(U[:,0:r],U[:,0:r].transpose()),self.feature_list[i])
                    Ui, Si, Vi = np.linalg.svd(act_feature)
                    self.feature_list[i]=Ui[:,:self.feature_list[i].shape[1]-r]

        print('-'*40)
        print('Gradient Constraints Summary')
        print('-'*40)
        for i in range(len(self.feature_list)):
            if self.project_type[i]=='remove' and (self.feature_list[i].shape[1] > (self.feature_list[i].shape[0]/2)):
                feature = self.feature_list[i]
                # ipdb.set_trace()
                U, S, V = np.linalg.svd(feature)
                new_feature = U[:,feature.shape[1]:]
                self.feature_list[i] = new_feature
                self.project_type[i] = 'retain'
            elif self.project_type[i]=='retain':
                assert self.feature_list[i].shape[1] <= (self.feature_list[i].shape[0]/2)
            print ('Layer {} : {}/{} type {}'.format(i+1,self.feature_list[i].shape[1], self.feature_list[i].shape[0], self.project_type[i]))
        print('-'*40)


    def _stage2_compact_classifier(self, task_size):
        for p in self._network.classifier_pool[:self._cur_task+1].parameters():
            p.requires_grad=True
            
        run_epochs = 5
        crct_num = self._total_classes    
        param_list = [p for p in self._network.classifier_pool.parameters() if p.requires_grad]
        network_params = [{'params': param_list, 'lr': 0.01,
                           'weight_decay': 0.0005}]
        optimizer = optim.SGD(network_params, lr=0.01, momentum=0.9, weight_decay=0.0005)
        # scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[4], gamma=lrate_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=run_epochs)
        self._network.to(self._device)

        # compensate the semantic drift in the prototypes with ADC
        if self.args["ADC"]:
            epoch = self.args["adv_epoch"]
            print('alpha: ',self.args["alpha"])

            for k, (_, data, label) in enumerate(self.train_loader):
                if k == 0:
                    x_min = data.min()
                    x_max = data.max()
                else:
                    if data.min() < x_min:
                        x_min = data.min()
                    if data.max() > x_max:
                        x_max = data.max()

            xx, yy, feats = [], [], []
            for _, data, label in self.train_loader:
                xx.append(data)
                yy.append(label)

            xx = torch.cat(xx, dim=0)
            yy = torch.cat(yy, dim=0)
            feats = torch.tensor(self._extract_vectors(self.train_loader, old=True)[0]).to(self._device)

            for class_idx in range(0, self._known_classes):
                d = torch.cdist(feats, torch.tensor(self._class_means[class_idx], dtype=torch.float).unsqueeze(0).to(self._device)).squeeze()
                closest = torch.argsort(d)[:self.args["sample_limit"]].cpu()
                x_top = xx[[closest]]
                y_top = yy[[closest]]
                
                idx_dataset = TensorDataset(x_top, y_top)
                loader = DataLoader(idx_dataset, batch_size=int(self.args["sample_limit"]), shuffle=False)

                attack = Attack(self._network, self.args["alpha"], loader, self._class_means[:self._known_classes], self._device, epoch, x_min, x_max, class_idx)
                
                x_, y_ = attack.run()
                if len(x_) > 0:
                    idx_dataset = TensorDataset(x_, y_)
                idx_loader = DataLoader(idx_dataset, batch_size=self.batch_size, shuffle=False)

                if idx_loader is not None:
                    vectors_old = self._extract_vectors_adv(idx_loader, old=True)[0]
                    vectors = self._extract_vectors_adv(idx_loader)[0]

                MU = np.asarray(torch.tensor(self._class_means[class_idx], dtype=torch.float).unsqueeze(0).cpu())
                gap = np.mean(vectors - vectors_old, axis=0)
                MU += gap
                self._class_means[class_idx] = MU[0]

                # self._class_covs[class_idx] = (torch.tensor(np.cov(vectors.T), dtype=torch.float64)+torch.eye(MU.shape[-1])*1e-5)
                    
        # compensate the semantic drift in the prototypes with SDC
        if self.args["SDC"]:
            emb_old = self._extract_vectors(self.train_loader, old=True, bases=self.base_list, types=self.base_type)[0]
            emb = self._extract_vectors(self.train_loader, bases=self.base_list, types=self.base_type)[0]
            MU = np.stack(self._class_means[:self._known_classes])
            gap = self.displacement(emb_old, emb, MU, self.args["sigma"])

            MU += gap
            self._class_means[:self._known_classes] = MU

        if self.args["BDC"]:
            for task_id in range(self._cur_task):
                base_list = []
                for p in range(len(self.feature_list)):
                    Uf=torch.Tensor(np.dot(self.feature_list[p][:,:self.task_r[task_id]],self.feature_list[p][:,:self.task_r[task_id]].transpose()))
                    base_list.append(Uf.to(self.__device))

                emb_old = self._extract_vectors(self.train_loader, old=True, bases=base_list, types=self.base_type)[0]
                emb = self._extract_vectors(self.train_loader, bases=base_list, types=self.base_type)[0]
                MU = np.stack(self._class_means[task_id*self.task_sizes[task_id]:(task_id+1)*self.task_sizes[task_id]])
                gap = self.displacement(emb_old, emb, MU, self.args["sigma"])

                MU += gap
                self._class_means[task_id*self.task_sizes[task_id]:(task_id+1)*self.task_sizes[task_id]] = MU

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)

        self._class_means = np.zeros((self._total_classes, self.feature_dim))
        self._class_covs = torch.zeros((self._total_classes, self.feature_dim, self.feature_dim))

        for class_idx in range(0, self._total_classes):
            idx_dataset = self.data_manager.get_dataset(np.arange(class_idx, class_idx+1), source='train', mode='test')
            idx_loader = DataLoader(idx_dataset, batch_size=64, shuffle=False, num_workers=4)
            vectors, _ = self._extract_vectors(idx_loader)
            class_mean = np.mean(vectors, axis=0) # vectors.mean(0)
            class_cov = torch.tensor(np.cov(vectors.T), dtype=torch.float64)+torch.eye(class_mean.shape[-1])*1e-5
            
            self._class_means[class_idx, :] = class_mean
            self._class_covs[class_idx, ...] = class_cov

        self._network.eval()
        for epoch in range(run_epochs):
            losses = 0.

            sampled_data = []
            sampled_label = []
            num_sampled_pcls = 256
        
            for c_id in range(crct_num):
                t_id = c_id//task_size
                decay = (t_id+1)/(self._cur_task+1)*0.1
                cls_mean = torch.tensor(self._class_means[c_id], dtype=torch.float64).to(self._device)*(0.9+decay)
                cls_cov = self._class_covs[c_id].to(self._device)

                try:
                    m = MultivariateNormal(cls_mean.float(), cls_cov.float())
                except:
                    cls_cov = cls_cov+torch.eye(cls_mean.shape[-1]).to(cls_cov.device)*1e-3

                    if self.dataset == 'ImageNet_A':
                        while True:
                            try:
                                torch.linalg.cholesky(cls_cov)
                                break
                            except:
                                cls_cov = cls_cov+torch.eye(cls_mean.shape[-1]).to(cls_cov.device)*1e-4
                    m = MultivariateNormal(cls_mean.float(), cls_cov.float())

                sampled_data_single = m.sample(sample_shape=(num_sampled_pcls,))
                sampled_data.append(sampled_data_single)                
                sampled_label.extend([c_id]*num_sampled_pcls)

            sampled_data = torch.cat(sampled_data, dim=0).float().to(self._device)
            sampled_label = torch.tensor(sampled_label).long().to(self._device)

            inputs = sampled_data
            targets= sampled_label

            sf_indexes = torch.randperm(inputs.size(0))
            inputs = inputs[sf_indexes]
            targets = targets[sf_indexes]

            
            for _iter in range(crct_num):
                inp = inputs[_iter*num_sampled_pcls:(_iter+1)*num_sampled_pcls]
                tgt = targets[_iter*num_sampled_pcls:(_iter+1)*num_sampled_pcls]
                outputs = self._network(inp, fc_only=True)
                logits = outputs

                if self.logit_norm is not None:
                    per_task_norm = []
                    prev_t_size = 0
                    cur_t_size = 0
                    for _ti in range(self._cur_task+1):
                        cur_t_size += self.task_sizes[_ti]
                        temp_norm = torch.norm(logits[:, prev_t_size:cur_t_size], p=2, dim=-1, keepdim=True) + 1e-7
                        per_task_norm.append(temp_norm)
                        prev_t_size += self.task_sizes[_ti]
                    per_task_norm = torch.cat(per_task_norm, dim=-1)
                    norms = per_task_norm.mean(dim=-1, keepdim=True)
                        
                    norms_all = torch.norm(logits[:, :crct_num], p=2, dim=-1, keepdim=True) + 1e-7
                    decoupled_logits = torch.div(logits[:, :crct_num], norms) / self.logit_norm
                    loss = F.cross_entropy(decoupled_logits, tgt)

                else:
                    loss = F.cross_entropy(logits[:, :crct_num], tgt)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

            scheduler.step()
            test_acc = self._compute_accuracy(self._network, self.test_loader)
            info = 'CA Task {} => Loss {:.3f}, Test_accy {:.3f}'.format(
                self._cur_task, losses/self._total_classes, test_acc)
            logging.info(info)




    def _compute_class_mean(self, data_manager, check_diff=False, oracle=False):
        # import ipdb
        # ipdb.set_trace()
        if hasattr(self, '_class_means') and self._class_means is not None and not check_diff:
            ori_classes = self._class_means.shape[0]
            assert ori_classes==self._known_classes
            new_class_means = np.zeros((self._total_classes, self.feature_dim))
            new_class_means[:self._known_classes] = self._class_means
            self._class_means = new_class_means
            # new_class_cov = np.zeros((self._total_classes, self.feature_dim, self.feature_dim))
            new_class_cov = torch.zeros((self._total_classes, self.feature_dim, self.feature_dim))
            new_class_cov[:self._known_classes] = self._class_covs
            self._class_covs = new_class_cov
        elif not check_diff:
            self._class_means = np.zeros((self._total_classes, self.feature_dim))
            # self._class_covs = np.zeros((self._total_classes, self.feature_dim, self.feature_dim))
            self._class_covs = torch.zeros((self._total_classes, self.feature_dim, self.feature_dim))

            # self._class_covs = []

        if check_diff:
            for class_idx in range(0, self._known_classes):
                data, targets, idx_dataset = data_manager.get_dataset(np.arange(class_idx, class_idx+1), source='train',
                                                                    mode='test', ret_data=True)
                idx_loader = DataLoader(idx_dataset, batch_size=64, shuffle=False, num_workers=4)
                # vectors, _ = self._extract_vectors_aug(idx_loader)
                vectors, _ = self._extract_vectors(idx_loader)
                class_mean = np.mean(vectors, axis=0)
                class_cov = torch.tensor(np.cov(vectors.T), dtype=torch.float64)+torch.eye(class_mean.shape[-1])*1e-4
                # class_cov = torch.cov(torch.tensor(vectors, dtype=torch.float64).T)
#                 centered_vectors = vectors - class_mean
#                 class_cov = torch.matmul(centered_vectors.T, centered_vectors) / (centered_vectors.size(0) - 1)
                if check_diff:
                    log_info = "cls {} sim: {}".format(class_idx, torch.cosine_similarity(torch.tensor(self._class_means[class_idx, :]).unsqueeze(0), torch.tensor(class_mean).unsqueeze(0)).item())
                    logging.info(log_info)
                    np.save('task_{}_cls_{}_mean.npy'.format(self._cur_task, class_idx), class_mean)
                    # print(class_idx, torch.cosine_similarity(torch.tensor(self._class_means[class_idx, :]).unsqueeze(0), torch.tensor(class_mean).unsqueeze(0)))

        if oracle:
            for class_idx in range(0, self._known_classes):
                data, targets, idx_dataset = data_manager.get_dataset(np.arange(class_idx, class_idx+1), source='train',
                                                                    mode='test', ret_data=True)
                idx_loader = DataLoader(idx_dataset, batch_size=64, shuffle=False, num_workers=4)
                vectors, _ = self._extract_vectors(idx_loader)

                # vectors = np.concatenate([vectors_aug, vectors])

                class_mean = np.mean(vectors, axis=0)
                class_cov = torch.tensor(np.cov(vectors.T), dtype=torch.float64)+torch.eye(class_mean.shape[-1])*1e-5
                # class_cov = torch.cov(torch.tensor(vectors, dtype=torch.float64).T)+torch.eye(class_mean.shape[-1])*1e-5
#                 centered_vectors = vectors - class_mean
#                 class_cov = torch.matmul(centered_vectors.T, centered_vectors) / (centered_vectors.size(0) - 1)+torch.eye(class_mean.shape[-1])*1e-5
                self._class_means[class_idx, :] = class_mean
                self._class_covs[class_idx, ...] = class_cov            

        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, idx_dataset = data_manager.get_dataset(np.arange(class_idx, class_idx+1), source='train',
                                                                  mode='test', ret_data=True)
            idx_loader = DataLoader(idx_dataset, batch_size=64, shuffle=False, num_workers=4)
            vectors, _ = self._extract_vectors(idx_loader)

            # vectors = np.concatenate([vectors_aug, vectors])

            class_mean = np.mean(vectors, axis=0)
            class_cov = torch.tensor(np.cov(vectors.T), dtype=torch.float64)+torch.eye(class_mean.shape[-1])*1e-4
#             centered_vectors = vectors - class_mean
#             class_cov = torch.matmul(centered_vectors.T, centered_vectors) / (centered_vectors.size(0) - 1)+torch.eye(class_mean.shape[-1])*1e-4

            if check_diff:
                log_info = "cls {} sim: {}".format(class_idx, torch.cosine_similarity(torch.tensor(self._class_means[class_idx, :]).unsqueeze(0), torch.tensor(class_mean).unsqueeze(0)).item())
                logging.info(log_info)
                np.save('task_{}_cls_{}_mean.npy'.format(self._cur_task, class_idx), class_mean)
                np.save('task_{}_cls_{}_mean_beforetrain.npy'.format(self._cur_task, class_idx), self._class_means[class_idx, :])
                # print(class_idx, torch.cosine_similarity(torch.tensor(self._class_means[class_idx, :]).unsqueeze(0), torch.tensor(class_mean).unsqueeze(0)))
            self._class_means[class_idx, :] = class_mean
            self._class_covs[class_idx, ...] = class_cov
            # self._class_covs.append(class_cov)

    def displacement(self, Y1, Y2, embedding_old, sigma):
        DY = (Y2 - Y1)
        distance = np.sum((np.tile(Y1[None, :, :], [embedding_old.shape[0], 1, 1])-np.tile(embedding_old[:, None, :], [1, Y1.shape[0], 1]))**2, axis=2)
        W = np.exp(-distance/(2*sigma ** 2)) +1e-5
        #print(W) # 1e-5
        W_norm = W/np.tile(np.sum(W, axis=1)[:, None], [1, W.shape[1]])
        displacement = np.sum(np.tile(W_norm[:, :, None], [1, 1, DY.shape[1]])*np.tile(DY[None, :, :], [W.shape[0], 1, 1]), axis=1)
        return displacement

    def _extract_vectors(self, loader, old=False, bases=None, types=None):
        self._network.to(self.__device)
        self._network.eval()
        vectors, targets = [], []
        for _, _inputs, _targets in loader:
            _targets = _targets.numpy()
            if isinstance(self._network, nn.DataParallel):
                if old:
                    _vectors = tensor2numpy(self._network.module.extract_vector_old(_inputs.to(self.__device), bases=bases, types=types))
                else:
                    _vectors = tensor2numpy(self._network.module.extract_vector(_inputs.to(self.__device), bases=bases, types=types))
            else:
                if old:
                    _vectors = tensor2numpy(self._network.extract_vector_old(_inputs.to(self.__device), bases=bases, types=types))
                else:
                    _vectors = tensor2numpy(self._network.extract_vector(_inputs.to(self.__device), bases=bases, types=types))

            vectors.append(_vectors)
            targets.append(_targets)
        self._network.to(self._device)

        return np.concatenate(vectors), np.concatenate(targets)


    def _get_real_mean_cov(self):
        real_protos = []
        real_covs = []
        for class_idx in range(0, self._total_classes):
            idx_dataset = self.data_manager.get_dataset(np.arange(class_idx, class_idx+1), source='train', mode='test')
            idx_loader = DataLoader(idx_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)
            vectors, _ = self._extract_vectors(idx_loader)
            class_mean = np.mean(vectors, axis=0) # vectors.mean(0)
            real_protos.append(torch.tensor(class_mean))
            class_cov = torch.tensor(np.cov(vectors.T), dtype=torch.float64)+torch.eye(class_mean.shape[-1])*1e-5
            real_covs.append(class_cov)

        # 计算均值误差
        mean_errors = []
        for rp, ep in zip(real_protos, self._class_means):
            error = torch.norm(rp - torch.tensor(ep), p=2)  # 欧氏距离
            mean_errors.append(error.item())
        # mean_error = torch.mean(torch.stack(mean_errors)).item()
        
        # 计算协方差误差
        cov_errors = []
        for rc, ec in zip(real_covs, self._class_covs):
            error = torch.norm(rc - ec, p='fro')  # 弗罗贝尼乌斯范数
            cov_errors.append(error.item())
        # cov_error = torch.mean(torch.stack(cov_errors)).item()

        return mean_errors, cov_errors

