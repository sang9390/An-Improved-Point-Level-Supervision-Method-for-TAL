import torch
import torch.nn as nn
import utils

import numpy as np

class Total_loss(nn.Module):
    def __init__(self, lambdas):
        super(Total_loss, self).__init__()
        self.tau = 0.1
        self.sampling_size = 3
        self.lambdas = lambdas
        self.ce_criterion = nn.BCELoss(reduction='none')

    ### start###
    # def forward(self, vid_score, cas_sigmoid_fuse, features, stored_info, label, point_anno, step):   # orginal_train_001
    def forward(self, vid_score, cas_sigmoid_fuse, features, stored_info, label, point_anno, step, bkg_ann): # modification_train_001
    ### end###

        loss = {}

        loss_vid = self.ce_criterion(vid_score, label)
        loss_vid = loss_vid.mean()

        point_anno = torch.cat((point_anno, torch.zeros((point_anno.shape[0], point_anno.shape[1], 1)).cuda()), dim=2)

        weighting_seq_act = point_anno.max(dim=2, keepdim=True)[0]
        num_actions = point_anno.max(dim=2)[0].sum(dim=1)

        focal_weight_act = (1 - cas_sigmoid_fuse) * point_anno + cas_sigmoid_fuse * (1 - point_anno)
        focal_weight_act = focal_weight_act ** 2

        loss_frame = (((focal_weight_act * self.ce_criterion(cas_sigmoid_fuse, point_anno) * weighting_seq_act).sum(
            dim=2)).sum(dim=1) / num_actions).mean()

        _, bkg_seed = utils.select_seed(cas_sigmoid_fuse.detach().cpu(), point_anno.detach().cpu())

        bkg_seed = bkg_seed.unsqueeze(-1).cuda()

        point_anno_bkg = torch.zeros_like(point_anno).cuda()
        point_anno_bkg[:, :, -1] = 1

        weighting_seq_bkg = bkg_seed
        num_bkg = bkg_seed.sum(dim=1)

        focal_weight_bkg = (1 - cas_sigmoid_fuse) * point_anno_bkg + cas_sigmoid_fuse * (1 - point_anno_bkg)
        focal_weight_bkg = focal_weight_bkg ** 2

        ### start###
        # loss_frame_bkg = (((focal_weight_bkg * self.ce_criterion(cas_sigmoid_fuse,point_anno_bkg) * weighting_seq_bkg).sum(dim=2)).sum(dim=1) / num_bkg).mean() # orgianl code
        loss_frame_bkg_000 = (((focal_weight_bkg * self.ce_criterion(cas_sigmoid_fuse, point_anno_bkg) * weighting_seq_bkg).sum(dim=2)).sum(dim=1) / num_bkg).mean() # modification code
        ### end###

        ### start###
        bkg_seed_001 = bkg_ann.cuda()
        if sum(sum(bkg_seed_001)) == 0:
            _, bkg_seed_001 = utils.select_seed(cas_sigmoid_fuse.detach().cpu(), point_anno.detach().cpu())
            bkg_seed_001 = bkg_seed_001.unsqueeze(-1).cuda()
        else:
            bkg_seed_001 = bkg_ann.cuda()
        ### end###

        ### start###
        point_anno_bkg_001 = torch.zeros_like(point_anno).cuda()
        point_anno_bkg_001[:, :, -1] = 1

        weighting_seq_bkg_001 = bkg_seed_001
        num_bkg_001 = bkg_seed_001.sum(dim=1)

        focal_weight_bkg_001 = (1 - cas_sigmoid_fuse) * point_anno_bkg_001 + cas_sigmoid_fuse * (1 - point_anno_bkg_001)
        focal_weight_bkg_001 = focal_weight_bkg_001 ** 2

        loss_frame_bkg_001 = (((focal_weight_bkg_001 * self.ce_criterion(cas_sigmoid_fuse, point_anno_bkg_001) * weighting_seq_bkg_001).sum(dim=2)).sum(dim=1) / num_bkg_001).mean()

        loss_frame_bkg = loss_frame_bkg_001 + loss_frame_bkg_000
        ### end###

        loss_score_act = 0
        loss_score_bkg = 0
        loss_feat = 0

        if len(stored_info['new_dense_anno'].shape) > 1:
            new_dense_anno = stored_info['new_dense_anno'].cuda()
            new_dense_anno = torch.cat((new_dense_anno, torch.zeros((new_dense_anno.shape[0], new_dense_anno.shape[1], 1)).cuda()), dim=2)

            act_idx_diff = new_dense_anno[:,1:] - new_dense_anno[:,:-1]
            loss_score_act = 0
            loss_feat = 0
            for b in range(new_dense_anno.shape[0]):
                gt_classes = torch.nonzero(label[b]).squeeze(1)
                act_count = 0
                loss_score_act_batch = 0
                loss_feat_batch = 0

                for c in gt_classes:
                    range_idx = torch.nonzero(act_idx_diff[b,:,c]).squeeze(1)
                    range_idx = range_idx.cpu().data.numpy().tolist()
                    if type(range_idx) is not list:
                        range_idx = [range_idx]
                    if len(range_idx) == 0:
                        continue
                    if act_idx_diff[b, range_idx[0], c] != 1:
                        range_idx = [-1] + range_idx
                    if act_idx_diff[b, range_idx[-1], c] != -1:
                        range_idx = range_idx + [act_idx_diff.shape[1] - 1]

                    label_lst = []
                    feature_lst = []

                    if range_idx[0] > -1:
                        start_bkg = 0
                        end_bkg = range_idx[0]
                        bkg_len = end_bkg - start_bkg + 1

                        label_lst.append(0)
                        feature_lst.append(utils.feature_sampling(features[b], start_bkg, end_bkg + 1, self.sampling_size))

                    for i in range(len(range_idx) // 2):
                        if range_idx[2*i + 1] - range_idx[2*i] < 1:
                            continue

                        label_lst.append(1)
                        feature_lst.append(utils.feature_sampling(features[b], range_idx[2*i] + 1, range_idx[2*i + 1] + 1, self.sampling_size))

                        if range_idx[2*i + 1] != act_idx_diff.shape[1] - 1:
                            start_bkg = range_idx[2*i + 1] + 1

                            if i == (len(range_idx) // 2 - 1):
                                end_bkg = act_idx_diff.shape[1] - 1
                            else:
                                end_bkg = range_idx[2*i + 2]

                            bkg_len = end_bkg - start_bkg + 1

                            label_lst.append(0)
                            feature_lst.append(utils.feature_sampling(features[b], start_bkg, end_bkg + 1, self.sampling_size))

                        start_act = range_idx[2*i] + 1
                        end_act = range_idx[2*i + 1]

                        complete_score_act = utils.get_oic_score(cas_sigmoid_fuse[b,:,c], start=start_act, end=end_act)

                        loss_score_act_batch += 1 - complete_score_act

                        act_count += 1

                    if sum(label_lst) > 1:
                        feature_lst = torch.stack(feature_lst, 0).clone()
                        feature_lst = feature_lst / torch.norm(feature_lst, dim=1, p=2).unsqueeze(1)
                        label_lst = torch.tensor(label_lst).cuda().float()

                        sim_matrix = torch.matmul(feature_lst, torch.transpose(feature_lst, 0, 1)) / self.tau

                        sim_matrix = torch.exp(sim_matrix)

                        sim_matrix = sim_matrix.clone().fill_diagonal_(0)

                        scores = (sim_matrix * label_lst.unsqueeze(1)).sum(dim=0) / sim_matrix.sum(dim=0)

                        loss_feat_batch = (-label_lst * torch.log(scores)).sum() / label_lst.sum()

                if act_count > 0:
                    loss_score_act += loss_score_act_batch / act_count
                    loss_feat += loss_feat_batch


            bkg_idx_diff = (1 - new_dense_anno[:,1:]) - (1 - new_dense_anno[:,:-1])
            loss_score_bkg = 0
            for b in range(new_dense_anno.shape[0]):
                gt_classes = torch.nonzero(label[b]).squeeze(1)
                loss_score_bkg_batch = 0
                bkg_count = 0

                for c in gt_classes:
                    range_idx = torch.nonzero(bkg_idx_diff[b,:,c]).squeeze(1)
                    range_idx = range_idx.cpu().data.numpy().tolist()
                    if type(range_idx) is not list:
                        range_idx = [range_idx]
                    if len(range_idx) == 0:
                        continue
                    if bkg_idx_diff[b, range_idx[0], c] != 1:
                        range_idx = [-1] + range_idx
                    if bkg_idx_diff[b, range_idx[-1], c] != -1:
                        range_idx = range_idx + [bkg_idx_diff.shape[1] - 1]

                    for i in range(len(range_idx) // 2):
                        if range_idx[2*i + 1] - range_idx[2*i] < 1:
                            continue

                        start_bkg = range_idx[2*i] + 1
                        end_bkg = range_idx[2*i + 1]

                        complete_score_bkg = utils.get_oic_score(1 - cas_sigmoid_fuse[b,:,c], start=start_bkg, end=end_bkg)

                        loss_score_bkg_batch += 1 - complete_score_bkg

                        bkg_count += 1

                if bkg_count > 0:
                    loss_score_bkg += loss_score_bkg_batch / bkg_count

            loss_score_act = loss_score_act / new_dense_anno.shape[0]
            loss_score_bkg = loss_score_bkg / new_dense_anno.shape[0]

            loss_feat = loss_feat / new_dense_anno.shape[0]

        loss_score = (loss_score_act + loss_score_bkg) ** 2

        loss_total = self.lambdas[0] * loss_vid + self.lambdas[1] * loss_frame + self.lambdas[2] * loss_frame_bkg + self.lambdas[3] * loss_score + self.lambdas[4] * loss_feat

        loss["loss_vid"] = loss_vid
        loss["loss_frame"] = loss_frame
        loss["loss_frame_bkg"] = loss_frame_bkg
        loss["loss_score_act"] = loss_score_act
        loss["loss_score_bkg"] = loss_score_bkg
        loss["loss_score"] = loss_score
        loss["loss_feat"] = loss_feat
        loss["loss_total"] = loss_total

        return loss_total, loss


def train(net, config, loader_iter, optimizer, criterion, logger, step):
    net.train()

    total_loss = {}
    total_cost = []

    optimizer.zero_grad()

    for _b in range(config.batch_size):

        ### start###
        # _, _data, _label, _point_anno, stored_info, _, _ = next(loader_iter)    #original_train.py_001
        _, _data, _label, _point_anno, stored_info, _, _, bkg_ann = next(loader_iter)   #modification_train.py_001
        ### end###

        _data = _data.cuda()
        _label = _label.cuda()
        _point_anno = _point_anno.cuda()

        ### start###
        # vid_score, cas_sigmoid_fuse, features = net(_data, _label) # original 001
        vid_score, cas_sigmoid_fuse, features = net(_data, _label, _point_anno, bkg_ann)  # modification 001
        ### end###

        ### start###
        # cost, loss = criterion(vid_score, cas_sigmoid_fuse, features, stored_info, _label, _point_anno, step) # original 002
        cost, loss = criterion(vid_score, cas_sigmoid_fuse, features, stored_info, _label, _point_anno, step, bkg_ann)   # modification 002
        ### end###

        total_cost.append(cost)

        for key in loss.keys():
            if not (key in total_loss):
                total_loss[key] = []

            if loss[key] > 0:
                total_loss[key] += [loss[key].detach().cpu().item()]
            else:
                total_loss[key] += [loss[key]]
    
    total_cost = sum(total_cost) / config.batch_size

    ### start###
    print(total_cost)
    ### end###

    total_cost.backward()
    optimizer.step()

    for key in total_loss.keys():
        logger.log_value("loss/" + key, sum(total_loss[key]) / config.batch_size, step)
