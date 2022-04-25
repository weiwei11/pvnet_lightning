import torch
import lib.csrc.ransac_voting.ransac_voting as ransac_voting
import numpy as np
import cv2
from matplotlib import pyplot as plt

from src.utils.leaf import visualize


def ransac_voting_layer(mask, vertex, round_hyp_num, inlier_thresh=0.999, confidence=0.99, max_iter=20,
                        min_num=5, max_num=30000):
    '''
    :param mask:      [b,h,w]
    :param vertex:    [b,h,w,vn,2]
    :param round_hyp_num:
    :param inlier_thresh:
    :return: [b,vn,2]
    '''
    b, h, w, vn, _ = vertex.shape
    batch_win_pts = []
    for bi in range(b):
        hyp_num = 0
        cur_mask = (mask[bi]).byte()
        foreground_num = torch.sum(cur_mask)

        # if too few points, just skip it
        if foreground_num < min_num:
            win_pts = torch.zeros([1, vn, 2], dtype=torch.float32, device=mask.device)
            batch_win_pts.append(win_pts)  # [1,vn,2]
            continue

        # if too many inliers, we randomly down sample it
        if foreground_num > max_num:
            selection = torch.zeros(cur_mask.shape, dtype=torch.float32, device=mask.device).uniform_(0, 1)
            selected_mask = (selection < (max_num / foreground_num.float()))
            cur_mask *= selected_mask

        coords = torch.nonzero(cur_mask, as_tuple=False).float()  # [tn,2]
        coords = coords[:, [1, 0]]
        direct = vertex[bi].masked_select(torch.unsqueeze(torch.unsqueeze(cur_mask, 2), 3))  # [tn,vn,2]
        direct = direct.view([coords.shape[0], vn, 2])
        tn = coords.shape[0]
        idxs = torch.zeros([round_hyp_num, vn, 2], dtype=torch.int32, device=mask.device).random_(0, direct.shape[0])
        all_win_ratio = torch.zeros([vn], dtype=torch.float32, device=mask.device)
        all_win_pts = torch.zeros([vn, 2], dtype=torch.float32, device=mask.device)

        cur_iter = 0
        while True:
            # generate hypothesis
            cur_hyp_pts = ransac_voting.generate_hypothesis(direct, coords, idxs)  # [hn,vn,2]

            # voting for hypothesis
            cur_inlier = torch.zeros([round_hyp_num, vn, tn], dtype=torch.uint8, device=mask.device)
            ransac_voting.voting_for_hypothesis(direct, coords, cur_hyp_pts, cur_inlier, inlier_thresh)  # [hn,vn,tn]

            # find max
            cur_inlier_counts = torch.sum(cur_inlier, 2)  # [hn,vn]
            cur_win_counts, cur_win_idx = torch.max(cur_inlier_counts, 0)  # [vn]
            cur_win_pts = cur_hyp_pts[cur_win_idx, torch.arange(vn)]
            cur_win_ratio = cur_win_counts.float() / tn

            # update best point
            larger_mask = all_win_ratio < cur_win_ratio
            all_win_pts[larger_mask, :] = cur_win_pts[larger_mask, :]
            all_win_ratio[larger_mask] = cur_win_ratio[larger_mask]

            # check confidence
            hyp_num += round_hyp_num
            cur_iter += 1
            cur_min_ratio = torch.min(all_win_ratio)
            if (1 - (1 - cur_min_ratio ** 2) ** hyp_num) > confidence or cur_iter > max_iter:
                break

        # compute mean intersection again
        normal = torch.zeros_like(direct)  # [tn,vn,2]
        normal[:, :, 0] = direct[:, :, 1]
        normal[:, :, 1] = -direct[:, :, 0]
        all_inlier = torch.zeros([1, vn, tn], dtype=torch.uint8, device=mask.device)
        all_win_pts = torch.unsqueeze(all_win_pts, 0)  # [1,vn,2]
        ransac_voting.voting_for_hypothesis(direct, coords, all_win_pts, all_inlier, inlier_thresh)  # [1,vn,tn]

        # coords [tn,2] normal [vn,tn,2]
        all_inlier = torch.squeeze(all_inlier.float(), 0)  # [vn,tn]
        normal = normal.permute(1, 0, 2)  # [vn,tn,2]
        normal = normal * torch.unsqueeze(all_inlier, 2)  # [vn,tn,2] outlier is all zero

        b = torch.sum(normal * torch.unsqueeze(coords, 0), 2)  # [vn,tn]
        ATA = torch.matmul(normal.permute(0, 2, 1), normal)  # [vn,2,2]
        ATb = torch.sum(normal * torch.unsqueeze(b, 2), 1)  # [vn,2]
        try:
            all_win_pts = torch.matmul(torch.inverse(ATA), torch.unsqueeze(ATb, 2))  # [vn,2,1]
            batch_win_pts.append(all_win_pts[None, :, :, 0])
        except:
            all_win_pts = torch.zeros([1, ATA.size(0), 2]).to(ATA.device)
            batch_win_pts.append(all_win_pts)

    batch_win_pts = torch.cat(batch_win_pts)

    return batch_win_pts


def b_inv(b_mat):
    '''
    code from
    https://stackoverflow.com/questions/46595157/how-to-apply-the-torch-inverse-function-of-pytorch-to-every-sample-in-the-batc
    :param b_mat:
    :return:
    '''
    eye = b_mat.new_ones(b_mat.size(-1)).diag().expand_as(b_mat)
    try:
        b_inv, _ = torch.solve(eye, b_mat)
    except:
        print('b_inv')
        b_inv = eye
    return b_inv


# @show_runtime
def ransac_voting_layer_v3(mask, vertex, round_hyp_num, inlier_thresh=0.999, confidence=0.99, max_iter=20,
                           min_num=5, max_num=30000):
    '''
    :param mask:      [b,h,w]
    :param vertex:    [b,h,w,vn,2]
    :param round_hyp_num:
    :param inlier_thresh:
    :return: [b,vn,2]
    '''
    b, h, w, vn, _ = vertex.shape
    batch_win_pts = []
    # hyp_kpts_list = []  # hypothesis keypoints record
    # hyp_kpts_vote_count = []

    for bi in range(b):
        hyp_num = 0
        cur_mask = (mask[bi]).byte()
        foreground_num = torch.sum(cur_mask)

        # if too few points, just skip it
        if foreground_num < min_num:
            # print('too few pixes for ransac voting!')
            win_pts = torch.zeros([1, vn, 2], dtype=torch.float32, device=mask.device)
            batch_win_pts.append(win_pts)  # [1,vn,2]
            continue

        # if too many inliers, we randomly down sample it
        if foreground_num > max_num:
            selection = torch.zeros(cur_mask.shape, dtype=torch.float32, device=mask.device).uniform_(0, 1)
            selected_mask = (selection < (max_num / foreground_num.float())).byte()
            cur_mask *= selected_mask

        coords = torch.nonzero(cur_mask, as_tuple=False).float()  # [tn,2]
        coords = coords[:, [1, 0]]
        direct = vertex[bi].masked_select(torch.unsqueeze(torch.unsqueeze(cur_mask.bool(), 2), 3))  # [tn,vn,2]
        direct = direct.view([coords.shape[0], vn, 2])
        tn = coords.shape[0]
        idxs = torch.zeros([round_hyp_num, vn, 2], dtype=torch.int32, device=mask.device).random_(0, direct.shape[0])
        # idxs = torch.zeros([round_hyp_num, 1, 2], dtype=torch.int32, device=mask.device).random_(0, direct.shape[0])
        # idxs = idxs.repeat([1, vn, 1])
        all_win_ratio = torch.zeros([vn], dtype=torch.float32, device=mask.device)
        all_win_pts = torch.zeros([vn, 2], dtype=torch.float32, device=mask.device)

        cur_iter = 0
        while True:
            # generate hypothesis
            cur_hyp_pts = ransac_voting.generate_hypothesis(direct, coords, idxs)  # [hn,vn,2]

            # voting for hypothesis
            cur_inlier = torch.zeros([round_hyp_num, vn, tn], dtype=torch.uint8, device=mask.device)
            ransac_voting.voting_for_hypothesis(direct, coords, cur_hyp_pts, cur_inlier, inlier_thresh)  # [hn,vn,tn]

            # find max
            cur_inlier_counts = torch.sum(cur_inlier, 2)  # [hn,vn]

            # v, v_idx = cur_inlier_counts.sort(dim=0, descending=True)
            # print(v[:20])
            # [print(cur_hyp_pts[v_idx[:10, vii], vii]) for vii in range(9)]
            # fig = visualize_utils.create_figure()
            # ax = fig.add_subplot(111)
            # [visualize_utils.draw_keypoints(ax, cur_hyp_pts[v_idx[:5, vii], vii].cpu().numpy(), '.y') for vii in range(9)]
            # [visualize_utils.draw_text(ax, cur_hyp_pts[v_idx[:5, vii], vii].cpu().numpy(), [str(vii)] * 5, 'y') for vii in range(9)]

            cur_win_counts, cur_win_idx = torch.max(cur_inlier_counts, 0)  # [vn]
            cur_win_pts = cur_hyp_pts[cur_win_idx, torch.arange(vn)]
            cur_win_ratio = cur_win_counts.float() / tn

            # record all hypothesis keypoints and vote count
            # hyp_kpts_list.append(cur_hyp_pts.detach().cpu().numpy())  # [hn, vn, 2]
            # hyp_kpts_vote_count.append(cur_inlier_counts.detach().cpu().numpy())  # [hn, vn]

            # image = np.zeros((128, 128), dtype=np.uint8)
            # kpt_score = np.zeros((vn, ), dtype=np.int)
            # all_score = cur_inlier_counts.detach().cpu().numpy()
            # all_pts = cur_hyp_pts.detach().cpu().numpy()
            # for pt_idx in range(vn):
            #     for hyp_idx in range(all_pts.shape[0]):
            #         pt = all_pts[hyp_idx, pt_idx, :]
            #         score = all_score[hyp_idx, pt_idx]
            #         if pt[0] < 0 or pt[0] >= 128 or pt[1] < 0 or pt[1] >= 128:
            #             continue
            #
            #         image[int(pt[1]), int(pt[0])] = image[int(pt[1]), int(pt[0])] + int(score)
            #         kpt_score[pt_idx] = kpt_score[pt_idx] + int(score)
            #     # print(kpt_score[pt_idx])
            #     plt.imshow(image)

            # update best point
            larger_mask = all_win_ratio < cur_win_ratio
            all_win_pts[larger_mask, :] = cur_win_pts[larger_mask, :]
            all_win_ratio[larger_mask] = cur_win_ratio[larger_mask]

            # check confidence
            hyp_num += round_hyp_num
            cur_iter += 1
            cur_min_ratio = torch.min(all_win_ratio)
            if (1 - (1 - cur_min_ratio ** 2) ** hyp_num) > confidence or cur_iter > max_iter:
                break

        # compute mean intersection again
        # see https://github.com/zju3dv/pvnet/issues/54
        normal = torch.zeros_like(direct)  # [tn,vn,2]
        normal[:, :, 0] = direct[:, :, 1]
        normal[:, :, 1] = -direct[:, :, 0]
        all_inlier = torch.zeros([1, vn, tn], dtype=torch.uint8, device=mask.device)
        all_win_pts = torch.unsqueeze(all_win_pts, 0)  # [1,vn,2]
        ransac_voting.voting_for_hypothesis(direct, coords, all_win_pts, all_inlier, inlier_thresh)  # [1,vn,tn]

        # coords [tn,2] normal [vn,tn,2]
        all_inlier = torch.squeeze(all_inlier.float(), 0)  # [vn,tn]
        normal = normal.permute(1, 0, 2)  # [vn,tn,2]
        normal = normal * torch.unsqueeze(all_inlier, 2)  # [vn,tn,2] outlier is all zero

        b = torch.sum(normal * torch.unsqueeze(coords, 0), 2)  # [vn,tn]
        ATA = torch.matmul(normal.permute(0, 2, 1), normal)  # [vn,2,2]
        ATb = torch.sum(normal * torch.unsqueeze(b, 2), 1)  # [vn,2]
        # try:
        all_win_pts = torch.matmul(b_inv(ATA), torch.unsqueeze(ATb, 2))  # [vn,2,1]
        # except:
        #    __import__('ipdb').set_trace()
        batch_win_pts.append(all_win_pts[None, :, :, 0])

    batch_win_pts = torch.cat(batch_win_pts)
    # return batch_win_pts, hyp_kpts_list, hyp_kpts_vote_count
    return batch_win_pts


def ransac_voting_layer_with_weight(mask, vertex, weight, round_hyp_num, inlier_thresh=0.999, confidence=0.99, max_iter=20,
                           min_num=5, max_num=30000):
    '''
    :param mask:      [b,h,w]
    :param vertex:    [b,h,w,vn,2]
    :param round_hyp_num:
    :param inlier_thresh:
    :return: [b,vn,2]
    '''
    b, h, w, vn, _ = vertex.shape
    batch_win_pts = []
    # hyp_kpts_list = []  # hypothesis keypoints record
    # hyp_kpts_vote_count = []

    for bi in range(b):
        hyp_num = 0
        cur_mask = (mask[bi]).byte()
        foreground_num = torch.sum(cur_mask)

        # if too few points, just skip it
        if foreground_num < min_num:
            # print('too few pixes for ransac voting!')
            win_pts = torch.zeros([1, vn, 2], dtype=torch.float32, device=mask.device)
            batch_win_pts.append(win_pts)  # [1,vn,2]
            continue

        # if too many inliers, we randomly down sample it
        if foreground_num > max_num:
            selection = torch.zeros(cur_mask.shape, dtype=torch.float32, device=mask.device).uniform_(0, 1)
            selected_mask = (selection < (max_num / foreground_num.float())).byte()
            cur_mask *= selected_mask

        coords = torch.nonzero(cur_mask, as_tuple=False).float()  # [tn,2]
        coords = coords[:, [1, 0]]
        direct = vertex[bi].masked_select(torch.unsqueeze(torch.unsqueeze(cur_mask.bool(), 2), 3))  # [tn,vn,2]
        direct = direct.view([coords.shape[0], vn, 2])
        w = weight[bi].masked_select(cur_mask.bool())  # [tn]
        tn = coords.shape[0]
        idxs = torch.zeros([round_hyp_num, vn, 2], dtype=torch.int32, device=mask.device).random_(0, direct.shape[0])
        # idxs = torch.zeros([round_hyp_num, 1, 2], dtype=torch.int32, device=mask.device).random_(0, direct.shape[0])
        # idxs = idxs.repeat([1, vn, 1])
        all_win_ratio = torch.zeros([vn], dtype=torch.float32, device=mask.device)
        all_win_pts = torch.zeros([vn, 2], dtype=torch.float32, device=mask.device)

        cur_iter = 0
        while True:
            # generate hypothesis
            cur_hyp_pts = ransac_voting.generate_hypothesis(direct, coords, idxs)  # [hn,vn,2]

            # voting for hypothesis
            cur_inlier = torch.zeros([round_hyp_num, vn, tn], dtype=torch.uint8, device=mask.device)
            ransac_voting.voting_for_hypothesis(direct, coords, cur_hyp_pts, cur_inlier, inlier_thresh)  # [hn,vn,tn]

            # find max
            cur_inlier_counts = torch.sum(cur_inlier * w[None, None, :], 2)  # [hn,vn]

            cur_win_counts, cur_win_idx = torch.max(cur_inlier_counts, 0)  # [vn]
            cur_win_pts = cur_hyp_pts[cur_win_idx, torch.arange(vn)]
            # cur_win_ratio = cur_win_counts.float() / tn
            cur_win_ratio = cur_win_counts.float() / w.sum()

            # update best point
            larger_mask = all_win_ratio < cur_win_ratio
            all_win_pts[larger_mask, :] = cur_win_pts[larger_mask, :]
            all_win_ratio[larger_mask] = cur_win_ratio[larger_mask]

            # check confidence
            hyp_num += round_hyp_num
            cur_iter += 1
            cur_min_ratio = torch.min(all_win_ratio)
            if (1 - (1 - cur_min_ratio ** 2) ** hyp_num) > confidence or cur_iter > max_iter:
                break

        # compute mean intersection again
        # see https://github.com/zju3dv/pvnet/issues/54
        normal = torch.zeros_like(direct)  # [tn,vn,2]
        normal[:, :, 0] = direct[:, :, 1]
        normal[:, :, 1] = -direct[:, :, 0]
        all_inlier = torch.zeros([1, vn, tn], dtype=torch.uint8, device=mask.device)
        all_win_pts = torch.unsqueeze(all_win_pts, 0)  # [1,vn,2]
        ransac_voting.voting_for_hypothesis(direct, coords, all_win_pts, all_inlier, inlier_thresh)  # [1,vn,tn]

        # coords [tn,2] normal [vn,tn,2]
        all_inlier = torch.squeeze(all_inlier.float(), 0)  # [vn,tn]
        all_w = all_inlier * w[None, :].float()  # [vn, tn]
        normal = normal.permute(1, 0, 2)  # [vn,tn,2]
        normal = normal * torch.unsqueeze(all_inlier, 2)  # [vn,tn,2] outlier is all zero

        b = torch.sum(normal * torch.unsqueeze(coords, 0), 2) * all_w  # [vn,tn]
        ATA = torch.matmul(normal.permute(0, 2, 1), normal * all_w[:, :, None])  # [vn,2,2]
        ATb = torch.sum(normal * torch.unsqueeze(b, 2), 1)  # [vn,2]
        # try:
        all_win_pts = torch.matmul(b_inv(ATA), torch.unsqueeze(ATb, 2))  # [vn,2,1]
        # except:
        #    __import__('ipdb').set_trace()
        batch_win_pts.append(all_win_pts[None, :, :, 0])

    batch_win_pts = torch.cat(batch_win_pts)
    return batch_win_pts


#@show_runtime
def new_voting_layer(mask, vertex, round_hyp_num, inlier_thresh=0.999, min_num=5, max_num=30000):
    '''
    :param mask:      [b,h,w]
    :param vertex:    [b,h,w,vn,2]
    :param round_hyp_num:
    :param inlier_thresh:
    :return: [b,vn,2]
    '''
    b, h, w, vn, _ = vertex.shape
    hyp_kpts_list = []  # hypothesis keypoints record
    hyp_kpts_vote_count = []

    for bi in range(b):
        cur_mask = (mask[bi]).byte()
        foreground_num = torch.sum(cur_mask)
        # print(foreground_num)
        # if too few points, just skip it
        if foreground_num < min_num:
            hyp_kpts = torch.zeros([1, vn, 2], dtype=torch.float32, device=mask.device)
            hyp_kpts_list.append(hyp_kpts)  # [1,vn,2]
            hyp_kpts_vote = torch.ones([1, vn], dtype=torch.float32, device=mask.device)
            hyp_kpts_vote_count.append(hyp_kpts_vote)
            continue

        # if too many inliers, we randomly down sample it
        if foreground_num > max_num:
            selection = torch.zeros(cur_mask.shape, dtype=torch.float32, device=mask.device).uniform_(0, 1)
            selected_mask = (selection < (max_num / foreground_num.float())).byte()
            cur_mask *= selected_mask

        coords = torch.nonzero(cur_mask, as_tuple=False).float()  # [tn,2]
        coords = coords[:, [1, 0]]
        direct = vertex[bi].masked_select(torch.unsqueeze(torch.unsqueeze(cur_mask.bool(), 2), 3))  # [tn,vn,2]
        direct = direct.view([coords.shape[0], vn, 2])
        tn = coords.shape[0]
        idxs = torch.zeros([round_hyp_num, vn, 2], dtype=torch.int32, device=mask.device).random_(0, direct.shape[0])

        # generate hypothesis
        cur_hyp_pts = ransac_voting.generate_hypothesis(direct, coords, idxs)  # [hn,vn,2]

        # voting for hypothesis
        cur_inlier = torch.zeros([round_hyp_num, vn, tn], dtype=torch.uint8, device=mask.device)
        ransac_voting.voting_for_hypothesis(direct, coords, cur_hyp_pts, cur_inlier, inlier_thresh)  # [hn,vn,tn]

        cur_inlier_counts = torch.sum(cur_inlier, 2)  # [hn,vn]

        # record all hypothesis keypoints and vote count
        hyp_kpts_list.append(cur_hyp_pts)  # [hn, vn, 2]
        hyp_kpts_vote_count.append(cur_inlier_counts)  # [hn, vn]

        # start_time = time.time()
        # image = np.zeros((480, 640), dtype=np.uint8)
        # kpt_score = np.zeros((vn, ), dtype=np.int)
        # all_score = cur_inlier_counts.detach().cpu().numpy()
        # all_pts = cur_hyp_pts.detach().cpu().numpy()
        # for pt_idx in range(vn):
        #     for hyp_idx in range(all_pts.shape[0]):
        #         pt = all_pts[hyp_idx, pt_idx, :]
        #         score = all_score[hyp_idx, pt_idx]
        #         if pt[0] < 0 or pt[0] >= 640 or pt[1] < 0 or pt[1] >= 480:
        #             continue
        #
        #         image[int(pt[1]), int(pt[0])] = image[int(pt[1]), int(pt[0])] + int(score)
        #         kpt_score[pt_idx] = kpt_score[pt_idx] + int(score)
        #     # print(kpt_score[pt_idx])
        #     # plt.imshow(image)
        #
        # # show
        # y, x = np.ogrid[0:480:480j, 0:640:640j]
        # newfunc = interpolate.interp2d(x, y, image, kind='cubic')
        # fnew = newfunc(np.linspace(0, 640, 160), np.linspace(0, 480, 120))
        # extent = [np.min(x), np.max(x), np.max(y), np.min(y)]
        # plt.imshow(fnew, extent=extent)
        # plt.colorbar()
        # print('show image time:', time.time()-start_time)

    return hyp_kpts_list, hyp_kpts_vote_count


#@show_runtime
def build_2d_3d_kpts_corresponding(hyp_kpts, vote_counts, kpts_3d, weight_ratio):
    # init
    hyp_num = hyp_kpts.shape[0]
    # hyp_kpts_tensor = torch.from_numpy(hyp_kpts).reshape((-1, hyp_kpts.shape[2])).cuda().float()
    # kpts_3d_tensor = torch.from_numpy(kpts_3d).unsqueeze(0).repeat((hyp_num, 1, 1)).reshape((-1, kpts_3d.shape[1])).cuda().float()
    hyp_kpts_tensor = torch.from_numpy(hyp_kpts).reshape((-1, hyp_kpts.shape[2])).cuda().float()
    kpts_3d_tensor = torch.from_numpy(kpts_3d).unsqueeze(0).repeat((hyp_num, 1, 1)).reshape((-1, kpts_3d.shape[1])).cuda().float()

    # generate 2D keypoints and 3D keypoints pairs by votes weighs
    votes_weight_list = np.floor(vote_counts.astype(np.float) * weight_ratio / np.sum(vote_counts, 0)).astype(np.int)
    element_num = torch.from_numpy(votes_weight_list).reshape((-1)).int().cuda()
    start_pos = element_num.cumsum(0).int() - element_num

    hyp_kpts_2d_tensor = ransac_voting.data_copy(hyp_kpts_tensor, start_pos, element_num)
    correspond_kpts_3d_tensor = ransac_voting.data_copy(kpts_3d_tensor, start_pos, element_num)

    # total_hyp_kpts_num = votes_weight_list.sum()
    # hyp_kpts_2d = np.zeros((total_hyp_kpts_num, 2))
    # correspond_kpts_3d = np.zeros((total_hyp_kpts_num, 3))
    # hyp_kpts_count = 0
    # for i_hyp in range(hyp_kpts.shape[0]):
    #     for i_kpts in range(hyp_kpts.shape[1]):
    #         cur_hyp_kpts_num = votes_weight_list[i_hyp, i_kpts]
    #         hyp_kpts_2d[hyp_kpts_count:hyp_kpts_count+cur_hyp_kpts_num, :] = np.repeat(hyp_kpts[i_hyp, i_kpts].reshape(1, -1), cur_hyp_kpts_num, axis=0)
    #         correspond_kpts_3d[hyp_kpts_count:hyp_kpts_count+cur_hyp_kpts_num, :] = np.repeat(kpts_3d[i_kpts, :].reshape(1, -1), cur_hyp_kpts_num, axis=0)
    #
    #         hyp_kpts_count += cur_hyp_kpts_num

    # print(np.sum(hyp_kpts_2d_tensor.cpu().numpy() - hyp_kpts_2d))
    # print(np.sum(correspond_kpts_3d_tensor.cpu().numpy() - correspond_kpts_3d))

    return hyp_kpts_2d_tensor.cpu().numpy(), correspond_kpts_3d_tensor.cpu().numpy()


#@show_runtime
def build_2d_3d_kpts_corresponding_v2(hyp_kpts, vote_counts, kpts_3d):
    # generate 2D keypoints and 3D keypoints pairs by votes weighs
    hyp_num = hyp_kpts.shape[0]
    hyp_kpts_2d = np.reshape(hyp_kpts, (-1, 2))
    correspond_kpts_3d = np.reshape(np.repeat(kpts_3d.reshape(1, -1, 3), hyp_num, axis=0), (-1, 3))

    return hyp_kpts_2d, correspond_kpts_3d


def ransac_voting_layer_v4(mask, vertex, round_hyp_num, inlier_thresh=0.999, confidence=0.99, max_iter=20,
                           min_num=5, max_num=30000):
    '''
    :param mask:      [b,h,w]
    :param vertex:    [b,h,w,vn,2]
    :param round_hyp_num:
    :param inlier_thresh:
    :return: [b,vn,2]
    '''
    fs = cv2.FileStorage("/home/ww/code/pvnet_cpp/build/ransac.xml", cv2.FileStorage_READ)

    b, h, w, vn, _ = vertex.shape
    batch_win_pts = []
    for bi in range(b):
        hyp_num = 0
        cur_mask = (mask[bi]).byte()
        foreground_num = torch.sum(cur_mask)

        # test
        print('foreground_num: ', foreground_num.detach().cpu().numpy())

        # if too few points, just skip it
        if foreground_num < min_num:
            win_pts = torch.zeros([1, vn, 2], dtype=torch.float32, device=mask.device)
            batch_win_pts.append(win_pts)  # [1,vn,2]
            continue

        # if too many inliers, we randomly down sample it
        if foreground_num > max_num:
            selection = torch.zeros(cur_mask.shape, dtype=torch.float32, device=mask.device).uniform_(0, 1)
            selected_mask = (selection < (max_num / foreground_num.float())).byte()
            cur_mask *= selected_mask

        # # test
        # cpp_cur_mask = fs.getNode('curmask').mat()
        # cur_mask = torch.from_numpy(np.reshape(cpp_cur_mask, cur_mask.shape)).byte().cuda()

        coords = torch.nonzero(cur_mask, as_tuple=False).float()  # [tn,2]
        coords = coords[:, [1, 0]]

        # # test
        # compute_dist.compute_dist(fs, 'coords', coords)

        direct = vertex[bi].masked_select(torch.unsqueeze(torch.unsqueeze(cur_mask.bool(), 2), 3))  # [tn,vn,2]
        direct = direct.view([coords.shape[0], vn, 2])

        # # test
        # compute_dist.compute_dist(fs, 'direct', direct)

        tn = coords.shape[0]
        idxs = torch.zeros([round_hyp_num, vn, 2], dtype=torch.int32, device=mask.device).random_(0, direct.shape[0])
        all_win_ratio = torch.zeros([vn], dtype=torch.float32, device=mask.device)
        all_win_pts = torch.zeros([vn, 2], dtype=torch.float32, device=mask.device)

        # # test
        # cpp_idxs = fs.getNode('idxs').mat()
        # idxs = torch.from_numpy(np.reshape(cpp_idxs, idxs.shape)).int().cuda()

        cur_iter = 0
        while True:
            # generate hypothesis
            cur_hyp_pts = ransac_voting.generate_hypothesis(direct, coords, idxs)  # [hn,vn,2]

            # voting for hypothesis
            cur_inlier = torch.zeros([round_hyp_num, vn, tn], dtype=torch.uint8, device=mask.device)
            ransac_voting.voting_for_hypothesis(direct, coords, cur_hyp_pts, cur_inlier, inlier_thresh)  # [hn,vn,tn]

            # find max
            cur_inlier_counts = torch.sum(cur_inlier, 2)  # [hn,vn]
            cur_win_counts, cur_win_idx = torch.max(cur_inlier_counts, 0)  # [vn]
            cur_win_pts = cur_hyp_pts[cur_win_idx, torch.arange(vn)]
            cur_win_ratio = cur_win_counts.float() / tn

            # # test
            # compute_dist.compute_dist(fs, 'cur_win_counts', cur_win_counts)
            # compute_dist.compute_dist(fs, 'cur_win_idx', cur_win_idx)
            # compute_dist.compute_dist(fs, 'cur_win_pts', cur_win_pts)
            # compute_dist.compute_dist(fs, 'cur_win_ratio', cur_win_ratio)

            # update best point
            larger_mask = all_win_ratio < cur_win_ratio
            all_win_pts[larger_mask, :] = cur_win_pts[larger_mask, :]
            all_win_ratio[larger_mask] = cur_win_ratio[larger_mask]

            # # test
            # compute_dist.compute_dist(fs, 'all_win_pts', all_win_pts)
            # compute_dist.compute_dist(fs, 'all_win_ratio', all_win_ratio)

            # check confidence
            hyp_num += round_hyp_num
            cur_iter += 1
            cur_min_ratio = torch.min(all_win_ratio)
            if (1 - (1 - cur_min_ratio ** 2) ** hyp_num) > confidence or cur_iter > max_iter:
                break

        # compute mean intersection again
        normal = torch.zeros_like(direct)  # [tn,vn,2]
        normal[:, :, 0] = direct[:, :, 1]
        normal[:, :, 1] = -direct[:, :, 0]
        all_inlier = torch.zeros([1, vn, tn], dtype=torch.uint8, device=mask.device)
        all_win_pts = torch.unsqueeze(all_win_pts, 0)  # [1,vn,2]
        ransac_voting.voting_for_hypothesis(direct, coords, all_win_pts, all_inlier, inlier_thresh)  # [1,vn,tn]

        # # test
        # compute_dist.compute_dist(fs, 'all_win_pts_last', all_win_pts)
        # compute_dist.compute_dist(fs, 'all_inlies', all_inlier)

        # coords [tn,2] normal [vn,tn,2]
        all_inlier = torch.squeeze(all_inlier.float(), 0)  # [vn,tn]
        normal = normal.permute(1, 0, 2)  # [vn,tn,2]
        normal = normal * torch.unsqueeze(all_inlier, 2)  # [vn,tn,2] outlier is all zero

        # # test
        # compute_dist.compute_dist(fs, 'normal', normal)

        b = torch.sum(normal * torch.unsqueeze(coords, 0), 2)  # [vn,tn]
        ATA = torch.matmul(normal.permute(0, 2, 1), normal)  # [vn,2,2]
        ATb = torch.sum(normal * torch.unsqueeze(b, 2), 1)  # [vn,2]

        # # test
        # compute_dist.compute_dist(fs, 'b', b)
        # compute_dist.compute_dist(fs, 'ATA', ATA)
        # compute_dist.compute_dist(fs, 'ATb', ATb)

        # try:
        all_win_pts = torch.matmul(b_inv(ATA), torch.unsqueeze(ATb, 2))  # [vn,2,1]

        # # test
        # compute_dist.compute_dist(fs, 'all_win_pts_finall', all_win_pts)
        # except:
        #    __import__('ipdb').set_trace()
        batch_win_pts.append(all_win_pts[None, :, :, 0])

    batch_win_pts = torch.cat(batch_win_pts)
    return batch_win_pts


def estimate_voting_distribution_with_mean(mask, vertex, mean, round_hyp_num=256, min_hyp_num=4096, topk=128, inlier_thresh=0.99, min_num=5, max_num=30000, output_hyp=False):
    b, h, w, vn, _ = vertex.shape
    all_hyp_pts, all_inlier_ratio = [], []
    for bi in range(b):
        k = 0
        cur_mask = mask[bi] == k + 1
        foreground = torch.sum(cur_mask)

        # if too few points, just skip it
        if foreground < min_num:
            cur_hyp_pts = torch.zeros([1, min_hyp_num, vn, 2], dtype=torch.float32, device=mask.device).float()
            all_hyp_pts.append(cur_hyp_pts)  # [1,vn,2]
            cur_inlier_ratio = torch.ones([1, min_hyp_num, vn], dtype=torch.int64, device=mask.device).float()
            all_inlier_ratio.append(cur_inlier_ratio)
            continue

        # if too many inliers, we randomly down sample it
        if foreground > max_num:
            selection = torch.zeros(cur_mask.shape, dtype=torch.float32, device=mask.device).uniform_(0, 1)
            selected_mask = (selection < (max_num / foreground.float()))
            cur_mask *= selected_mask
            foreground = torch.sum(cur_mask)

        coords = torch.nonzero(cur_mask, as_tuple=False).float()  # [tn,2]
        coords = coords[:, [1, 0]]
        direct = vertex[bi].masked_select(torch.unsqueeze(torch.unsqueeze(cur_mask.bool(), 2), 3))  # [tn,vn,2]
        direct = direct.view([coords.shape[0], vn, 2])
        tn = coords.shape[0]

        round_num = np.ceil(min_hyp_num/round_hyp_num)
        cur_hyp_pts = []
        cur_inlier_ratio = []
        for round_idx in range(int(round_num)):
            idxs = torch.zeros([round_hyp_num, vn, 2], dtype=torch.int32, device=mask.device).random_(0, direct.shape[0])

            # generate hypothesis
            hyp_pts = ransac_voting.generate_hypothesis(direct, coords, idxs)  # [hn,vn,2]

            # voting for hypothesis
            inlier = torch.zeros([round_hyp_num, vn, tn], dtype=torch.uint8, device=mask.device)
            ransac_voting.voting_for_hypothesis(direct, coords, hyp_pts, inlier, inlier_thresh)  # [hn,vn,tn]
            inlier_ratio = torch.sum(inlier, 2)                     # [hn,vn]
            inlier_ratio = inlier_ratio.float()/foreground.float()    # ratio

            cur_hyp_pts.append(hyp_pts)
            cur_inlier_ratio.append(inlier_ratio)

        cur_hyp_pts = torch.cat(cur_hyp_pts, 0)
        cur_inlier_ratio = torch.cat(cur_inlier_ratio, 0)
        all_hyp_pts.append(torch.unsqueeze(cur_hyp_pts, 0))
        all_inlier_ratio.append(torch.unsqueeze(cur_inlier_ratio, 0))

    all_hyp_pts = torch.cat(all_hyp_pts, 0)               # b,hn,vn,2
    all_inlier_ratio = torch.cat(all_inlier_ratio, 0)     # b,hn,vn

    # raw_hyp_pts=all_hyp_pts.permute(0,2,1,3).clone()
    # raw_hyp_ratio=all_inlier_ratio.permute(0,2,1).clone()

    all_hyp_pts = all_hyp_pts.permute(0, 2, 1, 3)            # b,vn,hn,2
    all_inlier_ratio = all_inlier_ratio.permute(0, 2, 1)    # b,vn,hn
    thresh = torch.max(all_inlier_ratio, 2)[0]-0.1         # b,vn
    all_inlier_ratio[all_inlier_ratio < torch.unsqueeze(thresh, 2)] = 0.0


    diff_pts = all_hyp_pts-torch.unsqueeze(mean, 2)                  # b,vn,hn,2
    weighted_diff_pts = diff_pts * torch.unsqueeze(all_inlier_ratio, 3)
    cov = torch.matmul(diff_pts.transpose(2, 3), weighted_diff_pts)  # b,vn,2,2
    cov /= torch.unsqueeze(torch.unsqueeze(torch.sum(all_inlier_ratio, 2), 2), 3)+1e-3 # b,vn,2,2

    # if output_hyp:
    #     return mean,cov,all_hyp_pts,all_inlier_ratio,raw_hyp_pts,raw_hyp_ratio

    return mean, cov


def ransac_radius_voting_layer(mask, vertex, round_hyp_num, inlier_thresh=0.999, confidence=0.99, max_iter=20,
                               min_num=5, max_num=30000):
    """
    :param mask:      [b,h,w]
    :param vertex:    [b,h,w,vn,2]
    :param round_hyp_num:
    :param inlier_thresh:
    :param confidence:
    :param max_iter:
    :param min_num:
    :param max_num:
    :return: [b,vn,2]
    """
    b, h, w, vn, _ = vertex.shape
    batch_win_pts = []

    for bi in range(b):
        hyp_num = 0
        cur_mask = (mask[bi]).byte()
        foreground_num = torch.sum(cur_mask)

        # if too few points, just skip it
        if foreground_num < min_num:
            # print('too few pixes for ransac voting!')
            win_pts = torch.zeros([1, vn, 2], dtype=torch.float32, device=mask.device)
            batch_win_pts.append(win_pts)  # [1,vn,2]
            continue

        all_win_ratio = torch.zeros([vn], dtype=torch.float32, device=mask.device)
        all_win_pts = torch.zeros([vn, 2], dtype=torch.float32, device=mask.device)

        cur_iter = 0
        while True:
            # if too many inliers, we randomly down sample it
            if foreground_num > max_num:
                selection = torch.zeros(cur_mask.shape, dtype=torch.float32, device=mask.device).uniform_(0, 1)
                selected_mask = (selection < (max_num / foreground_num.float())).byte()
                # cur_mask *= selected_mask
                cur_mask = mask[bi].byte() * selected_mask

            coords = torch.nonzero(cur_mask, as_tuple=False).float()  # [tn,2]
            coords = coords[:, [1, 0]]
            kpt_xy = vertex[bi].masked_select(torch.unsqueeze(torch.unsqueeze(cur_mask.bool(), 2), 3))  # [tn,vn,2]
            kpt_xy = kpt_xy.view([coords.shape[0], vn, 2])
            radius = kpt_xy.square().sum(dim=-1).sqrt()
            tn = coords.shape[0]

            idxs = torch.zeros([round_hyp_num, vn, 2], dtype=torch.int32, device=mask.device).random_(0, kpt_xy.shape[0])

            # generate hypothesis
            cur_hyp_pts = ransac_voting.generate_radius_hypothesis(radius, coords, idxs)  # [2*hn,vn,2]
            # import matplotlib.pyplot as plt
            # import matplotlib.patches as patches
            # fig = plt.figure()
            # ax = fig.add_subplot(111)
            # ax.imshow(mask[bi].cpu().numpy())
            # kpt_2d_cpu = cur_hyp_pts[:, 0, :].cpu().numpy()
            # coords_cpu = coords.cpu().numpy()
            # radius_cpu = radius.cpu().numpy()
            # ax.plot(kpt_2d_cpu[:, 0], kpt_2d_cpu[:, 1], '.')
            # for i in range(0, coords_cpu.shape[0], 3):
            #     ax.add_patch(patches.Circle((coords_cpu[i, 0], coords_cpu[i, 1]), radius_cpu[i, 0], fill=False))
            # plt.show()

            # voting for hypothesis
            cur_inlier = torch.zeros([2*round_hyp_num, vn, tn], dtype=torch.uint8, device=mask.device)
            ransac_voting.voting_for_radius_hypothesis(radius, coords, cur_hyp_pts, cur_inlier, inlier_thresh)  # [2*hn,vn,tn]

            # find max
            cur_inlier_counts = torch.sum(cur_inlier, 2)  # [hn,vn]
            cur_win_counts, cur_win_idx = torch.max(cur_inlier_counts, 0)  # [vn]
            cur_win_pts = cur_hyp_pts[cur_win_idx, torch.arange(vn)]
            cur_win_ratio = cur_win_counts.float() / tn

            # update best point
            larger_mask = all_win_ratio < cur_win_ratio
            all_win_pts[larger_mask, :] = cur_win_pts[larger_mask, :]
            all_win_ratio[larger_mask] = cur_win_ratio[larger_mask]
            # import matplotlib.pyplot as plt
            # import matplotlib.patches as patches
            # fig = plt.figure()
            # ax = fig.add_subplot(111)
            # ax.imshow(mask[bi].cpu().numpy())
            # all_win_pts_cpu = all_win_pts.cpu().numpy()
            # ax.plot(all_win_pts_cpu[:, 0], all_win_pts_cpu[:, 1], '.')
            # plt.show()

            # check confidence
            hyp_num += round_hyp_num
            cur_iter += 1
            cur_min_ratio = torch.min(all_win_ratio)
            if (1 - (1 - cur_min_ratio ** 2) ** hyp_num) > confidence or cur_iter > max_iter:
                break

        # compute mean intersection again
        all_inlier = torch.zeros([1, vn, tn], dtype=torch.uint8, device=mask.device)
        all_win_pts = torch.unsqueeze(all_win_pts, 0)  # [1,vn,2]
        ransac_voting.voting_for_radius_hypothesis(radius, coords, all_win_pts, all_inlier, inlier_thresh)  # [1,vn,tn]
        batch_win_pts.append(all_win_pts)

    batch_win_pts = torch.cat(batch_win_pts)
    return batch_win_pts

