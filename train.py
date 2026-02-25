import sys
import time
from scipy.optimize import linear_sum_assignment
sys.path.append('/')
from scipy import io
import os
import numpy as np
import torch
import argparse
from modules import dataset, network, loss, transform
from utils import yaml_config_hook, save_model, metric, initialization_utils
import torch
from Toolbox import Preprocessing
import csv
import os
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from thop import profile
# torch.autograd.set_detect_anomaly(True)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
def extract_features_for_tsne(test_loader, model, device):
    """
    从 test_loader 中抽取特征和标签，用于 t-SNE 可视化
    约定：model.forward_feature(x_modalities) -> [B, D] 的特征
    """
    model.eval()
    feats = []
    labels = []

    for step, (x, y) in enumerate(test_loader):
        # 同 inference 里的多模态 / 双增广兼容逻辑
        if isinstance(x, (list, tuple)) and len(x) > 0 and isinstance(x[0], (list, tuple)):
            x_modalities = [x_i.to(device) for x_i in x[0]]
        else:
            x_modalities = [x_i.to(device) for x_i in x]

        with torch.no_grad():
            # ⭐ 如果你 Net 里方法名不是 forward_feature，在这里改成你自己的：
            feat = model.forward_embedding(x_modalities)  # [B, D]

        feats.append(feat.cpu().numpy())
        labels.append(y.numpy())

        if step % 50 == 0:
            print(f"Step [{step}/{len(test_loader)}]\t Extracting features for t-SNE...")

    feats = np.concatenate(feats, axis=0)
    labels = np.concatenate(labels, axis=0)

    return feats, labels


def plot_tsne(features, labels, save_path, title=None, max_points=5000):
    """
    极简 t-SNE 绘图：
      - 去背景类 0
      - t-SNE 特征缩放到 [-1, 1]
      - 只保留坐标刻度数字，不要坐标轴名称、图例、标题
      - 坐标刻度固定为 [-1, -0.5, 0, 0.5, 1]
    """
    # 1️⃣ 过滤掉背景类 0
    mask = labels != 0
    features = features[mask]
    labels = labels[mask]

    if features.shape[0] == 0:
        print("No non-background samples to plot.")
        return

    # 2️⃣ 随机采样避免太慢
    N = features.shape[0]
    if N > max_points:
        idx = np.random.choice(N, max_points, replace=False)
        features = features[idx]
        labels = labels[idx]

    print(f"Running t-SNE on {features.shape[0]} samples ...")

    # 3️⃣ t-SNE 降维
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        init='pca',
        learning_rate='auto',
        random_state=42
    )
    feat_2d = tsne.fit_transform(features)

    # 4️⃣ 归一化到 [-1, 1]
    for i in range(2):
        max_abs = np.max(np.abs(feat_2d[:, i]))
        if max_abs > 0:
            feat_2d[:, i] /= max_abs

    # 5️⃣ 开始绘图
    fig, ax = plt.subplots(figsize=(6, 6))

    # 画点（不需要图例）
    classes = np.unique(labels)
    for c in classes:
        c_mask = (labels == c)
        ax.scatter(
            feat_2d[c_mask, 0],
            feat_2d[c_mask, 1],
            s=6,
            alpha=0.7
        )

    # 6️⃣ 坐标显示设置
    ticks = [-1, -0.5, 0, 0.5, 1]

    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    # 去掉坐标轴标签（只保留数字）
    ax.set_xlabel("")
    ax.set_ylabel("")

    # 固定坐标范围
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)

    # 7️⃣ 去掉上下左右边框（只保留刻度数字）
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)

    # 去掉网格 & title
    ax.grid(False)
    ax.set_title("")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"t-SNE figure saved to: {save_path}")


def get_hungarian_mapped_labels(test_loader, model, device):
    """
    使用与其他脚本相同的匈牙利对齐方式：
    从 test_loader 中拿 y_true 和 y_pred，
    用匈牙利算法把聚类标签对齐到真实标签，
    返回“对齐后的预测标签” y_pred_mapped（和其他方法一致）。
    """
    model.eval()
    y_pred_vector = []
    labels_vector = []

    for step, (x, y) in enumerate(test_loader):
        # 和 inference 里的取 x_modalities 一样
        if isinstance(x, (list, tuple)) and len(x) > 0 and isinstance(x[0], (list, tuple)):
            x_modalities = [x_i.to(device) for x_i in x[0]]
        else:
            x_modalities = [x_i.to(device) for x_i in x]

        with torch.no_grad():
            pred = model.forward_cluster(x_modalities)

        y_pred_vector.extend(pred.cpu().detach().numpy())
        labels_vector.extend(y.numpy())

    y_pred_vector = np.array(y_pred_vector)
    labels_vector = np.array(labels_vector)

    # 这里按“有标签像素”的情况处理（你的数据就是只保留有标签的样本）
    y_true = labels_vector
    y_pred = y_pred_vector

    true_classes = np.unique(y_true)
    pred_classes = np.unique(y_pred)

    n_true = len(true_classes)
    n_pred = len(pred_classes)

    # 构建混淆矩阵
    confusion_matrix = np.zeros((n_true, n_pred), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        ti = np.where(true_classes == t)[0][0]
        pi = np.where(pred_classes == p)[0][0]
        confusion_matrix[ti, pi] += 1

    # 匈牙利算法找最佳匹配：pred_class -> true_class
    row_ind, col_ind = linear_sum_assignment(-confusion_matrix)
    label_map = {pred_classes[c]: true_classes[r] for r, c in zip(col_ind, row_ind)}

    # 把每个预测标签映射到对应的真实类别编号
    y_pred_mapped = np.array([label_map.get(p, -1) for p in y_pred])

    return y_pred_mapped



def train(model, loss_op, train_loader, optimizer):
    model.train()
    loss_epoch = 0
    for step, ((x_1, x_2), y) in enumerate(train_loader):
        optimizer.zero_grad()
        x_list_1 = [x_i.to(DEVICE) for x_i in x_1]
        x_list_2 = [x_i.to(DEVICE) for x_i in x_2]
        y1, y2 = model(x_list_1, x_list_2)
        loss_, loss_con, loss_clu = loss_op(y1, y2, model.clustering_head.cluster_centers)
        loss_.backward()
        optimizer.step()
        if step % 50 == 0:
            print(f"Step [{step}/{len(train_loader)}]\t loss: "  f"{loss_.item():.6f}\t" f'CL:{loss_con.item():.6f}\t CLU: {loss_clu.item():.6f}')
        loss_epoch += loss_.item()
    return loss_epoch

def inference(test_loader, model, device, is_labeled_pixel):
    model.eval()
    y_pred_vector = []
    labels_vector = []
    for step, (x, y) in enumerate(test_loader):
        x_list = [x_i.to(device) for x_i in x]
        with torch.no_grad():
            pred = model.forward_cluster(x_list)
        y_pred_vector.extend(pred.cpu().detach().numpy())
        labels_vector.extend(y.numpy())
        if step % 50 == 0:
            print(f"Step [{step}/{len(test_loader)}]\t Computing features...")
    y_pred_vector = np.array(y_pred_vector)
    labels_vector = np.array(labels_vector)

    if is_labeled_pixel:
        acc, kappa, nmi, ari, pur, ca = metric.cluster_accuracy(labels_vector, y_pred_vector)
    else:
        indx_labeled = np.nonzero(labels_vector)[0]
        y_true = labels_vector[indx_labeled]
        y_pred = y_pred_vector[indx_labeled]
        true_classes = np.unique(y_true)
        pred_classes = np.unique(y_pred)
        n_true = len(true_classes)
        n_pred = len(pred_classes)
        confusion_matrix = np.zeros((n_true, n_pred), dtype=np.int64)
        for true_label, pred_label in zip(y_true, y_pred):
            true_idx = np.where(true_classes == true_label)[0][0]
            pred_idx = np.where(pred_classes == pred_label)[0][0]
            confusion_matrix[true_idx, pred_idx] += 1
        row_ind, col_ind = linear_sum_assignment(-confusion_matrix)
        label_map = {pred_classes[c]: true_classes[r] for r, c in zip(row_ind, col_ind)}
        y_pred_mapped = np.array([label_map.get(p, -1) for p in y_pred])
        acc, kappa, nmi, ari, pur, ca = metric.cluster_accuracy(y_true, y_pred_mapped)

    print('OA = {:.4f} Kappa = {:.4f} NMI = {:.4f} ARI = {:.4f} Purity = {:.4f}'.format(acc, kappa, nmi, ari, pur))
    # GT = io.loadmat(gt_path)
    # gt = GT['GT']  # Trento
    # gt = GT['gt']   # MUUFL&Augsburg
    # Preprocessing.Processor().show_class_map(y_pred_mapped, indx_labeled, gt)
    return acc, kappa, nmi, ari, pur, ca

# def inference(test_loader, model, device, is_labeled_pixel=True):
#     model.eval()
#     y_pred_vector = []
#     labels_vector = []
#
#     for step, (x, y) in enumerate(test_loader):
#         # 兼容两种情况：
#         # 1) x 是 [mod1, mod2, ...]
#         # 2) x 是 ( [mod1_aug1, mod2_aug1,...], [mod1_aug2, ...] )
#         if isinstance(x, (list, tuple)) and len(x) > 0 and isinstance(x[0], (list, tuple)):
#             # 如果是带两组 augmentation 的形式，推理时只用第一组就行
#             x_modalities = [x_i.to(device) for x_i in x[0]]
#         else:
#             # 普通情况：直接就是每个模态一个 tensor
#             x_modalities = [x_i.to(device) for x_i in x]
#
#         with torch.no_grad():
#             pred = model.forward_cluster(x_modalities)
#
#         y_pred_vector.extend(pred.cpu().detach().numpy())
#         labels_vector.extend(y.numpy())
#
#         if step % 50 == 0:
#             print(f"Step [{step}/{len(test_loader)}]\t Computing features...")
#
#     y_pred_vector = np.array(y_pred_vector)
#     labels_vector = np.array(labels_vector)
#
#     # ==============================
#     # 🔥 你现在的数据：只保留有标签的样本
#     #    所以可以直接做聚类指标
#     # ==============================
#     acc, kappa, nmi, ari, pur, ca = metric.cluster_accuracy(labels_vector, y_pred_vector)
#
#     print('OA = {:.4f} Kappa = {:.4f} NMI = {:.4f} ARI = {:.4f} Purity = {:.4f}'.format(
#         acc, kappa, nmi, ari, pur))
#     return acc, kappa, nmi, ari, pur, ca


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    config = yaml_config_hook("config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args()
    pretrain_path = args.model_path + '/pretrain'
    joint_train_path = args.model_path + '/joint-train'
    if not os.path.exists(pretrain_path):
        os.makedirs(pretrain_path)
    if not os.path.exists(joint_train_path):
        os.makedirs(joint_train_path)
    initialization_utils.set_global_random_seed(seed=args.seed)

    root = args.dataset_root

    # prepare data
    if args.dataset == "Trento":
        im_1, im_2 = 'Trento-HSI', 'Trento-Lidar'
        gt_ = 'gt'
        img_path = (root + im_1 + '.mat', root + im_2 + '.mat')
    elif args.dataset == "Augsburg":
        im_1, im_2 = 'data_HS_LR', 'data_SAR_HR'
        gt_ = 'gt'
        img_path = (root + im_1 + '.mat', root + im_2 + '.mat')
    elif args.dataset == "MUUFL":
        im_1, im_2 = 'HSI', 'LiDAR'
        gt_ = 'gt'
        img_path = (root + im_1 + '.mat', root + im_2 + '.mat')
    else:
        raise NotImplementedError
    gt_path = root + gt_ + '.mat'
    dataset_train = dataset.MultiModalDataset(gt_path, *img_path, patch_size=(args.image_size, args.image_size),
                                              transform=transform.Transforms(size=args.image_size),
                                              is_labeled=False)

    # dataset_train = dataset.MultiModalDataset(
    #     gt_path, *img_path,
    #     patch_size=(args.image_size, args.image_size),
    #     transform=transform.Transforms(size=args.image_size),
    #     is_labeled=False,
    #     noise_type=args.noise_type,
    #     noise_level=args.noise_level
    # )

    class_num = dataset_train.n_classes
    print('Processing %s ' % img_path[0])
    print(dataset_train.data_size, class_num)
    print(args)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
        prefetch_factor=4,
        pin_memory = True if DEVICE.type == "cuda" else False
    )
    # data_loader_train = torch.utils.data.DataLoader(
    #     dataset_train,
    #     batch_size=args.batch_size,
    #     shuffle=True,
    #     drop_last=True,
    #     num_workers=0,  # ← 关键
    #     pin_memory=False,  # Windows 下先关掉
    #     prefetch_factor=None  # num_workers=0 时不要设置
    # )


    # # test loader
    # dataset_test = dataset.MultiModalDatasettest(gt_path, *img_path,
    #                                          patch_size=(args.image_size, args.image_size),
    #                                          transform=None, is_labeled=args.is_labeled_pixel)
    # draw fig
    dataset_test = dataset.MultiModalDataset(gt_path, *img_path,
                                             patch_size=(args.image_size, args.image_size),
                                             transform=None, is_labeled=args.is_labeled_pixel)
    # dataset_test = dataset.MultiModalDataset(
    #     gt_path, *img_path,
    #     patch_size=(args.image_size, args.image_size),
    #     transform=None,
    #     is_labeled=args.is_labeled_pixel,
    #     noise_type=args.noise_type,
    #     noise_level=args.noise_level
    # )

    data_loader_test = torch.utils.data.DataLoader(dataset_test,
                                                batch_size =512,
                                                   shuffle=False,
                                                   drop_last=False, num_workers=args.workers,pin_memory = True if DEVICE.type == "cuda" else False)

    # data_loader_test = torch.utils.data.DataLoader(
    #     dataset_test,
    #     batch_size=512,
    #     shuffle=False,
    #     drop_last=False,
    #     num_workers=0,  # ← 关键
    #     pin_memory=False
    # )


    # ==========================================

    model = network.Net(dataset_train.in_channels, class_num, args.dim_emebeding)

    model = model.to(DEVICE)

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model =Net((63,2), 6, 128).to(device)

    # hsi = torch.randn(256, 63, 9, 9).to(DEVICE)
    # lidar = torch.randn(256, 2, 9, 9).to(DEVICE)
    #
    # flops, params = profile(model, inputs=((hsi, lidar),(hsi, lidar)))
    # print(f"FLOPs: {flops / 1e6:.2f}M")
    # print(f"Parameters: {params}")

    # optimizer / loss
    grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if 'clustering_head' not in n],
         'lr': args.learning_rate},
        {"params": model.clustering_head.cluster_centers, 'lr': args.learning_rate * args.lr_scale}
    ]
    optimizer = torch.optim.Adam(grouped_parameters, lr=args.learning_rate, weight_decay=args.weight_decay)

    # # ===== joint training ==========
    score_list = []
    each_class = []
    max_acc = 0
    best_ca = None
    best_metrics = None
    acc, kappa, nmi, ari, pur, ca = inference(data_loader_test, model, DEVICE, is_labeled_pixel=args.is_labeled_pixel)
    # acc, kappa, nmi, ari, pur, ca = inferenceallanddraw(data_loader_test, model, DEVICE, is_labeled_pixel=args.is_labeled_pixel)
    score_list.append([acc, kappa, nmi, ari, pur])
    print(f'initial accuracy: ACC={acc:.4f}')


    loss_op_joint = loss.JointLoss(args.batch_size,  # class_num,  #
                                   lambda_=args.contrastive_param,
                                   weight_clu=args.weight_clu_loss,
                                   regularization_coef=args.regularizer_coef, device=DEVICE)

    loss_history = []

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    print('start fine-tuning ...')
    start_time = time.time()
    for epoch in range(1, args.joint_train_epoch + 1):
        loss_epoch = train(model, loss_op_joint, data_loader_train, optimizer)
        print(f"Epoch [{epoch}/{args.joint_train_epoch}]\t Loss: {loss_epoch / len(data_loader_train)}")
        if epoch % 1 == 0:
            acc, kappa, nmi, ari, pur, ca = inference(data_loader_test, model, DEVICE, is_labeled_pixel=args.is_labeled_pixel)
            # acc, kappa, nmi, ari, pur, ca = inferenceallanddraw(data_loader_test, model, DEVICE,is_labeled_pixel=args.is_labeled_pixel)
            score_list.append([acc, kappa, nmi, ari, pur])
            each_class.append([ca])
            if acc > max_acc:
                max_acc = acc
                best_ca = ca
                best_metrics = [acc, kappa, nmi, ari, pur]
                print('Better acc')
            # save_model(joint_train_path, model, optimizer, epoch)
        loss_history.append(loss_epoch / len(data_loader_train))
        lr_scheduler.step()
    running_time = time.time() - start_time
    print(f'fine tuning time: {running_time:.3f} s')
    save_model(joint_train_path, model, optimizer, args.joint_train_epoch)
    print(loss_history)
    print(score_list)
    print(each_class)

    output_dir = os.path.join("OUTPUT", args.dataset)
    os.makedirs(output_dir, exist_ok=True)
    # 准备数据
    data = []

    if best_ca is not None:
        for idx, ca_val in enumerate(best_ca):
            data.append({'Class': f'Class_{idx}', 'MCPC': f"{ca_val * 100:.2f}"})

    metric_names = ['ACC', 'Kappa', 'NMI', 'ARI', 'Purity']
    for name, metric in zip(metric_names, best_metrics):
        data.append({'Class': name, 'MCPC': f"{metric * 100:.2f}"})

    data.append({'Class': 'Running Time', 'MCPC': f"{running_time:.2f}"})

    df = pd.DataFrame(data)

    csv_file = os.path.join(output_dir, 'best_results.csv')

    df.to_csv(csv_file, index=False)

    print(f"bestcsv: {csv_file}")
    print(df)
    # tsne_feat, tsne_label = extract_features_for_tsne(data_loader_test, model, DEVICE)

    # tsne_path = os.path.join(output_dir, 'tsne_best.png')
    # plot_tsne(
    #     tsne_feat,
    #     tsne_label,
    #     tsne_path,
    #     title=f'{args.dataset} t-SNE (best model)'
    # )
    # # 1️⃣ 抽特征（只用 feats，不用它返回的 labels）
    # tsne_feat, _ = extract_features_for_tsne(data_loader_test, model, DEVICE)
    #
    # # 2️⃣ 用匈牙利对齐后的预测标签做 t-SNE 颜色（和其他方法一致）
    # tsne_label = get_hungarian_mapped_labels(data_loader_test, model, DEVICE)
    #
    # tsne_path = os.path.join(output_dir, 'tsne_best.png')
    # plot_tsne(
    #     tsne_feat,
    #     tsne_label,
    #     tsne_path,
    #     title=f'{args.dataset} t-SNE (best model)'
    # )

