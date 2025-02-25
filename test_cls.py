import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import time
import numpy as np
# 训练记录相关函数
from torch.utils.tensorboard import SummaryWriter
import ipdb

# 调用自己的函数
from utils.common import setup_seed #模型初始化参数的随机种子
from data_preparation.data_load.SuDataset import SuDataset #数据装载
from models.pointnet2_cls import pointnet2_cls_ssg, pointnet2_cls_msg, cls_loss
# from models.new_mul_pointnet2 import mul_pointnet2
from models.mul_pointnet2 import mul_pointnet2
from models.mul_pointnet2_AT import mul_pointnet2_AT
from numpy.core._exceptions import _UFuncNoLoopError

def train_one_epoch(train_loader, model, loss_func, optimizer, device):
    losses, total_seen, total_correct = [], 0, 0
    print('start training...')
    # i = 0
    for data, labels in train_loader:
        # i+=1
        # print(i)
        optimizer.zero_grad()  # Important
        labels = labels.to(device)
        xyz, points = data[:, :, :3], data[:, :, 3:]
        pred = model(xyz.to(device), points.to(device))
        loss = loss_func(pred, labels)
        # ipdb.set_trace()
        loss.backward()
        optimizer.step()
        pred = torch.max(pred, dim=-1)[1]
        total_correct += torch.sum(pred == labels)
        total_seen += xyz.shape[0]
        losses.append(loss.item())

    return np.mean(losses), total_correct, total_seen, int(total_correct) / float(total_seen)

def test_one_epoch(test_loader, model, loss_func, device):
    print('start testing...')
    losses, total_seen, total_correct = [], 0, 0
    # ipdb.set_trace()
    for data, labels in test_loader: #########################################
        # ipdb.set_trace()
        labels = labels.to(device)
        xyz, points = data[:, :, :3], data[:, :, 3:]
        with torch.no_grad():
            pred = model(xyz.to(device), points.to(device))
            loss = loss_func(pred, labels)
        
            pred = torch.max(pred, dim=-1)[1]
            total_correct += torch.sum(pred == labels)
            total_seen += xyz.shape[0]
            losses.append(loss.item())
    return np.mean(losses), total_correct, total_seen, int(total_correct) / float(total_seen)

def train(train_loader, test_loader, model, loss_func, optimizer, scheduler, device, ngpus, nepoches, log_interval, log_dir, checkpoint_interval):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    checkpoint_dir = os.path.join(log_dir, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    tensorboard_dir = os.path.join(log_dir, 'tensorboard')
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)
    writer = SummaryWriter(tensorboard_dir)

    # for epoch in range(nepoches):
    for epoch in range(1):
        print('epoch:',epoch)
        # if ((epoch % checkpoint_interval == 0) & (epoch != 0)):
        if (epoch % checkpoint_interval == 0):
            
            # if ngpus > 1:
            #     torch.save(model.module.state_dict(), os.path.join(checkpoint_dir, "pointnet2_cls_%d.pth" % epoch))
            # else:
            #     torch.save(model.state_dict(), os.path.join(checkpoint_dir, "pointnet2_cls_%d.pth" % epoch))
            model.eval()
            lr = optimizer.state_dict()['param_groups'][0]['lr']
            start = time.time()
            loss, total_correct, total_seen, acc = test_one_epoch(test_loader, model, loss_func, device)
            final = time.time()
            # ipdb.set_trace()
            print(f"test_total_time：{final-start : .6f}s")
            print('Test  Epoch: {} / {}, lr: {:.6f}, Loss: {:.2f}, Corr: {}, Total: {}, Acc: {:.4f}'.format(epoch, nepoches, lr, loss, total_correct, total_seen, acc))
            # ipdb.set_trace()
            # writer.add_scalar('test loss', loss, epoch)
            # writer.add_scalar('test acc', acc, epoch)
            print('='*40)
        model.train()
        begin = time.time()
        # loss, total_correct, total_seen, acc = train_one_epoch(train_loader, model, loss_func, optimizer, device)
        # ipdb.set_trace()
        end = time.time()
        print(f"train_total_time：{end-begin : .6f}s")
        # ipdb.set_trace()
        # writer.add_scalar('train loss', loss, epoch)
        # writer.add_scalar('train acc', acc, epoch)
        if epoch % log_interval == 0:
            lr = optimizer.state_dict()['param_groups'][0]['lr']
            print('Train Epoch: {} / {}, lr: {:.6f}, Loss: {:.2f}, Corr: {}, Total: {}, Acc: {:.4f}'.format(epoch, nepoches, lr, loss, total_correct, total_seen, acc))
            # ipdb.set_trace()
        scheduler.step()




if __name__ == '__main__':

    Models = {
        'pointnet2_cls_ssg': pointnet2_cls_ssg,
        'pointnet2_cls_msg': pointnet2_cls_msg,
        'mul_pointnet2': mul_pointnet2,
        'mul_pointnet2_AT': mul_pointnet2_AT
    }
    parser = argparse.ArgumentParser()
    # =============================参数修改部分==========================================
    parser.add_argument('--data_root', type=str, default=r"F:\deep_learning\PKT_PCQA\kaiyuan\data\key_points_local16\test_key1024_local16", help='Root to the dataset')
    parser.add_argument('--data_aux_root', type=str, default=r'F:\deep_learning\PKT_PCQA\kaiyuan\data\txt\test', help='Root to the dataset txt file')
    # 训练轮次和batch_size
    parser.add_argument('--nepoches', type=int, default=200, help='Number of traing epoches')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    # 参与训练的点数********************************************
    parser.add_argument('--npoints', type=int, default=1024, help='Number of the training points')
    # 模型选择********************************************
    parser.add_argument('--model', type=str, default='mul_pointnet2_AT', help='Model name')
    parser.add_argument('--AT_type', type=str, default='scSE', help='Atteion model type: cSE, sSE, scSE')#当模型使用自注意力机制时使用
    parser.add_argument('--seed', type=int, default=1234, help='random seed')#设置模型初始化的随机数种子，以保证结果可复现，
    parser.add_argument('--pred_weights', type=str, default=r'F:\deep_learning\PKT_PCQA\kaiyuan\log_pths\test\checkpoints\pointnet2_cls_4.pth', help='Pretrained_Model')
    # 输出设置********************************************
    parser.add_argument('--nclasses', type=int, default=3, help='Number of classes')
    # 其他数据处理
    parser.add_argument('--augment', type=bool, default=False, help='Augment the train data')#？？
    # 模型训练设置
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learing rate')
    parser.add_argument('--step_size', type=int, default=30, help='StepLR step size')
    parser.add_argument('--dp', type=bool, default=True, help='Random input dropout during training')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='Initial learing rate')# ？？
    parser.add_argument('--gamma', type=float, default=0.7, help='StepLR gamma')# ？？
    # 硬件设置
    parser.add_argument('--gpus', type=str, default='0', help='Cuda ids')
    # 记录设置
    parser.add_argument('--checkpoint_interval', type=int, default=1, help='Checkpoint saved interval') #保存模型的间隔
    parser.add_argument('--log_dir', type=str, default='work_dirs', help='Train/val loss and accuracy logs')  #训练日志路径
    parser.add_argument('--log_interval', type=int, default=1, help='Print iterval') # 打印正确率间隔
    
    args = parser.parse_args()
    print(args)
    setup_seed(args.seed)

    device_ids = list(map(int, args.gpus.strip().split(','))) if ',' in args.gpus else [int(args.gpus)]
    ngpus = len(device_ids)

    # =============================数据装载部分==========================================
    # npoints是否有必要指定？
    su_train = SuDataset(data_root = args.data_root, data_aux_root=args.data_aux_root, split='train', npoints=args.npoints, augment=args.augment, dp=args.dp)
    su_test = SuDataset(data_root = args.data_root, data_aux_root=args.data_aux_root, split='test', npoints=args.npoints)#默认不做增强和丢点处理

    train_loader = DataLoader(dataset=su_train, batch_size=args.batch_size // ngpus, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(dataset=su_test, batch_size=32, shuffle=False, num_workers=1, pin_memory=True)
    print('Train set: {}'.format(len(su_train)))
    print('Test set: {}'.format(len(su_test)))

    # =============================模型设置==========================================
    Model = Models[args.model]
    if args.model == 'mul_pointnet2_AT':
        model = Model(6, args.nclasses, args.AT_type)
    else:
        model = Model(6, args.nclasses)

    # =============================训练部分======================================
    device = torch.device("cuda:{}".format(device_ids[0]) if torch.cuda.is_available() else "cpu")
    if ngpus > 1 and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, device_ids=device_ids)
    model = model.to(device)

    # 模型预训练
    PATH = args.pred_weights
    pretrained_dict = torch.load(PATH)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    # 损失函数***************************************************
    loss = cls_loss().to(device)
    # 优化方法***************************************************
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=args.decay_rate
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.7)

    tic = time.time()
    train(train_loader=train_loader,
          test_loader=test_loader,
          model=model,
          loss_func=loss,
          optimizer=optimizer,
          scheduler=scheduler,
          device=device,
          ngpus=ngpus,
          nepoches=args.nepoches,
          log_interval=args.log_interval,
          log_dir=args.log_dir,
          checkpoint_interval=args.checkpoint_interval,
          )
    toc = time.time()
    print('Training completed, {:.2f} minutes'.format((toc - tic) / 60))

    # 验证阶段 
    print('Validating the pretrained model...') 
    model.eval() 
    with torch.no_grad(): 
        test_loss, test_correct, test_seen, test_acc = test_one_epoch(test_loader, model, loss, device) 
        # test_loss = test_loss.item()
        # test_correct= test_correct.item()
        # test_seen = test_seen
        # test_acc = test_acc
    print('Validation completed. Loss: {:.2f}, Corr: {}, Total: {}, Acc: {:.4f}'.format(test_loss, test_correct, test_seen, test_acc))









