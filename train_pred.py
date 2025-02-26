import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import time
import numpy as np
from scipy.stats import spearmanr
# from scipy.stats.mstats import spearmanr
from audtorch.metrics.functional import pearsonr
# 训练记录相关函数
from torch.utils.tensorboard import SummaryWriter
# 调用自己的函数
from utils.common import setup_seed #模型初始化参数的随机种子
from data_preparation.data_load.SuDataset_Pred import SuDataset_Pred #数据装载
from models.pointnet2_pred import pointnet2_pred_ssg, pointnet2_pred_msg, pearson_loss
from models.mul_pointnet2_pred import mul_pointnet2_pred
# from models.new_mul_pointnet2_pred import mul_pointnet2_pred
from models.mul_pointnet2_AT_pred import mul_pointnet2_AT_pred
# from models.new_mul_pointnet2_AT_pred import mul_pointnet2_AT_pred
# import ipdb


def train_one_epoch(train_loader, model, loss_func, optimizer, device,epoch):
    losses,pcorrelation_list,scorrelation_list = [],[],[]
    print('start training...')
    model.train()   #设置模型为训练模式
    # i = 0
    for data, label0 in train_loader:
        # i = i + 1
        # print(i)

        label0 = label0.to(device)
        label = torch.unsqueeze(label0,1)
        optimizer.zero_grad()  # Important
        xyz, points = data[:, :, :3], data[:, :, 3:]
        pred = model(xyz.to(device), points.to(device))
        preds = pred.float().squeeze()
        labels = label.float().squeeze()
        
        #计算相关系数
        pcorrelation = pearsonr(preds, labels)
        # if pcorrelation=='nan':
        #     continue
                                                                                                                      
        loss = loss_func(preds, labels)
        loss.backward() # 得到每个参数的梯度值
        # nn.utils.clip_grad_norm_(model.parameters(),max_norm=0, norm_type=2)
        optimizer.step() # 参数更新
        losses.append(loss.item())
        # if epoch == 27:
        # print('dataset xyz:',xyz)
        # print('dataset points:',points)
        # ipdb.set_trace()
        # for name, parms in model.named_parameters():
        #     print('-->name:', name, '-->grad_requirs:',parms.requires_grad, '-->grad_value:', parms.grad)
        # ipdb.set_trace()
        # print('loss: ',loss.item())
        with torch.no_grad():
            preds = preds.cpu()
            labels = labels.cpu() 
            scorrelation = spearmanr(preds, labels)[0]

        pcorrelation_list.append(pcorrelation.item())
        scorrelation_list.append(scorrelation)
    print('Total Loss: {:.2f}, Cor: {}, Corp: {}'.format(np.mean(losses), np.mean(pcorrelation_list), np.mean(scorrelation_list)))
    return np.mean(losses), np.mean(pcorrelation_list), np.mean(scorrelation_list)


def test_one_epoch(test_loader, model, loss_func, device):
    print('start testing...')
    model.eval()         #设置模型为评估模式
    losses, pred_list, labels_list = [], torch.empty(0).to(device), torch.empty(0).to(device)
    total_seen = 0
    for data, label0 in test_loader: #########################################
        # ipdb.set_trace()
        label0 = label0.to(device)
        label = torch.unsqueeze(label0,1)
        xyz, points = data[:, :, :3], data[:, :, 3:]
        with torch.no_grad():
            pred = model(xyz.to(device), points.to(device))
            preds = pred.float().squeeze()
            labels = label.float().squeeze()
            total_seen += 1
            pred_list = torch.cat((pred_list,preds) ,dim = 0)
            labels_list = torch.cat((labels_list,labels) ,dim = 0)
    
    
    loss = loss_func(pred_list, labels_list)
    losses.append(loss.item())
    pcorrelation = pearsonr(pred_list, labels_list) #r： 相关系数 [-1，1]之间，p-value: p值, r越接近于1说明相关性越好
    pred_list = pred_list.cpu()
    labels_list = labels_list.cpu()        
    scorrelation = spearmanr(pred_list, labels_list)[0]
    return np.mean(losses), pcorrelation, scorrelation

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
    best_PLCC = -float('inf')
    best_epoch = -1

    train_losses = []
    test_losses = []

    for epoch in range(nepoches):
        print('epoch:',epoch)
        # if ((epoch % checkpoint_interval == 0) & (epoch != 0)):
        model.train()
        begin = time.time()
        train_loss, train_pcorrelation, train_scorrelation = train_one_epoch(train_loader, model, loss_func, optimizer, device, epoch)
        # loss1, pcorrelation1, scorrelation1 = test_one_epoch(train_loader, model, loss_func, device)
        # ipdb.set_trace()
        end = time.time()
        print(f"train_total_time：{end-begin : .6f}s")
        print('Train  Epoch: {} / {}, Loss: {:.2f}, PLCC: {}, SROCC: {}'.format(epoch, nepoches, train_loss, train_pcorrelation.item(), train_scorrelation))
        # ipdb.set_trace()
        writer.add_scalar('train loss', train_loss, epoch)
        writer.add_scalar('train PLCC', train_pcorrelation, epoch)
        writer.add_scalar('train SROCC', train_scorrelation, epoch)

        if (epoch % checkpoint_interval == 0):
            
            if ngpus > 1:
                torch.save(model.module.state_dict(), os.path.join(checkpoint_dir, "pointnet2_pred_%d.pth" % epoch))
            else:
                torch.save(model.state_dict(), os.path.join(checkpoint_dir, "pointnet2_pred_%d.pth" % epoch))
            model.eval()
            lr = optimizer.state_dict()['param_groups'][0]['lr']
            start = time.time()
            loss, pcorrelation, scorrelation = test_one_epoch(test_loader, model, loss_func, device)
            final = time.time()
            print(f"test_total_time：{final-start : .6f}s")
            # ipdb.set_trace()
            print('Test  Epoch: {} / {}, lr: {:.6f}, Loss: {:.2f}, PLCC: {}, SROCC: {}'.format(epoch, nepoches, lr, loss, pcorrelation.item(), scorrelation))
            writer.add_scalar('test loss', loss, epoch)
            writer.add_scalar('test PLCC', pcorrelation, epoch)
            writer.add_scalar('test SROCC', scorrelation, epoch)
            print('='*40)
            #更新最佳PLCC值和其对应的epoch
            if pcorrelation > best_PLCC:
                best_PLCC = pcorrelation
                best_epoch = epoch
            print(f'Best PLCC: {best_PLCC}, Epoch: {best_epoch}')

        # writer.add_scalar('final train loss', loss1, epoch)
        # writer.add_scalar('final train PLCC', pcorrelation1, epoch)
        # writer.add_scalar('final train SROCC', scorrelation1, epoch)
        # if epoch % log_interval == 0:
        #     lr = optimizer.state_dict()['param_groups'][0]['lr']
        #     print('Train Epoch: {} / {}, lr: {:.6f}, Final Loss: {:.2f}, Final PLCC: {}, Final SROCC: {}'.format(epoch, nepoches, lr, loss, pcorrelation, scorrelation))
        #     # ipdb.set_trace()
        # scheduler.step()
        if scheduler:
            scheduler.step()

if __name__ == '__main__':

    Models = {
        'pointnet2_pred_ssg': pointnet2_pred_ssg,
        'pointnet2_pred_msg': pointnet2_pred_msg,
        'mul_pointnet2_pred': mul_pointnet2_pred,
        'mul_pointnet2_AT_pred': mul_pointnet2_AT_pred
    }
    parser = argparse.ArgumentParser()
    # =============================参数修改部分==========================================
    parser.add_argument('--data_root', type=str, default=r"F:\deep_learning\PKT_PCQA\kaiyuan\data\key_points_local16\test_key1024_local16", help='Root to the dataset')
    parser.add_argument('--data_aux_root', type=str, default=r'F:\deep_learning\PKT_PCQA\kaiyuan\data\txt\test', help='Root to the dataset txt file')
    # 训练轮次和batch_size
    parser.add_argument('--nepoches', type=int, default=100, help='Number of traing epoches')#defult200
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    # 参与训练的点数********************************************
    parser.add_argument('--npoints', type=int, default=1024, help='Number of the training points')
    # 模型选择********************************************
    parser.add_argument('--model', type=str, default='mul_pointnet2_AT_pred', help='Model name')
    parser.add_argument('--AT_type', type=str, default='scSE', help='Atteion model type: cSE, sSE, scSE')#当模型使用自注意力机制时使用
    parser.add_argument('--seed', type=int, default=1234, help='random seed')#设置模型初始化的随机数种子，以保证结果可复现，
    # 预训练参数
    parser.add_argument('--pred_weights', type=str, default=r'F:\deep_learning\PKT_PCQA\kaiyuan\log_pths\test\test_cls_pth\checkpoints\pointnet2_cls_4.pth', help='Pretrained_Model')
    # 其他数据处理
    parser.add_argument('--augment', type=bool, default=False, help='Augment the train data')#？？
    # 模型训练设置
    parser.add_argument('--lr', type=float, default=0.0001, help='Initial learing rate')
    parser.add_argument('--step_size', type=int, default=20, help='StepLR step size')#每隔多少学习率下降
    parser.add_argument('--dp', type=bool, default=True, help='Random input dropout during training')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='Initial learing rate')# L2正则化系数
    parser.add_argument('--gamma', type=float, default=0.7, help='StepLR gamma')#学习率的变化
    # 硬件设置
    parser.add_argument('--gpus', type=str, default='0', help='Cuda ids')
    # 记录设置
    parser.add_argument('--checkpoint_interval', type=int, default=1, help='Checkpoint saved interval') #保存模型的间隔
    parser.add_argument('--log_dir', type=str, default=r'F:\deep_learning\PKT_PCQA\kaiyuan\log_pths\test\test_pred_pth', help='Train/val loss and accuracy logs')  #训练日志路径
    parser.add_argument('--log_interval', type=int, default=1, help='Print iterval') # 打印正确率间隔13 0.5847
    
    args = parser.parse_args()
    print(args)
    setup_seed(args.seed)

    device_ids = list(map(int, args.gpus.strip().split(','))) if ',' in args.gpus else [int(args.gpus)]
    ngpus = len(device_ids)

    # =============================数据装载部分==========================================
    # npoints是否有必要指定？
    su_train = SuDataset_Pred(data_root = args.data_root, data_aux_root=args.data_aux_root, split='train', npoints=args.npoints, augment=args.augment, dp=args.dp)
    su_test = SuDataset_Pred(data_root = args.data_root, data_aux_root=args.data_aux_root, split='test', npoints=args.npoints)#默认不做增强和丢点处理

    train_loader = DataLoader(dataset=su_train, batch_size=args.batch_size // ngpus, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(dataset=su_test, batch_size=32, shuffle=False, num_workers=1, pin_memory=True)
    print('Train set: {}'.format(len(su_train)))
    print('Test set: {}'.format(len(su_test)))

    # =============================模型设置==========================================
    Model = Models[args.model]
    if args.model == 'mul_pointnet2_AT_pred':
        model = Model(6, args.AT_type)
    else:
        model = Model(6)

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
    loss = pearson_loss().to(device)
    # 优化方法***************************************************
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=args.decay_rate
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.7)

    # optimizer = torch.optim.SGD(model.parameters(),lr=args.lr,momentum=0.9,weight_decay=args.decay_rate)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.7)


    
    
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