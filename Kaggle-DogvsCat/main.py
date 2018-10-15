#coding:utf8
from config import opt
import os
import torch as t
import models
import ipdb
from torch.autograd import Variable as V
from data.dataset import DogCat
from torch.utils.data import DataLoader
from torchnet import meter
from utils.visualize import Visualizer
from tqdm import tqdm


#@t.no_grad() # pytorch>=0.5
def test(**kwargs):
    opt._parse(kwargs)

    # configure model
    model = getattr(models, opt.model)().eval()
    if opt.load_model_path:
        model.load(opt.load_model_path)
    #model.to(opt.device)
    model.cuda() if opt.use_gpu == True else model
    # data
    train_data = DogCat(opt.test_data_root,test=True)
    test_dataloader = DataLoader(train_data,batch_size=opt.batch_size,shuffle=False,num_workers=opt.num_workers)
    results = []
    for ii,(data,path) in tqdm(enumerate(test_dataloader)):
        #input = data.to(opt.device)
        data = V(data)
        input = data.cuda() if opt.use_gpu==True else data
        score = model(input)
        probability = t.nn.functional.softmax(score,dim=1)[:,0].data.tolist()#softmax如果dim=1是按行进行softmax,如果dim=0就是按列进行softmax
        # label = score.max(dim = 1)[1].detach().tolist() X[:,0]就是取所有行的第0个数据, X[:,1] 就是取所有行的第1个数据
        
        batch_results = [(path_,probability_) for path_,probability_ in zip(path,probability) ]

        results += batch_results
    write_csv(results,opt.result_file)

    return results

def write_csv(results,file_name):
    import csv
    with open(file_name,'w') as f:
        writer = csv.writer(f)
        writer.writerow(['id','label'])
        writer.writerows(results)
    
def train(**kwargs):
    opt._parse(kwargs)
    vis = Visualizer(opt.env,port = opt.vis_port)

    # step1: configure model
    model = getattr(models, opt.model)()
    if opt.load_model_path:
        model.load(opt.load_model_path)
    #model.to(opt.device)#根据use_gpu选择将模型转移到gpu上还是cpu上
    model.cuda() if opt.use_gpu == True else model()

    # step2: data
    train_data = DogCat(opt.train_data_root,train=True)
    val_data = DogCat(opt.train_data_root,train=False)
    train_dataloader = DataLoader(train_data,opt.batch_size,
                        shuffle=True,num_workers=opt.num_workers)
    val_dataloader = DataLoader(val_data,opt.batch_size,
                        shuffle=False,num_workers=opt.num_workers)
    
    # step3: criterion and optimizer
    #criterion = t.nn.CrossEntropyLoss().to(opt.device)
    criterion = t.nn.CrossEntropyLoss().cuda() if opt.use_gpu == True else t.nn.CrossEntropyLoss()
    lr = opt.lr
    optimizer = model.get_optimizer(lr, opt.weight_decay)
        
    # step4: meters
    loss_meter = meter.AverageValueMeter()#meter可以帮助用户快速统计训练过程中的一些指标，AverageValueMeter计算所有数的平均值和标准差
    confusion_matrix = meter.ConfusionMeter(2)
    previous_loss = 1e10

    # train
    for epoch in range(opt.max_epoch):
        
        loss_meter.reset()
        confusion_matrix.reset()

        for ii,(data,label) in tqdm(enumerate(train_dataloader)):

            # train model 
            #input = data.to(opt.device)
            data = V(data)
            input = data.cuda() if opt.use_gpu == True else data
            #target = label.to(opt.device)
            label = V(label)
            target = label.cuda() if opt.use_gpu == True else label


            optimizer.zero_grad()
            score = model(input)
            loss = criterion(score,target)
            loss.backward()
            optimizer.step()
            
            
            # meters update and visualize
            loss_meter.add(loss.data[0])
            # detach 一下更安全保险
            confusion_matrix.add(score.data, target.data) 

            if (ii + 1)%opt.print_freq == 0:
                vis.plot('loss', loss_meter.value()[0])
                
                # 进入debug模式
                if os.path.exists(opt.debug_file):
                    import ipdb;
                    ipdb.set_trace()


        model.save()

        # validate and visualize
        val_cm,val_accuracy = val(model,val_dataloader)

        vis.plot('val_accuracy',val_accuracy)
        vis.log("epoch:{epoch},lr:{lr},loss:{loss},train_cm:{train_cm},val_cm:{val_cm}".format(
                    epoch = epoch,loss = loss_meter.value()[0],val_cm = str(val_cm.value()),train_cm=str(confusion_matrix.value()),lr=lr))
        
        # update learning rate
        if loss_meter.value()[0] > previous_loss:          
            lr = lr * opt.lr_decay
            # 第二种降低学习率的方法:不会有moment等信息的丢失
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        

        previous_loss = loss_meter.value()[0]

#@t.no_grad()
def val(model,dataloader):
    """
    计算模型在验证集上的准确率等信息
    """
    model.eval()
    confusion_matrix = meter.ConfusionMeter(2)
    for ii, (val_input, label) in tqdm(enumerate(dataloader)):
        #val_input = val_input.to(opt.device)
        val_input = V(val_input)
        val_input = val_input.cuda() if opt.use_gpu == True else val_input
        score = model(val_input)
        confusion_matrix.add(score.data.squeeze(), label.type(t.LongTensor))

    model.train()
    cm_value = confusion_matrix.value()
    accuracy = 100. * (cm_value[0][0] + cm_value[1][1]) / (cm_value.sum())
    return confusion_matrix, accuracy

def help():
    """
    打印帮助的信息： python file.py help
    """
    
    print("""
    usage : python file.py <function> [--args=value]
    <function> := train | test | help
    example: 
            python {0} train --env='env0701' --lr=0.01
            python {0} test --dataset='path/to/dataset/root/'
            python {0} help
    avaiable args:""".format(__file__))

    from inspect import getsource
    source = (getsource(opt.__class__))
    print(source)

if __name__=='__main__':
    import fire
    fire.Fire()
