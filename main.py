import torch.nn.functional as F
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
import time
from models import MMFN
import dataloader
from models import face_net, ges_net
from focal_loss import FocalLoss


def logprint(log_file, string):
    file_writer = open(log_file, 'a')
    file_writer.write('{:}\n'.format(string))
    file_writer.flush()

def main():
    '''time'''
    date = '1221/'
    hourandmin = '2132/'
    save_path_0 = './result_decision/' + date + hourandmin
    if os.path.exists(save_path_0) == False:
        os.makedirs(save_path_0)
    log_file = save_path_0 + '/log.txt'

    '''bs set'''
    batchsize_train = 4
    batchsize_eval = 1
    logprint(log_file,'train_bs = ' + str(batchsize_train))
    logprint(log_file, 'test_bs = ' + str(batchsize_eval))
    use_cuda = torch.cuda.is_available()

    '''Load full data, and than random split'''
    human = ['Sub1', 'Sub2', 'Sub3', 'Sub4', 'Sub5', 'Sub6', 'Sub7', 'Sub8', 'Sub9', 'Sub10']
    dataset = dataloader.Data_set(data_path='FETE dataset/',
                                     train_human=human,
                                     mode='both')
    for j in range(len(human)):
        save_path = save_path_0 + '/' + str(human[j])
        if os.path.exists(save_path) == False:
            os.makedirs(save_path)

        num = list(range(10))
        test_dataset = dataset[j]
        num.pop(j)
        train_dataset = []
        for i in num:
            train_dataset.extend(dataset[i])
        print('number of training set', len(train_dataset))
        print('number of testing set', len(test_dataset))
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batchsize_train, shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batchsize_eval, shuffle=False)
        print('读取训练数据集合完成,sample number:', len(train_dataset))
        print('读取test数据集合完成:', human[j], 'sample number:', len(test_dataset))

        num_epochs = 30
        lr_face = 1e-4
        lr_ges = 1e-3
        step_size = 10
        gamma = 1
        # declare0 and define an objet of MyCNN
        # fusion = fusion_network.Fusion()
        logprint(log_file, 'num_epochs = ' + str(num_epochs))
        logprint(log_file, 'learning rate face = ' + str(lr_face))
        logprint(log_file, 'learning rate ges = ' + str(lr_ges))
        logprint(log_file, 'step_size = ' + str(step_size))
        logprint(log_file, 'gamma = ' + str(gamma))

        video_model = face_net.C3D(num_classes=6, pretrained=True)
        touch_model = ges_net.Extractor(0.5,0.5)

        # load unimodal model weights
        video_model_path = os.path.join('models/face_pretrain_models/',
                                        "{}/model.pt".format(human[j]))
        video_model_checkpoint = torch.load(video_model_path) if use_cuda else \
            torch.load(video_model_path, map_location=torch.device('cpu'))
        video_model.load_state_dict(video_model_checkpoint)

        touch_model_path = os.path.join('models/ges_pretrain_models/',
                                        "{}/model.pt".format(human[j]))
        touch_model_checkpoint = torch.load(touch_model_path) if use_cuda else \
            torch.load(touch_model_path, map_location=torch.device('cpu'))
        touch_model.load_state_dict(touch_model_checkpoint)
        model_param = {
            "video": {
                "model": video_model,
                "id": 0
            },
            "touch": {
                "model": touch_model,
                "id": 1
            }
        }

        net = MMFN.MMFN(model_param)
        device = torch.device('cuda')

        optimizer = torch.optim.Adam([{'params': net.video_model_blocks.parameters(), 'lr':lr_face},
                           {'params': net.touch_model_blocks.parameters(), 'lr':lr_ges}])
        logprint(log_file, 'optimizer = ' + 'Adam')
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        logprint(log_file, 'lr_step_size = ' + str(step_size))
        logprint(log_file, 'lr_gamma = ' + str(gamma))
        losses, val_losses, accs, time, y_pre_one, y_ture_one, _ = fit(net, num_epochs, optimizer, device,
                                                                       train_loader, test_loader, lr_scheduler)


        np.save(save_path + '/train_loss.npy',losses)
        np.save(save_path + '/test_loss.npy', val_losses)
        np.save(save_path + '/test_acc.npy', accs)
        np.save(save_path + '/train_acc.npy', _)
        np.save(save_path + '/pre.npy', y_pre_one)
        np.save(save_path + '/true.npy', y_ture_one)
        show_curve(losses, "train loss",save_path)
        show_curve(val_losses, "test loss",save_path)
        show_curve(_, "train accuracy",save_path)
        show_curve(accs, "test accuracy",save_path)


def fit(model, num_epochs, optimizer, device, train_loader, test_loader, lr_scheduler):
    """
    train and evaluate an classifier num_epochs times.
    We use optimizer and cross entropy loss to train the model.
    Args:
        model: CNN network
        num_epochs: the number of training epochs
        optimizer: optimize the loss function
    """

    # loss and optimizer
    # loss_func = nn.CrossEntropyLoss()
    loss_func = FocalLoss()

    model.to(device)
    loss_func.to(device)

    # log train loss and test accuracy
    losses = []
    val_losses = []
    accs = []
    step_index = 0
    data_times = []
    end = time.time()
    predict = []
    lab = []
    train_accuracy = []
    best_model_acc = 0
    for epoch in range(num_epochs):
        # if epoch % 10 == 0:
        #     step_index += 1
        # lr = step_lr(optimizer, lr, epoch, 0.5, step_index)
        print('Epoch {}/{}:'.format(epoch + 1, num_epochs))
        for param_group in optimizer.param_groups:
            print(param_group['lr'])
        # train step
        loss, trainacc = train(model, train_loader, loss_func, optimizer, device)
        losses.append(loss)
        train_accuracy.append(trainacc)
        train_time = time.time() - end
        print(train_time)
        data_times.append(train_time)
        # evaluate step
        test_loss, accuracy, y_predict, label = evaluate(model, test_loader, loss_func, device)

        # if best_model_acc < accuracy[2]:
        #     best_model_acc = accuracy[2]
        #     torch.save(model.state_dict(), 'model/model' + str(epoch) + '.pt')
        #         print(label)
        val_losses.append(test_loss)
        accs.append(accuracy)
        for param_group in optimizer.param_groups:
            print(param_group['lr'])
        end = time.time()
        predict.append(np.array(y_predict))
        lab.append(np.array(label))
        lr_scheduler.step()

    # show curve
    lab = np.array(lab)
    predict = np.array(predict)

    print(lab.shape)
    return losses, val_losses, accs, time, predict, lab, train_accuracy


def show_curve(ys, title,save_path):
    """
    plot curlve for Loss and Accuacy
    Args:
        ys: loss or acc list
        title: loss or accuracy
    """
    x = np.array(range(len(ys)))
    y = np.array(ys)
    plt.plot(x, y, c='b')
    plt.axis()
    plt.title('{} curve'.format(title))
    plt.xlabel('epoch')
    plt.ylabel('{}'.format(title))
    # plt.show()
    f = os.path.join(save_path, '{}.png'.format(title))
    plt.savefig(f, dpi=400)
    # pylab.show()
    plt.cla()


def step_lr(optimizer, learning_rate, epoch, gamma, step_index):
    lr = learning_rate
    if (epoch % 10 == 0):  # &(epoch ==200):
        lr = learning_rate * gamma
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def train(model, train_loader, loss_func, optimizer, device):
    """
    train model using loss_fn and optimizer in an epoch.
    model: CNN networks
    train_loader: a Dataloader object with training data
    loss_func: loss function
    device: train on cpu or gpu device
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    ges_correct = 0
    face_correct = 0
    # train the model using minibatch
    for i, [ges,face, target] in enumerate(train_loader):
        ges = Variable(ges.type(torch.FloatTensor).cuda())
        face = Variable(face.type(torch.FloatTensor).cuda())
        target = Variable(target.long().cuda())

        # forward
        pred_face, pred_ges, pred_all, loss = model(ges,face,target)

        # backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        test_pred_face = torch.nn.functional.softmax(pred_face, dim=1)
        test_pred_ges = torch.nn.functional.softmax(pred_ges, dim=1)
        out_face = test_pred_face.data.max(1, keepdim=True)[1]
        out_ges = test_pred_ges.data.max(1, keepdim=True)[1]

        face_correct += out_face.eq(target.long().data.view_as(out_face)).cpu().sum()
        ges_correct += out_ges.eq(target.long().data.view_as(out_ges)).cpu().sum()

        total += target.size(0)

        # every 100 iteration, print loss
        if (i + 1) % 100 == 0:
            print("Step [{}/{}] Train Loss: {:.4f}"
                  .format(i + 1, len(train_loader), loss.item()))
    accuracy_face = face_correct / total
    accuracy_ges = ges_correct / total
    print("face acc: {:.4f}%\tges acc: {:.4f}%"
          .format((100 * accuracy_face), (100 * accuracy_ges)))
    return total_loss / len(train_loader), [accuracy_face, accuracy_ges]


def evaluate(model, val_loader, loss_func, device):
    """
    model: CNN networks
    val_loader: a Dataloader object with validation data
    device: evaluate on cpu or gpu device
    return classification accuracy of the model on val dataset
    """
    # evaluate the model
    model.eval()
    y_predict = []
    label = []
    feature = []
    # context-manager that disabled gradient computation
    with torch.no_grad():
        correct = 0
        total = 0
        valid_loss = 0
        ges_correct = 0
        face_correct = 0
        ave_correct = 0
        for i, [ges, face, target] in enumerate(val_loader):
            # device: cpu or gpu
            ges = Variable(ges.type(torch.FloatTensor).cuda())
            face = Variable(face.type(torch.FloatTensor).cuda())
            target = Variable(target.long().cuda())

            pred_face, pred_ges, pred_all, loss = model(ges, face, target)

            test_pred_face = torch.nn.functional.softmax(pred_face, dim=1)
            test_pred_ges = torch.nn.functional.softmax(pred_ges, dim=1)
            '''fusion pred'''
            test_pred_ave = torch.nn.functional.softmax(pred_all, dim=1)

            out_face = test_pred_face.data.max(1, keepdim=True)[1]
            out_ges = test_pred_ges.data.max(1, keepdim=True)[1]
            out_ave = test_pred_ave.data.max(1, keepdim=True)[1]
            face_correct += out_face.eq(target.long().data.view_as(out_face)).cpu().sum()
            ges_correct += out_ges.eq(target.long().data.view_as(out_ges)).cpu().sum()
            ave_correct += out_ave.eq(target.long().data.view_as(out_ave)).cpu().sum()



            valid_loss += loss.item()

            y_predict.extend(out_ave.cpu().numpy())
            label.extend(target.cpu().numpy())
            total += target.size(0)

        accuracy_face = face_correct / total
        accuracy_ges = ges_correct / total
        accuracy_ave = ave_correct / total

        print('test Loss: {:.4f} \t Accuracy face: {:.4f} %\t Accuracy ges: {:.4f} %\t Accuracy ave: {:.4f} %'.format(valid_loss / len(val_loader),
                                                                            100 * accuracy_face,
                                                                            100 * accuracy_ges,
                                                                            100 * accuracy_ave,))

        return valid_loss / len(val_loader), [accuracy_face,accuracy_ges,accuracy_ave], y_predict, label


if __name__ == '__main__':
    main()

