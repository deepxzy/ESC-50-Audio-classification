import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from models import RNN, MLP
from segdataset import SegDataset_mlp,SegDataset
import argparse
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str, default=r'data/sudoku.csv', help='root of data')
parser.add_argument('--save_root', type=str, default=r'checkpoints/mlp', help='root of data')
parser.add_argument('--epoch', type=int, default=10, help='epoch number')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--batch_size', type=int, default=16, help='training batch size')
parser.add_argument('--num_workers', type=int, default=8, help='num_workers')
parser.add_argument('--random_seed', type=int, default=10, help='random seed')
parser.add_argument('--model_kind', type=str, default='mlp', help='kind of model, i.e. cnn or rnn or mlp or transformer')
parser.add_argument('--is_shuffle_dataset', type=bool, default=True, help='shuffle dataset or not')
parser.add_argument('--test_split', type=float, default=0.1, help='ratio of the test set')
opt = parser.parse_args()

def main():
    if opt.model_kind == 'rnn':
        model = RNN()
        dataset = SegDataset()
        dataset1 = SegDataset(type='test')
    elif opt.model_kind == 'mlp':
        model = MLP()
        dataset = SegDataset_mlp()
        dataset1 = SegDataset_mlp(type='test')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    all_train_epoch_loss = []
    all_test_epoch_loss = []
    all_test_epoch_accuracy=[]
    sudoku_model = model.to(device)


    criterion = nn.CrossEntropyLoss().to(device)

    optimizer = optim.Adam(sudoku_model.parameters(), opt.lr)

    train_iter = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size,num_workers=opt.num_workers)
    test_iter = torch.utils.data.DataLoader(dataset1, batch_size=1,num_workers=opt.num_workers)

    for epo in range(opt.epoch):
        train_loss = 0
        sudoku_model.train()  # 启用batch normalization和drop out
        for index, (image, label) in enumerate(train_iter):
            image = image.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            output = sudoku_model(image)
            loss = criterion(output, label)
            loss.backward()
            iter_loss = loss.item()
            train_loss += iter_loss
            optimizer.step()

            if np.mod(index, 200) == 0:
                print('epoch {}, {}/{},train loss is {}'.format(epo, index, len(train_iter), iter_loss))

        # test
        test_loss = 0
        correct=0
        total=0
        sudoku_model.eval()
        with torch.no_grad():
            for index, (image, label) in enumerate(test_iter):
                image = image.to(device)
                label = label.to(device)
                optimizer.zero_grad()
                output = sudoku_model(image)
                loss = criterion(output, label)
                output=torch.argmax(output, dim=1)
                correct += (output == label).sum()
                total += len(label.view(-1))
                iter_loss = loss.item()
                test_loss += iter_loss
        accuracy=(correct/total).item()
        print('<---------------------------------------------------->')
        print('epoch: %f' % epo)
        print('epoch train loss = %f, epoch test loss = %f,accuracy =%.3f'
              % (train_loss / len(train_iter), test_loss / len(test_iter),accuracy))

        if np.mod(epo, 1) == 0:
            # 只存储模型参数
            torch.save(sudoku_model.state_dict(), opt.save_root+'/ep%03d-loss%.3f-val_loss%.3f.pth' % (
                (epo + 1), (train_loss / len(train_iter)), (test_loss / len(test_iter)))
                       )
            print('saving checkpoints/model_{}.pth'.format(epo))
        all_test_epoch_accuracy.append(accuracy)

        all_train_epoch_loss.append(train_loss / len(train_iter))
        all_test_epoch_loss.append(test_loss / len(test_iter))

    # plot
    plt.figure()
    plt.title('train_loss')
    plt.plot(all_train_epoch_loss)
    plt.xlabel('epoch')
    plt.savefig(opt.save_root+'/train_loss.png')

    plt.figure()
    plt.title('test_loss')
    plt.plot(all_test_epoch_loss)
    plt.xlabel('epoch')
    plt.savefig(opt.save_root+'/test_loss.png')

    plt.figure()
    plt.title('test_accury')
    plt.plot(all_test_epoch_accuracy)
    plt.xlabel('epoch')
    plt.savefig(opt.save_root+'/test_accury.png')
if __name__ == '__main__':
    main()