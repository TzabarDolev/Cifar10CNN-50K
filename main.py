import torch
import numpy
# import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn as nn

"""a series of data augmentations"""
augmentation = {
    "RandomHorizontalFlip": transforms.RandomHorizontalFlip(p=1),
    "Normalize": transforms.Normalize((0.4914, 0.4822, 0.4465),
                                      (0.2023, 0.1994, 0.2010)),
    "RandomAffine": transforms.RandomAffine(10, translate=None, scale=None,
                                            shear=None, resample=False, fillcolor=0),
    "RandomCrop": transforms.RandomCrop(32, padding=4),
    "CenterCrop": transforms.CenterCrop(32),
    "ColorJitterBrightness": transforms.ColorJitter(brightness=0.2),
    "ColorJitterContrast": transforms.ColorJitter(contrast=0.2),
    "ColorJitterSaturation": transforms.ColorJitter(saturation=0.2),
    "ColorJitterHue": transforms.ColorJitter(hue=0.2),
}

class OurCNN(nn.Module):
    """NN model"""
    def __init__(self):
        super(OurCNN, self).__init__()
        self.conv = nn.Sequential(
            # Layer 1
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.PReLU(),

            # Layer 2
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.PReLU(),

            # Layer 3
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.PReLU(),
            nn.MaxPool2d(2),

            # Layer 4
            nn.Conv2d(16, 28, kernel_size=3, padding=1),
            nn.PReLU(),

            # Layer 5
            nn.Conv2d(28, 28, kernel_size=3, padding=1),
            nn.PReLU(),

            # Layer 6
            nn.Conv2d(28, 28, kernel_size=3, padding=1),
            nn.BatchNorm2d(28),
            nn.PReLU(),
            nn.MaxPool2d(2),

            # Layer 7
            nn.Conv2d(28, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.MaxPool2d(2),
            nn.AvgPool2d(kernel_size=1, stride=1)

        )
        self.fc = nn.Sequential(
            nn.Linear(4 * 4 * 64, 10),
            nn.LogSoftmax()
        )
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        out = self.conv(x)
        """make 1D"""
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.fc(out)
        return out


use_gpu = torch.cuda.is_available()
device = torch.device('cuda:0' if use_gpu else 'cpu')

def to_cuda(x):
    if use_gpu:
        x = x.cuda()
    return x


def visualize(n, train_err, test_err, train_loss, test_loss, file_name):
    """create graphs for train and test error and loss"""
    f, (ax1, ax2) = plt.subplots(1, 2)

    ax1.plot(range(n), train_loss, test_loss)
    ax1.set_yticks(numpy.arange(0, 0.6, 0.05))
    ax1.set_title("Loss")
    ax1.set_ylabel('Loss')
    ax1.set_xlabel('Epochs')
    ax1.legend(["train", "test"])
    ax1.set_ylim(bottom=0, top=0.6)
    ax1.grid()

    ax2.plot(range(n), train_err, test_err)
    ax2.set_title("Error")
    ax2.set_yticks(numpy.arange(0, 0.6, 0.05))
    ax2.set_ylabel('Error')
    ax2.set_xlabel('Epochs')
    ax2.legend(["train", "test"])
    ax2.set_ylim(bottom=0)
    ax2.grid()

    f.tight_layout()
    plt.show()
    f.savefig(file_name)
    return


def testing_model(model, loader, criteria, optimizer, name):
    """testing the model"""
    correct = 0
    total = 0
    test_loss = 0

    if name == "train":
        """TRAIN"""
        for i, (images, labels) in enumerate(loader):
            images = to_cuda(images)
            labels = to_cuda(labels)

            optimizer.zero_grad()
            outputs = model(images)  # forward on the model for this batch
            loss = criteria(outputs, labels)
            loss.backward()
            optimizer.step()
            test_loss += loss

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += int((predicted.cuda() == labels).sum())

        """results per epoch"""
        avg_loss = test_loss / (i + 1)
        avg_err = 1 - (correct / total)

        return avg_err, avg_loss

    else:
        """TEST"""
        with torch.no_grad():
            for i, (images, labels) in enumerate(loader):
                images = to_cuda(images)
                labels = to_cuda(labels)

                outputs = model(images)
                loss = criteria(outputs, labels)
                test_loss += loss

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += int((predicted.cuda() == labels).sum())

        """results per epoch"""
        avg_loss = test_loss/(i+1)
        avg_err = 1 - (correct / total)

        return avg_err, avg_loss


def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp


def myround(x, base=5):
    return int(base * round(x/base))


def main():
    save_flag = False

    """Hyper parameters"""
    num_epochs = 450
    learning_rate = 0.00089
    batch_size = 119
    criterion_method = nn.NLLLoss()

    """Augmentation"""
    transform_train1 = transforms.Compose([
        augmentation['RandomCrop'],
        augmentation['ColorJitterBrightness'],
        augmentation['ColorJitterSaturation'],
        transforms.ToTensor(),
        augmentation['Normalize']
    ])

    transform_train2 = transforms.Compose([
        augmentation['ColorJitterHue'],
        augmentation['ColorJitterContrast'],
        augmentation['RandomHorizontalFlip'],
        transforms.ToTensor(),
        augmentation['Normalize']
    ])

    transform_train3 = transforms.Compose([
        augmentation['RandomCrop'],
        augmentation['RandomAffine'],
        transforms.ToTensor(),
        augmentation['Normalize']
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        augmentation['Normalize']
    ])

    """download the CIFAR10 train/test dataset"""
    "switch the download option to true if it's the first time using the code"
    train_dataset = torch.utils.data.ConcatDataset([
            dsets.CIFAR10(root='./data', train=True, transform=transform_train1, download=False),
            dsets.CIFAR10(root='./data', train=True, transform=transform_train2, download=False),
            dsets.CIFAR10(root='./data', train=True, transform=transform_train3, download=False)
    ])
    "switch the download option to true if it's the first time using the code"
    test_dataset = dsets.CIFAR10(root='./data', train=False, transform=transform_test, download=False)

    """Data Loader- allows iterating over our data in batches"""
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    """initializing the model"""
    model = OurCNN()
    model = to_cuda(model)
    print("Training Neural Net with %d parameters, lr: %f, batch size: %d, "
          % (get_n_params(model), learning_rate, batch_size))

    """Loss and Optimizer"""
    criterion = criterion_method
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-2)

    """Training the model"""
    test_error = []
    train_error = []
    train_loss = []
    test_loss = []
    max_epoch = 0

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=30, gamma=0.5)
    for epoch in range(num_epochs):
        scheduler.step()
        if epoch % 10 == 0:
            print('max epoch: ' + str(max_epoch))

        """TRAIN"""
        avg_train_err, avg_train_loss = testing_model(model, train_loader, criterion, optimizer, "train")
        train_error.append(avg_train_err)
        train_loss.append(avg_train_loss)
        # print('Epoch: [%d/%d], TRAIN: Avg.Loss: %0.4f, Avg.Acc: %f'
        #       % (epoch + 1, num_epochs, avg_train_loss, 1-avg_train_err))

        """TEST"""
        """testing the model -TEST"""
        avg_test_err, avg_test_loss = testing_model(model, test_loader, criterion, optimizer, "test")
        test_error.append(avg_test_err)
        test_loss.append(avg_test_loss)
        # print('Epoch: [%d/%d], TEST: Avg.Loss: %0.4f, Avg.Acc: %f'
        #       % (epoch + 1, num_epochs, avg_test_loss, 1-avg_test_err))

        if 1-avg_test_err > max_epoch:
            max_epoch = 1-avg_test_err
            if max_epoch > 0.86:
                torch.save(model.state_dict(), 'current_acc_'+str(int(max_epoch*1000))+'.pkl')
                print("Max Accuracy: %f" % max_epoch)
                save_flag = True

    print("Max Accuracy: %f" % max_epoch)
    print("")

    """Get Graphs"""
    # hyper_params = "Batch-Size: " + str(batch_size) + ", #-Epochs: " + str(num_epochs) + \
    #                ", Learning-Rate: " + str(learning_rate)
    if save_flag:
        file_name = (str(batch_size)+"-"+str(num_epochs)+"-"+str(learning_rate)).replace(".","")+".png"
        visualize(num_epochs, train_error, test_error, train_loss, test_loss, file_name)

    return


if __name__ == "__main__":
    main()
