from main import *


def evaluate():
    batch_size = 119

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        augmentation['Normalize']
    ])

    """load test-set and test-loader"""
    test_dataset = dsets.CIFAR10(root='./data', train=False, transform=transform_test, download=False)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
    """load the model"""

    model = OurCNN()
    """change pkl file name"""
    model.load_state_dict(torch.load('model.pkl'))
    model.eval()  # important

    regularization = nn.CrossEntropyLoss()

    """testing"""
    correct = 0
    total = 0
    test_loss = 0
    for i, (images, labels) in enumerate(test_loader):
        # images = to_cuda(images)
        # labels = to_cuda(labels)
        outputs = model(images)
        test_loss += regularization(outputs, labels)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += int((predicted == labels).sum())

    """results per epoch"""
    avg_acc = 100 * (correct / total)
    # avg_loss = test_loss/(i+1)

    print("TEST: Avg.Accuracy: %0.2f%%" % avg_acc)
    # print("Loss: %f" % avg_loss)

    return avg_acc


if __name__ == "__main__":
    acc = evaluate()
    print(acc)
