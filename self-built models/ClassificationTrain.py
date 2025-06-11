import os
import numpy as np
from load_datasets import get_all_data
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import ClassifyModel


os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(torch.__version__)

epochs = 50
batch_size = 4
n_class = 2
lr = 0.00001
h, w = 224, 224

save_model_name = './classfyModel.h5'
load_pretrain_weights = True
fp16 = False

x_train, y_train, x_test, y_test = get_all_data(h, w,'c')
print()


x_train = x_train.transpose([0, 3, 1, 2])
x_test = x_test.transpose([0, 3, 1, 2])
b, c, h, w = x_train.shape
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape, y_train.dtype, x_train.min(), x_train.max(),
      np.unique(y_train))

torch_train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train).long())
torch_test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test).long())
train_loader = torch.utils.data.DataLoader(dataset=torch_train_dataset, batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=torch_test_dataset, batch_size=batch_size,
                                          shuffle=False)

a = next(iter(train_loader))
print(a[0].shape, a[1].shape)


model = ClassifyModel.MyModel().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scaler = torch.cuda.amp.GradScaler()


valAccuracy=0
history = {'epoch': [], 'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}
for epoch in range(epochs):
    model.train()
    train_correct = 0
    train_total = 0
    train_total_loss = 0.
    for i, (x, y) in enumerate(train_loader):
        x = x.to(torch.float32).to(device)  # (b,c,h,w)
        y = y.to(torch.float32).to(device)  # (b,)
        if fp16:
            with torch.cuda.amp.autocast():
                y_pred = model(x)  # (b,n_class)
                train_loss = F.cross_entropy(y_pred, y)
        else:
            y_pred = model(x)  # (b,n_class)
            train_loss = F.cross_entropy(y_pred, y)

        train_total_loss += train_loss.item()
        train_total += y.size(0)
        _, y_pred = torch.max(y_pred, dim=1)
        _, y = torch.max(y, dim=1)
        train_correct += (y_pred == y).sum().item()

        if fp16:
            scaler.scale(train_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        test_correct = 0
        test_total = 0
        test_total_loss = 0.
        for x, y in test_loader:
            x = x.to(torch.float32).to(device)
            y = y.to(torch.float32).to(device)
            y_pred = model(x)
            test_loss = F.cross_entropy(y_pred, y)
            test_total_loss += test_loss.item()
            _, y_pred = torch.max(y_pred, dim=1)
            _, y = torch.max(y, dim=1)
            test_total += y.size(0)
            test_correct += (y_pred == y).sum().item()

    loss = train_total_loss / (i + 1)
    val_loss = test_total_loss / (i + 1)
    accuracy = train_correct / train_total
    val_accuracy = test_correct / test_total
    history['loss'].append(np.array(loss))
    history['val_loss'].append(np.array(val_loss))
    history['accuracy'].append(np.array(accuracy))
    history['val_accuracy'].append(np.array(val_accuracy))
    history['epoch'].append(epoch)
    print('epochs:%s/%s:' % (epoch + 1, epochs),
          'loss:%.6f' % history['loss'][epoch], 'accuracy:%.6f' % history['accuracy'][epoch],
          'val_loss:%.6f' % history['val_loss'][epoch], 'val_accuracy:%.6f' % history['val_accuracy'][epoch])

    if val_accuracy>valAccuracy and val_loss<0.5:
        valAccuracy=val_accuracy
        print('saved to:%s' % save_model_name)
        torch.save(model.state_dict(), save_model_name)
x=[i for i in range(1,epochs+1)]
plt.plot(x, history['loss'])
plt.title('train_loss',fontsize=20)
plt.xlabel('epoch',fontsize=15)
plt.ylabel('loss', fontsize=15)
plt.show()

plt.plot(x, history['accuracy'])
plt.title('train_acc',fontsize=20)
plt.xlabel('epoch',fontsize=15)
plt.ylabel('accuracy', fontsize=15)
plt.show()