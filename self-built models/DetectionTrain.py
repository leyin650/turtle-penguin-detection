import torchvision.transforms

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from load_datasets import get_all_data
import numpy as np
from DetectionModel import MyModel
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def IOU(predicted_boxes, target_boxes):

    pred_x1, pred_y1, pred_x2, pred_y2 = predicted_boxes[:, 0], predicted_boxes[:, 1], predicted_boxes[:,
                                                                                       2], predicted_boxes[:, 3]
    true_x1, true_y1, true_x2, true_y2 = target_boxes[:, 0], target_boxes[:, 1], target_boxes[:, 2], target_boxes[:, 3]


    xi1 = torch.max(pred_x1, true_x1)
    yi1 = torch.max(pred_y1, true_y1)
    xi2 = torch.min(pred_x2, true_x2)
    yi2 = torch.min(pred_y2, true_y2)


    inter_area = torch.clamp(xi2 - xi1, min=0) * torch.clamp(yi2 - yi1, min=0)
    pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
    true_area = (true_x2 - true_x1) * (true_y2 - true_y1)
    union_area = pred_area + true_area - inter_area


    iou = inter_area / union_area

    return iou

epochs = 300
batch_size = 10
lr = 0.00001
h, w = 256, 256


x_train, y_train, x_test, y_test = get_all_data(h, w, 'd')

x_train = x_train.transpose([0, 3, 1, 2])
x_test = x_test.transpose([0, 3, 1, 2])
b, c, h, w = x_train.shape


torch_train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
torch_test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))
train_loader = torch.utils.data.DataLoader(dataset=torch_train_dataset, batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=torch_test_dataset, batch_size=batch_size,
                                          shuffle=False)
a = next(iter(train_loader))
print(a[0].shape, a[1].shape)

model = MyModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # 定义优化器
scaler = torch.cuda.amp.GradScaler()  # 梯度缩放


detection_model_name = 'detectionModel.h5'
train_loss_index = 0.
history = {'epoch': [], 'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': [],'iou':[]}  # 创建保存的中间过程字典
for epoch in range(epochs):
    model.train()
    train_correct = 0
    train_total = 0
    train_total_loss = 0.
    for i, (x, y) in enumerate(train_loader):
        x = x.to(torch.float32).to(device)  # (b,c,h,w)
        y = y.to(torch.float32).to(device)

        y_pred = model(x)  # (b,n_class)

        train_loss = nn.SmoothL1Loss()(y_pred, y)

        train_loss.requires_grad_(True)
        train_total_loss += train_loss.item()

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

    model.eval()
    test_total_iou = 0.
    with torch.no_grad():
        test_correct = 0
        test_total = 0
        test_total_loss = 0.
        for x, y in test_loader:
            x = x.to(torch.float32).to(device)
            y = y.to(torch.float32).to(device)
            y_pred = model(x)
            test_loss = nn.SmoothL1Loss()(y_pred, y)

            for i in range(len(y)):
                y[i,2]=y[i, 0] + y[i, 2]
                y[i, 3] = y[i, 1] + y[i, 3]
                y_pred[i, 2] = y_pred[i, 0] + y_pred[i, 2]
                y_pred[i, 3] = y_pred[i, 1] + y_pred[i, 3]

            ious = IOU(y, y_pred)
            for iou in ious:
                test_total_iou +=iou
            test_total_loss += test_loss.item()

    loss = train_total_loss / 500
    val_loss = test_total_loss / 72
    history['loss'].append(np.array(loss))
    history['val_loss'].append(np.array(val_loss))
    history['epoch'].append(epoch)
    history['iou'].append(test_total_iou.cpu()/72)
    print('epochs:%s/%s:' % (epoch + 1, epochs), 'loss:%.6f' % history['loss'][epoch],
          'val_loss:%.6f' % history['val_loss'][epoch], 'iou:%.6f' % (test_total_iou/72))
    if train_loss_index > loss:
        train_loss_index = loss
        print('save in :%s' % detection_model_name)
        torch.save(model.state_dict(), detection_model_name)

