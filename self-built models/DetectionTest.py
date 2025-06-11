from DetectionModel import MyModel
import torch
import torch.nn as nn
from load_datasets import get_all_data
import math
import os
import numpy as np
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



def standard_deviation(data):

    mean = sum(data) / len(data)

    variance = sum([((x - mean) ** 2) for x in data]) / len(data)

    std_deviation = math.sqrt(variance)
    return std_deviation


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 10
h, w = 256, 256



_, _, x_test, y_test = get_all_data(h, w,'d')

x_test = x_test.transpose([0, 3, 1, 2])



torch_test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))

test_loader = torch.utils.data.DataLoader(dataset=torch_test_dataset, batch_size=batch_size,
                                          shuffle=False)




detection_model_name='detectionModel.h5'

model=MyModel().to(device)
try:
    model.load_state_dict(torch.load(detection_model_name, map_location=device))
    print('load model')
except:
    print('failed')
    pass
model.eval()
total_distance=0
distance_list=[]
y_pred_list=[]
y_true_list=[]
iou_std=[]
with torch.no_grad():
    test_total_iou = 0.
    test_loss_total=0
    for x, y in test_loader:
        x = x.to(torch.float32).to(device)
        y = y.to(torch.float32).to(device)

        y_pred = model(x)
        test_loss = nn.SmoothL1Loss()(y_pred, y)
        test_loss_total+=test_loss.item()

        for i in range(len(y)):
            y_true_list.append([y[i, 0].to("cpu"),y[i, 1].to("cpu"),y[i, 2].to("cpu"),y[i, 3].to("cpu")])
            y_pred_list.append([y_pred[i, 0].to("cpu"),y_pred[i, 1].to("cpu"),y_pred[i, 2].to("cpu"),y_pred[i, 3].to("cpu")])



        for i in range(len(y)):
            y[i, 2] = y[i, 0] + y[i, 2]
            y[i, 3] = y[i, 1] + y[i, 3]
            y_pred[i, 2] = y_pred[i, 0] + y_pred[i, 2]
            y_pred[i, 3] = y_pred[i, 1] + y_pred[i, 3]

        for i in range(len(y)):
            distance = math.sqrt(((y_pred[i, 0] + (y_pred[i, 2])*0.5) - (y[i, 0] + (y[i, 2])*0.5)) ** 2 + ((y_pred[i, 1] + (y_pred[i, 3])*0.5) - (y[i, 1] + (y[i, 3])*0.5)) ** 2)
            total_distance+=distance
            distance_list.append(distance)
        ious=IOU(y,y_pred)
        for iou in ious:
            test_total_iou += iou
            iou_std.append(iou)



print("iou : {}".format(test_total_iou / 72))
print("loss : {}".format(test_loss_total / 72))
print("mean distance : {}".format(total_distance / 72))
print("std distance : {}".format(standard_deviation(distance_list)))
print("std iou : {}".format(standard_deviation(iou_std)))



