
import numpy as np
from sklearn.metrics import f1_score,recall_score,confusion_matrix
from load_datasets import get_all_data
import torch
import time
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import ClassifyModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(torch.__version__)

batch_size = 2
n_class = 2
h, w = 224, 224

save_model_name = './classfyModel.h5'  # 模型保存的名字

x_train, y_train, x_test, y_test = get_all_data(h, w,'c')

x_train = x_train.transpose([0, 3, 1, 2])
x_test = x_test.transpose([0, 3, 1, 2])
b, c, h, w = x_train.shape
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape, y_train.dtype, x_train.min(), x_train.max())

torch_test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test).long())
test_loader = torch.utils.data.DataLoader(dataset=torch_test_dataset, batch_size=batch_size, shuffle=False)
model = ClassifyModel.MyModel().to(device)

try:
    model.load_state_dict(torch.load(save_model_name, map_location=device))
    print('load model')
except:
    print('failed')
    pass

print('predicting')
model.eval()
t = time.time()
y_pred_list = []
totoal_loss=0
with torch.no_grad():
    for (x, y) in test_loader:
        y_pred = model(x.to(torch.float32).to(device))
        test_loss = F.cross_entropy(y_pred, y.to(torch.float32).to(device))
        y_pred=y_pred.cpu().detach().numpy()  # (b,n_class)

        totoal_loss +=test_loss
        y_pred_list.append(y_pred)
y_pred = np.concatenate(y_pred_list, axis=0)

y_pred = np.argmax(y_pred, -1)
y_test = np.argmax(y_test, -1)
cm=confusion_matrix(y_test,y_pred)
print(y_pred)
print('test loss:', totoal_loss/72)
print('test_acc:', np.mean(np.equal(y_pred, y_test)), ':%sseconds' % (time.time() - t))
print('f1 score:', f1_score(y_pred.tolist(),y_test.tolist()))
print('confusion matrix',cm)

