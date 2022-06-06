import torch
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
#
# x_data = np.array([[0, 0], [1, 0], [1, 1], [0, 0], [0, 0], [0, 1]])
# x_data_torch = torch.from_numpy(x_data).float()
#
# y_data = np.array([0,1,2,0,0,2])
# y_data_torch = torch.from_numpy(y_data).float()
#
# num_features = 2
# num_classes = 3
# n_hidden_1 = 5
#
#
# W1 = torch.randn(num_features, n_hidden_1, requires_grad=True)
# B1 = torch.randn(n_hidden_1, requires_grad=True)
#
# Wout = torch.randn(n_hidden_1, num_classes, requires_grad=True)
# Bout = torch.randn(num_classes, requires_grad=True)
#
#
# learning_rate = 0.01
# no_of_epochs = 1000
#
# for epoch in range(no_of_epochs):
#     z1 = torch.add(torch.matmul(x_data_torch, W1), B1)
#     Zout = torch.add(torch.matmul(F.relu(z1), Wout), Bout)
#
#     log_softmax = F.log_softmax(Zout, dim=1)
#     loss = F.nll_loss(log_softmax, y_data_torch.long())
#
#     loss.backward()
#     with torch.no_grad():
#         W1.data -= learning_rate * W1.grad.data
#         B1.data -= learning_rate * B1.grad.data
#         Wout.data -= learning_rate * Wout.grad.data
#         Bout.data -= learning_rate * Bout.grad.data
#
#     W1.grad.data.zero_()
#     B1.grad.data.zero_()
#     Wout.grad.data.zero_()
#     Bout.grad.data.zero_()
#
#
#     if epoch % 100 == 0:
#         with torch.no_grad():
#             z1 = torch.add(torch.matmul(x_data_torch ,W1),B1)
#             Zout = torch.add(torch.matmul(F.relu(z1) ,Wout),Bout)
#             predicted = torch.argmax(Zout, 1)
#             train_acc = accuracy_score(predicted.numpy(),y_data)
#             print('Epoch: %d, loss: %.4f, train_acc: %.3f' %(epoch + 1, loss.item() , train_acc))
#
# print("Finished")
# # Result
# print('Predicted :', predicted.numpy())
# print('Truth :', y_data)
# print('Accuracy : %.2f' %train_acc)


x = torch.rand(1,4)
y = torch.ones(1,4)





