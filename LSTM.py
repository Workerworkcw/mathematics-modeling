import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt


# Define LSTM Neural Networks
class LstmRNN(nn.Module):
    """
        Parameters：
        - input_size: feature size,输入数据的特征大小。
        - hidden_size: number of hidden units,隐藏单元数量。决定了 LSTM 能够学习到的特征表示的复杂度。默认为 1
        - output_size: number of output,模型的输出大小。
        - num_layers: layers of LSTM to stack,LSTM 层的堆叠数量。增加层数可以增加模型的表达能力，但也可能导致过拟合和训练时间增加。
    """

    def __init__(self, input_size, hidden_size=1, output_size=1, num_layers=1):
        super().__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)  # utilize the LSTM model in torch.nn
        # 创建一个全连接层（线性层），用于将 LSTM 的输出映射到最终的输出大小。这个线性层将接收 LSTM 的隐藏状态作为输入，并输出预测结果。
        self.forwardCalculation = nn.Linear(hidden_size, output_size)

    def forward(self, _x):
        x, _ = self.lstm(_x)  # _x is input, size (seq_len, batch, input_size)
        s, b, h = x.shape  # x is output, size (seq_len, batch, hidden_size)
        x = x.view(s * b, h)
        x = self.forwardCalculation(x)
        x = x.view(s, b, -1)
        return x


if __name__ == '__main__':
    # create database
    data_len = 200
    t = np.linspace(0, 12 * np.pi, data_len)
    sin_t = np.sin(t)
    cos_t = np.cos(t)

    dataset = np.zeros((data_len, 2))
    dataset[:, 0] = sin_t
    dataset[:, 1] = cos_t
    dataset = dataset.astype('float32')

    # plot part of the original dataset
    plt.figure()
    plt.plot(t[0:60], dataset[0:60, 0], label='sin(t)')
    plt.plot(t[0:60], dataset[0:60, 1], label='cos(t)')
    plt.plot([2.5, 2.5], [-1.3, 0.55], 'r--', label='t = 2.5')  # t = 2.5
    plt.plot([6.8, 6.8], [-1.3, 0.85], 'm--', label='t = 6.8')  # t = 6.8
    plt.xlabel('t')
    plt.ylim(-1.2, 1.2)
    plt.ylabel('sin(t) and cos(t)')
    plt.legend(loc='upper right')

    # choose dataset for training and testing
    train_data_ratio = 0.5  # Choose 50% of the data for testing
    train_data_len = int(data_len * train_data_ratio)
    train_x = dataset[:train_data_len, 0]
    train_y = dataset[:train_data_len, 1]
    INPUT_FEATURES_NUM = 1  # input_size
    OUTPUT_FEATURES_NUM = 1     # output_size
    t_for_training = t[:train_data_len]

    test_x = dataset[train_data_len:, 0]
    test_y = dataset[train_data_len:, 1]
    t_for_testing = t[train_data_len:]

    # ----------------- train -------------------
    # 分成5批，每批有INPUT_FEATURES_NUM个特征数量
    train_x_tensor = train_x.reshape(-1, 5, INPUT_FEATURES_NUM)  # set batch size to 5
    train_y_tensor = train_y.reshape(-1, 5, OUTPUT_FEATURES_NUM)  # set batch size to 5

    # transfer data to pytorch tensor
    # 转换数据类型
    train_x_tensor = torch.from_numpy(train_x_tensor)
    train_y_tensor = torch.from_numpy(train_y_tensor)

    lstm_model = LstmRNN(INPUT_FEATURES_NUM, 16, output_size=OUTPUT_FEATURES_NUM, num_layers=1)  # 16 hidden units
    print('LSTM model:', lstm_model)    # 结构信息和各个模块的信息
    print('model.parameters:', lstm_model.parameters)   # 模型的参数信息

    loss_function = nn.MSELoss()    # 损失函数，均方差
    # 优化器 Adam，lstm_model.parameters() 表示将 LSTM 模型的所有可学习参数传递给优化器
    # lr=1e-2 设置学习率为 0.01
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=1e-2)

    max_epochs = 10000
    for epoch in range(max_epochs):
        # 将训练数据输入，得到输出
        output = lstm_model(train_x_tensor)
        # 损失函数计算真实值和预测值的损失
        loss = loss_function(output, train_y_tensor)
        # 反向传播，计算损失相对模型参数的梯度
        loss.backward()
        # 根据计算得到的梯度，使用优化器（这里是 Adam 优化器 torch.optim.Adam）更新模型的参数，朝着减小损失的方向调整参数值
        optimizer.step()
        # 清空上一轮迭代中存储的梯度信息，为下一轮迭代做准备。
        optimizer.zero_grad()

        if loss.item() < 1e-4:
            print('Epoch [{}/{}], Loss: {:.5f}'.format(epoch + 1, max_epochs, loss.item()))
            print("The loss value is reached")
            break
        elif (epoch + 1) % 100 == 0:
            print('Epoch: [{}/{}], Loss:{:.5f}'.format(epoch + 1, max_epochs, loss.item()))

    # prediction on training dataset
    predictive_y_for_training = lstm_model(train_x_tensor)
    predictive_y_for_training = predictive_y_for_training.view(-1, OUTPUT_FEATURES_NUM).data.numpy()

    # torch.save(lstm_model.state_dict(), 'model_params.pkl') # save model parameters to files

    # ----------------- test -------------------
    # lstm_model.load_state_dict(torch.load('model_params.pkl'))  # load model parameters from files
    lstm_model = lstm_model.eval()  # switch to testing model

    # prediction on test dataset
    test_x_tensor = test_x.reshape(-1, 5,
                                   INPUT_FEATURES_NUM)  # set batch size to 5, the same value with the training set
    test_x_tensor = torch.from_numpy(test_x_tensor)

    predictive_y_for_testing = lstm_model(test_x_tensor)
    predictive_y_for_testing = predictive_y_for_testing.view(-1, OUTPUT_FEATURES_NUM).data.numpy()

    # ----------------- plot -------------------
    plt.figure()
    plt.plot(t_for_training, train_x, 'g', label='sin_trn')
    plt.plot(t_for_training, train_y, 'b', label='ref_cos_trn')
    plt.plot(t_for_training, predictive_y_for_training, 'y--', label='pre_cos_trn')

    plt.plot(t_for_testing, test_x, 'c', label='sin_tst')
    plt.plot(t_for_testing, test_y, 'k', label='ref_cos_tst')
    plt.plot(t_for_testing, predictive_y_for_testing, 'm--', label='pre_cos_tst')

    plt.plot([t[train_data_len], t[train_data_len]], [-1.2, 4.0], 'r--', label='separation line')  # separation line

    plt.xlabel('t')
    plt.ylabel('sin(t) and cos(t)')
    plt.xlim(t[0], t[-1])
    plt.ylim(-1.2, 4)
    plt.legend(loc='upper right')
    plt.text(14, 2, "train", size=15, alpha=1.0)
    plt.text(20, 2, "test", size=15, alpha=1.0)

    plt.show()