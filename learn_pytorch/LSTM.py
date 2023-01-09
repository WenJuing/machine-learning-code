import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_num, num_classes):
        """Args:
            input_dim: width of input image.
            hidden_dim: num of RNN neurons.
            layer_num: num of RNN layers.
        """
        super().__init__()
        # batch_first=True表示B在数据的第一个维度
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_num, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        # x: [B, H, W], out: [B, H, W], h_n: [layer_num, B, hidden_dim]
        # b_n为隐藏层的输出，None表示h0使用全0初始化
        x, _ = self.lstm(x, None)    
        # 选择最后一个时间点的x输出
        x = self.fc(x[:, -1, :])
        
        return x
        
        