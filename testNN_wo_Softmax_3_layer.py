# testNN_wo_Softmax_3_layer.py
import torch
import torch.nn as nn

class testNN_wo_Softmax_3_layer(nn.Module):
    def __init__(self, outputClasses):
        super(testNN_wo_Softmax_3_layer, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(64 * 8 * 8, outputClasses)  # 64 채널, 8x8 출력
        self._initialize_weights()  # He 초기화

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)      # 출력: (B, 32, 16, 16)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.pool(x)      # 출력: (B, 64, 8, 8)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.relu(x)

        x = x.view(x.size(0), -1)  # flatten
        x = self.fc(x)
        return x

if __name__ == '__main__':
    # 간단한 모델 구조 확인
    model = testNN_wo_Softmax_3_layer(outputClasses=10)
    print(model)
