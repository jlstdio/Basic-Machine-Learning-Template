# main.py
import torch
import torch.nn as nn
import torch.optim as optim
from prepare import get_dataloaders
from testNN_wo_Softmax_3_layer import testNN_wo_Softmax_3_layer
from train import train_model

def main():
    # 하이퍼파라미터 설정
    batch_size = 128
    epochs = 50
    learning_rate = 1e-3
    num_classes = 10  # CIFAR-10

    device = 'cuda:3' if torch.cuda.is_available() else 'cpu'
    
    # 데이터셋 준비 (iid)
    train_loader, test_loader = get_dataloaders(batch_size=batch_size)
    
    # 모델 생성
    model = testNN_wo_Softmax_3_layer(outputClasses=num_classes)
    
    # 손실 함수 및 옵티마이저 정의
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 학습 시작
    train_model(model, train_loader, test_loader, criterion, optimizer, epochs, device)

if __name__ == '__main__':
    main()
