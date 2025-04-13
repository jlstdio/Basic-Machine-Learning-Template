# evaluation.py
import torch

def evaluate_model(model, test_loader, device='cuda'):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            # 가장 높은 값을 가진 클래스 예측
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    acc = correct / total
    return acc

if __name__ == '__main__':
    # 테스트용 코드: 모델과 dataloader가 준비되었을 때 평가 함수 호출 가능
    pass
