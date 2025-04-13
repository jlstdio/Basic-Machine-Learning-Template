# prepare.py
import torch
import torchvision
import torchvision.transforms as transforms

def get_dataloaders(batch_size=128):
    # 학습/테스트 이미지에 대한 전처리 정의
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # CIFAR-10 데이터셋 로드
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    # DataLoader 생성 (iid 분포를 위해 shuffle=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, test_loader

if __name__ == '__main__':
    train_loader, test_loader = get_dataloaders()
    print("학습 데이터셋 크기:", len(train_loader))
    print("테스트 데이터셋 크기:", len(test_loader))
