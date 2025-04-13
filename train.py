# train.py
import torch
from evaluation import evaluate_model

def train_model(model, train_loader, test_loader, criterion, optimizer, epochs, device):
    model.to(device)
    best_acc = 0.0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        total = 0
        correct = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        epoch_loss = running_loss / total
        train_acc = correct / total
        test_acc = evaluate_model(model, test_loader, device)
        
        print(f'Epoch [{epoch+1}/{epochs}] - Loss: {epoch_loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
        torch.save(model.state_dict(), f'saved_models/model_epoch_{epoch}.pth')
        
        # 최고 성능 모델 저장
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), 'best_model.pth')
    
    print("Training complete. Best Test Accuracy: {:.4f}".format(best_acc))
