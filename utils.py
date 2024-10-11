import torch
import sys

def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, device, num_epochs=25):
    best_model_wts = model.state_dict()
    best_acc = 0.0

    train_acc_history = []
    val_acc_history = []

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('=' * 50)
        sub_epoch = 0

        # Mỗi epoch có 2 pha: training và validation
        for phase in ['Training Data', 'Validation Data']:
            if phase == 'Training Data':
                model.train()  # Đặt mô hình ở chế độ huấn luyện
            else:
                model.eval()   # Đặt mô hình ở chế độ đánh giá

            running_loss = 0.0
            running_corrects = 0

            # Lặp qua dữ liệu
            for inputs, labels in dataloaders[phase]:
                sys.stdout.write(f'\r{phase} - Batch {str(sub_epoch+1):<3}|{len(dataloaders[phase])}')

                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward
                # Track history nếu ở pha huấn luyện
                with torch.set_grad_enabled(phase == 'Training Data'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward + optimize chỉ ở pha huấn luyện
                    if phase == 'Training Data':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                sub_epoch += 1

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'\n{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Lưu lại lịch sử accuracy
            if phase == 'Training Data':
                train_acc_history.append(epoch_acc)
            else:
                val_acc_history.append(epoch_acc)
                # Deep copy mô hình nếu đạt accuracy tốt nhất
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = model.state_dict()

        print()

    print(f'\nTraining complete. Best validation Acc: {best_acc:.4f}')

    # Tải lại trọng số tốt nhất
    model.load_state_dict(best_model_wts)
    return model, train_acc_history, val_acc_history

def evaluate_model(model, dataloader, dataset_sizes, device):
    model.eval()
    running_corrects = 0

    with torch.no_grad():
        for inputs, labels in dataloader['Testing Data']:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            running_corrects += torch.sum(preds == labels.data)

    test_acc = running_corrects.double() / dataset_sizes['Testing Data']
    print(f'Test Acc: {test_acc:.4f}')

class_names = ["Training Data", "Validation Data", "Testing Data"]