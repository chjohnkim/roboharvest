import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from PIL import Image
from model import SimpleCNN
from data_utils import CustomImageDataset
from tqdm import tqdm
import os
import cv2 as cv
import click
import time 

def train(model, criterion, optimizer, train_loader, val_loader, epochs=10, checkpoint_path='best_model.pth'):
    best_val_loss = float('inf')

    for epoch in range(epochs):
        print(f"Epoch [{epoch+1}/{epochs}]")

        # Training phase
        model.train()
        running_train_loss = 0.0
        correct_train = 0
        total_train = 0
        train_loader_tqdm = tqdm(train_loader, desc='Training', leave=False)
        
        for inputs, labels in train_loader_tqdm:
            inputs, labels = inputs.cuda(), labels.cuda().float().unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Calculate running loss and accuracy
            running_train_loss += loss.item()
            predicted = torch.round(outputs)  # Get binary predictions (0 or 1)
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)

            train_loader_tqdm.set_postfix({
                'Loss': running_train_loss / len(train_loader_tqdm),
                'Acc': correct_train / total_train * 100
            })
        
        avg_train_loss = running_train_loss / len(train_loader)
        train_accuracy = correct_train / total_train * 100

        # Validation phase
        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0
        val_loader_tqdm = tqdm(val_loader, desc='Validation', leave=False)
        
        with torch.no_grad():
            for inputs, labels in val_loader_tqdm:
                inputs, labels = inputs.cuda(), labels.cuda().float().unsqueeze(1)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()
                
                # Calculate validation accuracy
                predicted = torch.round(outputs)  # Get binary predictions (0 or 1)
                correct_val += (predicted == labels).sum().item()
                total_val += labels.size(0)

                val_loader_tqdm.set_postfix({
                    'Val Loss': running_val_loss / len(val_loader_tqdm),
                    'Val Acc': correct_val / total_val * 100
                })

        avg_val_loss = running_val_loss / len(val_loader)
        val_accuracy = correct_val / total_val * 100

        print(f'Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%')
        print(f'                     Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')

        # Save the model if validation loss improves
        if avg_val_loss < best_val_loss:
            print(f'  Validation loss improved from {best_val_loss:.4f} to {avg_val_loss:.4f}. Saving model...')
            torch.save(model.state_dict(), checkpoint_path)
            best_val_loss = avg_val_loss

# Inference function
def inference(model, image_path, transform, checkpoint_path='best_model.pth'):
    model.load_state_dict(torch.load(checkpoint_path))
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).cuda()
    model.eval()
    with torch.no_grad():
        output = model(image)
        prediction = torch.round(output).item()
        return int(prediction)

# Inference function for a video input
def inference_on_video(model, video_path, transform, checkpoint_path='best_model.pth'):
    # Load the best model
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    
    # Check if video exists
    if not os.path.exists(video_path):
        print(f"Video file {video_path} not found.")
        return

    # Open the video file
    cap = cv.VideoCapture(video_path)
    inference_time_sum = 0
    count = 0
    while cap.isOpened():
        success, img = cap.read()
        if not success:
            break
        print(img.shape)
        print(type(img))
        #print(img)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        print(img.shape)
        print(type(img))
        print(img)
        image = Image.fromarray(img).convert("RGB")
        t0 = time.time()
        image_tensor = transform(image).unsqueeze(0).cuda()
        # Perform inference
        with torch.no_grad():
            output = model(image_tensor)
            prediction = torch.round(output).item()
            label = "grasped" if prediction == 1 else "not_grasped"
        inference_time = time.time() - t0
        inference_time_sum += inference_time
        count += 1
        print(f"Avg inference time: {inference_time_sum / count:.4f} s")

        # Display the image with the prediction using cv2
        #image_cv = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        # Resize the image for display
        image_cv = cv.resize(img, (512, 512))
        image_cv = cv.cvtColor(image_cv, cv.COLOR_RGB2BGR)
        cv.putText(image_cv, label, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv.imshow('Image', image_cv)
        cv.waitKey(1)
    cv.destroyAllWindows()  

@click.command()
@click.option('--test', default=False, help='Test mode')
@click.option('--video_path', default=None, help='Path to video file')
@click.option('--train_dir', default=None, help='Path to train dataset file')
def main(test, train_dir, video_path):
    # If test, video_path cannot be None
    if test and video_path is None:
        print("Please provide a video file path for testing.")
        return
    # If train, train_dir cannot be None
    if not test and train_dir is None:
        print("Please provide a directory path for training.")
        return

    # Define common variables for both training and inference
    model = SimpleCNN().cuda()
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    if test:
        inference_on_video(model, video_path, transform, checkpoint_path='best_model.pth')
    else:
        # Define dataset and data transformations
        dataset = CustomImageDataset(train_dir, transform=transform)

        # Split dataset into training and validation sets (80% train, 20% val)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Train the model
        train(model, criterion, optimizer, train_loader, val_loader, epochs=10, checkpoint_path='best_model.pth')

    
# Main
if __name__ == "__main__":
    main()