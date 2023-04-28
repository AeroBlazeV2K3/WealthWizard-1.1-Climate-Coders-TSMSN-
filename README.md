# WealthWizard-1.1-Climate-Coders-TSMSN-
# The Project of Climate Coders NMS Hackathon

# NOTE: GO TO THE WEBSITE TO LOOK AT THE STEPS FOR RUNNING THE NEURAL NET

# Website Link: https://climatecoders24.wixsite.com/climate-coders
# Google Colab Link: https://colab.research.google.com/drive/1lZtmBDFy0ovBIpHX99n7-myiHAOHakdO?usp=sharing

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv

data = pd.read_csv('adult.csv')

edu_map = {
    'Preschool': 0,
    '1st-4th': 2,
    '5th-6th': 3,
    '7th-8th': 4,
    '9th': 5,
    '10th': 6,
    '11th': 7,
    '12th': 8,
    'HS-grad': 9,
    'Some-college': 10,
    'Assoc-voc': 11,
    'Assoc-acdm': 12,
    'Bachelors': 13,
    'Masters': 14,
    'Prof-school': 15,
    'Doctorate': 16
}

if 'education' in data:
    data['education-num'] = data['education'].map(edu_map)
    data.drop(['education'], axis=1, inplace=True)


X = data['education-num'].values.astype(np.float32).reshape(-1, 1)
y = (data['income'].values == '>50K').astype(np.float32).reshape(-1, 1)

X = (X - np.mean(X)) / np.std(X)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1, 20)
        self.fc2 = nn.Linear(20, 100)
        self.fc3 = nn.Linear(100, 20)
        self.fc4 = nn.Linear(20, 5)
        self.fc5 = nn.Linear(5, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.sigmoid(self.fc5(x))
        return x

net = Net()
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.2)

num_epochs = 250
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = net(torch.from_numpy(X))
    loss = criterion(outputs, torch.from_numpy(y))
    loss.backward()
    optimizer.step()

    if (epoch+1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
        
    # Save the state dictionary of the network at epoch 500
    if epoch == 100:
        torch.save(net.state_dict(), 'model.pth')

X_test = torch.from_numpy(np.arange(0, 30, 0.1).astype(np.float32).reshape(-1, 1))
with torch.no_grad():
    y_pred = net(X_test)

y_pred_binary = (y_pred > 0.5).numpy().astype(np.float32)
y_test = (X_test.numpy() > 9).astype(np.float32) # Assume HS-grad means >50K
accuracy = np.mean(y_pred_binary == y_test)
accuracy1 = accuracy * 100
accuracy2 = f"{accuracy1:.2f}%"

plt.plot(X_test.numpy(), y_pred.numpy())
plt.xlabel('Education level (number of years)')
plt.ylabel('Probability of income >50K')
accuracy_text = f"Accuracy: {accuracy2}"
plt.text(0.01, 0.01, accuracy_text, transform=plt.gca().transAxes, fontsize=8, verticalalignment='bottom')
plt.figure(figsize=(10,8))
plt.show()
