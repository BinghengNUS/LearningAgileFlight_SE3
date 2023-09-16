## this file is for neural network training
from quad_nn import *
# Device configuration
device = torch.device('cpu')#torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters 
input_size = 9 
hidden_size = 64 
output_size = 7
num_epochs = 3
batch_size = 10000
learning_rate = 2e-5

FILE = "nn_pre.pth"
model = network(input_size, hidden_size, hidden_size,output_size).to(device)
# model = torch.load(FILE)
# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  

for epoch in range(num_epochs):
    for i in range(batch_size):  
        

        inputs = nn_sample()
        outputs  = torch.tensor(t_output(inputs), dtype=torch.float).to(device)
        
        # Forward pass
        pre_outputs = model(inputs)
        #print(inputs,' ',pre_outputs)
        loss = criterion(pre_outputs, outputs)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{batch_size}], Loss: {loss.item():.4f}')

#save model
torch.save(model, FILE)

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    n_loss = 0
    for i in range(100):
        inputs = nn_sample()

        ## obtain the expected output
        outputs  = torch.tensor(t_output(inputs), dtype=torch.float).to(device)
        
        # Forward pass
        pre_outputs = model(inputs)
        loss = criterion(pre_outputs, outputs).cpu().data.numpy()
        # max returns (value ,index)
        #_, predicted = torch.max(outputs.data, 1)
        n_loss += loss

    
    print(n_loss/100)

a = nn_sample()
print(a,' ',model(a))

