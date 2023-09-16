## this file is for neural network training
from quad_nn import *
from quad_policy import *
from multiprocessing import Process, Array
import numpy as np

# Device configuration
device = torch.device('cpu')
#torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Hyper-parameters 
input_size = 18 
hidden_size = 128 
output_size = 7
num_epochs = 16*100
batch_size = 50
num_cores = 16
learning_rate = 1e-6

file1 = "nn_deep2_0"
model1 = torch.load(file1)

FILE = "nn3_1.pth"
model = network(input_size, hidden_size, hidden_size,output_size).to(device)
# model = torch.load(FILE)
# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  

def traj(inputs, outputs, state_traj):
    gate_point = np.array([[-inputs[7]/2,0,1],[inputs[7]/2,0,1],[inputs[7]/2,0,-1],[-inputs[7]/2,0,-1]])
    gate1 = gate(gate_point)
    gate_point = gate1.rotate_y_out(inputs[8])

    quad1 = run_quad(goal_pos=inputs[3:6],ini_r=inputs[0:3].tolist(),ini_q=toQuaternion(inputs[6],[0,0,1]))
    quad1.init_obstacle(gate_point.reshape(12))

    quad1.get_input(ini_state=quad1.ini_state,tra_pos=outputs[0:3],tra_ang=outputs[3:6],t=outputs[6],Ulast=[0,0,0,0])

    state_t = np.reshape(quad1.sol1['state_traj_opt'],(batch_size+1)*13)
    state_traj[:] = state_t

if __name__ == '__main__':
    for epoch in range(int(num_epochs/num_cores)):
        n_inputs = []
        n_outputs = []
        n_out = []
        n_traj = []
        n_process = []
        for _ in range(num_cores):
            # sample
            inputs = nn_sample()
            # forward pass
            outputs = model1(inputs)
            out = outputs.data.numpy()
            # create shared variables
            state_traj = Array('d',np.zeros((batch_size+1)*13))
            # collection
            n_inputs.append(inputs)
            n_outputs.append(outputs)
            n_out.append(out)
            n_traj.append(state_traj)

        for j in range(num_cores):
            p = Process(target=traj,args=(n_inputs[j],n_out[j],n_traj[j]))
            p.start()
            n_process.append(p)

        for process in n_process:
            process.join()

        loss_n = 0
        for k in range(num_cores):
            state_traj = np.reshape(n_traj[k],[(batch_size+1),13])
            for i in range(batch_size):  
        
                inputs = np.zeros(18)
                inputs[0:13] = state_traj[i,:] # in world frame
                inputs[13:16] = n_inputs[k][3:6] # final position
                inputs[16:18] = n_inputs[k][7:9] # gap information, width and pitch angle

                out = np.zeros(7)
                out[0:6] = n_out[k][0:6]
                out[6] = n_out[k][6]-i*0.10
        
                t_out = torch.tensor(out, dtype=torch.float).to(device)
                # Forward pass
                pre_outputs = model(inputs)
                #print(inputs,' ',pre_outputs)
                loss = criterion(pre_outputs, t_out)
                loss_t = loss.data.numpy()
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_n += loss_t
        
        print (f'Epoch [{(epoch+1)*num_cores}/{num_epochs}], Loss: {loss_n/(batch_size*num_cores):.4f}')

#save model
torch.save(model, "nn3_1.pth")


