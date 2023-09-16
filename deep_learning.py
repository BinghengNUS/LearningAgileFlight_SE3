## this file is for deep learning

from quad_nn import *
from quad_model import *
from quad_policy import *
from multiprocessing import Process, Array
# initialization


## deep learning
# Hyper-parameters 

num_epochs = 100
batch_size = 100 # 100
learning_rate = 1e-4
num_cores =10 #5

FILE = "nn_pre.pth"
model = torch.load(FILE)
Every_reward = np.zeros((num_epochs,batch_size))


# for multiprocessing, obtain the gradient
def grad(inputs, outputs, gra):
    gate_point = np.array([[-inputs[7]/2,0,1],[inputs[7]/2,0,1],[inputs[7]/2,0,-1],[-inputs[7]/2,0,-1]])
    gate1 = gate(gate_point)
    gate_point = gate1.rotate_y_out(inputs[8])

    quad1 = run_quad(goal_pos=inputs[3:6],ini_r=inputs[0:3].tolist(),ini_q=toQuaternion(inputs[6],[0,0,1]))
    quad1.init_obstacle(gate_point.reshape(12))

    gra[:] = quad1.sol_gradient(quad1.ini_state,outputs[0:3],outputs[3:6],outputs[6])

if __name__ == '__main__':
    for k in range(5):
        FILE = "nn_pre"
        model = torch.load(FILE)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        Iteration = []
        Mean_r = []
        for epoch in range(num_epochs):
        #move = gate1.plane_move()
            evalue = 0
            Iteration += [epoch+1]
            for i in range(int(batch_size/num_cores)):
                n_inputs = []
                n_outputs = []
                n_out = []
                n_gra = []
                n_process = []
                for _ in range(num_cores):
                # sample
                    inputs = nn_sample()
                # forward pass
                    outputs = model(inputs)
                    out = outputs.data.numpy()
                # create shared variables
                    gra = Array('d',np.zeros(8))
                # collection
                    n_inputs.append(inputs)
                    n_outputs.append(outputs)
                    n_out.append(out)
                    n_gra.append(gra)

                #calculate gradient and loss
                for j in range(num_cores):
                    p = Process(target=grad,args=(n_inputs[j],n_out[j],n_gra[j]))
                    p.start()
                    n_process.append(p)
        
                for process in n_process:
                    process.join()

                # Backward and optimize
                for j in range(num_cores):                
                    outputs = model(n_inputs[j])
                    loss = model.myloss(outputs,n_gra[j][0:7])        

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    evalue += n_gra[j][7]
                    Every_reward[epoch,j+num_cores*i]=n_gra[j][7]

                if (i+1)%1 == 0:
                    print (f'Iterate: {k}, Epoch [{epoch+1}/{num_epochs}], Step [{(i+1)*num_cores}/{batch_size}], Reward: {n_gra[0][7]:.4f}')
            # change state
            mean_reward = evalue/batch_size # evalue/int(batch_size/num_cores)
            Mean_r += [mean_reward]
            print('evaluation: ',mean_reward)
            np.save('Iteration',Iteration)
            np.save('Mean_Reward'+str(k),Mean_r)
            np.save('Every_reward'+str(k),Every_reward)
        torch.save(model, "nn_deep2_"+str(k))