##this file is to obtain the optimal solution

from casadi import *
import numpy
from scipy import interpolate
import casadi

class OCSys:

    def __init__(self, project_name="my optimal control system"):
        self.project_name = project_name

    def setAuxvarVariable(self, auxvar=None):
        if auxvar is None or auxvar.numel() == 0:
            self.auxvar = SX.sym('auxvar')
        else:
            self.auxvar = auxvar
        self.n_auxvar = self.auxvar.numel()

    def setStateVariable(self, state, state_lb=[], state_ub=[]):
        self.state = state
        self.n_state = self.state.numel()
        if len(state_lb) == self.n_state:
            self.state_lb = state_lb
        else:
            self.state_lb = self.n_state * [-1e20]

        if len(state_ub) == self.n_state:
            self.state_ub = state_ub
        else:
            self.state_ub = self.n_state * [1e20]

    def setControlVariable(self, control, control_lb=[], control_ub=[]):
        self.control = control
        self.n_control = self.control.numel()

        if len(control_lb) == self.n_control:
            self.control_lb = control_lb
        else:
            self.control_lb = self.n_control * [-1e20]

        if len(control_ub) == self.n_control:
            self.control_ub = control_ub
        else:
            self.control_ub = self.n_control * [1e20]

    def setDyn(self, f, dt):
        if not hasattr(self, 'auxvar'):
            self.setAuxvarVariable()

        #self.dyn = casadi.Function('f',[self.state, self.control],[f])
        self.dyn = self.state + dt * f
        self.dyn_fn = casadi.Function('dynamics', [self.state, self.control, self.auxvar], [self.dyn])
        #M = 4
        #DT = dt/4
        #X0 = casadi.SX.sym("X", self.n_state)
        #U = casadi.SX.sym("U", self.n_control)
        # #
        #X = X0
        #for _ in range(M):
            # --------- RK4------------
        #    k1 =DT*self.dyn(X, U)
        #    k2 =DT*self.dyn(X+0.5*k1, U)
        #    k3 =DT*self.dyn(X+0.5*k2, U)
        #    k4 =DT*self.dyn(X+k3, U)
            #
        #    X = X + (k1 + 2*k2 + 2*k3 + k4)/6        
        ## Fold
        #self.dyn_fn = casadi.Function('dyn', [X0, U], [X])

    def setthrustcost(self, thrust_cost):
        if not hasattr(self, 'auxvar'):
            self.setAuxvarVariable()

        assert thrust_cost.numel() == 1, "thrust_cost must be a scalar function"        

        self.thrust_cost = thrust_cost
        self.thrust_cost_fn = casadi.Function('thrust_cost',[self.control, self.auxvar], [self.thrust_cost])

    def setPathCost(self, path_cost):
        if not hasattr(self, 'auxvar'):
            self.setAuxvarVariable()

        assert path_cost.numel() == 1, "path_cost must be a scalar function"

        self.path_cost = path_cost
        self.path_cost_fn = casadi.Function('path_cost', [self.state,  self.auxvar], [self.path_cost])

    def setFinalCost(self, final_cost):
        if not hasattr(self, 'auxvar'):
            self.setAuxvarVariable()

        assert final_cost.numel() == 1, "final_cost must be a scalar function"

        self.final_cost = final_cost
        self.final_cost_fn = casadi.Function('final_cost', [self.state, self.auxvar], [self.final_cost])
    
    def setTraCost(self, tra_cost, t = 3.0):
        self.t = t
        self.tra_cost = tra_cost
        self.tra_cost_fn = casadi.Function('tra_cost', [self.state, self.auxvar], [self.tra_cost])


    def ocSolver(self, ini_state, Ulast=None, horizon=None, auxvar_value=1, print_level=0, dt = 0.1,costate_option=0):
        assert hasattr(self, 'state'), "Define the state variable first!"
        assert hasattr(self, 'control'), "Define the control variable first!"
        assert hasattr(self, 'dyn'), "Define the system dynamics first!"
        assert hasattr(self, 'path_cost'), "Define the running cost function first!"
        assert hasattr(self, 'final_cost'), "Define the final cost function first!"

        if type(ini_state) == numpy.ndarray:
            ini_state = ini_state.flatten().tolist()

        # Start with an empty NLP
        w = []
        w0 = []
        lbw = []
        ubw = []
        J = 0
        g = []
        lbg = []
        ubg = []

        # "Lift" initial conditions
        Xk = MX.sym('X0', self.n_state)
        w += [Xk]
        lbw += ini_state
        ubw += ini_state
        w0 += ini_state
        if Ulast is not None:
            Ulast = Ulast
        else:
            Ulast = np.array([0,0,0,0])
        
        # Formulate the NLP
        for k in range(int(horizon)):
            # New NLP variable for the control
            Uk = MX.sym('U_' + str(k), self.n_control)
            w += [Uk]
            lbw += self.control_lb
            ubw += self.control_ub
            w0 += [0.5 * (x + y) for x, y in zip(self.control_lb, self.control_ub)]

            #calculate weight
            weight = 60*casadi.exp(-10*(dt*k-self.t)**2) #gamma should increase as the flight duration decreases
             
            # Integrate till the end of the interval
            Xnext = self.dyn_fn(Xk, Uk,auxvar_value)
            Ck = weight*self.tra_cost_fn(Xk, auxvar_value) + self.path_cost_fn(Xk, auxvar_value)\
                +self.thrust_cost_fn(Uk, auxvar_value) + 1*dot(Uk-Ulast,Uk-Ulast)
            J = J + Ck

            # New NLP variable for state at end of interval
            Xk = MX.sym('X_' + str(k + 1), self.n_state)
            w += [Xk]
            lbw += self.state_lb
            ubw += self.state_ub
            w0 += [0.5 * (x + y) for x, y in zip(self.state_lb, self.state_ub)]
            Ulast = Uk

            # Add equality constraint
            g += [Xnext - Xk]
            lbg += self.n_state * [0]
            ubg += self.n_state * [0]

        # Adding the final cost
        J = J + self.final_cost_fn(Xk, auxvar_value)

        # Create an NLP solver and solve it
        opts = {'ipopt.print_level': print_level, 'ipopt.sb': 'yes', 'print_time': print_level}
        prob = {'f': J, 'x': vertcat(*w), 'g': vertcat(*g)}
        solver = nlpsol('solver', 'ipopt', prob, opts)
        # Solve the NLP
        sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
        w_opt = sol['x'].full().flatten()

        # take the optimal control and state
        sol_traj = numpy.concatenate((w_opt, self.n_control * [0]))
        sol_traj = numpy.reshape(sol_traj, (-1, self.n_state + self.n_control))
        state_traj_opt = sol_traj[:, 0:self.n_state]
        control_traj_opt = numpy.delete(sol_traj[:, self.n_state:], -1, 0)
        time = numpy.array([k for k in range(horizon + 1)])

        # Compute the costates using two options
        if costate_option == 0:
            # Default option, which directly obtains the costates from the NLP solver
            costate_traj_opt = numpy.reshape(sol['lam_g'].full().flatten(), (-1, self.n_state))
        else:
            # Another option, which solve the costates by the Pontryagin's Maximum Principle
            # The variable name is consistent with the notations used in the PDP paper
            dfx_fun = casadi.Function('dfx', [self.state, self.control, self.auxvar], [jacobian(self.dyn, self.state)])
            dhx_fun = casadi.Function('dhx', [self.state, self.auxvar], [jacobian(self.final_cost, self.state)])
            dcx_fun = casadi.Function('dcx', [self.state, self.control, self.auxvar],
                                      [jacobian(self.path_cost, self.state)])
            costate_traj_opt = numpy.zeros((horizon, self.n_state))
            costate_traj_opt[-1, :] = dhx_fun(state_traj_opt[-1, :], auxvar_value)
            for k in range(horizon - 1, 0, -1):
                costate_traj_opt[k - 1, :] = dcx_fun(state_traj_opt[k, :], control_traj_opt[k, :],
                                                     auxvar_value).full() + numpy.dot(
                    numpy.transpose(dfx_fun(state_traj_opt[k, :], control_traj_opt[k, :], auxvar_value).full()),
                    costate_traj_opt[k, :])

        # output
        opt_sol = {"state_traj_opt": state_traj_opt,
                   "control_traj_opt": control_traj_opt,
                   "costate_traj_opt": costate_traj_opt,
                   'auxvar_value': auxvar_value,
                   "time": time,
                   "horizon": horizon,
                   "cost": sol['f'].full()}

        return opt_sol

    # def diffPMP(self):
    #     assert hasattr(self, 'state'), "Define the state variable first!"
    #     assert hasattr(self, 'control'), "Define the control variable first!"
    #     assert hasattr(self, 'dyn'), "Define the system dynamics first!"
    #     assert hasattr(self, 'path_cost'), "Define the running cost/reward function first!"
    #     assert hasattr(self, 'final_cost'), "Define the final cost/reward function first!"

    #     # Define the Hamiltonian function
    #     self.costate = casadi.SX.sym('lambda', self.state.numel())
    #     self.path_Hamil = self.path_cost + dot(self.dyn, self.costate)  # path Hamiltonian
    #     self.final_Hamil = self.final_cost  # final Hamiltonian

    #     # Differentiating dynamics; notations here are consistent with the PDP paper
    #     self.dfx = jacobian(self.dyn, self.state)
    #     self.dfx_fn = casadi.Function('dfx', [self.state, self.control, self.auxvar], [self.dfx])
    #     self.dfu = jacobian(self.dyn, self.control)
    #     self.dfu_fn = casadi.Function('dfu', [self.state, self.control, self.auxvar], [self.dfu])
    #     self.dfe = jacobian(self.dyn, self.auxvar)
    #     self.dfe_fn = casadi.Function('dfe', [self.state, self.control, self.auxvar], [self.dfe])

    #     # First-order derivative of path Hamiltonian
    #     self.dHx = jacobian(self.path_Hamil, self.state).T
    #     self.dHx_fn = casadi.Function('dHx', [self.state, self.control, self.costate, self.auxvar], [self.dHx])
    #     self.dHu = jacobian(self.path_Hamil, self.control).T
    #     self.dHu_fn = casadi.Function('dHu', [self.state, self.control, self.costate, self.auxvar], [self.dHu])

    #     # Second-order derivative of path Hamiltonian
    #     self.ddHxx = jacobian(self.dHx, self.state)
    #     self.ddHxx_fn = casadi.Function('ddHxx', [self.state, self.control, self.costate, self.auxvar], [self.ddHxx])
    #     self.ddHxu = jacobian(self.dHx, self.control)
    #     self.ddHxu_fn = casadi.Function('ddHxu', [self.state, self.control, self.costate, self.auxvar], [self.ddHxu])
    #     self.ddHxe = jacobian(self.dHx, self.auxvar)
    #     self.ddHxe_fn = casadi.Function('ddHxe', [self.state, self.control, self.costate, self.auxvar], [self.ddHxe])
    #     self.ddHux = jacobian(self.dHu, self.state)
    #     self.ddHux_fn = casadi.Function('ddHux', [self.state, self.control, self.costate, self.auxvar], [self.ddHux])
    #     self.ddHuu = jacobian(self.dHu, self.control)
    #     self.ddHuu_fn = casadi.Function('ddHuu', [self.state, self.control, self.costate, self.auxvar], [self.ddHuu])
    #     self.ddHue = jacobian(self.dHu, self.auxvar)
    #     self.ddHue_fn = casadi.Function('ddHue', [self.state, self.control, self.costate, self.auxvar], [self.ddHue])

    #     # First-order derivative of final Hamiltonian
    #     self.dhx = jacobian(self.final_Hamil, self.state).T
    #     self.dhx_fn = casadi.Function('dhx', [self.state, self.auxvar], [self.dhx])

    #     # second order differential of path Hamiltonian
    #     self.ddhxx = jacobian(self.dhx, self.state)
    #     self.ddhxx_fn = casadi.Function('ddhxx', [self.state, self.auxvar], [self.ddhxx])
    #     self.ddhxe = jacobian(self.dhx, self.auxvar)
    #     self.ddhxe_fn = casadi.Function('ddhxe', [self.state, self.auxvar], [self.ddhxe])

    # def getAuxSys(self, state_traj_opt, control_traj_opt, costate_traj_opt, auxvar_value=1):
    #     statement = [hasattr(self, 'dfx_fn'), hasattr(self, 'dfu_fn'), hasattr(self, 'dfe_fn'),
    #                  hasattr(self, 'ddHxx_fn'), \
    #                  hasattr(self, 'ddHxu_fn'), hasattr(self, 'ddHxe_fn'), hasattr(self, 'ddHux_fn'),
    #                  hasattr(self, 'ddHuu_fn'), \
    #                  hasattr(self, 'ddHue_fn'), hasattr(self, 'ddhxx_fn'), hasattr(self, 'ddhxe_fn'), ]
    #     if not all(statement):
    #         self.diffPMP()

    #     # Initialize the coefficient matrices of the auxiliary control system: note that all the notations used here are
    #     # consistent with the notations defined in the PDP paper.
    #     dynF, dynG, dynE = [], [], []
    #     matHxx, matHxu, matHxe, matHux, matHuu, matHue, mathxx, mathxe = [], [], [], [], [], [], [], []

    #     # Solve the above coefficient matrices
    #     for t in range(numpy.size(control_traj_opt, 0)):
    #         curr_x = state_traj_opt[t, :]
    #         curr_u = control_traj_opt[t, :]
    #         next_lambda = costate_traj_opt[t, :]
    #         dynF += [self.dfx_fn(curr_x, curr_u, auxvar_value).full()]
    #         dynG += [self.dfu_fn(curr_x, curr_u, auxvar_value).full()]
    #         dynE += [self.dfe_fn(curr_x, curr_u, auxvar_value).full()]
    #         matHxx += [self.ddHxx_fn(curr_x, curr_u, next_lambda, auxvar_value).full()]
    #         matHxu += [self.ddHxu_fn(curr_x, curr_u, next_lambda, auxvar_value).full()]
    #         matHxe += [self.ddHxe_fn(curr_x, curr_u, next_lambda, auxvar_value).full()]
    #         matHux += [self.ddHux_fn(curr_x, curr_u, next_lambda, auxvar_value).full()]
    #         matHuu += [self.ddHuu_fn(curr_x, curr_u, next_lambda, auxvar_value).full()]
    #         matHue += [self.ddHue_fn(curr_x, curr_u, next_lambda, auxvar_value).full()]
    #     mathxx = [self.ddhxx_fn(state_traj_opt[-1, :], auxvar_value).full()]
    #     mathxe = [self.ddhxe_fn(state_traj_opt[-1, :], auxvar_value).full()]

    #     auxSys = {"dynF": dynF,
    #               "dynG": dynG,
    #               "dynE": dynE,
    #               "Hxx": matHxx,
    #               "Hxu": matHxu,
    #               "Hxe": matHxe,
    #               "Hux": matHux,
    #               "Huu": matHuu,
    #               "Hue": matHue,
    #               "hxx": mathxx,
    #               "hxe": mathxe}
    #     return auxSys