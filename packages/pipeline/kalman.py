"""Constant-velocity wrist filter in pixels; predictions never create finger poses."""
import numpy as np

class WristKalman:
    def __init__(self,x,y,t):
        self.x=np.array([x,y,0.,0.]);self.P=np.diag([25.,25.,40000.,40000.]);self.t=t
    def forecast(self,t):
        dt=max(0.,min((t-self.t)/1000,.6))
        F=np.eye(4);F[0,2]=dt;F[1,3]=dt
        G=np.array([[dt*dt/2,0],[0,dt*dt/2],[dt,0],[0,dt]])
        return F@self.x,F@self.P@F.T+G@G.T*250000.
    def update(self,x,y,t,noise=25.):
        state,cov=self.forecast(t);H=np.eye(2,4);R=np.eye(2)*noise
        K=np.linalg.solve(H@cov@H.T+R,H@cov).T
        self.x=state+K@(np.array([x,y])-H@state)
        I=np.eye(4)-K@H;self.P=I@cov@I.T+K@R@K.T;self.t=t
    def point(self,t):
        state,cov=self.forecast(t)
        return float(state[0]),float(state[1]),float(np.sqrt(max(cov[0,0],cov[1,1])))
