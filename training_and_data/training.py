def training(t, observations, rewards, future_rewards, true_policies, actions, k):
    ### Loss functions useds
    def mse(a,b):
        return (a-b)**2
    def centropy(a,b):
        return np.sum(a * np.log(b))
    
    """takes the obs up to time t"""
    """At each subsequent time state upTo k..."""
    """predict pi and v"""
    """Take real action, and predict r"""
    """compare these to true pis, true future_rewards and true r"""
    
    o = observations[:t] #all observations up to time t
    s = representation(o) #first representation of state at time t
    loss_p = 0
    loss_v = 0
    loss_r = 0
    for i in range(k):
        ##
        true_p = true_policies[t+k]
        true_fr = future_rewards[t+k]
        p, v = predict_P_and_V(s)
        loss_p += centropy(p,true_p)
        loss_v = mse(true_fr, v)
        s, r = dynamics(s, actions[t+k])
        loss_r += mse(r, rewards[t+k])