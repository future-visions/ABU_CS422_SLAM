import numpy as np

#x: initial state
#u: external input
#z: measurement
#F: next state matrix
#P: initial variance
#R: Measurement variance
#H: Measurement function matrix
#Q: Process variance
def predict(x, u, z, F, P, R, H=None, Q=None):

    #INITIALIZATION
    i_p=np.eye(*P.shape)
    if H is None:
        H=np.ones(x.shape)
    if Q is None:
        Q=np.zeros(P.shape)

    #PREDICTION
    x_n=np.add(np.matmul(F, x), u)
    P=np.matmul(np.matmul(F, P), F.transpose())+Q

    #MEASUREMENTS
    z_n=np.matmul(H, x_n)
    err_z_z_n=np.subtract(z,z_n)
    h_t=H.transpose()
    Knum=np.matmul(P, h_t)
    Kden=np.add(np.matmul(np.matmul(H, P), h_t),R)
    K=np.matmul(Knum,np.linalg.inv(Kden))

    #UPDATE
    x_n=np.add(x_n, np.matmul(K, err_z_z_n))
    p_n=np.matmul(np.subtract(i_p, np.matmul(K,H)),P)


    return x_n, p_n