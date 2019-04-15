import numpy as np

#x: initial state
#u: external input
#z: measurement
#F: next state matrix
#P: initial variance
#R: Measurement variance
#g: Confidence level for validation gate
#H: Measurement function matrix
#Q: Process variance
def predict(x, u, z, F, P, R, g, H=None, Q=None, round=False):
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
    Kden=np.add(np.matmul(np.matmul(H, P), h_t),R) #S matrix
    S_inv=np.linalg.inv(Kden)
    K=np.matmul(Knum,S_inv) # Filter Gain W matrix

    #UPDATE
    x_n=np.add(x_n, np.matmul(K, err_z_z_n))
    p_n=np.matmul(np.subtract(i_p, np.matmul(H, K)),P)

    #P ESTIMATION CHANGES IN EKF FOR NUMERICAL ROUNDING PROBLEMS
    i_wh=np.subtract(i_p, np.matmul(K, H))
    i_wh_t=i_wh.transpose()
    wrwt=np.matmul(np.matmul(K,R),K.transpose())
    P_N=np.matmul(np.matmul(np.matmul(i_wh, P),i_wh_t),wrwt)

    #print(err_z_z_n.shape, S_inv.shape)
    e_sq = np.matmul(np.matmul(err_z_z_n, S_inv), err_z_z_n.transpose()) # EXTENDED KALMAN FILTER FEATURE
    if not e_sq <= g ** 2: # VALIDATION GATE
        raise Exception('VALIDATION GATE: MEASUREMENT EXCEEDS EXPECTED LEVELS')

    if round:
        return x_n, P_N
    return x_n, p_n