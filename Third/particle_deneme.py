import numpy as np
import particle_filter as pf
import matplotlib.pyplot as plt

def move(p, r, d, trv, rv, ms):
    try:
        nr, _= p.shape
    except:
        q,= p.shape
        p=np.reshape(p, (1, q))
        nr, _ = p.shape
    for i in range(nr):
        r_n=(r+p[i,2]+np.sqrt(rv)*np.random.rand())%(2*np.pi)
        d_n=(d+np.sqrt(trv)*np.random.rand())
        x_n=(p[i,0]+np.cos(r_n)*d_n)%ms
        y_n=(p[i,1]+np.sin(r_n)*d_n)%ms
        p[i]=np.array([x_n, y_n, r_n])
    return p

def measure(r, l, mv):
    p=r[:2]
    nl,_=l.shape
    v=np.ones((nl,1))*p
    d2=l-v
    d=np.sqrt(np.square(d2[:,0])+np.square(d2[:,1]))+np.sqrt(mv)*np.random.rand()
    return d

N=100
mapSize=100
trv=1
rv=1
mv=10
d=5
r=0.05

p=np.zeros((N,3))
p[:,0]=np.random.rand(N)*mapSize
p[:,1]=np.random.rand(N)*mapSize
p[:,2]=np.random.rand(N)*2*np.pi

landmarks=np.array([[10,10],[90,90],[10,90],[90,10]])
robot=np.array([40,50, np.random.rand()*2*np.pi])
prediction=np.mean(p, axis=0)

plt.plot(landmarks[:,0], landmarks[:,1], linestyle="None", marker="o")
lineR,=plt.plot(robot[0], robot[1], linestyle="None", marker="P")
linePar,=plt.plot(p[:,0], p[:,1],linestyle="None", marker=".", markersize=1)
linePre,=plt.plot(prediction[0], prediction[1], linestyle="None", marker="+")
plt.xlim(0, mapSize)
plt.ylim(0, mapSize)
plt.legend(["Landmarks", "Robot", "Particles", "Prediction"], loc=9)
plt.title("State: %d"%(0))
plt.savefig("%d.png"%(0))

for q in range(10):
    d_n = d+np.abs(np.sqrt(1)*np.random.rand())
    robot = move(robot, r, d_n,trv,rv,mapSize).reshape((3,))
    z=measure(robot, landmarks, mv)
    p=move(p, r, d_n, trv, rv, mapSize)
    p, prediction = pf.particleFilter(p, z, mv, landmarks)
    print(q)
    """
    plt.pause(0.1)
    lineR.set_data(robot[0], robot[1])
    linePar.set_data(p[:,0], p[:,1])
    linePre.set_data(prediction[0], prediction[1])
    """
    plt.cla()
    plt.plot(landmarks[:, 0], landmarks[:, 1], linestyle="None", marker="o")
    lineR, = plt.plot(robot[0], robot[1], linestyle="None", marker="P")
    linePar, = plt.plot(p[:, 0], p[:, 1], linestyle="None", marker=".", markersize=1)
    linePre, = plt.plot(prediction[0], prediction[1], linestyle="None", marker="+")
    plt.xlim(0,mapSize)
    plt.ylim(0,mapSize)
    plt.legend(["Landmarks", "Robot", "Particles", "Prediction"], loc=9)
    plt.title("State: %d" % (q+1))
    plt.savefig("%d.png" % (q+1))

print("Done")
#plt.show()