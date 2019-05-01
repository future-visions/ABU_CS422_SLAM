import numpy as np

#p: particles
#z: measurements
#mv: measurement variance
#l: landmarks

def particleFilter(p, z, mv, l):
    w = mweights(p, z, l, mv)
    nr, _ = p.shape
    maxw=np.amax(w)
    p_new=np.zeros(p.shape)
    ind=np.random.randint(nr,size=1)
    b=0
    i=0
    while i<nr:
        b+=np.random.rand()*2*maxw
        while b>w[ind]:
            b-=w[ind]
            ind=(ind+1)%(nr)
        p_new[i]=p[ind]
        i+=1
    return p_new, np.mean(p, axis=0)

def normpdf(x, mu, sigma):
    u = (x-mu)*1.0/abs(sigma)
    y = (1.0/(np.sqrt(2*np.pi)*abs(sigma)))*np.exp(-u*u/2.0)
    return y

def mweights(p,z,l,mv):
    nr, _ = p.shape
    w = np.zeros((nr,1))
    i = 0
    while i < nr:
        x,y=p[i,0], p[i,1]
        pr=1
        j=0
        lnr, _=l.shape
        while j<lnr:
            err=(x-l[j,0])**2+(y-l[j,1])**2
            d=np.sqrt(err)
            pr*=normpdf(z[j],d,np.sqrt(mv))
            j+=1
        w[i]=pr
        i += 1
    return w