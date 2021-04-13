import numpy as np
import time
from scipy import interpolate
import ranmedi1D
import ranmedi2D
import ranmedi3D
import GPy as gpy
import vtuIO

class RanmediVTU(object):
    def __init__(self, vtufile, xi=250, eps=20, lx=512, ly=512, lz=512, kappa=0.1, seed=42, meditype="gaussian", dim=2):
        self.dim = dim
        self.lx = lx
        self.ly= ly
        self.lz = lz
        print(f"Gerating random field for {dim} dimensions")
        if dim == 1:
            self.rm = ranmedi1D.Ranmedi1D(xi,eps,lx=lx,kappa=0.1, seed=42, meditype="gaussian")
        elif dim == 2:
            self.rm = ranmedi2D.Ranmedi2D(xi,eps,lx=lx, lz=ly, kappa=0.1, seed=42, meditype="gaussian")
        elif dim == 3:
            self.rm = ranmedi3D.Ranmedi3D(xi,eps,lx=lx, ly=ly, lz=lz, kappa=0.1, seed=42, meditype="gaussian")
        print("reading in VTU file")
        self.vtufile = vtuIO.VTUIO(vtufile)
        self.points = self.vtufile.points
        if dim == 1:
            self.X = np.zeros((lx,1))
            self.y = np.zeros((lx,2))
            indx = 0
            for i in range(lx):
                self.X[indx,0] = i
                self.y[indx,0] = self.rm.ranmedi[i,j]
                indx += 1
        elif dim == 2:
            self.X = np.zeros((lx*ly,2))
            self.y = np.zeros((lx*ly,2))
            indx = 0
            for i in range(lx):
                for j in range(ly):
                    self.X[indx,0] = i
                    self.X[indx,1] = j
                    self.y[indx,0] = self.rm.ranmedi[i,j]
                    indx += 1

        elif dim == 3:
            self.X = np.zeros((lx*ly*lz,3))
            self.y = np.zeros((lx*ly*lz,2))
            indx = 0
            for i in range(lx):
                for j in range(ly):
                    for k in range(lz):
                        self.X[indx,0] = i
                        self.X[indx,1] = j
                        self.X[indx,2] = k
                        self.y[indx,0] = self.rm.ranmedi[i,j]
                        indx += 1

        if dim ==2:
            self.field = self.scipyinterpd2d()
#        m = self.createGP()
#        self.field = self.runGP(m)
    def scipyinterpd2d(self):
        start = time.time()
        field = np.zeros(len(self.points[:,0]))
        x = [i for i in range(self.lx)]
        y = [i for i in range(self.ly)]
        xx, yy = np.meshgrid(x,y)
        f = interpolate.interp2d(x, y, self.rm.ranmedi, kind='cubic')
        x_offset = np.min(self.points[:,0])
        x_max = np.max(self.points[:,0])
        if (self.dim == 2):
            y_offset = np.min(self.points[:,1])
            y_max = np.max(self.points[:,1])
        X_predict = self.points
        X_predict[:,0] = self.points[:,0] - x_offset
        X_predict[:,0] = (X_predict[:,0] * self.lx / (x_max-x_offset))
        if (self.dim == 2):
            X_predict[:,1] = self.points[:,1] - y_offset
            X_predict[:,1] = (X_predict[:,1] * self.ly / (y_max-y_offset))
        for i in range(len(self.points[:,0])):
            field[i] = f(X_predict[i,0], X_predict[i,1])
        stop = time.time()
        diff = stop-start
        print(f"interpolation took {diff}s")
        return field
    def createGP(self):
        print("Generating gaussian proxy model for interpolation")
        start = time.time()
        ker = gpy.kern.Matern52(self.dim,ARD=True)
        self.X[:,0] = self.X[:,0]/self.lx - 0.5
        if (self.dim == 2) or (self.dim == 3):
            self.X[:,1] = self.X[:,1]/self.ly - 0.5
        if self.dim == 3:
            self.X[:,2] = self.X[:,2]/self.lz - 0.5
        m = gpy.models.GPRegression(self.X, self.y, ker, noise_var=1.e-3)
        m.optimize(optimizer='lbfgs', messages=True, max_iters = 100)
        stop = time.time()
        diff = stop-start
        print(f"proxy building took {diff}s")
        return m
    def runGP(self, m, scalex=1, scaley=1):
        x_offset = np.min(self.points[:,0])
        x_max = np.max(self.points[:,0])
        if (self.dim == 2) or (self.dim == 3):
            y_offset = np.min(self.points[:,1])
            y_max = np.max(self.points[:,1])
        if self.dim == 3:
            z_offset = np.min(self.points[:,2])
            z_max = np.max(self.points[:,2])
        X_predict = self.points
        X_predict[:,0] = self.points[:,0] - x_offset
        X_predict[:,0] = (X_predict[:,0] / (x_max-x_offset)) - 0.5
        if (self.dim == 2) or (self.dim == 3):
            X_predict[:,1] = self.points[:,1] - y_offset
            X_predict[:,1] = (X_predict[:,1]  / (y_max-y_offset)) - 0.5
        if self.dim == 3:
            X_predict[:,2] = self.points[:,2] - z_offset
            X_predict[:,2] = (X_predict[:,2] / (z_max-z_offset)) - 0.5
        y_predict, _ = m.predict(X_predict)
        return y_predict

    def writefield(self,fieldname, ofilename):
        print("writing VTU file")
        self.vtufile.writeField(self.field, fieldname, ofilename)

if __name__== '__main__':
    rm = RanmediVTU("risk.vtu")
    rm.writefield("gaussian_noise", "out.vtu")
