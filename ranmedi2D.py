# ranmedi - generating 2D random field from PSDF
# Contributing author: Joerg Buchwald
# This software is distributed under the GNU General Public License

import numpy as np
import ranmedi
import matplotlib.pyplot as plt

class Ranmedi2D(ranmedi.Ranmedi):
    def __init__(self, xi, eps, lx=512, lz=512, kappa=0.1, seed=42, meditype="gaussian"):
        self.xi = xi
        self.eps = eps
        self.kappa = kappa
        self.lx = lx
        self.lz = lz
        np.random.seed(seed)
        self.psdf = np.zeros((lx,lz), dtype=np.complex128)
        if meditype == "gaussian":
            self.gaussian()
            self.label = f"{meditype}, xi={xi}, eps={eps}"
        elif meditype == "exponential":
            self.exponential()
            self.label = f"{meditype}, xi={xi}, eps={eps}"
        elif meditype == "vkarman":
            self.vkarman()
            self.label = f"{meditype}, xi={xi}, eps={eps}, kappa={kappa}"
        self.ranmedi = self.getfield()
    def plot(self):
        im=plt.imshow(self.ranmedi, cmap=plt.cm.jet)
        plt.colorbar(im)
        plt.title(self.label)
        plt.show()
    def krsq(self,i,j):
        return (i*2*np.pi/self.lx)**2+(j*2*np.pi/self.lz)**2
    def getfield(self):
        return np.real((np.fft.ifft2(self.psdf)))
    def fluk1(self, psdfi):
        return np.sqrt(psdfi) * np.exp(1.j * self.random)
    def fluk2(self,psdfi):
        return -np.conjugate(np.sqrt(psdfi) * np.exp(1.j*self.random))

    def gaussian(self):
        for i in range(0, int(self.lx/2)):
            for j in range(0, int(self.lz/2)):
                psdfi = self.eps**2*np.pi*(self.xi)**2 * np.exp(-self.krsq(i,j) * (self.xi)**2/4)
                self.psdf[i,j] = self.fluk1(psdfi)
                self.psdf[i,self.lz-j-1] = self.fluk2(psdfi)
                self.psdf[self.lx-i-1,j] = self.fluk2(psdfi)

    def exponential(self):
        for i in range(0, int(self.lx/2)):
            for j in range(0, int(self.lz/2)):
                psdfi = self.eps**2 * 2 * np.pi*(self.xi)**2 / (1.+self.krsq(i,j) * self.xi**2)**1.5
                self.psdf[i,j] = self.fluk1(psdfi)
                self.psdf[i,self.lz-j-1] = self.fluk2(psdfi)
                self.psdf[self.lx-i-1,j] = self.fluk2(psdfi)

    def vkarman(self):
        for i in range(0, int(self.lx/2)):
            for j in range(0, int(self.lz/2)):
                psdfi = self.eps**2 * 2*np.pi * (self.xi)**2 / (1.+self.krsq(i,j) * self.xi**2)**(1+self.kappa)
                self.psdf[i,j] = self.fluk1(psdfi)
                self.psdf[i,self.lz-j-1] = self.fluk2(psdfi)
                self.psdf[self.lx-i-1,j] = self.fluk2(psdfi)

if __name__=='__main__':
    rm = Ranmedi2D(15, 20, lx=32, lz=32)
    rm.plot()
    rm = Ranmedi2D(15, 20, meditype="exponential")
    rm.plot()
    rm = Ranmedi2D(15, 20, meditype="vkarman")
    rm.plot()
