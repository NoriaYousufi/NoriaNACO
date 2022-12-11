'''Example of the OneMax problem, using binary images.

Requirements:
    pip install ioh>=0.3.3 matplotlib
'''

import os

import numpy as np
import matplotlib.pyplot as plt
import ioh

class OneMaxImage(ioh.problem.Integer):
    def __init__(self, instance=1):
        instances = os.listdir("images")
        if instance not in range(1, len(instances) + 1):
            raise AttributeError(f"Instance: {instance} not found!")
        
        image = plt.imread(os.path.join("images", instances[instance-1]))
        rgba = np.array([0.2989, 0.5870, 0.1140, 1.0])
        grey = image.dot(rgba)
        self.image = (grey > .1).astype(int)
        self.shape = self.image.shape
        self.xopt = self.image.ravel()
        self.ax1 = None
        self.ax2 = None
        self.f = None
        super().__init__("OneMaxImage", self.image.size, instance, False)


    def evaluate(self, x: np.ndarray):
        return (x == self.xopt).sum() #OneMax; try as LeadingOnes as well.

    def show(self):
        if self.ax1 is None and self.ax2 is None:
            self.f, (self.ax1, self.ax2) = plt.subplots(1, 2)
            plt.ion()
            self.ax2.set_title("Target")
            self.ax2.imshow(self.image)
            plt.show()

        self.ax1.clear()
        self.f.suptitle(f"Evaluations: {self.state.evaluations}, y:{self.state.current_best.y}")
        self.ax1.set_title("Current best")
        self.ax1.imshow(self.state.current_best.x.reshape(self.shape))
        
        plt.draw()
        plt.pause(0.01)

    def reset(self):
        super().reset()
        self.ax1 = None
        self.ax2 = None
        self.f = None


def randomSearch(problem):
    '''Implementation of a basic random search.'''

    dim = problem.meta_data.n_variables
    x = np.random.randint(0, 2, dim)
    y = float("-inf")
    for e in range(int(1e3 * dim)):
        if e % ((dim) // 10) == 0:
            problem.show()
        
        xn = np.random.randint(0, 2, dim)
        yn = problem(xn)

        if (yn > y):
            y = yn
            x = xn.copy()
        
        if y == dim:
            break
        

if __name__ == "__main__":
    for i in range(1, 11):
        problem = OneMaxImage(i)
        randomSearch(problem)
