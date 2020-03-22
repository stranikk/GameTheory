import random
import numpy as np
import matplotlib.pyplot as pp
from math import sin, cos, pi, sqrt
from mpl_toolkits.mplot3d import Axes3D

def example():
    print("Example input:")
    print("x = 3")
    print("y = 2")
    print("z = 1")
    print("radial_search = 0.1")
    print("n_points = 50")
    print("nsearch = 100\n")

class Sphere:
    def __init__(self, x_x, y_y, z_z):
        self.x_x = x_x
        self.y_y = y_y
        self.z_z = z_z

    def rand_coordinate(self):
        alpha = random.random() * pi
        betta = random.random() * 2 *pi
        return self.coordinate(alpha, betta)

    def coordinate(self, alpha, betta):
        x = self.x_x * sin(alpha) * cos(betta)
        y = self.y_y * sin(alpha) * sin(betta)
        z = self.z_z * cos(alpha)
        return np.array([x, y, z])

def plot_figure_and_calc_res(x,y,z,radial_search,n_points,nsearch):
    e = Sphere(x, y, z)

    points = []
    sph = pp.figure()
    base_sph = sph.add_subplot(111, projection='3d')
    r = np.linspace(0, pi, 20)
    p = np.linspace(0, 2*pi, 20)
    R, P = np.meshgrid(r, p)

    X = x * np.sin(R) * np.cos(P)
    Y = y * np.sin(R) * np.sin(P)
    Z = z * np.cos(R)

    base_sph.plot_wireframe(X, Y, Z, cmap=pp.cm.plasma) # Figure ploting 
    while len(points) != n_points:
        rand_point = e.rand_coordinate()
        normal = True
        for point in points:
            dist = np.linalg.norm(point - rand_point)
            if dist <= 2 * radial_search:
                normal = False
                break
        if not normal:
            continue

        base_sph.scatter(*rand_point, c='m') # points drow
        points.append(rand_point)

    pp.show()

    result_win = 0
    result_loss = 0

    for i in range(nsearch):
        rand_point = e.rand_coordinate()
        flag = False
        for point in points:
            dist = np.linalg.norm(point - rand_point)
            if dist <= radial_search:
                flag = True
                break

        if flag == False:
            result_loss = result_loss + 1
        else:
            result_win = result_win + 1

    print("Number of wins: ", result_win)
    print("Number of loss: ", result_loss)
    print("Percent wins: ", 100 * result_win / nsearch)
    print("Percent loss: ", 100 * result_loss / nsearch)

if __name__ == "__main__":
    example()
    x = int(input('Input The Size Of The Coordinate x: '))
    y = int(input('Input The Size Of The Coordinate y: '))
    z = int(input('Input The Size Of The Coordinate z: '))
    radial_search = float(input('Input radial search: '))
    n_points = int(input('Input numbers of distributed points on the sphere: '))
    nsearch = int(input('Input Number of search attempts: '))

    plot_figure_and_calc_res(x,y,z,radial_search,n_points,nsearch)
    


