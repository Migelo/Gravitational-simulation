import numpy as np
import plotly.graph_objs as px
import plotly.express as ex
import matplotlib.pyplot as plt

import plotly.offline
from tqdm import tqdm

class objec:
    def __init__(self, pos, mass, name, colour, vx=0, vy=0, vz=0):
        self.colour = colour
        self.x, self.y, self.z = pos
        self.mass = mass
        self.vx = vx
        self.vy = vy
        self.vz = vz
        self.name = name
        self.pos = []

    def movemement(self, objects, day):
        fx = fy = fz = 0
        for body in objects:
            if body != self:
                if (self.x, self.y, self.z) == (body.x, body.y, body.z):
                    print(self.name, body.name)
                    raise ValueError("collison")

                dx = self.x -body.x
                dy = self.y -body.y
                dz = self.z -body.z

                d = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
                dxy = np.sqrt(dx ** 2 + dy ** 2)

                F = -self.mass * body.mass / (d ** 2)

                theta = np.arccos(dz / d)
                cosphi = dx /dxy
                sinphi = dy / dxy

                f1 = F * np.sin(theta) * cosphi
                f2 = F * np.sin(theta) * sinphi
                f3 = F * np.cos(theta)

                fx += f1
                fy += f2
                fz += f3

        self.vx += fx / self.mass * day
        self.vy += fy / self.mass * day
        self.vz += fz / self.mass * day

        self.x += self.vx * day
        self.y += self.vy * day
        self.z += self.vz * day

        self.pos.append((self.x, self.y, self.z))

    def info(self):
        print(f'{self.name} Position = {self.x},{self.y},{self.z} \nVelocity = {self.vx},{self.vy},{self.vz} \n')


def main():
    #defining each object

    Sun = objec((0, 0, 0), 1, "Sun", "#d40404")
    Mercury = objec((0.3598099242, 0.039723385, 0), 1.65962e-7, "Mercury", "#66696e", 0, 0, 1.611)
    Venus = objec((0.72246, 0.027754, 0), 2.080442433e-6, "Venus", "#eb2409", 0, 0, 1.174)
    Earth = objec((0.99961, 0.02792163872, 0), 3.00348959632e-6, "Earth", "#0985eb", 0, 0, 1)
    Mars = objec((1.523329, 0.0452114, 0), 3.2314228261e-7, "Mars", "#eb7215", 0, 0, 0.802)
    Jupiter = objec((5.20292, 0.027242719, 0), 9.545098039e-4, "Jupiter", "#735020", 0, 0, 0.434)
    Saturn = objec((9.53782, 0.1498320, 0), 2.858019105e-4, "Saturn", "#f5c584", 0, 0, 0.323)
    Uranus = objec((19.17707, 0.334737, 0), 4.364957265e-5, "Uranus", "#1385d6", 0, 0, 0.228)
    Neptune = objec((30.057756, 0.36724304, 0), 5.149874309e-5, "Neptune", "#0e3bcc", 0, 0, 0.182)


    objects = [Sun, Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune]
    #objects = [Sun, Earth, Mars]


    time = np.arange(0, 1 * 2 *np.pi, 2 * np.pi/365)

    day = time[1]
    trace = []

    for i, _ in tqdm(enumerate(time), total=time.size):
        for obj in objects:
            obj.movemement(objects, day)

    for obj in objects:
        obj.pos = np.array(obj.pos)
        trace.append(px.Scatter3d(
                x = obj.pos[:, 0],
                y = obj.pos[:, 1],
                z = obj.pos[:, 2],
                mode='lines',
                line=dict(
                color=obj.colour,
                width=15
                ),
                ))

    fig = px.Figure(data=trace)
    plotly.offline.plot(fig)
    plt.show()


main()