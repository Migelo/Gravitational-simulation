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
        self.previousvelocity = [(self.vx, self.vy, self.vz), (self.vx, self.vy, self.vz)]

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

    def energycalc(self, object2):
        self.energy = self.mass * ((0.5 * (self.vx ** 2 + self.vy ** 2 + self.vz ** 2))\
                                       - (object2.mass / np.sqrt((object2.z - self.z) ** 2 + (object2.y - self.y) ** 2 + (object2.x - self.x) ** 2)))


    def info(self):
        print(f'{self.name} Position = {self.x},{self.y},{self.z} \nVelocity = {self.vx},{self.vy},{self.vz}')


def main():
    #defining each object
    Sun = objec((0, 0, 0), 1, "Sun", "#d40404")
    Earth = objec((1, 0, 0), 3.00348959632e-6, "Earth", "#0985eb", 0, 0, 1)

    objects = [Sun, Earth]

    time = np.arange(0, 100*2*np.pi, 2*np.pi/365)

    day = time[1]

    trace = []

    energy = []
    en = 0

    for i, _ in tqdm(enumerate(time), total=time.size):
        for obj in objects:
            obj.movemement(objects, day)
            for body in objects:
                if body != obj:
                    obj.energycalc(body)
            en += obj.energy
        energy.append(en)

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
    plt.xlabel("Time")
    plt.ylabel("Energy")
    plt.plot(energy)
    plt.show()

main()
