import signal
import sys
import time

import numpy as np
from scipy.optimize import minimize_scalar

import matplotlib.pyplot as plt
import matplotlib.path as mpath
from matplotlib.patches import Circle, Wedge, PathPatch


def signal_handler(sig, frame):
    plt.close('all')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)


class Ball:
    def __init__(self, x=0.0, y=0.0, mass=1.0, radius=0.2, v_x=0.0, v_y=0.0, omega=0.0, color="red"):
        self.color = color
        self.radius = radius
        self.mass = mass
        self.inertia = 0.4 * self.mass * self.radius**2

        # state
        self.pos = np.array([x, y])
        self.v = np.array([v_x, v_y])
        self.omega = omega

class Surface:
    def __init__(self, type="line", **params):
        self.type = type
        self.params = params

    def distance_and_normal(self, point):
        """
        assumptions 
            1. circle surface uses euclidian distance
            2. others surfaces uses vertical distance
        """
        x0, y0 = point
        dx = 0.001

        if self.type == "circle":
            cx = self.params.get("center_x", 0)
            cy = self.params.get("center_y", 0)
            r = self.params.get("radius", 10)

            vec = np.array([x0 - cx, y0 - cy])
            dist = np.linalg.norm(vec)
            normal = vec / dist
            return dist - r, normal

        if self.type == "line":
            k = self.params.get("k", 0)
            b = self.params.get("b", 0)

            # surface_y = k * x0 + b
            surface_y = self.surface_y(x0)
            distance = y0 - surface_y
            normal = np.array([-k, 1]) / np.sqrt(1 + k**2)
            return distance, normal

        if self.type == "parabola":
            a = self.params.get("a", 0.01)
            b = self.params.get("b", 0)
            c = self.params.get("c", 0)

            # surface_y = a * x0**2 + b * x0 + c
            surface_y = self.surface_y(x0)
            distance = y0 - surface_y
            dy_dx = 2 * a * x0 + b

            normal = np.array([-dy_dx, 1]) / np.sqrt(1 + dy_dx**2)
            return distance, normal

        if self.type == "sine":
            A = self.params.get("A", 5)
            freq = self.params.get("freq", 0.1)
            phase = self.params.get("phase", 0)
            offset = self.params.get("offset", 0)

            # surface_y = A * np.sin(freq*x0 + phase) + offset
            surface_y = self.surface_y(x0)
            distance = y0 - surface_y

            dy_dx = A * freq * np.cos(freq * x0 + phase)

            normal = np.array([-dy_dx, 1]) / np.sqrt(1 + dy_dx**2)
            return distance, normal

        return y0, np.array([0.0, 1.0])

    def surface_y(self, x):
        if self.type == "line":
            k = self.params.get("k", 0)
            b = self.params.get("b", 0)
            return k * x + b
            
        elif self.type == "parabola":
            a = self.params.get("a", 0.01)
            b = self.params.get("b", 0)
            c = self.params.get("c", 0)
            return a * x**2 + b * x + c
            
        elif self.type == "sine":
            A = self.params.get("A", 5)
            freq = self.params.get("freq", 0.1)
            phase = self.params.get("phase", 0)
            offset = self.params.get("offset", 0)
            return A * np.sin(freq * x + phase) + offset
        else:
            return 0

class PhysicsEngine:
    def __init__(self, gravity=[0, -9.81]):
        self.gravity = np.array(gravity)
        self.spring_constant = 100.0
        self.damping_constant = 1.0 

    def update(self, objects, surface, dt):
        for obj in objects:
            Fg = self.gravity * obj.mass
            F_rolling = -0.1 * obj.mass * abs(self.gravity[1]) * np.sign(obj.v[0])
            F_total = Fg + np.array([F_rolling, 0])

            F_collision, M_collision = self._compute_collision_force(obj, surface)
            F_total += F_collision

            M_friction = -0.1 * np.sign(obj.omega) * obj.omega
            M_total = M_friction + M_collision

            dv = F_total / obj.mass * dt
            domega = M_total / obj.inertia * dt

            obj.v += dv
            obj.pos += obj.v * dt
            obj.omega += domega * dt

    def _compute_collision_force(self, obj, surface):
        distance, normal = surface.distance_and_normal(obj.pos) 
        penetration = obj.radius - abs(distance)

        if penetration > 0:
            normal = normal if distance >= 0 else -normal
            
            F_spring = self.spring_constant * penetration * normal
            
            v_n = np.dot(obj.v, normal)
            
            F_damp = -self.damping_constant * v_n * normal
            
            t = np.array([-normal[1], normal[0]])
            v_t = np.dot(obj.v, t) + obj.omega * obj.radius
            
            mu = 0.1
            F_friction_magnitude = mu * np.linalg.norm(F_spring)
            F_friction = -F_friction_magnitude * np.sign(v_t) * t
            
            M_friction = obj.radius * np.cross(t, F_friction)
            
            F_total = F_spring + F_damp + F_friction
            return F_total, M_friction
        
        return np.array([0.0, 0.0]), 0.0

class Renderer:
    def __init__(self, xdim=1000, ydim=1000):
        self.xdim = xdim
        self.ydim = ydim
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.ax.set_xlim(-xdim//2, xdim//2)
        self.ax.set_ylim(-ydim//2, ydim//2)
        self.ax.set_aspect('equal')
        self.list_obj_patch = []
        self.surface_patch = None

    def update(self, objects, surface, dt=0.0001):
        if not self.list_obj_patch:
            for obj in objects:
                patch = Circle(obj.pos, obj.radius, color=obj.color)
                self.ax.add_patch(patch)
                self.list_obj_patch.append(patch)
        else:
            for obj, obj_to_update in zip(objects, self.list_obj_patch):
                obj_to_update.center = obj.pos

        if self.surface_patch is None:
            if surface.type == "circle":
                center_x = surface.params.get("center_x", 0)
                center_y = surface.params.get("center_y", 0)
                radius = surface.params.get("radius", 10)

                theta = np.linspace(0, 2 * np.pi, 100)
                x_surface = center_x + radius * np.cos(theta)
                y_surface = center_y + radius * np.sin(theta)                
            else:
                x_surface = np.linspace(-self.xdim//2, self.xdim//2, 1000)
                y_surface = [surface.surface_y(x) for x in x_surface]

            self.surface_patch, = self.ax.plot(x_surface, y_surface, 'k-', linewidth=2)

        plt.pause(dt)

class World:
    def __init__(self, physics, renderer):
        self.physics = physics
        self.renderer = renderer
        self.surface = None
        self.objects = []
        self.time = 0
        self.dt = 0.005

    def set_surface(self, surface):
        self.surface = surface

    def add_object(self, obj):
        self.objects.append(obj)

    def step(self, i):
        self.physics.update(self.objects, self.surface, self.dt)
        if i % 10 == 0:
            self.renderer.update(self.objects, self.surface)
        self.time += self.dt

    def run(self, steps):
        for i in range(steps):
            self.step(i)

def main():

    surface = Surface(type="line", k=0.0707, b=0.0)  # kx + b
    # surface = Surface(type="parabola", a=0.005, b=0, c=0.0)    # ax**2 + bx  +c
    # surface = Surface(type="sine", A=10, freq=0.05, phase=0, offset=0) # A sin(freq x + phase) + offset
    # surface = Surface(type="circle", center_x=0, center_y=0, radius=100)

    physics = PhysicsEngine(gravity=[0.0, -9.81])
    renderer = Renderer()

    world = World(physics, renderer)
    world.set_surface(surface)

    for y0 in map(float, range(-200, 300, 100)):
        for x0 in map(float, range(-200, 300, 100)):
            ball = Ball(x=x0, y=y0, radius=3 + 2 * np.random.rand(), 
                    v_x=2 * np.random.rand() - 1, v_y=0, 
                    omega=5 * np.random.rand() - 2.5, 
                    color=np.random.choice(['red', 'blue', 'green', 'orange', 'purple']))
            world.add_object(ball)

    world.run(10000)
    plt.show()

if __name__ == "__main__":
    main()
