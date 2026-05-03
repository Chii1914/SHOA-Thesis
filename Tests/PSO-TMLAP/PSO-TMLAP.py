import random, math

class Problem:
    def __init__(self):
        self.n_clients = 6
        self.n_hubs = 3

        # Matriz de distancias cliente-hub
        self.distancias = [
            [5, 8, 11],
            [7, 4, 10],
            [6, 6, 6],
            [9, 5, 7],
            [8, 9, 5],
            [4, 7, 12]
        ]

        # Costo fijo por abrir cada hub
        self.costs = [20, 25, 15]

        # Capacidad máxima de atención por hub
        self.capacidad = [2, 3, 2]

        # Distancia máxima tolerada cliente–hub
        self.D_max = 8

    def check(self, x):
        # Derivar hubs desde asignaciones
        hubs = []
        for _ in range(self.n_hubs):
            hubs.append(0)
        for h in x:
            hubs[h] = 1

        # Verificar capacidad y distancia máxima
        conteo = [0] * self.n_hubs
        for c in range(self.n_clients):
            h = x[c]

            # Verificar distancia máxima
            if self.distancias[c][h] > self.D_max:
                return False

            # Hub debe estar activo (ya se cumple por construcción)
            conteo[h] += 1
            if conteo[h] > self.capacidad[h]:
                return False

        return True

    def fit(self, x):
        # Derivar hubs desde asignaciones
        hubs = []
        for _ in range(self.n_hubs):
            hubs.append(0)

        for h in x:
            hubs[h] = 1

        # Calcular distancia total de asignación
        total = 0
        for c in range(self.n_clients):
            total += self.distancias[c][x[c]]

        # Agregar costo fijo de hubs activados
        for j in range(self.n_hubs):
            if hubs[j] == 1:
                total += self.costs[j]

        return total

    def keep_domain(self, v):
        probs = []
        for _ in range(self.n_hubs):
            probs.append(1 / (1 + math.exp(-(v + random.gauss(0, 0.5)))))

        total = sum(probs)
        normalized = [p / total for p in probs]

        # Muestreo proporcional
        r = random.random()
        acc = 0
        for i, p in enumerate(normalized):
            acc += p
            if r <= acc:
                return i
        return self.n_hubs - 1

class Particle:
    def __init__(self):
        self.p = Problem()
        # Creamos un atributo dimesion de valor n_clients dado desde el problema
        self.dimension = self.p.n_clients

        self.position = []
        self.velocity = []
        self.p_best = []
        # Crear una solución inicial sin validar
        for _ in range(self.dimension):
            self.position.append(random.randint(0, self.p.n_hubs - 1))
            self.velocity.append(0)
        
        self.update_p_best()

    def update_p_best(self):
        self.p_best = self.position.copy()
        
    def is_feasible(self):
        return self.p.check(self.position)

    def fitness(self):
        return self.p.fit(self.position)

    def fitness_p_best(self):
        return self.p.fit(self.p_best)

    def is_better_than_p_best(self):
        # < para problemas de minimización
        return self.fitness() < self.fitness_p_best()

    def is_better_than(self, g):
        # < para problemas de minimización
        return self.fitness_p_best() < g.fitness_p_best()

    def move(self, g, theta, alpha, beta):
        for j in range(self.dimension):
            self.velocity[j] = (self.velocity[j] * theta +
                         alpha * random.random() * (g.p_best[j] - self.position[j]) +
                         beta * random.random() * (self.p_best[j] - self.position[j]))
            self.position[j] = self.p.keep_domain(self.velocity[j])

    def copy(self, other):
        if isinstance(other, Particle):
            self.position = other.position.copy()
            self.velocity = other.velocity.copy()
            self.p_best = other.p_best.copy()

    def __str__(self):
        return f"p_best: {self.p_best}, fitness {self.fitness_p_best()}"

class PSO:
    def __init__(self):
        self.max_iter = 25
        self.n_particles = 10
        self.theta = 0.7
        self.alpha = 2
        self.beta = 2
        self.swarm = []
        self.g = None

    def random(self):
        for _ in range(self.n_particles):
            feasible = False
            while not feasible:
                p = Particle()
                feasible = p.is_feasible() 
            self.swarm.append(p)
        
        self.g = self.swarm[0]
        for i in range(1, self.n_particles):
            if self.swarm[i].is_better_than(self.g):
                self.g.copy(self.swarm[i])

        self.show_results(0)

    def evolve(self):
        t = 1
        p = Particle()
        while t <= self.max_iter:
            for i in range(1, self.n_particles):
                feasible = False
                while not feasible:
                    p.copy(self.swarm[i])                    
                    p.move(self.g, self.theta, self.alpha, self.beta)
                    feasible = p.is_feasible()

                self.swarm[i].copy(p)
                if self.swarm[i].is_better_than_p_best():
                    self.swarm[i].update_p_best()
                if self.swarm[i].is_better_than(self.g):
                    self.g.copy(self.swarm[i])

            self.show_results(t)
            t += 1
        
    def show_results(self, t):
        print(f"t: {t}, g_best: {self.g}")

    def solve(self):
        self.random()
        self.evolve()

if __name__ == "__main__":
    # Ejecutar
    PSO().solve()