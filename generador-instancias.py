import math
import random

def generar_instancia_desde_hubs(n_hubs=15, D_max=10):
    """
    Genera una instancia donde los clientes se crean como "clusters" 
    alrededor de los hubs, garantizando factibilidad espacial.
    """
    
    # ==========================================
    # PARÁMETROS CONFIGURABLES
    # ==========================================
    tamano_grilla_x = 30
    tamano_grilla_y = 30
    
    costo_fijo_min = 15
    costo_fijo_max = 30
    factor_holgura = 1.2 # 20% extra de capacidad total
    
    # ¿Cuántos clientes queremos generar por cada hub? 
    # (El número real será aleatorio dentro de este rango para cada hub)
    min_clientes_por_hub = 3
    max_clientes_por_hub = 6
    
    # ==========================================
    # 1. GENERACIÓN DE HUBS
    # ==========================================
    hubs_coords = []
    costs = []
    
    for _ in range(n_hubs):
        # Ubicamos los hubs aleatoriamente en el mapa
        x_h = random.randint(0, tamano_grilla_x)
        y_h = random.randint(0, tamano_grilla_y)
        hubs_coords.append((x_h, y_h))
        
        # Asignamos su costo fijo de apertura
        costs.append(random.randint(costo_fijo_min, costo_fijo_max))

    # ==========================================
    # 2. GENERACIÓN DE CLIENTES (En función de los Hubs)
    # ==========================================
    clientes_coords = []
    
    for hub_x, hub_y in hubs_coords:
        # Decidimos cuántos clientes "pertenecerán" inicialmente a este hub
        n_clientes_cluster = random.randint(min_clientes_por_hub, max_clientes_por_hub)
        
        for _ in range(n_clientes_cluster):
            # Generamos al cliente dentro del radio D_max del hub actual
            # Usamos coordenadas polares para asegurar que caiga en el círculo
            angulo = random.uniform(0, 2 * math.pi)
            radio = random.uniform(0, D_max)
            
            x_c = round(hub_x + radio * math.cos(angulo))
            y_c = round(hub_y + radio * math.sin(angulo))
            
            clientes_coords.append((x_c, y_c))
            
    n_clients = len(clientes_coords)

    # ==========================================
    # 3. ASIGNACIÓN DE CAPACIDADES
    # ==========================================
    # Ahora que sabemos cuántos clientes hay en total, repartimos la capacidad
    capacidad = []
    capacidad_total_objetivo = math.ceil(n_clients * factor_holgura)
    capacidad_base = capacidad_total_objetivo // n_hubs
    resto_capacidad = capacidad_total_objetivo % n_hubs
    
    for i in range(n_hubs):
        cap = capacidad_base + (1 if i < resto_capacidad else 0)
        capacidad.append(cap)

    # ==========================================
    # 4. CÁLCULO DE MATRIZ DE DISTANCIAS
    # ==========================================
    distancias = []
    for c_x, c_y in clientes_coords:
        fila_distancias = []
        for h_x, h_y in hubs_coords:
            # Distancia euclidiana redondeada
            dist = round(math.hypot(c_x - h_x, c_y - h_y))
            fila_distancias.append(dist)
        distancias.append(fila_distancias)

    # ==========================================
    # 5. FORMATEO DE SALIDA
    # ==========================================
    print(f"self.n_clientes = {n_clients}")
    print(f"self.n_hubs = {n_hubs}\n")
    
    print(f"# Matriz de distancias cliente-hub ({n_clients} x {n_hubs})")
    print("self.distancias = [")
    for i, fila in enumerate(distancias):
        fila_str = "    [" + ", ".join(map(str, fila)) + "]"
        if i < len(distancias) - 1:
            print(fila_str + ",")
        else:
            print(fila_str)
    print("]\n")
    
    print("# Costo fijo por abrir cada hub")
    print(f"self.costs = {costs}\n")
    
    print(f"# Capacidad máxima por hub (total {sum(capacidad)} >= {n_clients})")
    print(f"self.capacidad = {capacidad}\n")
    
    print("# Distancia máxima tolerada")
    print(f"self.D_max = {D_max}")

# Pruébalo aquí cambiando la cantidad de hubs
if __name__ == "__main__":
    generar_instancia_desde_hubs(n_hubs=50, D_max=12)