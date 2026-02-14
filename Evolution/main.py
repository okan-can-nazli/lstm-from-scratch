import pygame
import sys
import random
import math

# --- Config ---
WIDTH, HEIGHT = 1000, 700
FPS = 60
BACKGROUND = (20, 20, 20)

# Colors
PREY_COLOR = (50, 200, 50)     # Green
PRED_COLOR = (200, 50, 50)     # Red
PLANT_COLOR = (0, 255, 100)    # Bright Green
MEAT_COLOR = (150, 0, 0)       # Dark Red (Dead body)

# Settings
PLANT_GROWTH_RATE = 2  # % chance per frame a plant spreads
MAX_PLANTS = 150       # Don't let screen get too crowded

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
font = pygame.font.SysFont("Arial", 16)

class Agent:
    def __init__(self, x, y, species, dna=None):
        self.x, self.y = x, y
        self.species = species 
        self.energy = 100 if species == "prey" else 150
        self.alive = True
        
        # --- DNA ---
        if dna:
            self.speed = dna['speed'] + random.uniform(-0.5, 0.5)
            self.vision = dna['vision'] + random.uniform(-10, 10)
            self.size = dna['size'] + random.uniform(-1, 1)
        else:
            self.speed = random.uniform(2, 5)
            self.vision = random.uniform(50, 150)
            self.size = random.uniform(10, 20)

        # Limits
        self.speed = max(1, min(7, self.speed))
        self.vision = max(30, min(300, self.vision))
        self.size = max(5, min(40, self.size))

        self.rect = pygame.Rect(x, y, self.size, self.size)
        self.vel_x = random.choice([-1, 1])
        self.vel_y = random.choice([-1, 1])

    def update(self, plants, meat_list, agents):
        # Energy Cost
        cost = (self.speed ** 1.6) * 0.05 + (self.size * 0.01)
        if self.species == "predator": cost *= 1.2
        self.energy -= cost

        if self.energy <= 0:
            self.alive = False
            return None # Dies

        # --- SENSORY & MOVEMENT ---
        # 1. Prey looks for Plants
        target = None
        min_dist = self.vision

        cx, cy = self.rect.centerx, self.rect.centery

        if self.species == "prey":
            # Find closest plant
            for p in plants:
                d = math.hypot(p.centerx - cx, p.centery - cy)
                if d < min_dist:
                    min_dist = d
                    target = p
        
        elif self.species == "predator":
            # Find closest Prey OR Meat
            # Check Meat first (It's free food!)
            for m in meat_list:
                d = math.hypot(m.centerx - cx, m.centery - cy)
                if d < min_dist:
                    min_dist = d
                    target = m
            
            # If no meat, check for live prey
            if not target:
                for a in agents:
                    if a.species == "prey" and a.alive:
                        d = math.hypot(a.x - cx, a.y - cy)
                        if d < min_dist:
                            min_dist = d
                            target = a

        # Move logic
        if target:
            # Move towards target
            dx = (target.x if hasattr(target, 'x') else target.centerx) - self.x
            dy = (target.y if hasattr(target, 'y') else target.centery) - self.y
            dist = math.hypot(dx, dy)
            if dist != 0:
                self.vel_x = (dx/dist) * self.speed
                self.vel_y = (dy/dist) * self.speed
        else:
            # Wander
            if random.random() < 0.05:
                self.vel_x = random.choice([-self.speed, self.speed])
                self.vel_y = random.choice([-self.speed, self.speed])

        self.x += self.vel_x
        self.y += self.vel_y
        
        # Clamp
        self.x = max(0, min(WIDTH-self.size, self.x))
        self.y = max(0, min(HEIGHT-self.size, self.y))
        self.rect.topleft = (self.x, self.y)

        # --- REPRODUCTION ---
        if self.energy > 200:
            self.energy -= 100
            dna = {'speed': self.speed, 'vision': self.vision, 'size': self.size}
            return Agent(self.x, self.y, self.species, dna)
        return None

    def draw(self, surface):
        color = PREY_COLOR if self.species == "prey" else PRED_COLOR
        # Dim color if starving
        if self.energy < 30: 
            color = (max(0, color[0]-50), max(0, color[1]-50), max(0, color[2]-50))
        pygame.draw.rect(surface, color, self.rect)

# --- Simulation Setup ---
agents = []
plant_list = []
meat_list = []

# Initial Spawn
for _ in range(20): agents.append(Agent(random.randint(0, WIDTH), random.randint(0, HEIGHT), "prey"))
for _ in range(4): agents.append(Agent(random.randint(0, WIDTH), random.randint(0, HEIGHT), "predator"))
for _ in range(50): plant_list.append(pygame.Rect(random.randint(0, WIDTH), random.randint(0, HEIGHT), 8, 8))

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT: running = False

    screen.fill(BACKGROUND)

    # --- 1. ORGANIC PLANT GROWTH ---
    # Chance for existing plants to spread seeds nearby
    if len(plant_list) < MAX_PLANTS:
        if len(plant_list) > 0 and random.randint(0, 100) < PLANT_GROWTH_RATE:
            parent = random.choice(plant_list)
            # Spawn near parent (creates clusters)
            nx = parent.x + random.randint(-30, 30)
            ny = parent.y + random.randint(-30, 30)
            # Keep on screen
            nx = max(0, min(WIDTH-10, nx))
            ny = max(0, min(HEIGHT-10, ny))
            plant_list.append(pygame.Rect(nx, ny, 8, 8))
        elif len(plant_list) == 0:
            # If all plants eaten, spawn random one to restart life
            plant_list.append(pygame.Rect(random.randint(0, WIDTH), random.randint(0, HEIGHT), 8, 8))

    # --- Draw Environment ---
    for p in plant_list: pygame.draw.rect(screen, PLANT_COLOR, p)
    for m in meat_list: pygame.draw.rect(screen, MEAT_COLOR, m)

    # --- Update Agents ---
    new_babies = []
    
    for agent in agents:
        baby = agent.update(plant_list, meat_list, agents)
        if baby: new_babies.append(baby)
        
        if agent.alive:
            # COLLISION / EATING
            if agent.species == "prey":
                # Eat Plants
                idx = agent.rect.collidelist(plant_list)
                if idx != -1:
                    del plant_list[idx]
                    agent.energy += 40
            
            elif agent.species == "predator":
                # Eat Meat (Prioritize Scavenging)
                idx = agent.rect.collidelist(meat_list)
                if idx != -1:
                    del meat_list[idx]
                    agent.energy += 50
                else:
                    # Hunt Live Prey
                    # Simple collision check with all prey
                    for prey in agents:
                        if prey.species == "prey" and prey.alive:
                            if agent.rect.colliderect(prey.rect):
                                prey.alive = False # Kill
                                agent.energy += 80
                                # No meat created if eaten alive (devoured)
            
            agent.draw(screen)
        else:
            # AGENT DIED NATURALLY (Starvation) -> Turn into Meat
            # Add a meat block where they died
            meat_rect = pygame.Rect(agent.x, agent.y, 15, 15)
            meat_list.append(meat_rect)

    # Cleanup Lists
    agents.extend(new_babies)
    agents = [a for a in agents if a.alive]
    
    # Meat rots away over time (Simulated by random removal to keep list small)
    if len(meat_list) > 20 and random.random() < 0.05:
        del meat_list[0]

    # --- HUD ---
    stats = [
        f"Prey: {len([a for a in agents if a.species == 'prey'])}",
        f"Predators: {len([a for a in agents if a.species == 'predator'])}",
        f"Plants: {len(plant_list)}",
        f"Meat: {len(meat_list)}"
    ]
    for i, line in enumerate(stats):
        screen.blit(font.render(line, True, (200, 200, 200)), (10, 10 + i*20))

    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()
sys.exit()