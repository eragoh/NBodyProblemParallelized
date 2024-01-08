#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>
#include <random>
#include <fstream>

#define BODIES 512
#define STEPS 100000
#define TIME_STEP 1 // Time step in seconds
#define GRAVITATIONAL_CONSTANT 6.67430e-11 // Time step in seconds
#define STEPS_BETWEEN_WRITING 20000

struct alignas(64) Body {
    double x, y, z;
    double mass;
    double vx, vy, vz;
};

void add_randomized_bodies(std::vector<Body> &bodies, int N) {
    std::random_device rd;
    std::mt19937 gen(rd());

    // Set ranges for body attributes based on realistic astronomical values
    std::uniform_real_distribution<> dist_x(1.0e9, 5.0e12); // Position x
    std::uniform_real_distribution<> dist_y(-5.0e12, 5.0e12); // Position y
    std::uniform_real_distribution<> dist_z(-5.0e12, 5.0e12); // Position z
    std::uniform_real_distribution<> dist_mass(1.0e21, 1.0e28); // Mass
    std::uniform_real_distribution<> dist_vx(-5.0e4, 5.0e4); // Velocity x
    std::uniform_real_distribution<> dist_vy(-5.0e4, 5.0e4); // Velocity y
    std::uniform_real_distribution<> dist_vz(-5.0e4, 5.0e4); // Velocity z

    for (int i = 0; i < N; ++i) {
        Body new_body;
        new_body.x = dist_x(gen);
        new_body.y = dist_y(gen);
        new_body.z = dist_z(gen);
        new_body.mass = dist_mass(gen);
        new_body.vx = dist_vx(gen);
        new_body.vy = dist_vy(gen);
        new_body.vz = dist_vz(gen);

        bodies.push_back(new_body);
    }
}

void update_forces(std::vector<Body>& bodies) {
    #pragma omp for schedule(guided, 1) nowait
    for (int i = 0; i < bodies.size(); ++i) {
        double fx = 0.0, fy = 0.0, fz = 0.0;
        for (int j = 0; j < bodies.size(); ++j) {
            if (i == j) continue;
            
            double dx = bodies[j].x - bodies[i].x;
            double dy = bodies[j].y - bodies[i].y;
            double dz = bodies[j].z - bodies[i].z;
            
            double dist = std::sqrt(dx * dx + dy * dy + dz * dz);
            double force = GRAVITATIONAL_CONSTANT * bodies[i].mass * bodies[j].mass / (dist * dist * dist);
            
            fx += force * dx;
            fy += force * dy;
            fz += force * dz;
        }

        bodies[i].vx += fx / bodies[i].mass;
        bodies[i].vy += fy / bodies[i].mass;
        bodies[i].vz += fz / bodies[i].mass;
    }
}

void update_positions(std::vector<Body>& bodies) {
    #pragma omp for schedule(guided, 1) nowait
    for (int i = 0; i < bodies.size(); ++i) {
        bodies[i].x += bodies[i].vx * TIME_STEP;
        bodies[i].y += bodies[i].vy * TIME_STEP;
        bodies[i].z += bodies[i].vz * TIME_STEP;
    }
}

int main() {
    char affinity[256];
    omp_get_affinity_format(affinity, sizeof(affinity));
    printf("OpenMP  -- Max threads: %d  -- Affinity: %s\n", omp_get_max_threads(), affinity);
    printf("OpenMP  -- sizeof(Body): %d\n", sizeof(Body));
    omp_set_num_threads(16);
    
    std::vector<Body> bodies;

    bodies.push_back({0, 0, 0, 1.989e30, 0, 0, 0}); // Sun
    bodies.push_back({57.9e9, 0, 0, 3.3011e23, 0, 47.87e3, 0}); // Mercury
    bodies.push_back({108.2e9, 0, 0, 4.8675e24, 0, 35.02e3, 0}); // Venus
    bodies.push_back({149.6e9, 0, 0, 5.972e24, 0, 29.78e3, 0}); // Earth
    bodies.push_back({227.9e9, 0, 0, 6.4171e23, 0, 24.077e3, 0}); // Mars
    bodies.push_back({778.6e9, 0, 0, 1.898e27, 0, 13.07e3, 0}); // Jupiter
    bodies.push_back({1.433e12, 0, 0, 5.683e26, 0, 9.69e3, 0}); // Saturn
    bodies.push_back({2.872e12, 0, 0, 8.681e25, 0, 6.81e3, 0}); // Uranus
    // bodies.push_back({4.495e12, 0, 0, 1.024e26, 0, 5.43e3, 0}); // Neptune
    add_randomized_bodies(bodies, BODIES-8);
    
    // Create and open a new CSV file
    std::ofstream csvfile("../../visualizer/body_positions.csv");


    // Main simulation loop
    for (int step = 0; step < STEPS; ++step) {
        // Write positions to CSV file
        if (step % STEPS_BETWEEN_WRITING == 0) {
            for (const auto& body : bodies) {
                csvfile << body.x << "," << body.y << "," << body.z;
                if (&body != &bodies.back()) {
                    csvfile << ",";
                }
            }
            csvfile << "\n";
            printf("%.2f%\n", step*100.0/STEPS);
        }

        #pragma omp parallel
        {
            //printf("threads: %d\n", omp_get_thread_num());
            update_forces(bodies);
            #pragma omp barrier
            update_positions(bodies);
        }
    }

    csvfile.close();  // Close the CSV file
    return 0;
}

 // getconf LEVEL1_DCACHE_LINESIZE     = 64
 // sizeof(Body)                       = 56  (without alignas(64))
 // printf("%bodies[1-0] = %p\n", &bodies[1] - &bodies[0]);      = 7 (proof for possible False Sharing)

 // Optimizations:
 // guided for, alignas(64) prevents False Sharing, barrier for synchronization
