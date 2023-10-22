#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>
#include <fstream>
#define STEPS 100000000

struct Body {
    double x, y, z;
    double mass;
    double vx, vy, vz;
};

void update_forces(std::vector<Body>& bodies) {
    const double G = 6.67430e-11;
    #pragma omp parallel for
    for (int i = 0; i < bodies.size(); ++i) {
        double fx = 0.0, fy = 0.0, fz = 0.0;
        for (int j = 0; j < bodies.size(); ++j) {
            if (i == j) continue;
            
            double dx = bodies[j].x - bodies[i].x;
            double dy = bodies[j].y - bodies[i].y;
            double dz = bodies[j].z - bodies[i].z;
            
            double dist = std::sqrt(dx * dx + dy * dy + dz * dz);
            double force = G * bodies[i].mass * bodies[j].mass / (dist * dist * dist);
            
            fx += force * dx;
            fy += force * dy;
            fz += force * dz;
        }
        bodies[i].vx += fx / bodies[i].mass;
        bodies[i].vy += fy / bodies[i].mass;
        bodies[i].vz += fz / bodies[i].mass;
    }
}

void update_positions(std::vector<Body>& bodies, double dt) {
    #pragma omp parallel for
    for (int i = 0; i < bodies.size(); ++i) {
        bodies[i].x += bodies[i].vx * dt;
        bodies[i].y += bodies[i].vy * dt;
        bodies[i].z += bodies[i].vz * dt;
    }
}

int main() {
    // Initialize some sample bodies: Sun, Earth, Mars
    std::vector<Body> bodies;
    bodies.push_back({0, 0, 0, 1.989e30, 0, 0, 0}); // Sun
    bodies.push_back({1.496e11, 0, 0, 5.972e24, 0, 29.78e3, 0}); // Earth
    bodies.push_back({2.279e11, 0, 0, 6.4171e23, 0, 24e3, 0}); // Mars
    
    const double dt = 1;  // Time step in seconds
    
    // Create and open a new CSV file
    std::ofstream csvfile("../../visualizer/body_positions.csv");
    
    // Main simulation loop
    for (int step = 0; step < STEPS; ++step) {
        // Write positions to CSV file
        if (step % 20 == 0) {
            for (const auto& body : bodies) {
                csvfile << body.x << "," << body.y << "," << body.z;
                if (&body != &bodies.back()) {
                    csvfile << ",";
                }
            }
        csvfile << "\n";
        }
        update_forces(bodies);
        update_positions(bodies, dt);
    }

    csvfile.close();  // Close the CSV file
    return 0;
}
