#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <fstream>

#define BODIES 512
#define STEPS 100000
#define TIME_STEP 1 // Time step in seconds
#define GRAVITATIONAL_CONSTANT 6.67430e-11 // Time step in seconds
#define STEPS_BETWEEN_WRITING 20000

struct Vector3 {
    double x, y, z;
};

struct Body {
    double x, y, z;
    double mass;
    double vx, vy, vz;
};

__host__
void add_randomized_bodies(std::vector<Body> &bodies, int N) {
    std::random_device rd;
    std::mt19937 gen(rd());

    // Set ranges for body attributes based on realistic astronomical values
    std::uniform_real_distribution<> dist_x(1.0e9, 5.0e12); // Position x
    std::uniform_real_distribution<> dist_y(-5.0e5, 5.0e5); // Position y
    std::uniform_real_distribution<> dist_z(-5.0e5, 5.0e5); // Position z
    std::uniform_real_distribution<> dist_mass(1.0e21, 1.0e28); // Mass
    std::uniform_real_distribution<> dist_vx(-5.0e3, 5.0e3); // Velocity x
    std::uniform_real_distribution<> dist_vy(-5.0e3, 5.0e3); // Velocity y
    std::uniform_real_distribution<> dist_vz(-5.0e3, 5.0e3); // Velocity z

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

__host__
void errorexit(const char *s) {
    printf("\n%s",s);
    exit(EXIT_FAILURE);	 	
}

__host__
void write_bodies_to_csv(std::vector<Body> &bodies, std::ofstream &csvfile) {
    for (const auto& body : bodies) {
        csvfile << body.x << "," << body.y << "," << body.z;
        if (&body != &bodies.back()) {
            csvfile << ",";
        }
    }
    csvfile << "\n";
}

__global__
void update_forces(Body *bodies) {
    Body& my_body = bodies[blockIdx.x];
    double dx;
    double dy;
    double dz;
    double force;
    __shared__ double fxs[BODIES];
    __shared__ double fys[BODIES];
    __shared__ double fzs[BODIES];

    if (blockIdx.x == threadIdx.x) {
        fxs[threadIdx.x] = 0;
        fys[threadIdx.x] = 0;
        fzs[threadIdx.x] = 0;
    }
    else {
        dx = bodies[threadIdx.x].x - my_body.x;
        dy = bodies[threadIdx.x].y - my_body.y;
        dz = bodies[threadIdx.x].z - my_body.z;
        
        double dist = std::sqrt(dx * dx + dy * dy + dz * dz);
        force = GRAVITATIONAL_CONSTANT * my_body.mass * bodies[threadIdx.x].mass / (dist * dist * dist);
        fxs[threadIdx.x] = force * dx;
        fys[threadIdx.x] = force * dy;
        fzs[threadIdx.x] = force * dz;
    }

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            fxs[threadIdx.x] += fxs[threadIdx.x + s];
            fys[threadIdx.x] += fys[threadIdx.x + s];
            fzs[threadIdx.x] += fzs[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        my_body.vx += fxs[threadIdx.x] / my_body.mass;
        my_body.vy += fys[threadIdx.x] / my_body.mass;
        my_body.vz += fzs[threadIdx.x] / my_body.mass;
    }
}

__global__
void update_positions(Body *bodies) {
    Body& my_body = bodies[blockIdx.x];
    my_body.x += my_body.vx * TIME_STEP;
    my_body.y += my_body.vy * TIME_STEP;
    my_body.z += my_body.vz * TIME_STEP;
}

int main() {
    printf("CUDA\n");

    int threadsinblock = BODIES;
    int blocksingrid = BODIES;

    std::vector<Body> h_bodies;
    Body *d_bodies;

    h_bodies.push_back({0, 0, 0, 1.989e30, 0, 0, 0}); // Sun
    h_bodies.push_back({57.9e9, 0, 0, 3.3011e23, 0, 47.87e3, 0}); // Mercury
    h_bodies.push_back({108.2e9, 0, 0, 4.8675e24, 0, 35.02e3, 0}); // Venus
    h_bodies.push_back({149.6e9, 0, 0, 5.972e24, 0, 29.78e3, 0}); // Earth
    h_bodies.push_back({227.9e9, 0, 0, 6.4171e23, 0, 24.077e3, 0}); // Mars
    h_bodies.push_back({778.6e9, 0, 0, 1.898e27, 0, 13.07e3, 0}); // Jupiter
    h_bodies.push_back({1.433e12, 0, 0, 5.683e26, 0, 9.69e3, 0}); // Saturn
    h_bodies.push_back({2.872e12, 0, 0, 8.681e25, 0, 6.81e3, 0}); // Uranus

    add_randomized_bodies(h_bodies, BODIES-8);
    
    //device memory allocation (GPU)
    if (cudaSuccess!=cudaMalloc(&d_bodies, h_bodies.size() * sizeof(Body)))
      errorexit("Error allocating memory on the GPU");

    cudaMemcpy(d_bodies, h_bodies.data(), h_bodies.size() * sizeof(Body), cudaMemcpyHostToDevice);

    // Create and open a new CSV file
    std::ofstream csvfile("../../visualizer/body_positions.csv");

    // First Writing and Calculations
    update_forces<<<blocksingrid,threadsinblock>>>(d_bodies);
    write_bodies_to_csv(h_bodies, csvfile);
    cudaDeviceSynchronize();
    update_positions<<<blocksingrid,1>>>(d_bodies);
    cudaDeviceSynchronize();

    // Main simulation loop
    for (int step = 1; step < STEPS; step++) {
        update_forces<<<blocksingrid,threadsinblock>>>(d_bodies);
        if (step % STEPS_BETWEEN_WRITING == 0) {
            cudaMemcpy(h_bodies.data(), d_bodies, h_bodies.size() * sizeof(Body), cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
            write_bodies_to_csv(h_bodies, csvfile);
            printf("%.2f%\n", step*100.0/STEPS);
        }
        cudaDeviceSynchronize();
        update_positions<<<blocksingrid,1>>>(d_bodies);
        cudaDeviceSynchronize();
    }

    cudaFree(d_bodies);
    csvfile.close();  // Close the CSV file
    return 0;
}
