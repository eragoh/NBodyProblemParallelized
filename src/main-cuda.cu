#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>
#include <random>
#include <fstream>

#define STEPS 1000
#define TIME_STEP 1 // Time step in seconds
#define GRAVITATIONAL_CONSTANT 6.67430e-11 // Time step in seconds
#define STEPS_BETWEEN_WRITING 20

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
void update_forces_and_positions(Body *bodies, int n) {
    int my_index=blockIdx.x*blockDim.x+threadIdx.x;
    for (int j = 0; j < STEPS_BETWEEN_WRITING; ++j) {
        double fx = 0.0, fy = 0.0, fz = 0.0;
        for (int i = 0; i < n; ++i) {
            if (my_index == i) continue;
            
            double dx = bodies[i].x - bodies[my_index].x;
            double dy = bodies[i].y - bodies[my_index].y;
            double dz = bodies[i].z - bodies[my_index].z;
            
            double dist = std::sqrt(dx * dx + dy * dy + dz * dz);
            double force = GRAVITATIONAL_CONSTANT * bodies[my_index].mass * bodies[i].mass / (dist * dist * dist);
            
            fx += force * dx;
            fy += force * dy;
            fz += force * dz;
        }
        __syncthreads();
        bodies[my_index].vx += fx / bodies[my_index].mass;
        bodies[my_index].vy += fy / bodies[my_index].mass;
        bodies[my_index].vz += fz / bodies[my_index].mass;

        bodies[my_index].x += bodies[my_index].vx * TIME_STEP;
        bodies[my_index].y += bodies[my_index].vy * TIME_STEP;
        bodies[my_index].z += bodies[my_index].vz * TIME_STEP;
        __syncthreads();
    }
}

int main() {
    printf("CUDA\n");

    int threadsinblock = 1024;
    int blocksingrid = 1;

    cudaStream_t calc_stream, copy_stream;
    cudaStreamCreate(&calc_stream);
    cudaStreamCreate(&copy_stream);

    std::vector<Body> h_bodies;
    Body *d_bodies, *d_bodies2;

    h_bodies.push_back({0, 0, 0, 1.989e30, 0, 0, 0}); // Sun
    h_bodies.push_back({57.9e9, 0, 0, 3.3011e23, 0, 47.87e3, 0}); // Mercury
    h_bodies.push_back({108.2e9, 0, 0, 4.8675e24, 0, 35.02e3, 0}); // Venus
    h_bodies.push_back({149.6e9, 0, 0, 5.972e24, 0, 29.78e3, 0}); // Earth
    h_bodies.push_back({227.9e9, 0, 0, 6.4171e23, 0, 24.077e3, 0}); // Mars
    h_bodies.push_back({778.6e9, 0, 0, 1.898e27, 0, 13.07e3, 0}); // Jupiter
    h_bodies.push_back({1.433e12, 0, 0, 5.683e26, 0, 9.69e3, 0}); // Saturn
    h_bodies.push_back({2.872e12, 0, 0, 8.681e25, 0, 6.81e3, 0}); // Uranus
    h_bodies.push_back({4.495e12, 0, 0, 1.024e26, 0, 5.43e3, 0}); // Neptune

    add_randomized_bodies(h_bodies, 1015);
    
    //device memory allocation (GPU)
    if (cudaSuccess!=cudaMalloc(&d_bodies, h_bodies.size() * sizeof(Body)))
      errorexit("Error allocating memory on the GPU");
    
    if (cudaSuccess!=cudaMalloc(&d_bodies2, h_bodies.size() * sizeof(Body)))
      errorexit("Error allocating memory on the GPU");

    cudaMemcpy(d_bodies, h_bodies.data(), h_bodies.size() * sizeof(Body), cudaMemcpyHostToDevice);

    // Create and open a new CSV file
    std::ofstream csvfile("../../visualizer/body_positions.csv");

    // First Writing and Calculations
    update_forces_and_positions<<<blocksingrid,threadsinblock, 0, calc_stream>>>(d_bodies, h_bodies.size());
    write_bodies_to_csv(h_bodies, csvfile);

    // Main simulation loop
    for (int step = 1; step < STEPS; step+=STEPS_BETWEEN_WRITING) {
        std::swap(d_bodies, d_bodies2);
        printf("%f%\n", step*100.0/STEPS);
        cudaStreamSynchronize(calc_stream);
        update_forces_and_positions<<<blocksingrid,threadsinblock, 0, calc_stream>>>(d_bodies, h_bodies.size());
        cudaMemcpyAsync(h_bodies.data(), d_bodies2, h_bodies.size() * sizeof(Body), cudaMemcpyDeviceToHost, copy_stream);
        cudaStreamSynchronize(copy_stream);
        write_bodies_to_csv(h_bodies, csvfile);
    }

    cudaFree(d_bodies);
    cudaFree(d_bodies2);
    csvfile.close();  // Close the CSV file
    return 0;
}


// użyć pamięci shared
// 2 streamy
// dynamic paralellism można spróbować żeby bardziej GPU użyć // ewentualnie więcej bloków ogarnąć
// vector3 zamiast double x,y,z może coś dać

// czasy:
// x przed pętlą w samym kernelu (wykonywałem wcześniej 20 razy kernel, a teraz jest pętla w kernelu na 20 iteracji)
// x jeden strumień
// 12:52 dwa strumienie, nałożenie obliczeń i memcpyAsync (trzeba wyczaić co trwa krócej, bo może copy czeka np. 3 krotność swojego czasu działania na calc'a -- można
//                                                     by to pewnie poprawić poprzez zwiększenie liczby strumieni copy lub zmniejszyć STEPS_BETWEEN_WRITING czy coś)



// wygląda na to że memcpyAsync nie zabiera właściwie wcale czasu, a na calc_stream zawsze czekam w synchronizacji ...