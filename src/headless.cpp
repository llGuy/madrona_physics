#include "mgr.hpp"

#include <chrono>
#include <signal.h>
#include <madrona/macros.hpp>

using namespace madPhysics;

struct HeadlessWrapper {
    Manager *mgr;
    uint32_t numWorlds;
    uint32_t numSteps;

    void run()
    {
        auto start = std::chrono::system_clock::now();

        for (uint32_t i = 0; i < numSteps; ++i) {
            mgr->step();
        }

        auto end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed = end - start;

        float fps = (double)numSteps * (double)numWorlds / elapsed.count();
        printf("FPS %f\n", fps);
        printf("Average step time: %f ms\n", 1000.0f * elapsed.count() / (double)numSteps);
    }
};

int main(int argc, char *argv[])
{
    uint32_t num_worlds = 32;

    Manager *mgr = new Manager (Manager::Config {
        .execMode = madrona::ExecMode::CUDA,
        .gpuID = 0,
        .numWorlds = (uint32_t)num_worlds,
        .randSeed = 5
    });
    
    HeadlessWrapper wrapper = {
        .mgr = mgr,
        .numWorlds = num_worlds,
        .numSteps = 1000,
    };

    wrapper.run();

    printf("Finished simulation\n");

    return 0;
}
