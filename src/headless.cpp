#include "mgr.hpp"

#include <signal.h>
#include <madrona/macros.hpp>

using namespace madPhysics;

struct HeadlessWrapper {
    Manager *mgr;
    uint32_t numSteps;

    void run()
    {
        for (uint32_t i = 0; i < numSteps; ++i) {
            printf("step: %u\n", i);
            mgr->step();
        }
    }
};

int main(int argc, char *argv[])
{
    Manager *mgr = new Manager (Manager::Config {
        .execMode = madrona::ExecMode::CPU,
        .gpuID = 0,
        .numWorlds = (uint32_t)128,
        .randSeed = 5
    });
    
    HeadlessWrapper wrapper = {
        .mgr = mgr,
        .numSteps = 8,
    };

    wrapper.run();

    printf("Finished simulation\n");

    return 0;
}
