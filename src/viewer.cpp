#include <madrona/viz/viewer.hpp>
#include <madrona/render/render_mgr.hpp>
#include <madrona/window.hpp>

#include "sim.hpp"
#include "mgr.hpp"

#include <filesystem>
#include <fstream>
#include <imgui.h>

#include <stb_image_write.h>

using namespace madrona;
using namespace madrona::viz;

int main(int argc, char *argv[])
{
    using namespace madPhysics;

    uint32_t num_worlds = 1;
    madrona::ExecMode exec_mode = madrona::ExecMode::CUDA;

    if (argc < 3) {
        printf("./stick_viewer [cpu|cuda] [num_worlds]\n");
        return -1;
    } else {
        if (!strcmp(argv[1], "cuda")) {
            exec_mode = madrona::ExecMode::CUDA;
        } else if (!strcmp(argv[1], "cpu")) {
            exec_mode = madrona::ExecMode::CPU;
        } else {
            FATAL("Invalid exec mode\n");
        }

        num_worlds = std::stoi(argv[2]);
    }

    WindowManager wm {};
    WindowHandle window = wm.makeWindow("Stick Viewer", 
            2000, 1000);

    render::GPUHandle render_gpu = wm.initGPU(0, { window.get() });

    // Create the simulation manager
    Manager mgr({
        .execMode = exec_mode,
        .gpuID = 0,
        .numWorlds = num_worlds,
        .randSeed = 5,
        .extRenderAPI = wm.gpuAPIManager().backend(),
        .extRenderDev = render_gpu.device(),
    });

    float camera_move_speed = 10.f;

    // Create the viewer viewer
    viz::Viewer viewer(mgr.getRenderManager(), window.get(), {
        .numWorlds = num_worlds,
        .simTickRate = 60,
        .cameraMoveSpeed = camera_move_speed * 7.f,
        .cameraPosition = { 41.899895f, -57.452969f, 33.152081f },
        .cameraRotation = { 0.944346f, -0.054453f, -0.018675f, 0.323878f },
    });

    uint64_t step_iter = 0;

    // Main loop for the viewer viewer
    viewer.loop(
        [&mgr](CountT /* world_idx */, const Viewer::UserInput &/* input */) {
            // No input
        }
        , [&mgr](CountT /* world_idx */, CountT /* agent_idx */,
               const Viewer::UserInput & /* input */) {
            // No input
        }, [&]() {
            auto start = std::chrono::system_clock::now();

            mgr.step();

            auto end = std::chrono::system_clock::now();
            std::chrono::duration<double> elapsed = end - start;
            float fps = (double)1 * (double)num_worlds / elapsed.count();
            printf("FPS %f\n", fps);

            printf("step %llu!\n", step_iter);
            step_iter++;
        }, [&]() {
            // No ImGui windows for now
        });
}
