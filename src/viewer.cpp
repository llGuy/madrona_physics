#include <madrona/viz/viewer.hpp>
#include <madrona/render/render_mgr.hpp>
#include <madrona/window.hpp>

#include "sim.hpp"
#include "mgr.hpp"
#include "types.hpp"

#include <filesystem>
#include <fstream>
#include <imgui.h>

#include <stb_image_write.h>

using namespace madrona;
using namespace madrona::viz;

int main(int argc, char *argv[])
{
    using namespace madEscape;

    uint32_t num_worlds = 1;
    if (argc < 2) {
        printf("./stick_viewer [num_worlds]\n");
        return -1;
    } else {
        num_worlds = std::stoi(argv[1]);
    }

    WindowManager wm {};
    WindowHandle window = wm.makeWindow("Stick Viewer", 
            2730/2, 1536/2);

    render::GPUHandle render_gpu = wm.initGPU(0, { window.get() });

    // Create the simulation manager
    Manager mgr({
        .execMode = madrona::ExecMode::CUDA,
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
        .simTickRate = 120,
        .cameraMoveSpeed = camera_move_speed * 7.f,
        .cameraPosition = { 41.899895f, -57.452969f, 33.152081f },
        .cameraRotation = { 0.944346f, -0.054453f, -0.018675f, 0.323878f },
    });

    // Main loop for the viewer viewer
    viewer.loop(
        [&mgr](CountT world_idx, const Viewer::UserInput &input) {
            // No input
        }
        , [&mgr](CountT world_idx, CountT agent_idx,
               const Viewer::UserInput &input) {
            // No input
        }, [&]() {
            mgr.step();
        }, [&]() {
            // No ImGui windows for now
        });
}
