#include "mgr.hpp"

#include <signal.h>
#include <madrona/macros.hpp>
#include <madrona/window.hpp>
#include <madrona/viz/viewer.hpp>
#include <madrona/py/bindings.hpp>

#include <nanobind/ndarray.h>

namespace nb = nanobind;

using namespace madrona;
using namespace madrona::viz;

struct CVXSolveData {
    bool init;
    nb::callable call;
};

float *cvxSolveCall(void *vdata,
                    uint32_t total_num_dofs,
                    float *mass,
                    float *tau)
{
    CVXSolveData *data = (CVXSolveData *)vdata;

    if (!data->init) {
        return nullptr;
    }

    using Tensor = nb::ndarray<float, nb::numpy, nb::shape<>>;

    Tensor m_tensor(
        mass,
        { total_num_dofs, total_num_dofs },
        {},
        {},
        nb::dtype<float>(),
        nb::device::cpu::value
    );

    Tensor tau_tensor(
        tau,
        { total_num_dofs },
        {},
        {},
        nb::dtype<float>(),
        nb::device::cpu::value
    );

    float *ret_data = new float[total_num_dofs];
    for (int i = 0; i < total_num_dofs; ++i) {
        ret_data[i] = 0.f;
    }

    Tensor ret_tensor(
        ret_data,
        { total_num_dofs },
        {},
        {},
        nb::dtype<float>(),
        nb::device::cpu::value
    );

    data->call(m_tensor, tau_tensor, ret_tensor);

    return ret_data;
}

namespace madPhysics {

struct AppWrapper {
    WindowManager wm;
    WindowHandle window;
    render::GPUHandle renderGPU;
    uint32_t numWorlds;
    Manager *mgr;
    CVXSolveData *solveData;
    madrona::phys::CVXSolve *solve;

    void run()
    {
        float camera_move_speed = 10.f;

        // Create the viewer viewer
        viz::Viewer viewer(mgr->getRenderManager(), window.get(), {
            .numWorlds = numWorlds,
            .simTickRate = 60,
            .cameraMoveSpeed = camera_move_speed * 7.f,
            .cameraPosition = { 41.899895f, -57.452969f, 33.152081f },
            .cameraRotation = { 0.944346f, -0.054453f, -0.018675f, 0.323878f },
        });

        // Main loop for the viewer viewer
        viewer.loop(
            [this](CountT /* world_idx */, const Viewer::UserInput &/* input */) {
                // No input
            }
            , [this](CountT /* world_idx */, CountT /* agent_idx */,
                   const Viewer::UserInput & /* input */) {
                // No input
            }, [this]() {
                mgr->step();
            }, [this]() {
                // No ImGui windows for now
            });
    }
};

template <typename PyFn>
void run(AppWrapper *self, PyFn &&py_call_fn)
{
    self->solve->fn = &py_call_fn;
    self->run();
}

NB_MODULE(madrona_stick, m) {
    madrona::py::setupMadronaSubmodule(m);

    nb::class_<AppWrapper> (m, "PhysicsApp")
        .def("__init__", [](AppWrapper *self,
                            int64_t num_worlds) {
            WindowManager wm {};
            WindowHandle window = wm.makeWindow("Stick Viewer",
                    2730, 1536);
            render::GPUHandle render_gpu = wm.initGPU(0, { window.get() });

            CVXSolveData *solve_data = new CVXSolveData {
                false
            };

            madrona::phys::CVXSolve *solve_fn = new madrona::phys::CVXSolve {
                cvxSolveCall, solve_data, 0
            };

            Manager *mgr = new Manager (Manager::Config {
                .execMode = madrona::ExecMode::CPU,
                .gpuID = 0,
                .numWorlds = (uint32_t)num_worlds,
                .randSeed = 5,
                .extRenderAPI = wm.gpuAPIManager().backend(),
                .extRenderDev = render_gpu.device(),
                .cvxSolve = solve_fn
            });

            new (self) AppWrapper {
                .wm = std::move(wm),
                .window = std::move(window),
                .renderGPU = std::move(render_gpu),
                .numWorlds = (uint32_t)num_worlds,
                .mgr = mgr,
                .solveData = solve_data,
                .solve = solve_fn,
            };
        })
        .def("run", [](AppWrapper *self,
                       nb::callable cvx_solve) {
#if 1
            self->solveData->init = true;
            self->solveData->call = cvx_solve;

            self->run();
#endif
        })
    ;
}
    
}
