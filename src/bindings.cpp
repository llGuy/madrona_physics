#if 1
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
                    uint32_t num_contact_pts,
                    uint32_t num_equality_rows,
                    float h,
                    float *mass,
                    float *free_acc,
                    float *vel,
                    float *J_c,
                    float *J_e,
                    float *mu,
                    float *penetrations)
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

    Tensor free_acc_tensor(
        free_acc,
        { total_num_dofs },
        {},
        {},
        nb::dtype<float>(),
        nb::device::cpu::value
    );

    Tensor vel_tensor(
        vel,
        { total_num_dofs },
        {},
        {},
        nb::dtype<float>(),
        nb::device::cpu::value
    );

    Tensor J_tensor(
        J_c,
        { total_num_dofs, 3 * num_contact_pts },
        {},
        {},
        nb::dtype<float>(),
        nb::device::cpu::value
    );

    Tensor J_e_tensor(
        J_e,
        { total_num_dofs, num_equality_rows },
        {},
        {},
        nb::dtype<float>(),
        nb::device::cpu::value
    );

    Tensor mu_tensor(
        mu,
        { num_contact_pts },
        {},
        {},
        nb::dtype<float>(),
        nb::device::cpu::value
    );

    Tensor pen_tensor(
        penetrations,
        { num_contact_pts },
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

    data->call(m_tensor, free_acc_tensor, vel_tensor, J_tensor,
        J_e_tensor, mu_tensor, pen_tensor, h, ret_tensor);

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
            .simTickRate = 10,
            .cameraMoveSpeed = camera_move_speed * 7.f,
            .cameraPosition = { 41.899895f, -57.452969f, 33.152081f },
            .cameraRotation = { 0.944346f, -0.054453f, -0.018675f, 0.323878f },
        });

        uint32_t step_i = 0;

        // Main loop for the viewer viewer
        viewer.loop(
            [this](CountT /* world_idx */, const Viewer::UserInput &/* input */) {
                // No input
            }
            , [this](CountT /* world_idx */, CountT /* agent_idx */,
                   const Viewer::UserInput & /* input */) {
                // No input
            }, [this, &step_i]() {
                mgr->step();

                printf("step: %u\n", step_i);
                if (step_i == 75) {
                    // exit(0);
                }
                step_i++;
            }, [this]() {
                // No ImGui windows for now
            });
    }
};

struct HeadlessWrapper {
    Manager *mgr;
    uint32_t numWorlds;
    uint32_t numSteps;

    CVXSolveData *solveData;
    madrona::phys::CVXSolve *solve;

    void run()
    {
        for (uint32_t i = 0; i < numSteps; ++i) {
            printf("step: %u\n", i);
            mgr->step();
        }
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

    nb::class_<HeadlessWrapper> (m, "HeadlessRun")
        .def("__init__", [](HeadlessWrapper *self,
                            int64_t num_worlds,
                            int64_t num_steps) {
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
                .cvxSolve = solve_fn,
                .headlessMode = true
            });

            new (self) HeadlessWrapper {
                .mgr = mgr,
                .numWorlds = (uint32_t)num_worlds,
                .numSteps = (uint32_t)num_steps,
                .solveData = solve_data,
                .solve = solve_fn,
            };
        })
        .def("run", [](HeadlessWrapper *self,
                       nb::callable cvx_solve) {
            self->solveData->init = true;
            self->solveData->call = cvx_solve;

            self->run();
        })
    ;

    nb::class_<AppWrapper> (m, "PhysicsApp")
        .def("__init__", [](AppWrapper *self,
                            int64_t num_worlds) {
            WindowManager wm {};
            WindowHandle window = wm.makeWindow("Stick Viewer",
                    2000, 1000);
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
            self->solveData->init = true;
            self->solveData->call = cvx_solve;

            self->run();
        })
    ;
}
    
}
#endif
