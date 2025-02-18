#include "mgr.hpp"
#include "sim.hpp"
#include "load.hpp"

#include <random>
#include <numeric>
#include <algorithm>

#include <madrona/utils.hpp>
#include <madrona/importer.hpp>
#include <madrona/physics_loader.hpp>
#include <madrona/tracing.hpp>
#include <madrona/mw_cpu.hpp>
#include <madrona/render/api.hpp>
#include <madrona/physics_assets.hpp>

#include <array>
#include <charconv>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <string>

#include <madrona/render/asset_processor.hpp>

#ifdef MADRONA_CUDA_SUPPORT
#include <madrona/mw_gpu.hpp>
#include <madrona/cuda_utils.hpp>
#endif

#define MADRONA_VIEWER

using namespace madrona;
using namespace madrona::math;
using namespace madrona::phys;
using namespace madrona::render;
using namespace madrona::imp;

namespace fs = std::filesystem;

namespace madPhysics {

struct RenderGPUState {
    render::APILibHandle apiLib;
    render::APIManager apiMgr;
    render::GPUHandle gpu;
};


static inline Optional<RenderGPUState> initRenderGPUState(
    const Manager::Config &mgr_cfg)
{
    if (!mgr_cfg.headlessMode) {
        if (mgr_cfg.extRenderDev) {
            return Optional<RenderGPUState>::none();
        }
    }

#ifdef MGR_DISABLE_VULKAN
    return Optional<RenderGPUState>::none();
#endif

    auto render_api_lib = render::APIManager::loadDefaultLib();
    render::APIManager render_api_mgr(render_api_lib.lib());
    render::GPUHandle gpu = render_api_mgr.initGPU(mgr_cfg.gpuID);

    return RenderGPUState {
        .apiLib = std::move(render_api_lib),
        .apiMgr = std::move(render_api_mgr),
        .gpu = std::move(gpu),
    };
}

static inline Optional<render::RenderManager> initRenderManager(
    const Manager::Config &mgr_cfg,
    const Optional<RenderGPUState> &render_gpu_state)
{
    if (mgr_cfg.headlessMode) {
        return Optional<render::RenderManager>::none();
    }

    if (!mgr_cfg.headlessMode) {
        if (!mgr_cfg.extRenderDev) {
            return Optional<render::RenderManager>::none();
        }
    }

    render::APIBackend *render_api;
    render::GPUDevice *render_dev;

    if (render_gpu_state.has_value()) {
        render_api = render_gpu_state->apiMgr.backend();
        render_dev = render_gpu_state->gpu.device();
    } else {
        render_api = mgr_cfg.extRenderAPI;
        render_dev = mgr_cfg.extRenderDev;
    }

    return render::RenderManager(render_api, render_dev, {
        .enableBatchRenderer = false,
        .renderMode = render::RenderManager::Config::RenderMode::RGBD,
        .agentViewWidth = 64,
        .agentViewHeight = 64,
        .numWorlds = mgr_cfg.numWorlds,
        .maxViewsPerWorld = 4, // Just some dummy number - not used
        .maxInstancesPerWorld = 100,
        .execMode = mgr_cfg.execMode,
        .voxelCfg = {},
    });
}

URDFExport loadAssets(
        AssetLoader &asset_loader,
        PhysicsLoader &physics_loader,
        Optional<RenderManager> &render_mgr)
{
#if 1
    uint32_t stick_idx = asset_loader.addGlobalAsset(
        (std::filesystem::path(DATA_DIR) / "cylinder_long_render.obj").string(),
        (std::filesystem::path(DATA_DIR) / "cylinder_long.obj").string());

#if 1
    uint32_t disk_idx = asset_loader.addGlobalAsset(
        (std::filesystem::path(DATA_DIR) / "disk_render.obj").string(),
        (std::filesystem::path(DATA_DIR) / "disk.obj").string());
#endif

#endif

#if 1
    // Add a URDF
    uint32_t urdf_idx = asset_loader.addURDF(
        (std::filesystem::path(DATA_DIR) / "urdf/franka_lnd.urdf"));
#endif

    std::vector extra_materials = {
        SourceMaterial { Vector4{0.4f, 0.4f, 0.4f, 0.0f}, -1, 0.8f, 0.2f },
        SourceMaterial { Vector4{1.0f, 0.1f, 0.1f, 0.0f}, -1, 0.8f, 0.2f },
    };

    std::vector mat_overrides = {
        AssetLoader::MaterialOverride { 0, stick_idx },
        AssetLoader::MaterialOverride { 0, disk_idx },
        AssetLoader::MaterialOverride { 0, 1 },
        AssetLoader::MaterialOverride { 1, 2 },
        AssetLoader::MaterialOverride { 0, 0 },
    };

    URDFExport urdf_export = asset_loader.finish(physics_loader,
                        extra_materials,
                        mat_overrides,
                        render_mgr);

    if (render_mgr.has_value()) {
        render_mgr->configureLighting({
                { true, 
                math::Vector3{1.0f, 1.0f, -2.0f}, 
                math::Vector3{1.0f, 1.0f, 1.0f} }
                });
    }

    return urdf_export;
}

struct Manager::Impl {
    Config cfg;
    Optional<RenderGPUState> renderGPUState;
    Optional<render::RenderManager> renderMgr;
    CVXSolve *cvxSolve;

    Action *agentActionsBuffer;

    inline Impl(const Manager::Config &mgr_cfg,
                Optional<RenderGPUState> &&render_gpu_state,
                Optional<render::RenderManager> &&render_mgr,
                CVXSolve *cvx_solve,
                Action *action_buffer)
        : cfg(mgr_cfg),
          renderGPUState(std::move(render_gpu_state)),
          renderMgr(std::move(render_mgr)),
          cvxSolve(cvx_solve),
          agentActionsBuffer(action_buffer)
    {}

    inline virtual ~Impl() {}

    virtual void init() = 0;
    virtual void run() = 0;

    static inline Impl * init(const Config &cfg);
};

#ifdef MADRONA_CUDA_SUPPORT
struct Manager::CUDAImpl final : Manager::Impl {
    MWCudaExecutor gpuExec;
    MWCudaLaunchGraph initGraph;
    MWCudaLaunchGraph stepGraph;

    inline CUDAImpl(const Manager::Config &mgr_cfg,
                   Optional<RenderGPUState> &&render_gpu_state,
                   Optional<render::RenderManager> &&render_mgr,
                   MWCudaExecutor &&gpu_exec,
                   MWCudaLaunchGraph &&init_graph,
                   MWCudaLaunchGraph &&step_graph,
                   Action *action_buffer)
        : Impl(mgr_cfg,
               std::move(render_gpu_state), std::move(render_mgr),
               nullptr, action_buffer),
          gpuExec(std::move(gpu_exec)),
          initGraph(std::move(init_graph)),
          stepGraph(std::move(step_graph))
    {}

    inline virtual ~CUDAImpl() final {}

    inline virtual void init()
    {
        gpuExec.run(initGraph);
    }

    inline virtual void run()
    {
        gpuExec.run(stepGraph);
    }
};
#endif

struct Manager::CPUImpl final : Manager::Impl {
    using TaskGraphT =
        TaskGraphExecutor<Engine, Sim, Sim::Config, Sim::WorldInit>;

    TaskGraphT cpuExec;
    PhysicsLoader physLoader;

    inline CPUImpl(const Manager::Config &mgr_cfg,
                   PhysicsLoader &&phys_loader,
                   Optional<RenderGPUState> &&render_gpu_state,
                   CVXSolve *cvx_solve,
                   Optional<render::RenderManager> &&render_mgr,
                   TaskGraphT &&cpu_exec,
                   Action *action_buffer)
        : Impl(mgr_cfg,
               std::move(render_gpu_state), std::move(render_mgr),
               cvx_solve, action_buffer),
          cpuExec(std::move(cpu_exec)),
          physLoader(std::move(phys_loader))
    {}

    inline virtual ~CPUImpl() final {}

    inline virtual void init()
    {
        cpuExec.runTaskGraph(TaskGraphID::Init, nullptr);
    }

    inline virtual void run()
    {
        cpuExec.runTaskGraph(TaskGraphID::Step, cvxSolve);
    }
};

Manager::Impl * Manager::Impl::init(
    const Manager::Config &mgr_cfg)
{
    Sim::Config sim_cfg;
    sim_cfg.autoReset = false;
    sim_cfg.initRandKey = rand::initKey(mgr_cfg.randSeed);
    sim_cfg.cvxSolve = mgr_cfg.cvxSolve;
    sim_cfg.envType = mgr_cfg.envType;

    AssetLoader asset_loader (
        BuiltinAssets {
            .renderCubePath = 
                (fs::path(DATA_DIR) / "cube_render.obj").string(),
            .physicsCubePath = 
                (fs::path(DATA_DIR) / "cube_collision.obj").string(),

            .renderSpherePath = 
                (fs::path(DATA_DIR) / "sphere_render.obj").string(),
            // Just pass a dummy thing here
            .physicsSpherePath = 
                (fs::path(DATA_DIR) / "cube_collision.obj").string(),

            .renderPlanePath = 
                (fs::path(DATA_DIR) / "plane.obj").string(),
        }
    );

    switch (mgr_cfg.execMode) {
    case ExecMode::CUDA: {
#ifdef MADRONA_CUDA_SUPPORT
        CUcontext cu_ctx = MWCudaExecutor::initCUDA(mgr_cfg.gpuID);

        Optional<RenderGPUState> render_gpu_state =
            initRenderGPUState(mgr_cfg);
        Optional<render::RenderManager> render_mgr =
            initRenderManager(mgr_cfg, render_gpu_state);

        PhysicsLoader phys_loader(mgr_cfg.execMode, 100);

        URDFExport urdf_export = loadAssets(asset_loader, phys_loader, render_mgr);
        URDFExport urdf_cpy = urdf_export.makeGPUCopy(urdf_export);

        sim_cfg.numModelConfigs = urdf_cpy.numModelConfigs;
        sim_cfg.modelConfigs = urdf_cpy.modelConfigs;
        sim_cfg.modelData = urdf_cpy.modelData;

        ObjectManager *phys_obj_mgr = &phys_loader.getObjectManager();
        sim_cfg.rigidBodyObjMgr = phys_obj_mgr;

        if (render_mgr.has_value()) {
            sim_cfg.renderBridge = render_mgr->bridge();
        } else {
            sim_cfg.renderBridge = nullptr;
        }

        HeapArray<Sim::WorldInit> world_inits(mgr_cfg.numWorlds);

        MWCudaExecutor gpu_exec({
            .worldInitPtr = world_inits.data(),
            .numWorldInitBytes = sizeof(Sim::WorldInit),
            .userConfigPtr = (void *)&sim_cfg,
            .numUserConfigBytes = sizeof(Sim::Config),
            .numWorldDataBytes = sizeof(Sim),
            .worldDataAlignment = alignof(Sim),
            .numWorlds = mgr_cfg.numWorlds,
            .numTaskGraphs = (uint32_t)TaskGraphID::NumTaskGraphs,
            .numExportedBuffers = (uint32_t)ExportID::NumExports, 
        }, {
            { STICK_SRC_LIST },
            { STICK_COMPILE_FLAGS },
            CompileConfig::OptMode::LTO,
        }, cu_ctx, 
        // No ray tracing for batch rendering.
        Optional<madrona::CudaBatchRenderConfig>::none());

        MWCudaLaunchGraph step_graph = gpu_exec.buildLaunchGraph(
                TaskGraphID::Step);
        MWCudaLaunchGraph init_graph = gpu_exec.buildLaunchGraph(
                TaskGraphID::Init);

        Action *actions = (Action *)gpu_exec.getExported((uint32_t)ExportID::Action);

        return new CUDAImpl {
            mgr_cfg,
            std::move(render_gpu_state),
            std::move(render_mgr),
            std::move(gpu_exec),
            std::move(init_graph),
            std::move(step_graph),
            actions,
        };
#else
        FATAL("Madrona was not compiled with CUDA support");
#endif
    } break;
    case ExecMode::CPU: {
        // Hello
        PhysicsLoader phys_loader(ExecMode::CPU, 100);
        //loadPhysicsAssets(phys_loader);

        Optional<RenderGPUState> render_gpu_state =
            initRenderGPUState(mgr_cfg);

        Optional<render::RenderManager> render_mgr =
            initRenderManager(mgr_cfg, render_gpu_state);

        URDFExport urdf_export = loadAssets(asset_loader, phys_loader, render_mgr);
        URDFExport urdf_cpy = urdf_export.makeCPUCopy(urdf_export);

        sim_cfg.numModelConfigs = urdf_cpy.numModelConfigs;
        sim_cfg.modelConfigs = urdf_cpy.modelConfigs;
        sim_cfg.modelData = urdf_cpy.modelData;

        ObjectManager *phys_obj_mgr = &phys_loader.getObjectManager();
        sim_cfg.rigidBodyObjMgr = phys_obj_mgr;

        if (render_mgr.has_value()) {
            sim_cfg.renderBridge = render_mgr->bridge();
        } else {
            sim_cfg.renderBridge = nullptr;
        }

        HeapArray<Sim::WorldInit> world_inits(mgr_cfg.numWorlds);

        CPUImpl::TaskGraphT cpu_exec {
            ThreadPoolExecutor::Config {
                .numWorlds = mgr_cfg.numWorlds,
                .numExportedBuffers = (uint32_t)ExportID::NumExports,
            },
            sim_cfg,
            world_inits.data(),
            (uint32_t)TaskGraphID::NumTaskGraphs,
        };

        Action *actions = (Action *)cpu_exec.getExported((uint32_t)ExportID::Action);

        return new CPUImpl {
            mgr_cfg,
            std::move(phys_loader),
            std::move(render_gpu_state),
            mgr_cfg.cvxSolve,
            std::move(render_mgr),
            std::move(cpu_exec),
            actions
        };
    } break;
    default: MADRONA_UNREACHABLE();
    }
}

Manager::Manager(const Config &cfg)
    : impl_(Impl::init(cfg))
{
    impl_->init();
    step();
}

Manager::~Manager() {}

void Manager::step()
{
    impl_->run();

    if (impl_->renderMgr.has_value()) {
        impl_->renderMgr->readECS();
    }
}

render::RenderManager & Manager::getRenderManager()
{
    return *impl_->renderMgr;
}

void Manager::setAction(int32_t agent_idx,
                        int32_t move_amount,
                        bool visualize_colliders)
{
    Action action { 
        .v = move_amount,
        .vizColliders = visualize_colliders
    };

    auto *action_ptr = impl_->agentActionsBuffer +
        agent_idx;

    if (impl_->cfg.execMode == ExecMode::CUDA) {
#ifdef MADRONA_CUDA_SUPPORT
        cudaMemcpy(action_ptr, &action, sizeof(Action),
                   cudaMemcpyHostToDevice);
#endif
    } else {
        *action_ptr = action;
    }
}

}
