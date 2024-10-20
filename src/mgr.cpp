#include "mgr.hpp"
#include "sim.hpp"
#include "import.hpp"

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

namespace madEscape {

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
        .maxInstancesPerWorld = 32,
        .execMode = mgr_cfg.execMode,
        .voxelCfg = {},
    });
}

static imp::ImportedAssets loadAssets(
        Optional<render::RenderManager> &render_mgr)
{
    std::array<std::string, (size_t)SimObject::NumObjects> render_asset_paths;

    render_asset_paths[(size_t)SimObject::Stick] =
        (std::filesystem::path(DATA_DIR) / "cylinder.obj").string();
    render_asset_paths[(size_t)SimObject::Plane] =
        (std::filesystem::path(DATA_DIR) / "plane.obj").string();

    std::array<const char *, (size_t)SimObject::NumObjects> render_asset_cstrs;
    for (size_t i = 0; i < render_asset_paths.size(); i++) {
        render_asset_cstrs[i] = render_asset_paths[i].c_str();
    }

    imp::AssetImporter importer;

    std::array<char, 1024> import_err;
    auto render_assets = importer.importFromDisk(
        render_asset_cstrs, Span<char>(import_err.data(), import_err.size()),
        true);

    std::vector<imp::SourceMaterial> materials = {
        { math::Vector4{0.4f, 0.4f, 0.4f, 0.0f}, -1, 0.8f, 0.2f,},
        { math::Vector4{1.0f, 0.1f, 0.1f, 0.0f}, -1, 0.8f, 0.2f,},
    };

    for (auto &mat : materials) {
        render_assets->materials.push_back(mat);
    }

    // Override materials
    render_assets->objects[0].meshes[0].materialIDX = 0;
    render_assets->objects[1].meshes[0].materialIDX = 1;

    if (render_mgr.has_value()) {
        render_mgr->loadObjects(render_assets->objects,
                Span(materials.data(), materials.size()), 
                {},
                true);

        render_mgr->configureLighting({
            { true, math::Vector3{1.0f, 1.0f, -2.0f}, math::Vector3{1.0f, 1.0f, 1.0f} }
        });
    }

    return std::move(*render_assets);
}

struct Manager::Impl {
    Config cfg;
    Optional<RenderGPUState> renderGPUState;
    Optional<render::RenderManager> renderMgr;

    inline Impl(const Manager::Config &mgr_cfg,
                Optional<RenderGPUState> &&render_gpu_state,
                Optional<render::RenderManager> &&render_mgr)
        : cfg(mgr_cfg),
          renderGPUState(std::move(render_gpu_state)),
          renderMgr(std::move(render_mgr))
    {}

    inline virtual ~Impl() {}

    virtual void run() = 0;

    static inline Impl * init(const Config &cfg);
};

#ifdef MADRONA_CUDA_SUPPORT
struct Manager::CUDAImpl final : Manager::Impl {
    MWCudaExecutor gpuExec;
    MWCudaLaunchGraph stepGraph;

    inline CUDAImpl(const Manager::Config &mgr_cfg,
                   Optional<RenderGPUState> &&render_gpu_state,
                   Optional<render::RenderManager> &&render_mgr,
                   MWCudaExecutor &&gpu_exec,
                   MWCudaLaunchGraph &&step_graph)
        : Impl(mgr_cfg,
               std::move(render_gpu_state), std::move(render_mgr)),
          gpuExec(std::move(gpu_exec)),
          stepGraph(std::move(step_graph))
    {}

    inline virtual ~CUDAImpl() final {}

    inline virtual void run()
    {
        gpuExec.run(stepGraph);
    }
};
#else
static_assert(false, "CPU backend hasn't been implemented yet");
#endif

Manager::Impl * Manager::Impl::init(
    const Manager::Config &mgr_cfg)
{
    Sim::Config sim_cfg;
    sim_cfg.autoReset = false;
    sim_cfg.initRandKey = rand::initKey(mgr_cfg.randSeed);

    switch (mgr_cfg.execMode) {
    case ExecMode::CUDA: {
#ifdef MADRONA_CUDA_SUPPORT
        CUcontext cu_ctx = MWCudaExecutor::initCUDA(mgr_cfg.gpuID);

        Optional<RenderGPUState> render_gpu_state =
            initRenderGPUState(mgr_cfg);

        Optional<render::RenderManager> render_mgr =
            initRenderManager(mgr_cfg, render_gpu_state);

        auto imported_assets = loadAssets(
                render_mgr);

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

        return new CUDAImpl {
            mgr_cfg,
            std::move(render_gpu_state),
            std::move(render_mgr),
            std::move(gpu_exec),
            std::move(step_graph),
        };
#else
        FATAL("Madrona was not compiled with CUDA support");
#endif
    } break;
    case ExecMode::CPU: {
        FATAL("CPU backend not implemented yet");
    } break;
    default: MADRONA_UNREACHABLE();
    }
}

Manager::Manager(const Config &cfg)
    : impl_(Impl::init(cfg))
{
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

}
