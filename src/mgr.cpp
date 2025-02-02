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

#if 0
static imp::ImportedAssets loadRenderAssets(
        Optional<render::RenderManager> &render_mgr)
{
    std::array<std::string, (size_t)SimObject::NumObjects> render_asset_paths;

    render_asset_paths[(size_t)SimObject::Stick] =
        (std::filesystem::path(DATA_DIR) / "cylinder_long_render.obj").string();
    render_asset_paths[(size_t)SimObject::Sphere] =
        (std::filesystem::path(DATA_DIR) / "sphere_render.obj").string();
    render_asset_paths[(size_t)SimObject::Capsule] =
        (std::filesystem::path(DATA_DIR) / "capsule_render.obj").string();
    render_asset_paths[(size_t)SimObject::Cube] =
        (std::filesystem::path(DATA_DIR) / "cube_render.obj").string();
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
    render_assets->objects[(CountT)SimObject::Stick].meshes[0].materialIDX = 0;
    render_assets->objects[(CountT)SimObject::Sphere].meshes[0].materialIDX = 0;
    render_assets->objects[(CountT)SimObject::Capsule].meshes[0].materialIDX = 0;
    render_assets->objects[(CountT)SimObject::Plane].meshes[0].materialIDX = 1;
    render_assets->objects[(CountT)SimObject::Cube].meshes[0].materialIDX = 0;

    if (render_mgr.has_value()) {
        render_mgr->loadObjects(render_assets->objects,
                Span(materials.data(), materials.size()), 
                {},
                true);

        render_mgr->configureLighting({
            { true, 
              math::Vector3{1.0f, 1.0f, -2.0f}, 
              math::Vector3{1.0f, 1.0f, 1.0f} }
        });
    }

    return std::move(*render_assets);
}

std::vector<const char *> makeCStrings(const std::vector<std::string> &paths)
{
    std::vector<const char *> asset_cstrs;
    for (size_t i = 0; i < paths.size(); i++) {
        asset_cstrs[i] = paths[i].c_str();
    }
    return asset_cstrs;
}

static void loadPhysicsAssets(PhysicsLoader &loader)
{
    imp::AssetImporter importer;


    std::vector<std::string> asset_paths;
    asset_paths.push_back((std::filesystem::path(DATA_DIR) / "cylinder_long.obj").string());
    asset_paths.push_back((std::filesystem::path(DATA_DIR) / "sphere.obj").string());
    asset_paths.push_back((std::filesystem::path(DATA_DIR) / "capsule.obj").string());
    asset_paths.push_back((std::filesystem::path(DATA_DIR) / "cube_collision.obj").string());

    std::vector<const char *> asset_cstrs = makeCStrings(asset_paths);

    char import_err_buffer[4096];
    auto imported_hulls = importer.importFromDisk(
        asset_cstrs, import_err_buffer, true);

    if (!imported_hulls.has_value()) {
        FATAL("%s", import_err_buffer);
    }

    DynArray<imp::SourceMesh> src_convex_hulls(
        imported_hulls->objects.size());

    DynArray<DynArray<SourceCollisionPrimitive>> prim_arrays(0);
    
    HeapArray<SourceCollisionObject> src_objs(
        (CountT)SimObject::NumObjects);

    auto setupHull = [&](SimObject obj_id, float inv_mass,
                         RigidBodyFrictionData friction) {
        auto meshes = imported_hulls->objects[(CountT) obj_id].meshes;
        DynArray<SourceCollisionPrimitive> prims(meshes.size());

        for (const imp::SourceMesh &mesh : meshes) {
            src_convex_hulls.push_back(mesh);
            prims.push_back({
                .type = CollisionPrimitive::Type::Hull,
                .hullInput = {
                    .hullIDX = uint32_t(src_convex_hulls.size() - 1),
                },
            });
        }

        prim_arrays.emplace_back(std::move(prims));

        src_objs[(CountT)obj_id] = SourceCollisionObject {
            .prims = Span<const SourceCollisionPrimitive>(prim_arrays.back()),
            .invMass = inv_mass,
            .friction = friction,
        };
    };

    setupHull(SimObject::Stick, 0.5f, {
        .muS = 0.5f,
        .muD = 2.f,
    });
    setupHull(SimObject::Sphere, 0.5f, {
        .muS = 0.5f,
        .muD = 2.f,
    });
    setupHull(SimObject::Capsule, 0.5f, {
        .muS = 0.5f,
        .muD = 2.f,
    });
    setupHull(SimObject::Cube, 0.5f, {
        .muS = 0.5f,
        .muD = 2.f,
    });

    SourceCollisionPrimitive plane_prim {
        .type = CollisionPrimitive::Type::Plane,
    };

    src_objs[(CountT)SimObject::Plane] = {
        .prims = Span<const SourceCollisionPrimitive>(&plane_prim, 1),
        .invMass = 0.f,
        .friction = {
            .muS = 2.f,
            .muD = 2.f,
        },
    };


    StackAlloc tmp_alloc;
    RigidBodyAssets rigid_body_assets;
    CountT num_rigid_body_data_bytes;
    void *rigid_body_data = RigidBodyAssets::processRigidBodyAssets(
        src_convex_hulls,
        src_objs,
        false,
        tmp_alloc,
        &rigid_body_assets,
        &num_rigid_body_data_bytes);

    if (rigid_body_data == nullptr) {
        FATAL("Invalid collision hull input");
    }

    loader.loadRigidBodies(rigid_body_assets);
    free(rigid_body_data);
}
#endif

void loadAssets(
        AssetLoader &asset_loader,
        PhysicsLoader &physics_loader,
        Optional<RenderManager> &render_mgr)
{
    uint32_t stick_idx = asset_loader.addGlobalAsset(
        (std::filesystem::path(DATA_DIR) / "cylinder_long_render.obj").string(),
        (std::filesystem::path(DATA_DIR) / "cylinder_long.obj").string());

    assert(stick_idx == (uint32_t)SimObject::Stick);

    std::vector extra_materials = {
        SourceMaterial { Vector4{0.4f, 0.4f, 0.4f, 0.0f}, -1, 0.8f, 0.2f },
        SourceMaterial { Vector4{1.0f, 0.1f, 0.1f, 0.0f}, -1, 0.8f, 0.2f },
    };

    std::vector mat_overrides = {
        AssetLoader::MaterialOverride { 0, stick_idx },
        AssetLoader::MaterialOverride { 0, 1 },
        AssetLoader::MaterialOverride { 1, 2 },
        AssetLoader::MaterialOverride { 0, 0 },
    };

    asset_loader.finish(physics_loader,
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
}

struct Manager::Impl {
    Config cfg;
    Optional<RenderGPUState> renderGPUState;
    Optional<render::RenderManager> renderMgr;
    CVXSolve *cvxSolve;

    inline Impl(const Manager::Config &mgr_cfg,
                Optional<RenderGPUState> &&render_gpu_state,
                Optional<render::RenderManager> &&render_mgr,
                CVXSolve *cvx_solve)
        : cfg(mgr_cfg),
          renderGPUState(std::move(render_gpu_state)),
          renderMgr(std::move(render_mgr)),
          cvxSolve(cvx_solve)
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
                   MWCudaLaunchGraph &&step_graph)
        : Impl(mgr_cfg,
               std::move(render_gpu_state), std::move(render_mgr),
               nullptr),
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
                   TaskGraphT &&cpu_exec)
        : Impl(mgr_cfg,
               std::move(render_gpu_state), std::move(render_mgr),
               cvx_solve),
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

        PhysicsLoader phys_loader(mgr_cfg.execMode, 10);

        loadAssets(asset_loader, phys_loader, render_mgr);

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

        return new CUDAImpl {
            mgr_cfg,
            std::move(render_gpu_state),
            std::move(render_mgr),
            std::move(gpu_exec),
            std::move(init_graph),
            std::move(step_graph),
        };
#else
        FATAL("Madrona was not compiled with CUDA support");
#endif
    } break;
    case ExecMode::CPU: {
        // Hello
        PhysicsLoader phys_loader(ExecMode::CPU, 10);
        //loadPhysicsAssets(phys_loader);

        Optional<RenderGPUState> render_gpu_state =
            initRenderGPUState(mgr_cfg);

        Optional<render::RenderManager> render_mgr =
            initRenderManager(mgr_cfg, render_gpu_state);

        loadAssets(asset_loader, phys_loader, render_mgr);

#if 0
        auto imported_assets = loadRenderAssets(
                render_mgr);
#endif


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

        return new CPUImpl {
            mgr_cfg,
            std::move(phys_loader),
            std::move(render_gpu_state),
            mgr_cfg.cvxSolve,
            std::move(render_mgr),
            std::move(cpu_exec)
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

}
