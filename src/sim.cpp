#include <algorithm>
#include <madrona/mw_gpu_entry.hpp>

#include "sim.hpp"
#include "consts.hpp"

#include <madrona/cvphysics.hpp>

#ifdef MADRONA_GPU_MODE
#include <madrona/mw_gpu/host_print.hpp>
#define LOG(...) mwGPU::HostPrint::log(__VA_ARGS__)
#else
#define LOG(...)
#endif

using namespace madrona;
using namespace madrona::math;
using namespace madrona::phys;

namespace RenderingSystem = madrona::render::RenderingSystem;
namespace PhysicsSystem = madrona::phys::PhysicsSystem;

namespace madPhysics {

constexpr inline CountT numPhysicsSubsteps = 1;
constexpr inline auto physicsSolverSelector = PhysicsSystem::Solver::Convex;

// 32-byte range map unit
struct RMUnit32 {
    uint8_t data[32];
};

inline void testRM(Engine &ctx,
                   TestSingleton &s)
{
    if (ctx.data().freed == 0) {
        auto &testRM = ctx.data().testRM;

        RMUnit32 *units = ctx.memoryRangePointer<RMUnit32>(testRM);
        uint32_t *data = (uint32_t *)units;

        assert(data[0] == 42 * (ctx.worldID().idx + 1));
        assert(data[1] == 43 * (ctx.worldID().idx + 1));
        assert(data[2] == 44 * (ctx.worldID().idx + 1));

        if (s.v == 2) {
            if (ctx.worldID().idx < 6) {
                ctx.freeMemoryRange(ctx.data().testRM);
                ctx.data().freed = 1;
            }
        }
    }

    if (s.v == 4 && ctx.data().freed == 1) {
        if (ctx.worldID().idx < 4) {
            auto &testRM = ctx.data().testRM = ctx.allocMemoryRange<RMUnit32>(3);

            RMUnit32 *units = ctx.memoryRangePointer<RMUnit32>(ctx.data().testRM);
            uint32_t *data = (uint32_t *)units;

            data[0] = 42 * (ctx.worldID().idx + 1);
            data[1] = 43 * (ctx.worldID().idx + 1);
            data[2] = 44 * (ctx.worldID().idx + 1);

            ctx.data().freed = 0;
        }
    }

    s.v++;
}

// Register all the ECS components and archetypes that will be
// used in the simulation
void Sim::registerTypes(ECSRegistry &registry, const Config &cfg)
{
    base::registerTypes(registry);

    PhysicsSystem::registerTypes(registry, physicsSolverSelector);
    RenderingSystem::registerTypes(registry, cfg.renderBridge);

    registry.registerArchetype<DynamicObject>();
    registry.registerArchetype<TestEntity>();

    registry.registerMemoryRangeElement<RMUnit32>();

    registry.registerSingleton<TestSingleton>();
}

#ifdef MADRONA_GPU_MODE
template <typename ArchetypeT>
TaskGraph::NodeID queueSortByWorld(TaskGraph::Builder &builder,
                                   Span<const TaskGraph::NodeID> deps)
{
    auto sort_sys =
        builder.addToGraph<SortArchetypeNode<ArchetypeT, WorldID>>(
            deps);
    auto post_sort_reset_tmp =
        builder.addToGraph<ResetTmpAllocNode>({sort_sys});

    return post_sort_reset_tmp;
}

template <typename RangeMapUnitT>
TaskGraph::NodeID queueSortRangeMap(TaskGraph::Builder &builder,
                                    Span<const TaskGraph::NodeID> deps)
{
    auto sort_sys =
        builder.addToGraph<SortMemoryRangeNode<RMUnit32>>(deps);
    auto post_sort_reset_tmp =
        builder.addToGraph<ResetTmpAllocNode>({sort_sys});

    return post_sort_reset_tmp;
}

#endif

static void setupStepTasks(TaskGraphBuilder &builder, 
                           const Sim::Config &cfg)
{
    (void)cfg;

#if 1
    auto broadphase_setup_sys = phys::PhysicsSystem::setupBroadphaseTasks(
            builder, {});

    auto substep_sys = PhysicsSystem::setupPhysicsStepTasks(builder,
        {broadphase_setup_sys}, numPhysicsSubsteps, physicsSolverSelector);

    auto physics_cleanup = phys::PhysicsSystem::setupCleanupTasks(
        builder, {substep_sys});
#endif

#if 0
    auto test_sys = builder.addToGraph<ParallelForNode<Engine,
        testRM,
            TestSingleton
       >>({});
#endif

    // For now the step does nothing but just setup the rendering tasks
    // for the visualizer.
    if (cfg.renderBridge) {
        auto render_sys = RenderingSystem::setupTasks(
                builder, {physics_cleanup});
    }

#if 0
#ifdef MADRONA_GPU_MODE
    queueSortRangeMap<RMUnit32>(builder, {test_sys});
#endif
#endif
}

static void setupInitTasks(TaskGraphBuilder &builder,
                           const Sim::Config &cfg)
{
    auto node = phys::PhysicsSystem::setupInitTasks(
            builder, {}, physicsSolverSelector);
    (void)node;
}

// Build the task graph
void Sim::setupTasks(TaskGraphManager &taskgraph_mgr, const Config &cfg)
{
    setupInitTasks(taskgraph_mgr.init(TaskGraphID::Init), cfg);
    setupStepTasks(taskgraph_mgr.init(TaskGraphID::Step), cfg);
}

static Entity makeDynObject(Engine &ctx,
                            Vector3 pos,
                            Quat rot,
                            Diag3x3 scale,
                            ResponseType type,
                            SimObject obj,
                            uint32_t num_dofs)
{
    Entity e = ctx.makeRenderableEntity<DynamicObject>();
    ctx.get<Position>(e) = pos;
    ctx.get<Rotation>(e) = rot.normalize();
    ctx.get<Scale>(e) = scale;
    ObjectID e_obj_id = ObjectID { (int32_t)obj };
    ctx.get<ObjectID>(e) = e_obj_id;

    ctx.get<phys::broadphase::LeafID>(e) =
        PhysicsSystem::registerEntity(ctx, e, e_obj_id,
                                      num_dofs,
                                      physicsSolverSelector);

    ctx.get<Velocity>(e) = {
        Vector3::zero(),
        Vector3::zero(),
    };
    ctx.get<ResponseType>(e) = type;
    ctx.get<ExternalForce>(e) = Vector3::zero();
    ctx.get<ExternalTorque>(e) = Vector3::zero();

    return e;
}

void Sim::makePhysicsObjects(Engine &ctx,
                             const Config &cfg)
{
#if 1
    PhysicsSystem::init(ctx, cfg.rigidBodyObjMgr,
            consts::deltaT, 1,
            -9.8f * math::up, 100,
            physicsSolverSelector,
            (CVXSolve *)cfg.cvxSolve);
    RenderingSystem::init(ctx, cfg.renderBridge);

    for (int j = 0; j < 3; ++j) {
        for (int i = 0; i < 3; ++i) {
            float random_angle = ctx.data().rng.sampleUniform() *
                math::pi * 2.f;

            Entity grp = cv::makeCVBodyGroup(ctx, 1);
            Entity e = makeDynObject(ctx,
                              Vector3{ (float)j * 20.f, 0.f, 60.0f + (float)i * 10.0f + (float) j * 4.f },
                              // Quat::angleAxis((float)(ctx.worldID().idx + 1) * ((float)i + (float)j), { 0.f, 0.f, 1.f }) * 
                              Quat::angleAxis(random_angle, {0.f, 0.f, 1.f}) *
                              Quat::angleAxis(
                                  ///* (float)ctx.worldID().idx */ 8.f * (float)i,
                                  math::pi / 2.f,
                                  { 1.f, 0.f, 0.f }),
                              // Quat::angleAxis(
                                  // 1.f, { 1.f, 1.f, 1.f }),
                              Diag3x3{ 1.f, 1.f, 1.f },
                              ResponseType::Dynamic,
                              SimObject::Stick,
                              6);
            cv::setCVGroupRoot(ctx, grp, e);
        }
    }

    { // Make the plane
        plane = makeDynObject(ctx,
                              Vector3{ 0.f, 0.f, 1.f },
                              Quat::angleAxis(0.5f, { 0.f, 0.f, 1.f }),
                              Diag3x3{ 0.03f, 0.03f, 0.03f },
                              ResponseType::Static,
                              SimObject::Plane,
                              0);
    }
#endif

#if 0
    PhysicsSystem::init(ctx, cfg.rigidBodyObjMgr,
                        consts::deltaT, 1,
                        -9.8f * math::up, 100,
                        physicsSolverSelector,
                        (CVXSolve *)cfg.cvxSolve);
    RenderingSystem::init(ctx, cfg.renderBridge);

    for (int i = 0; i < 3; ++i) {
        Entity grp = cv::makeCVBodyGroup(ctx, 1);
        Entity e = makeDynObject(ctx,
                          Vector3{ 20.f, 0.f, 60.0f + (float)i * 4.0f },
                          Quat::angleAxis(1.f, { 1.f, 1.f, 1.f }),
                          Diag3x3{ 1.f, 1.f, 1.f },
                          ResponseType::Dynamic,
                          SimObject::Stick,
                          6);
        cv::setCVGroupRoot(ctx, grp, e);
    }

    { // Make the articulated sticks
        stickBodyGrp = cv::makeCVBodyGroup(ctx, 3);



        // This instruction and the following just create the bodies.
        // After the bodies have been created, we need to set the
        // relationship between the two.
        stickRoot = makeDynObject(ctx,
                              Vector3{ 0.f, 0.f, 60.0f },
                              Quat::angleAxis(0.f, { 0.f, 0.f, 1.f }),
                              Diag3x3{ 1.f, 1.f, 1.f },
                              ResponseType::Dynamic,
                              SimObject::Stick,
                              6);

        stickChild = makeDynObject(ctx,
                              Vector3{ 0.f, 0.f, 40.01f },
                              Quat::angleAxis(0.5f, { 2.f, 1.f, 1.f }),
                              Diag3x3{ 1.f, 1.f, 1.f },
                              ResponseType::Dynamic,
                              SimObject::Stick,
                              3);

        Entity stickChild2 = makeDynObject(ctx,
                              Vector3{ 0.f, 0.f, 40.01f },
                              Quat::angleAxis(0.5f, { 2.f, 1.f, 1.f }),
                              Diag3x3{ 1.f, 1.f, 1.f },
                              ResponseType::Dynamic,
                              SimObject::Stick,
                              3);


        // Configure the root of the armature
        cv::setCVGroupRoot(ctx, stickBodyGrp, stickRoot);

        // Configure the parent/child relationship
        cv::setCVEntityParentBall(ctx,
                                 stickBodyGrp,
                                 stickRoot,
                                 stickChild,
                                 Vector3 { 0.f, 0.f, 16.f },
                                 Vector3 { 0.f, 0.f, 16.f });
                                 // Vector3 { 1.f, 0.f, 0.f });
        cv::setCVEntityParentBall(ctx,
                                 stickBodyGrp,
                                 stickChild,
                                 stickChild2,
                                 Vector3 { 0.f, 0.f, 16.f },
                                 Vector3 { 0.f, 0.f, 16.f });
    }

    { // Make the plane
        plane = makeDynObject(ctx,
                              Vector3{ 0.f, 0.f, 1.f },
                              Quat::angleAxis(0.5f, { 0.f, 0.f, 1.f }),
                              Diag3x3{ 0.01f, 0.01f, 0.01f },
                              ResponseType::Static,
                              SimObject::Plane,
                              0);
    }
#endif
}

void Sim::makeRangeMapTest(Engine &ctx)
{
    testRM = ctx.allocMemoryRange<RMUnit32>(3);

    RMUnit32 *units = ctx.memoryRangePointer<RMUnit32>(testRM);
    uint32_t *data = (uint32_t *)units;

    data[0] = 42 * (ctx.worldID().idx + 1);
    data[1] = 43 * (ctx.worldID().idx + 1);
    data[2] = 44 * (ctx.worldID().idx + 1);
}

Sim::Sim(Engine &ctx,
         const Config &cfg,
         const WorldInit &)
    : WorldBase(ctx)
{
    ctx.data().initRandKey = cfg.initRandKey;
    ctx.data().rng = RNG(rand::split_i(ctx.data().initRandKey,
        0, (uint32_t)ctx.worldID().idx));

    ctx.singleton<TestSingleton>().v = 0;

    ctx.data().freed = 0;

#if 1
    makePhysicsObjects(ctx, cfg);
#else
    makeRangeMapTest(ctx);
#endif

    Entity test = ctx.makeEntity<TestEntity>();
}

// This declaration is needed for the GPU backend in order to generate the
// CUDA kernel for world initialization, which needs to be specialized to the
// application's world data type (Sim) and config and initialization types.
// On the CPU it is a no-op.
MADRONA_BUILD_MWGPU_ENTRY(Engine, Sim, Sim::Config, Sim::WorldInit);

}
