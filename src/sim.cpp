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
// namespace PhysicsSystem = madrona::phys::PhysicsSystem;

namespace madPhysics {

constexpr inline CountT numPhysicsSubsteps = 1;
constexpr inline auto physicsSolverSelector = PhysicsSystem::Solver::Convex;

// Register all the ECS components and archetypes that will be
// used in the simulation
void Sim::registerTypes(ECSRegistry &registry, const Config &cfg)
{
    base::registerTypes(registry);

    PhysicsSystem::registerTypes(registry, physicsSolverSelector);
    RenderingSystem::registerTypes(registry, cfg.renderBridge);

    registry.registerArchetype<DynamicObject>();
}

#define DYNAMIC_MOVEMENT

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

    // For now the step does nothing but just setup the rendering tasks
    // for the visualizer.
    auto render_sys = RenderingSystem::setupTasks(builder, {physics_cleanup});

    (void)render_sys;
}

// Build the task graph
void Sim::setupTasks(TaskGraphManager &taskgraph_mgr, const Config &cfg)
{
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

Sim::Sim(Engine &ctx,
         const Config &cfg,
         const WorldInit &)
    : WorldBase(ctx)
{
    ctx.data().initRandKey = cfg.initRandKey;
    ctx.data().rng = RNG(rand::split_i(ctx.data().initRandKey,
        0, (uint32_t)ctx.worldID().idx));

    PhysicsSystem::init(ctx, cfg.rigidBodyObjMgr,
                        consts::deltaT, 1,
                        -9.8f * math::up, 100,
                        physicsSolverSelector,
                        cfg.cvxSolve);
    RenderingSystem::init(ctx, cfg.renderBridge);

    { // Make the articulated sticks
        stickBodyGrp = cv::makeCVBodyGroup(ctx);



        // This instruction and the following just create the bodies.
        // After the bodies have been created, we need to set the
        // relationship between the two.
        stickRoot = makeDynObject(ctx,
                              Vector3{ 0.f, 0.f, 40.01f },
                              Quat::angleAxis(0.5f, { 1.f, 1.f, 1.f }),
                              Diag3x3{ 1.f, 1.f, 1.f },
                              ResponseType::Dynamic,
                              SimObject::Stick,
                              6);

        stickChild = makeDynObject(ctx,
                              Vector3{ 0.f, 0.f, 40.01f },
                              Quat::angleAxis(0.5f, { 1.f, 1.f, 1.f }),
                              Diag3x3{ 1.f, 1.f, 1.f },
                              ResponseType::Dynamic,
                              SimObject::Stick,
                              1);



        // Configure the root of the armature
        cv::setCVGroupRoot(ctx, stickBodyGrp, stickRoot);

        // Configure the parent/child relationship
        cv::setCVEntityParentHinge(ctx,
                                 stickBodyGrp,
                                 stickRoot,
                                 stickChild,
                                 Vector3 { 0.f, 0.f, 15.f },
                                 Vector3 { 0.f, 0.f, 15.f },
                                 Vector3 { 1.f, 0.f, 0.f });
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
}

// This declaration is needed for the GPU backend in order to generate the
// CUDA kernel for world initialization, which needs to be specialized to the
// application's world data type (Sim) and config and initialization types.
// On the CPU it is a no-op.
MADRONA_BUILD_MWGPU_ENTRY(Engine, Sim, Sim::Config, Sim::WorldInit);

}
