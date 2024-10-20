#include <algorithm>
#include <madrona/mw_gpu_entry.hpp>

#include "sim.hpp"

#ifdef MADRONA_GPU_MODE
#include <madrona/mw_gpu/host_print.hpp>
#define LOG(...) mwGPU::HostPrint::log(__VA_ARGS__)
#else
#define LOG(...)
#endif

using namespace madrona;
using namespace madrona::math;

namespace RenderingSystem = madrona::render::RenderingSystem;

namespace madEscape {

// Register all the ECS components and archetypes that will be
// used in the simulation
void Sim::registerTypes(ECSRegistry &registry, const Config &cfg)
{
    base::registerTypes(registry);

    RenderingSystem::registerTypes(registry, cfg.renderBridge);

    registry.registerArchetype<RigidBody>();
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

    // For now the step does nothing but just setup the rendering tasks
    // for the visualizer.
    RenderingSystem::setupTasks(builder, {});
}

// Build the task graph
void Sim::setupTasks(TaskGraphManager &taskgraph_mgr, const Config &cfg)
{
    setupStepTasks(taskgraph_mgr.init(TaskGraphID::Step), cfg);
}

Sim::Sim(Engine &ctx,
         const Config &cfg,
         const WorldInit &)
    : WorldBase(ctx)
{
    ctx.data().initRandKey = cfg.initRandKey;
    ctx.data().rng = RNG(rand::split_i(ctx.data().initRandKey,
        0, (uint32_t)ctx.worldID().idx));

    RenderingSystem::init(ctx, cfg.renderBridge);



    { // Make the stick
        stick = ctx.makeRenderableEntity<RigidBody>();
        ctx.get<Position>(stick) = Vector3 {
            0.f, 0.f, 40.f,
        };
        ctx.get<Rotation>(stick) = Quat::angleAxis(
                0.5f, { 1.f, 1.f, 1.f, });
        ctx.get<Scale>(stick) = Diag3x3 { 1.f, 1.f, 10.f, };
        ctx.get<ObjectID>(stick) = { (uint32_t)SimObject::Stick };
    }

    { // Make the plane
        plane = ctx.makeRenderableEntity<RigidBody>();
        ctx.get<Position>(plane) = Vector3 {
            0.f, 0.f, 1.f,
        };
        ctx.get<Rotation>(plane) = Quat::angleAxis(
                0.f, { 0.f, 0.f, 1.f, });
        ctx.get<Scale>(plane) = Diag3x3 { 0.01f, 0.01f, 0.01f, };
        ctx.get<ObjectID>(plane) = { (uint32_t)SimObject::Plane };
    }
}

// This declaration is needed for the GPU backend in order to generate the
// CUDA kernel for world initialization, which needs to be specialized to the
// application's world data type (Sim) and config and initialization types.
// On the CPU it is a no-op.
MADRONA_BUILD_MWGPU_ENTRY(Engine, Sim, Sim::Config, Sim::WorldInit);

}
