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

// Register all the ECS components and archetypes that will be
// used in the simulation
void Sim::registerTypes(ECSRegistry &registry, const Config &cfg)
{
    base::registerTypes(registry);

    RenderingSystem::registerTypes(registry, cfg.renderBridge);
    PhysicsSystem::registerTypes(registry, physicsSolverSelector);
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
    if (cfg.renderBridge) {
        auto render_sys = RenderingSystem::setupTasks(
                builder, {physics_cleanup});
    }
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

static void createStick(Engine &ctx,
                        Vector3 position,
                        Quat rotation)
{
    Entity grp = cv::makeBodyGroup(ctx, 1);

    Entity l0;

    float stick_mass = PhysicsSystem::getObjectMass(
            ctx, (int32_t)SimObject::Stick);
    Diag3x3 stick_inertia = PhysicsSystem::getObjectInertia(
            ctx, (int32_t)SimObject::Stick);
    float stick_mus = PhysicsSystem::getObjectMuS(
            ctx, (int32_t)SimObject::Stick);

    { // Create the body
        l0 = cv::makeBody(
            ctx,
            grp,
            cv::BodyDesc {
                .type = cv::DofType::FreeBody,
                .initialPos = position,
                .initialRot = rotation,
                .responseType = phys::ResponseType::Dynamic,
                .numCollisionObjs = 1,
                .numVisualObjs = 1,
                .mass = stick_mass,
                .inertia = stick_inertia,
                .muS = stick_mus,
            });
    }

    { // Now, we need to create the collision / visual objects
        cv::attachCollision(
            ctx, grp, l0, 0,
            cv::CollisionDesc {
                .objID = (uint32_t)SimObject::Stick,
                .offset = Vector3::all(0.f),
                .rotation = Quat::id(),
                .scale = Diag3x3::id(),
            });
        cv::attachVisual(
            ctx, grp, l0, 0,
            cv::VisualDesc {
                .objID = (uint32_t)SimObject::Stick,
                .offset = Vector3::all(0.f),
                .rotation = Quat::id(),
                .scale = Diag3x3::id(),
            });
    }

    { // Now, we need to specify the relationship between these links
        // Set the root
        cv::setRoot(ctx, grp, l0);
    }
}

static void createExampleBodyGroup(Engine &ctx)
{
    // Example of creating an articulated rigid body
    Entity grp = cv::makeBodyGroup(ctx, 3);

    Entity l0, l1, l2;

    float stick_mass = PhysicsSystem::getObjectMass(
            ctx, (int32_t)SimObject::Stick);
    Diag3x3 stick_inertia = PhysicsSystem::getObjectInertia(
            ctx, (int32_t)SimObject::Stick);
    float stick_mus = PhysicsSystem::getObjectMuS(
            ctx, (int32_t)SimObject::Stick);

    { // First step is to create all the links
        l0 = cv::makeBody(
            ctx,
            grp,
            cv::BodyDesc {
                .type = cv::DofType::FreeBody,
                .initialPos = Vector3 { 0.f, 0.f, 60.0f },
                .initialRot = Quat::angleAxis(0.f, { 0.f, 0.f, 1.f }),
                .responseType = phys::ResponseType::Dynamic,
                .numCollisionObjs = 1,
                .numVisualObjs = 1,
                .mass = stick_mass,
                .inertia = stick_inertia,
                .muS = stick_mus,
            });

        l1 = cv::makeBody(
            ctx,
            grp,
            cv::BodyDesc {
                .type = cv::DofType::Hinge,
                .initialPos = Vector3 { 0.f, 0.f, 40.01f },
                .initialRot = Quat::angleAxis(0.5f, { 2.f, 1.f, 1.f }),
                .responseType = phys::ResponseType::Dynamic,
                .numCollisionObjs = 1,
                .numVisualObjs = 1,
                .mass = stick_mass,
                .inertia = stick_inertia,
                .muS = stick_mus,
            });

        l2 = cv::makeBody(
            ctx,
            grp,
            cv::BodyDesc {
                .type = cv::DofType::Hinge,
                .initialPos = Vector3 { 0.f, 0.f, 40.01f },
                .initialRot = Quat::angleAxis(0.5f, { 2.f, 1.f, 1.f }),
                .responseType = phys::ResponseType::Dynamic,
                .numCollisionObjs = 1,
                .numVisualObjs = 1,
                .mass = stick_mass,
                .inertia = stick_inertia,
                .muS = stick_mus,
            });
    }

    { // Now, we need to create the collision / visual objects
        cv::attachCollision(
            ctx, grp, l0, 0,
            cv::CollisionDesc {
                .objID = (uint32_t)SimObject::Stick,
                .offset = Vector3::all(0.f),
                .rotation = Quat::id(),
                .scale = Diag3x3::id(),
            });
        cv::attachVisual(
            ctx, grp, l0, 0,
            cv::VisualDesc {
                .objID = (uint32_t)SimObject::Stick,
                .offset = Vector3::all(0.f),
                .rotation = Quat::id(),
                .scale = Diag3x3::id(),
            });

        cv::attachCollision(
            ctx, grp, l1, 0,
            cv::CollisionDesc {
                .objID = (uint32_t)SimObject::Stick,
                .offset = Vector3::all(0.f),
                .rotation = Quat::id(),
                .scale = Diag3x3::id(),
            });
        cv::attachVisual(
            ctx, grp, l1, 0,
            cv::VisualDesc {
                .objID = (uint32_t)SimObject::Stick,
                .offset = Vector3::all(0.f),
                .rotation = Quat::id(),
                .scale = Diag3x3::id(),
            });

        cv::attachCollision(
            ctx, grp, l2, 0,
            cv::CollisionDesc {
                .objID = (uint32_t)SimObject::Stick,
                .offset = Vector3::all(0.f),
                .rotation = Quat::id(),
                .scale = Diag3x3::id(),
            });
        cv::attachVisual(
            ctx, grp, l2, 0,
            cv::VisualDesc {
                .objID = (uint32_t)SimObject::Stick,
                .offset = Vector3::all(0.f),
                .rotation = Quat::id(),
                .scale = Diag3x3::id(),
            });
    }

    { // Now, we need to specify the relationship between these links
        // Set the root
        cv::setRoot(ctx, grp, l0);

        cv::joinBodies(
            ctx, grp, l0, l1,
            cv::JointHinge {
                .relPositionParent = Vector3 { 0.f, 0.f, 16.f },
                .relPositionChild = Vector3 { 0.f, 0.f, 16.f },
                .hingeAxis = Vector3 { 1.f, 0.f, 0.f },
            });

        cv::joinBodies(
            ctx, grp, l1, l2,
            cv::JointHinge {
                .relPositionParent = Vector3 { 0.f, 0.f, 16.f },
                .relPositionChild = Vector3 { 0.f, 0.f, 16.f },
                .hingeAxis = Vector3 { 1.f, 0.f, 0.f },
            });
    }

    { // Attach some joint limits
        cv::attachLimit(
            ctx, grp, l1,
            cv::HingeLimit {
                .lower = 0.1f,
                .upper = math::pi / 2.f
            });

        cv::attachLimit(
            ctx, grp, l2,
            cv::HingeLimit {
                .lower = 0.1f,
                .upper = math::pi / 2.f
            });
    }
}

static void createFloorPlane(Engine &ctx)
{
    Entity grp = cv::makeBodyGroup(ctx, 1);

    Entity l0;

    float plane_mass = PhysicsSystem::getObjectMass(
            ctx, (int32_t)SimObject::Plane);
    Diag3x3 plane_inertia = PhysicsSystem::getObjectInertia(
            ctx, (int32_t)SimObject::Plane);
    float plane_mus = PhysicsSystem::getObjectMuS(
            ctx, (int32_t)SimObject::Plane);

    { // Create the body
        l0 = cv::makeBody(
            ctx,
            grp,
            cv::BodyDesc {
                .type = cv::DofType::FixedBody,
                .initialPos = Vector3 { 0.f, 0.f, 1.f },
                .initialRot = Quat::angleAxis(0.5f, { 0.f, 0.f, 1.f }),
                .responseType = phys::ResponseType::Static,
                .numCollisionObjs = 1,
                .numVisualObjs = 1,
                .mass = plane_mass,
                .inertia = plane_inertia,
                .muS = plane_mus,
            });
    }

    { // Now, we need to create the collision / visual objects
        cv::attachCollision(
            ctx, grp, l0, 0,
            cv::CollisionDesc {
                .objID = (uint32_t)SimObject::Plane,
                .offset = Vector3::all(0.f),
                .rotation = Quat::id(),
                .scale = Diag3x3 { 0.03f, 0.03f, 0.03f },
            });
        cv::attachVisual(
            ctx, grp, l0, 0,
            cv::VisualDesc {
                .objID = (uint32_t)SimObject::Plane,
                .offset = Vector3::all(0.f),
                .rotation = Quat::id(),
                .scale = Diag3x3 { 0.03f, 0.03f, 0.03f },
            });
    }

    { // Now, we need to specify the relationship between these links
        // Set the root
        cv::setRoot(ctx, grp, l0);
    }
}

static void makeExampleConfig0(Engine &ctx,
                               const Sim::Config &cfg)
{
    for (int j = 0; j < 3; ++j) {
        for (int i = 0; i < 3; ++i) {
            float random_angle = ctx.data().rng.sampleUniform() *
                math::pi * 2.f;

            Vector3 pos = { 
                (float)j * 20.f,
                0.f,
                60.0f + (float)i * 10.0f + (float) j * 4.f 
            };


            Quat rot = 
                Quat::angleAxis(random_angle, {0.f, 0.f, 1.f}) *
                Quat::angleAxis(
                    math::pi / 2.f,
                    { 1.f, 0.f, 0.f });

            createStick(ctx, pos, rot);
        }
    }
}

void Sim::makePhysicsObjects(Engine &ctx,
                             const Config &cfg)
{
    PhysicsSystem::init(ctx, cfg.rigidBodyObjMgr,
            consts::deltaT, 1,
            -9.8f * math::up, 100,
            physicsSolverSelector,
            (CVXSolve *)cfg.cvxSolve);

    // createRigidBody(ctx);
    createExampleBodyGroup(ctx);

    // makeExampleConfig0(ctx, cfg);
    createFloorPlane(ctx);
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

    makePhysicsObjects(ctx, cfg);
}

// This declaration is needed for the GPU backend in order to generate the
// CUDA kernel for world initialization, which needs to be specialized to the
// application's world data type (Sim) and config and initialization types.
// On the CPU it is a no-op.
MADRONA_BUILD_MWGPU_ENTRY(Engine, Sim, Sim::Config, Sim::WorldInit);

}
