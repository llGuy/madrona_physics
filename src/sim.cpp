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

static void createObject(Engine &ctx,
                         Vector3 position,
                         Quat rotation,
                         Diag3x3 scale,
                         SimObject obj)
{
    Entity grp = cv::makeBodyGroup(ctx, 1);

    Entity l0;

    float obj_mass = PhysicsSystem::getObjectMass(
            ctx, (int32_t)obj);
    Diag3x3 obj_inertia = PhysicsSystem::getObjectInertia(
            ctx, (int32_t)obj);
    float obj_mus = PhysicsSystem::getObjectMuS(
            ctx, (int32_t)obj);

    printf("Cube mu = %f\n", obj_mus);

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
                .mass = obj_mass,
                .inertia = obj_inertia,
                .muS = obj_mus,
            });
    }

    { // Now, we need to create the collision / visual objects
        cv::attachCollision(
            ctx, grp, l0, 0,
            cv::CollisionDesc {
                .objID = (uint32_t)obj,
                .offset = Vector3::all(0.f),
                .rotation = Quat::id(),
                .scale = scale,
            });
        cv::attachVisual(
            ctx, grp, l0, 0,
            cv::VisualDesc {
                .objID = (uint32_t)obj,
                .offset = Vector3::all(0.f),
                .rotation = Quat::id(),
                .scale = scale,
            });
    }

    { // Now, we need to specify the relationship between these links
        // Set the root
        cv::setRoot(ctx, grp, l0);
    }
}

static void createExampleSlider(Engine &ctx)
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
                .type = cv::DofType::Slider,
                .initialPos = Vector3 { 0.f, 0.f, 50.0f },
                .initialRot = Quat::angleAxis(0.5f, { 0.f, 1.f, 0.f }),
                .responseType = phys::ResponseType::Dynamic,
                .numCollisionObjs = 0,
                .numVisualObjs = 1,
                .mass = 0.01f,
                .inertia = { 1.f, 1.f, 1.f },
                .muS = 1.f,
            });

        l2 = cv::makeBody(
            ctx,
            grp,
            cv::BodyDesc {
                .type = cv::DofType::Ball,
                .initialPos = Vector3 { 40.1f, 0.f, 0.f },
                .initialRot = Quat::angleAxis(math::pi / 2.f, { 1.f, 0.f, 0.f }),
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

        cv::attachVisual(
            ctx, grp, l1, 0,
            cv::VisualDesc {
                .objID = (uint32_t)SimObject::Sphere,
                .offset = Vector3::all(0.f),
                .rotation = Quat::id(),
                .scale = { 0.5f, 0.5f, 0.5f },
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
            cv::JointSlider {
                .relPositionParent = Vector3 { 0.f, 0.f, 0.f },
                .relPositionChild = Vector3 { 0.f, 0.f, 0.f },
                .relParentRotation = Quat::id(),
                .slideVector = Vector3 { 0.f, 0.f, 1.f },
            });

        cv::joinBodies(
            ctx, grp, l1, l2,
            cv::JointHinge {
                .relPositionParent = Vector3 { 3.f, 0.f, 0.f },
                .relPositionChild = Vector3 { 0.f, 0.f, 16.f },
                .relParentRotation = Quat::id(),
                .hingeAxis = Vector3 { 1.f, 0.f, 0.f },
            });
    }

    { // Attach some joint limits
        cv::attachLimit(
            ctx, grp, l1,
            cv::SliderLimit {
                .lower = -8.f,
                .upper = 8.f
            });
    }
}

static void createExampleBodyGroup0(Engine &ctx)
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
                .relParentRotation = Quat::id(),
                .hingeAxis = Vector3 { 1.f, 0.f, 0.f },
            });

        cv::joinBodies(
            ctx, grp, l1, l2,
            cv::JointHinge {
                .relPositionParent = Vector3 { 0.f, 0.f, 16.f },
                .relPositionChild = Vector3 { 0.f, 0.f, 16.f },
                .relParentRotation = Quat::id(),
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

static void createExampleBodyGroup1(Engine &ctx)
{
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
                .relParentRotation = Quat::id(),
                .hingeAxis = Vector3 { 1.f, 0.f, 0.f },
            });

        cv::joinBodies(
            ctx, grp, l1, l2,
            cv::JointHinge {
                .relPositionParent = Vector3 { 0.f, 0.f, 16.f },
                .relPositionChild = Vector3 { 0.f, 0.f, 16.f },
                .relParentRotation = Quat::angleAxis(math::pi / 8.f, { 1.f, 0.f, 0.f }),
                .hingeAxis = Vector3 { 1.f, 0.f, 0.f },
            });
    }

#if 1
    { // Set joint limits
        cv::attachLimit(
            ctx, grp, l1,
            cv::SliderLimit {
                .lower = 0.f,
                .upper = math::pi / 8.f
            });

        cv::attachLimit(
            ctx, grp, l2,
            cv::SliderLimit {
                .lower = 0.f,
                .upper = math::pi / 8.f
            });
    }
#endif
}

// This body group has multiple children
static void createExampleBodyGroup2(Engine &ctx)
{
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
                .relPositionChild = Vector3 { 0.f, 1.f, 16.f },
                .relParentRotation = Quat::angleAxis(math::pi / 8.f, { -1.f, 0.f, 0.f }),
                .hingeAxis = Vector3 { 1.f, 0.f, 0.f },
            });

        cv::joinBodies(
            ctx, grp, l0, l2,
            cv::JointHinge {
                .relPositionParent = Vector3 { 0.f, 0.f, 16.f },
                .relPositionChild = Vector3 { 0.f, -1.f, 16.f },
                .relParentRotation = Quat::angleAxis(math::pi / 8.f, { 1.f, 0.f, 0.f }),
                .hingeAxis = Vector3 { 1.f, 0.f, 0.f },
            });
    }

#if 1
    { // Set joint limits
        cv::attachLimit(
            ctx, grp, l1,
            cv::SliderLimit {
                .lower = 0.f,
                .upper = math::pi / 8.f
            });

        cv::attachLimit(
            ctx, grp, l2,
            cv::SliderLimit {
                .lower = 0.f,
                .upper = math::pi / 8.f
            });
    }
#endif
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

    // Set mu to 1 for now (TODO: require some form of priority)
    plane_mus = 1.f;

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
    Diag3x3 id_scale = { 1.f, 1.f, 1.f };

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

            createObject(ctx, pos, rot, id_scale, SimObject::Stick);
        }
    }
}

Entity createURDFModel(Engine &ctx,
                       const Sim::Config &cfg)
{
    Entity urdf_model = cv::loadModel(
            ctx, cfg.modelConfigs[0], cfg.modelData,
            Vector3 { 0.f, 0.f, 30.f },
            Quat::id());

    return urdf_model;
}

void Sim::makePhysicsObjects(Engine &ctx,
                             const Config &cfg)
{
    PhysicsSystem::init(ctx, cfg.rigidBodyObjMgr,
            consts::deltaT, 1,
            -9.8f * math::up, 100,
            physicsSolverSelector,
            (CVXSolve *)cfg.cvxSolve);

#if 1
    createExampleBodyGroup2(ctx);

    createObject(
            ctx, 
            { -40.f, -40.f, 10.f },
            Quat::angleAxis(math::pi / 8.f, { 1.f, 1.f, 1.f }),
            { 4.f, 4.f, 4.f },
            SimObject::Cube);

    createObject(
            ctx, 
            { -40.f, -40.f, 25.f },
            Quat::angleAxis(math::pi / 8.f, { -1.f, 1.f, 1.f }),
            { 4.f, 4.f, 4.f },
            SimObject::Cube);
#else
    createURDFModel(ctx, cfg);
#endif

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
