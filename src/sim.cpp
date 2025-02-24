#include <algorithm>
#include <madrona/mw_gpu_entry.hpp>

#include "sim.hpp"
#include "consts.hpp"

#include <madrona/cv_load.hpp>
#include <madrona/cv_physics.hpp>

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

    registry.registerComponent<Action>();
    registry.registerArchetype<ActorArchetype>();

    registry.exportColumn<ActorArchetype, Action>(
            ExportID::Action);
}

void actionTask(Engine &ctx,
                Action &action)
{
    switch (ctx.data().envType) {
    case EnvType::Car: {
        if (action.v == 1) {
            Entity hinge = ctx.data().carHinge;
            cv::addHingeExternalForce(ctx, hinge, 70.f);
        } else if (action.v == -1) {
            Entity hinge = ctx.data().carHinge;
            cv::addHingeExternalForce(ctx, hinge, -70.f);
        } else {
            Entity hinge = ctx.data().carHinge;
            cv::addHingeExternalForce(ctx, hinge, 0.f);
        }
    } break;

    default: {
    } break;
    }

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

    auto action_node = builder.addToGraph<ParallelForNode<Engine, actionTask,
         Action
        >>({});

#if 1
    auto broadphase_setup_sys = phys::PhysicsSystem::setupBroadphaseTasks(
            builder, {action_node});

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
    (void)cfg;

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
    Entity grp = cv::makeBodyGroup(ctx, 1, 2.f);

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
                .numLimits = 0,
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

static void createCar2(Engine &ctx)
{
    Entity grp = cv::makeBodyGroup(ctx, 4);
    
    Entity free_body, link0, wheel0, wheel1;

    { // Create the bodies
        free_body = cv::makeBody(
                ctx,
                grp,
                cv::BodyDesc {
                    .type = cv::DofType::FreeBody,
                    .initialPos = Vector3 { 0.f, 0.f, 15.f },
                    .initialRot = Quat::angleAxis(
                            math::pi / 2.f, { 1.f, 0.f, 0.f }),
                    .responseType = phys::ResponseType::Dynamic,
                    .numCollisionObjs = 0,
                    .numVisualObjs = 1,
                    .mass = 10.f,
                    .inertia = Diag3x3::uniform(10000.f),
                    .muS = 1.f
                });

        link0 = cv::makeBody(
                ctx,
                grp,
                cv::BodyDesc {
                    .type = cv::DofType::Hinge,
                    .responseType = phys::ResponseType::Dynamic,
                    .numCollisionObjs = 1,
                    .numVisualObjs = 1,
                    .mass = 1.f,
                    .inertia = { 1.f, 1.f, 1.f },
                    .muS = 1.f
                });

        wheel0 = cv::makeBody(
                ctx,
                grp,
                cv::BodyDesc {
                    .type = cv::DofType::FixedBody,
                    .initialPos = Vector3 { 0.f, 0.f, 0.f },
                    .initialRot = Quat::id(),
                    .responseType = phys::ResponseType::Dynamic,
                    .numCollisionObjs = 1,
                    .numVisualObjs = 1,
                    .mass = 1.f,
                    .inertia = { 1.f, 1.f, 1.f },
                    .muS = 1.f
                });

        wheel1 = cv::makeBody(
                ctx,
                grp,
                cv::BodyDesc {
                    .type = cv::DofType::FixedBody,
                    .initialPos = Vector3 { 0.f, 0.f, 0.f },
                    .initialRot = Quat::id(),
                    .responseType = phys::ResponseType::Dynamic,
                    .numCollisionObjs = 1,
                    .numVisualObjs = 1,
                    .mass = 1.f,
                    .inertia = { 1.f, 1.f, 1.f },
                    .muS = 1.f
                });
    }

    { // Attach collision / visual objects
        cv::attachVisual(
            ctx, grp, free_body, 0,
            cv::VisualDesc {
                .objID = (uint32_t)SimObject::Cube,
                .offset = Vector3::all(0.f),
                .rotation = Quat::id(),
                .scale = Diag3x3::uniform(10.f)
            });
        cv::attachCollision(
            ctx, grp, link0, 0,
            cv::CollisionDesc {
                .objID = (uint32_t)SimObject::Stick,
                .offset = Vector3::all(0.f),
                .rotation = Quat::id(),
                .scale = Diag3x3::id(),
            });
        cv::attachVisual(
            ctx, grp, link0, 0,
            cv::VisualDesc {
                .objID = (uint32_t)SimObject::Stick,
                .offset = Vector3::all(0.f),
                .rotation = Quat::id(),
                .scale = Diag3x3::id(),
            });

        cv::attachCollision(
            ctx, grp, wheel0, 0,
            cv::CollisionDesc {
                .objID = (uint32_t)SimObject::Disk,
                .offset = Vector3::all(0.f),
                .rotation = Quat::id(),
                .scale = Diag3x3::id(),
            });
        cv::attachVisual(
            ctx, grp, wheel0, 0,
            cv::VisualDesc {
                .objID = (uint32_t)SimObject::Disk,
                .offset = Vector3::all(0.f),
                .rotation = Quat::id(),
                .scale = Diag3x3::id(),
            });

        cv::attachCollision(
            ctx, grp, wheel1, 0,
            cv::CollisionDesc {
                .objID = (uint32_t)SimObject::Disk,
                .offset = Vector3::all(0.f),
                .rotation = Quat::id(),
                .scale = Diag3x3::id(),
            });
        cv::attachVisual(
            ctx, grp, wheel1, 0,
            cv::VisualDesc {
                .objID = (uint32_t)SimObject::Disk,
                .offset = Vector3::all(0.f),
                .rotation = Quat::id(),
                .scale = Diag3x3::id(),
            });
    }

    { // Create hierarchy
        cv::setRoot(ctx, grp, free_body);

        cv::joinBodies(
            ctx, grp, free_body, link0,
            cv::JointHinge {
                .relPositionParent = Vector3 { 0.f, 0.f, 0.f },
                .relPositionChild = Vector3 { 0.f, 0.f, 0.f },
                .relParentRotation = Quat::id(),
                .hingeAxis = Vector3 { 0.f, 0.f, 1.f },
            });

        cv::joinBodies(
            ctx, grp, link0, wheel0,
            cv::JointFixed {
                .relPositionParent = Vector3 { 0.f, 0.f, 16.f },
                .relPositionChild = Vector3 { 0.f, 0.f, 3.f },
                .relParentRotation = Quat::id(),
            });

        cv::joinBodies(
            ctx, grp, link0, wheel1,
            cv::JointFixed {
                .relPositionParent = Vector3 { 0.f, 0.f, -16.f },
                .relPositionChild = Vector3 { 0.f, 0.f, -3.f },
                .relParentRotation = Quat::id(),
            });
    }

    ctx.data().carHinge = link0;
}

static void createFloorPlane(Engine &ctx, bool visible)
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
                .initialRot = Quat::angleAxis(0.0f, { 0.f, 0.f, 1.f }),
                .responseType = phys::ResponseType::Static,
                .numCollisionObjs = 1,
                .numVisualObjs = visible ? 1u : 0u,
                .numLimits = 0,
                .mass = plane_mass,
                .inertia = plane_inertia,
                .muS = 0.8f,
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

        if (visible) {
            cv::attachVisual(
                ctx, grp, l0, 0,
                cv::VisualDesc {
                    .objID = (uint32_t)SimObject::Plane,
                    .offset = Vector3::all(0.f),
                    .rotation = Quat::id(),
                    .scale = Diag3x3 { 0.03f, 0.03f, 0.03f },
                });
        }
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

            createObject(
                    ctx, 
                    pos + Vector3 { 0.f, 0.f, 10.f },
                    rot,
                    { 4.f, 4.f, 4.f },
                    SimObject::Cube);
        }
    }
}

Entity createURDFModel(Engine &ctx,
                       const Sim::Config &cfg)
{
    Entity urdf_model1 = cv::loadModel(
            ctx, cfg.modelConfigs[0], cfg.modelData,
            Vector3 { 0.f, 0.f, 0.f },
            Quat::id(),
            40.f);
    // cv::setColliderVisualizer(ctx, urdf_model1, false);

    return urdf_model1;
}

void Sim::makePhysicsObjects(Engine &ctx,
                             const Config &cfg)
{
    PhysicsSystem::init(ctx, cfg.rigidBodyObjMgr,
            consts::deltaT, 1,
            -9.8f * math::up, 100,
            physicsSolverSelector,
            (CVXSolve *)cfg.cvxSolve);

    switch (cfg.envType) {
    case EnvType::URDFTest: {
        createURDFModel(ctx, cfg);

        createFloorPlane(ctx, false);
    } break;

    case EnvType::Car: {
        createCar2(ctx);

        Entity actor = ctx.makeEntity<ActorArchetype>();
        ctx.get<Action>(actor).v = 0;

        createFloorPlane(ctx, true);
    } break;

    case EnvType::FallingObjects: {
        makeExampleConfig0(ctx, cfg);
        createFloorPlane(ctx, true);
    } break;
    }
}

Sim::Sim(Engine &ctx,
         const Config &cfg,
         const WorldInit &)
    : WorldBase(ctx)
{
    initRandKey = cfg.initRandKey;
    rng = RNG(rand::split_i(ctx.data().initRandKey,
        0, (uint32_t)ctx.worldID().idx));

    envType = cfg.envType;

    RenderingSystem::init(ctx, cfg.renderBridge);

    makePhysicsObjects(ctx, cfg);
}

// This declaration is needed for the GPU backend in order to generate the
// CUDA kernel for world initialization, which needs to be specialized to the
// application's world data type (Sim) and config and initialization types.
// On the CPU it is a no-op.
MADRONA_BUILD_MWGPU_ENTRY(Engine, Sim, Sim::Config, Sim::WorldInit);

}
