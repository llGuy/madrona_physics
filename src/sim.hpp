#pragma once

#include <madrona/taskgraph_builder.hpp>
#include <madrona/custom_context.hpp>
#include <madrona/rand.hpp>
#include <madrona/physics.hpp>
#include <madrona/cvphysics.hpp>
#include <madrona/render/ecs.hpp>

namespace madPhysics {

class Engine;

// Include several madrona types into the simulator namespace for convenience
using madrona::Entity;
using madrona::RandKey;
using madrona::CountT;
using madrona::base::Position;
using madrona::base::Rotation;
using madrona::base::Scale;
using madrona::base::ObjectID;

// This enum is used by the Sim and Manager classes to track the export slots
// for each component exported to the training code.
enum class ExportID : uint32_t {
    Action,
    NumExports,
};

enum class TaskGraphID : uint32_t {
    Init,
    Step,
    NumTaskGraphs,
};

enum class SimObject : uint32_t {
    Cube,
    Sphere,
    Plane,
    Stick,
    Disk,
    NumObjects
};

struct Action {
    int v;
    bool vizColliders;
};

struct ActorArchetype : public madrona::Archetype<
    Action
> {};

enum class EnvType {
    URDFTest,
    Car,
    FallingObjects,
};

// The Sim class encapsulates the per-world state of the simulation.
// Sim is always available by calling ctx.data() given a reference
// to the Engine / Context object that is passed to each ECS system.
//
// Per-World state that is frequently accessed but only used by a few
// ECS systems should be put in a singleton component rather than
// in this class in order to ensure efficient access patterns.
struct Sim : public madrona::WorldBase {
    struct Config {
        bool autoReset;
        RandKey initRandKey;

        madrona::phys::ObjectManager *rigidBodyObjMgr;
        const madrona::render::RenderECSBridge *renderBridge;

        void *cvxSolve;

        uint32_t numModelConfigs;
        madrona::phys::cv::ModelConfig *modelConfigs;
        madrona::phys::cv::ModelData modelData;

        EnvType envType;
    };

    struct WorldInit {};

    static void registerTypes(madrona::ECSRegistry &registry,
                              const Config &cfg);

    static void setupTasks(madrona::TaskGraphManager &taskgraph_mgr,
                           const Config &cfg);

    Sim(Engine &ctx,
        const Config &cfg,
        const WorldInit &);

    void makePhysicsObjects(Engine &ctx, const Config &cfg);

    madrona::RandKey initRandKey;
    madrona::RNG rng;

    Entity stickBodyGrp;
    Entity stickRoot;
    Entity stickChild;
    Entity plane;

    Entity carHinge;

    Entity urdf;
    
    EnvType envType;
};

class Engine : public ::madrona::CustomContext<Engine, Sim> {
public:
    using CustomContext::CustomContext;

    // These are convenience helpers for creating renderable
    // entities when rendering isn't necessarily enabled
    template <typename ArchetypeT>
    inline madrona::Entity makeRenderableEntity();
    inline void destroyRenderableEntity(Entity e);
};

}

#include "sim.inl"
