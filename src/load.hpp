#pragma once

#include <string>
#include <madrona/optional.hpp>
#include <madrona/cv_load.hpp>
#include <madrona/physics_loader.hpp>
#include <madrona/render/render_mgr.hpp>

namespace madPhysics {

// These are assets which are bulitin and always present.
struct BuiltinAssets {
    // Cube of side length 1 (object ID 0 for everything)
    std::string renderCubePath;
    std::string physicsCubePath;

    // Sphere of radius 1 (object ID 1 for everything)
    std::string renderSpherePath;
    std::string physicsSpherePath;

    // Path to plane to be rendered (object ID 2 for everything)
    std::string renderPlanePath;
};

struct URDFExport {
    madrona::phys::cv::ModelData modelData;
    uint32_t numModelConfigs;
    madrona::phys::cv::ModelConfig *modelConfigs;

    URDFExport makeCPUCopy(URDFExport urdf_export);
    URDFExport makeGPUCopy(URDFExport urdf_export);
};

// This does both rendering and physics asset loading
struct AssetLoader {
    AssetLoader(const BuiltinAssets &builtins);
    ~AssetLoader();

    struct Impl;
    std::unique_ptr<Impl> impl;

    // Adds both render and physics asset
    uint32_t addGlobalAsset(const std::string &render_path,
                            const std::string &physics_path);

    // Returns the index of the ModelConfig struct.
    // These need to be added after all the global assets
    uint32_t addURDF(const std::string &path);

    struct MaterialOverride {
        uint32_t extraMatIndex;
        uint32_t objID;
    };

    URDFExport finish(
            madrona::phys::PhysicsLoader &loader,
            const std::vector<madrona::imp::SourceMaterial> &extra_materials,
            const std::vector<MaterialOverride> &mat_overrides,
            madrona::Optional<madrona::render::RenderManager> &render_mgr);
};
    
}
