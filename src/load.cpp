#include "load.hpp"

#include <madrona/urdf.hpp>
#include <madrona/importer.hpp>

using namespace madrona;
using namespace madrona::imp;
using namespace madrona::math;
using namespace madrona::phys;
using namespace madrona::render;

namespace madPhysics {

struct AssetLoader::Impl {
    std::vector<std::string> renderAssetPaths;
    std::vector<std::string> physicsAssetPaths;

    imp::URDFLoader urdf;

    inline Impl(const BuiltinAssets &bi)
    {
        // Cube
        renderAssetPaths.push_back(bi.renderCubePath);
        physicsAssetPaths.push_back(bi.physicsCubePath);

        // Sphere (push a dummy object for physics)
        renderAssetPaths.push_back(bi.renderSpherePath);
        physicsAssetPaths.push_back(bi.renderPlanePath);

        // Plane (push a dummy object for physics)
        renderAssetPaths.push_back(bi.renderPlanePath);
        physicsAssetPaths.push_back(bi.renderPlanePath);
    }
};

AssetLoader::AssetLoader(const BuiltinAssets &builtins)
    : impl(new AssetLoader::Impl(builtins))
{
}

AssetLoader::~AssetLoader()
{
}

// Adds both render and physics asset
uint32_t AssetLoader::addGlobalAsset(
        const std::string &render_path,
        const std::string &physics_path)
{
    uint32_t idx = (uint32_t)impl->renderAssetPaths.size();
    impl->renderAssetPaths.push_back(render_path);
    impl->physicsAssetPaths.push_back(physics_path);
    return idx;
}

uint32_t AssetLoader::addURDF(const std::string &path)
{
    URDFLoader::BuiltinPrimitives prims = {
        .cubeRenderIdx = 0,
        .cubePhysicsIdx = 0,
        .planeRenderIdx = 2,
        .planePhysicsIdx = 2,
        .sphereRenderIdx = 1,
        .spherePhysicsIdx = 1,
    };

    return impl->urdf.load(
        path.c_str(),
        prims,
        impl->renderAssetPaths,
        impl->physicsAssetPaths);
}

static std::vector<const char *> makeCStrings(
        const std::vector<std::string> &paths)
{
    std::vector<const char *> asset_cstrs;
    for (size_t i = 0; i < paths.size(); i++) {
        asset_cstrs.push_back(paths[i].c_str());
    }
    return asset_cstrs;
}

AssetLoader::URDFExport AssetLoader::finish(
        PhysicsLoader &loader,
        const std::vector<SourceMaterial> &extra_materials,
        const std::vector<MaterialOverride> &mat_overrides,
        Optional<RenderManager> &render_mgr)
{
    AssetImporter physics_importer;

    URDFExport urdf_export;

    { // Physics first
        // TODO Post process some of the strings for URDF
        std::vector<const char *> physics_cstrs =
            makeCStrings(impl->physicsAssetPaths);

        char import_err_buffer[4096];
        auto imported_hulls = physics_importer.importFromDisk(
            physics_cstrs, import_err_buffer, true);

        if (!imported_hulls.has_value()) {
            FATAL("%s", import_err_buffer);
        }

        DynArray<imp::SourceMesh> src_convex_hulls(
            imported_hulls->objects.size());

        DynArray<DynArray<SourceCollisionPrimitive>> prim_arrays(0);
        DynArray<SourceCollisionObject> src_objs(0);

        auto setupHull = [&](uint32_t obj_id) {
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

            src_objs.push_back(SourceCollisionObject {
                .prims = Span<const SourceCollisionPrimitive>(
                        prim_arrays.back()),
                .invMass = 0.5f,
                .friction = { 0.5f, 2.f }
            });
        };

        // Cube
        setupHull(0);

        // Sphere
        SourceCollisionPrimitive sphere_prim {
            .type = CollisionPrimitive::Type::Sphere,
        };
        src_objs.push_back(SourceCollisionObject {
            .prims = Span<const SourceCollisionPrimitive>(&sphere_prim, 1),
        });

        // Plane
        SourceCollisionPrimitive plane_prim {
            .type = CollisionPrimitive::Type::Plane,
        };
        src_objs.push_back(SourceCollisionObject {
            .prims = Span<const SourceCollisionPrimitive>(&plane_prim, 1),
        });

        for (uint32_t i = 3; i < imported_hulls->objects.size(); ++i) {
            setupHull(i);
        }

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

        urdf_export.modelData = impl->urdf.getModelData();
        urdf_export.modelConfigs =
            impl->urdf.getModelConfigs(urdf_export.numModelConfigs);
    }

    AssetImporter render_importer;

    if (render_mgr.has_value()) { // Rendering second
        std::vector<const char *> render_cstrs =
            makeCStrings(impl->renderAssetPaths);

        std::array<char, 1024> import_err;
        auto render_assets = render_importer.importFromDisk(
            render_cstrs, Span<char>(import_err.data(), import_err.size()),
            true);

        uint32_t extra_mat_offset = (uint32_t)render_assets->materials.size();

        for (auto &mat : extra_materials) {
            render_assets->materials.push_back(mat);
        }

        for (auto mat_override : mat_overrides) {
            render_assets->objects[mat_override.objID].meshes[0].materialIDX = 
                mat_override.extraMatIndex + extra_mat_offset;
        }

        render_mgr->loadObjects(render_assets->objects,
                Span(render_assets->materials.data(), render_assets->materials.size()), 
                {},
                true);
    }

    return urdf_export;
}
    
}
