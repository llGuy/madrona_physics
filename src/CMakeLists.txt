set(STICK_SIMULATOR_SRCS
    types.hpp
    sim.hpp sim.inl sim.cpp
)

SET(STICK_COMPILE_FLAGS
    -v)

add_library(stick_cpu_impl STATIC
    ${STICK_SIMULATOR_SRCS}
)

target_link_libraries(stick_cpu_impl
    PUBLIC
        madrona_mw_core
    PRIVATE
        madrona_common
        madrona_mw_physics
        madrona_rendering_system
)

add_library(stick_mgr STATIC
    mgr.hpp mgr.cpp
)

target_link_libraries(stick_mgr 
    PUBLIC
        madrona_python_utils
    PRIVATE
        stick_cpu_impl
        madrona_mw_cpu
        madrona_common
        madrona_importer
        madrona_physics_loader
        madrona_render
        madrona_render_asset_processor
)

if (TARGET madrona_mw_gpu)
    madrona_build_compile_defns(
        OUT_TARGET
            stick_gpu_srcs
        SOURCES_DEFN
            STICK_SRC_LIST
        FLAGS_DEFN
            STICK_COMPILE_FLAGS 
        SRCS
            ${STICK_SIMULATOR_SRCS}
    )

    target_link_libraries(stick_mgr PRIVATE
        stick_gpu_srcs
        madrona_mw_gpu
    )
endif ()

target_compile_definitions(stick_mgr PRIVATE
    -DDATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../data/"
)

if (TARGET madrona_viz)
    add_executable(stick_viewer viewer.cpp)
    target_link_libraries(stick_viewer 
        PUBLIC
        stick_mgr
        PRIVATE
        madrona_mw_core madrona_viz madrona_cuda stb)

    target_compile_definitions(stick_viewer PUBLIC
        -DMADRONA_VIEWER
    )

    if (TARGET madrona_cuda)
        target_link_libraries(stick_viewer PRIVATE madrona_cuda)
    endif()
    target_compile_definitions(stick_viewer PRIVATE
        -DDATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../data/"
    )
endif()
