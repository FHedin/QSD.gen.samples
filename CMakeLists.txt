# Copyright (c) 2016-2018, Florent Hédin, Tony Lelièvre, and École des Ponts - ParisTech
#All rights reserved.
#
#The 3-clause BSD license is applied to this software.
#
#See LICENSE.txt

cmake_minimum_required (VERSION 2.8)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING
      "Choose the type of build, options are: Debug | Release (default) | RelWithDebInfo | MinSizeRel"
      FORCE)
endif()

set(TGT "QSD.gen.samples" CACHE STRING "The executable name")

project(${TGT} C CXX)

# Some of the following are required for building the project, if not found you will have to install them manually 
find_package(OpenMM REQUIRED)
find_package(MPI REQUIRED)
find_package(LuaJIT REQUIRED)
find_package(sqlite3 REQUIRED)

# list of include folders with .h/.hpp files
include_directories(
# found by cmake
${MPI_INCLUDE_PATH}
${OpenMM_INCLUDE_DIR}
${LUA_INCLUDE_DIR}
# project files
include
# external headers (dependancies provided with this project)
external/sol2
)

# list all source files
set(
SRCS
# core files
src/logger.cpp
src/main.cpp
src/rand.cpp
# dynamics files
src/observable.cpp
src/GelmanRubin.cpp
src/runSim.cpp
src/QSD_samples_generator.cpp
# MD engine interfaces files
src/omm_interface.cpp
# utilities files
src/mpi_utils.cpp
src/lua_interface.cpp
# external dependancies provided with this project
external/luasqlite3/lsqlite3.c
)

add_executable(${TGT} ${SRCS})

# more warnings for c
set(cWarnings "-Wall -Wextra")

# more warnings for c++
# set(cxxWarnings "-Wall -Wextra -Wformat=2 -Wshadow -Wconversion -Wuseless-cast")
# no-reorder is for ignoring some warnings from the omm library
set(cxxWarnings "-Wall -Wextra -Wno-reorder")

set(cFeatures   "-fstack-protector-all -fstack-check")
set(cxxFeatures "-fstack-protector-all -fstack-check")

# enabling -ffast-math may provide extra performance by enabling supplementary vectorization

set(cFeatures   "-ffast-math")
# set(cxxFeatures "-ffast-math")
# This is required if we try to bind the executable to a binary version of the OpenMM library compiled either with  g++ < 5 or with clang
set(cxxFeatures "-ffast-math -D_GLIBCXX_USE_CXX11_ABI=0")


#set(cFeatures   "-ffast-math -fopt-info-vec")
#set(cxxFeatures "-ffast-math -fopt-info-vec")

# may be required when using clang
#set(cxxFeatures "-stdlib=libstdc++")

# C compiler flags
set(CMAKE_C_FLAGS_DEBUG          "-std=c11 ${cWarnings} ${cFeatures} -O0 -g -DDEBUG_BUILD")
set(CMAKE_C_FLAGS_RELEASE        "-std=c11 ${cWarnings} ${cFeatures} -O3    -march=native")
set(CMAKE_C_FLAGS_RELWITHDEBINFO "-std=c11 ${cWarnings} ${cFeatures} -O2 -g -march=native")
set(CMAKE_C_FLAGS_MINSIZEREL     "-std=c11 ${cWarnings} ${cFeatures} -Os -s -march=native")

# C++ compiler flags
set(CMAKE_CXX_FLAGS_DEBUG          "-std=c++14 ${cxxWarnings} ${cxxFeatures} -O0 -g -DDEBUG_BUILD")
set(CMAKE_CXX_FLAGS_RELEASE        "-std=c++14 ${cxxWarnings} ${cxxFeatures} -O3    -march=native")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-std=c++14 ${cxxWarnings} ${cxxFeatures} -O2 -g -march=native")
set(CMAKE_CXX_FLAGS_MINSIZEREL     "-std=c++14 ${cxxWarnings} ${cxxFeatures} -Os -s -march=native")

target_link_libraries(${TGT} ${OpenMM_LIBRARY} ${LUA_LIBRARIES} ${MPI_LIBRARIES} ${SQLITE3_LIBRARY} dl pthread)

