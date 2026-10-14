# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "/mnt/c/Users/ashpr/OneDrive/Desktop/Ashu/Programs/AntiGravProjects/LLMSimulator/src/dram/ramulator2/ext/yaml-cpp")
  file(MAKE_DIRECTORY "/mnt/c/Users/ashpr/OneDrive/Desktop/Ashu/Programs/AntiGravProjects/LLMSimulator/src/dram/ramulator2/ext/yaml-cpp")
endif()
file(MAKE_DIRECTORY
  "/mnt/c/Users/ashpr/OneDrive/Desktop/Ashu/Programs/AntiGravProjects/LLMSimulator/build/_deps/yaml-cpp-build"
  "/mnt/c/Users/ashpr/OneDrive/Desktop/Ashu/Programs/AntiGravProjects/LLMSimulator/build/_deps/yaml-cpp-subbuild/yaml-cpp-populate-prefix"
  "/mnt/c/Users/ashpr/OneDrive/Desktop/Ashu/Programs/AntiGravProjects/LLMSimulator/build/_deps/yaml-cpp-subbuild/yaml-cpp-populate-prefix/tmp"
  "/mnt/c/Users/ashpr/OneDrive/Desktop/Ashu/Programs/AntiGravProjects/LLMSimulator/build/_deps/yaml-cpp-subbuild/yaml-cpp-populate-prefix/src/yaml-cpp-populate-stamp"
  "/mnt/c/Users/ashpr/OneDrive/Desktop/Ashu/Programs/AntiGravProjects/LLMSimulator/build/_deps/yaml-cpp-subbuild/yaml-cpp-populate-prefix/src"
  "/mnt/c/Users/ashpr/OneDrive/Desktop/Ashu/Programs/AntiGravProjects/LLMSimulator/build/_deps/yaml-cpp-subbuild/yaml-cpp-populate-prefix/src/yaml-cpp-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/mnt/c/Users/ashpr/OneDrive/Desktop/Ashu/Programs/AntiGravProjects/LLMSimulator/build/_deps/yaml-cpp-subbuild/yaml-cpp-populate-prefix/src/yaml-cpp-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/mnt/c/Users/ashpr/OneDrive/Desktop/Ashu/Programs/AntiGravProjects/LLMSimulator/build/_deps/yaml-cpp-subbuild/yaml-cpp-populate-prefix/src/yaml-cpp-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
