cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)

project(MRtoCT)

#-----------------------------------------------------------------------------
# Extension meta-information  (replace YOUR_GITHUB_USER)
set(EXTENSION_HOMEPAGE "https://github.com/YOUR_GITHUB_USER/SlicerMRtoCT")
set(EXTENSION_CATEGORY "Radiotherapy")
set(EXTENSION_CONTRIBUTORS "Kevin Estiven Giraldo Paniagua")
set(EXTENSION_DESCRIPTION "Generates a synthetic CT from a brain MR with a deep-learning model (SAdaNGF), for radiotherapy research. Non-commercial license, not for clinical use.")
set(EXTENSION_ICONURL "https://raw.githubusercontent.com/YOUR_GITHUB_USER/SlicerMRtoCT/main/MRtoCT/Resources/Icons/MRtoCT.png")
set(EXTENSION_SCREENSHOTURLS "https://raw.githubusercontent.com/YOUR_GITHUB_USER/SlicerMRtoCT/main/Screenshots/1.png")
set(EXTENSION_DEPENDS "PyTorch")

#-----------------------------------------------------------------------------
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
add_subdirectory(MRtoCT)

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
