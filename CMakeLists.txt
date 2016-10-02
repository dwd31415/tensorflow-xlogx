cmake_minimum_required(VERSION 2.8)

project (cpp_example)
set(CMAKE_BUILD_TYPE Release)

execute_process(COMMAND bash "-c" "python3 -c 'import tensorflow as tf; print(tf.sysconfig.get_include())'" OUTPUT_VARIABLE TF_INC)

set(CMAKE_CXX_FLAGS  "-fPIC -D_GLIBCXX_USE_CXX11_ABI=0")

if(APPLE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -undefined dynamic_lookup -std=c++11")
elseif(UNIX)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
endif()

message(STATUS ${TF_INC})
find_package(CUDA)

set(SRC_FILE_NAME src/example.cc)

if (CUDA_FOUND)
  set(CUDA_NVCC_FLAGS -I ${TF_INC} -D_FORCE_INLINES -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC)
  cuda_compile(OP_O src/example.cu)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -D GOOGLE_CUDA=1")
else()
  set(OP_O "")
endif ()

include_directories(${TF_INC})
add_library(cpp_example SHARED ${SRC_FILE_NAME} ${OP_O})
