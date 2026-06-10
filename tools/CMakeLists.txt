cmake_minimum_required(VERSION 3.18.1)
project(kws_smoke_test LANGUAGES CXX)

set(KWS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/..")
set(KWS_LIB "${KWS_ROOT}/dist/${ANDROID_ABI}/lib/libhey_siri_kws.so")

if(NOT EXISTS "${KWS_LIB}")
  message(FATAL_ERROR "Missing ${KWS_LIB}. Run scripts/build.ps1 first.")
endif()

add_library(hey_siri_kws SHARED IMPORTED GLOBAL)
set_target_properties(hey_siri_kws PROPERTIES IMPORTED_LOCATION "${KWS_LIB}")

find_library(LOG_LIB log)

function(add_kws_tool target source)
  add_executable(${target} ${source})
  target_include_directories(${target} PRIVATE "${KWS_ROOT}/include")
  target_compile_options(${target} PRIVATE -Wall -Wextra -O2)
  target_link_libraries(${target} hey_siri_kws ${LOG_LIB})
endfunction()

add_kws_tool(kws_smoke_test smoke_test.cpp)
add_kws_tool(kws_bench_test bench_test.cpp)
