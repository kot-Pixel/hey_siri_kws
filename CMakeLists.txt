cmake_minimum_required(VERSION 3.18.1)
project(hey_siri_kws LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(USE_TFLITE_STATIC "Link prebuilt static TFLite from third_party/tflite-static" ON)

set(TFLITE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/tflite" CACHE PATH "TensorFlow Lite headers (dynamic fallback)")
set(TFLITE_STATIC_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/tflite-static" CACHE PATH "Prebuilt static TFLite (WSL build)")

add_library(hey_siri_kws SHARED
    src/hey_siri_kws.cpp
    src/model_data.cc
    src/mfcc_extractor.cpp
    src/mfcc/rfft.cc
)

find_library(LOG_LIB log)

set(TFLITE_STATIC_LIB_DIR "${TFLITE_STATIC_ROOT}/${ANDROID_ABI}/lib")
set(TFLITE_STATIC_MARKER "${TFLITE_STATIC_LIB_DIR}/libtensorflow-lite.a")

if(USE_TFLITE_STATIC AND EXISTS "${TFLITE_STATIC_MARKER}")
  message(STATUS "Using static TFLite from ${TFLITE_STATIC_LIB_DIR}")

  set(TFLITE_LIBS_FILE "${TFLITE_STATIC_ROOT}/${ANDROID_ABI}/lib/libs.txt")
  if(EXISTS "${TFLITE_LIBS_FILE}")
    file(STRINGS "${TFLITE_LIBS_FILE}" TFLITE_LIB_NAMES)
    set(TFLITE_STATIC_LIBS)
    foreach(name IN LISTS TFLITE_LIB_NAMES)
      list(APPEND TFLITE_STATIC_LIBS "${TFLITE_STATIC_LIB_DIR}/${name}")
    endforeach()
  else()
    file(GLOB TFLITE_STATIC_LIBS CONFIGURE_DEPENDS "${TFLITE_STATIC_LIB_DIR}/*.a")
    if(TFLITE_STATIC_LIBS)
      list(FILTER TFLITE_STATIC_LIBS EXCLUDE REGEX ".*/libfft2d_fft4f2d\\.a$")
    endif()
  endif()
  if(NOT TFLITE_STATIC_LIBS)
    message(FATAL_ERROR "No static TFLite libs under ${TFLITE_STATIC_LIB_DIR}. Run scripts/build_tflite_static.sh in WSL.")
  endif()

  target_include_directories(hey_siri_kws
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
      $<INSTALL_INTERFACE:include>
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/src
      ${TFLITE_STATIC_ROOT}/include
  )

  target_link_libraries(hey_siri_kws
    -Wl,--start-group
    ${TFLITE_STATIC_LIBS}
    -Wl,--end-group
    ${LOG_LIB}
    m
    z
  )
else()
  message(STATUS "Using dynamic TFLite from ${TFLITE_ROOT}/lib/${ANDROID_ABI}")

  target_include_directories(hey_siri_kws
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
      $<INSTALL_INTERFACE:include>
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/src
      ${TFLITE_ROOT}/include
  )

  set(TFLITE_LIB_DIR "${TFLITE_ROOT}/lib/${ANDROID_ABI}")
  if(EXISTS "${TFLITE_LIB_DIR}/libtensorflowlite_jni.so")
    set(TFLITE_CORE_LIB "${TFLITE_LIB_DIR}/libtensorflowlite_jni.so")
  elseif(EXISTS "${TFLITE_LIB_DIR}/libtensorflowlite.so")
    set(TFLITE_CORE_LIB "${TFLITE_LIB_DIR}/libtensorflowlite.so")
  else()
    message(FATAL_ERROR "Missing TFLite core lib for ${ANDROID_ABI}. Run scripts/prepare_deps.ps1 or scripts/build_tflite_static.sh")
  endif()

  target_link_libraries(hey_siri_kws
    ${LOG_LIB}
    ${TFLITE_CORE_LIB}
  )
endif()

target_compile_options(hey_siri_kws PRIVATE -Wall -Wextra -O3)

install(TARGETS hey_siri_kws
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)
install(FILES include/hey_siri_kws.h DESTINATION include)
