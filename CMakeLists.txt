cmake_minimum_required(VERSION 3.18.1)
project(hey_siri_kws LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(TFLITE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/tflite" CACHE PATH "TensorFlow Lite root")

add_library(hey_siri_kws SHARED
    src/hey_siri_kws.cpp
    src/flex_delegate_loader.cpp
    src/model_data.cc
)

target_include_directories(hey_siri_kws
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${TFLITE_ROOT}/include
)

find_library(LOG_LIB log)

set(TFLITE_LIB_DIR "${TFLITE_ROOT}/lib/${ANDROID_ABI}")
if(EXISTS "${TFLITE_LIB_DIR}/libtensorflowlite_jni.so")
  set(TFLITE_CORE_LIB "${TFLITE_LIB_DIR}/libtensorflowlite_jni.so")
elseif(EXISTS "${TFLITE_LIB_DIR}/libtensorflowlite.so")
  set(TFLITE_CORE_LIB "${TFLITE_LIB_DIR}/libtensorflowlite.so")
else()
  message(FATAL_ERROR "Missing TFLite core lib for ${ANDROID_ABI}. Run scripts/prepare_deps.ps1")
endif()

target_link_libraries(hey_siri_kws
    ${LOG_LIB}
    ${TFLITE_CORE_LIB}
)

if(EXISTS "${TFLITE_ROOT}/lib/${ANDROID_ABI}/libtensorflowlite_flex_jni.so")
  target_link_libraries(hey_siri_kws
      ${TFLITE_ROOT}/lib/${ANDROID_ABI}/libtensorflowlite_flex_jni.so)
elseif(EXISTS "${TFLITE_ROOT}/lib/${ANDROID_ABI}/libtensorflowlite_flex.so")
  target_link_libraries(hey_siri_kws
      ${TFLITE_ROOT}/lib/${ANDROID_ABI}/libtensorflowlite_flex.so)
else()
  message(FATAL_ERROR "Missing Flex delegate for ${ANDROID_ABI}. Run scripts/prepare_deps.ps1")
endif()

target_compile_options(hey_siri_kws PRIVATE -Wall -Wextra -O3)

# Install: libhey_siri_kws.so + hey_siri_kws.h
install(TARGETS hey_siri_kws
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)
install(FILES include/hey_siri_kws.h DESTINATION include)
