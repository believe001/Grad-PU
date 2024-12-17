# CMake generated Testfile for 
# Source directory: /home/lh/code/Grad-PU/evaluation_code
# Build directory: /home/lh/code/Grad-PU/evaluation_code
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(compilation_of__evaluation "/usr/bin/cmake" "--build" "/home/lh/code/Grad-PU/evaluation_code" "--target" "evaluation")
set_tests_properties(compilation_of__evaluation PROPERTIES  LABELS "Distance_2_Tests")
add_test(execution___of__evaluation "/home/lh/code/Grad-PU/evaluation_code/evaluation")
set_tests_properties(execution___of__evaluation PROPERTIES  DEPENDS "compilation_of__evaluation" LABELS "Distance_2_Tests" WORKING_DIRECTORY "/home/lh/code/Grad-PU/evaluation_code")
