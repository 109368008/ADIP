# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.18

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Disable VCS-based implicit rules.
% : %,v


# Disable VCS-based implicit rules.
% : RCS/%


# Disable VCS-based implicit rules.
% : RCS/%,v


# Disable VCS-based implicit rules.
% : SCCS/s.%


# Disable VCS-based implicit rules.
% : s.%


.SUFFIXES: .hpux_make_needs_suffix_list


# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /home/spie/.local/lib/python3.6/site-packages/cmake/data/bin/cmake

# The command to remove a file.
RM = /home/spie/.local/lib/python3.6/site-packages/cmake/data/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = "/home/spie/Desktop/ADIP/homework/HW#1_109368008"

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = "/home/spie/Desktop/ADIP/homework/HW#1_109368008/build"

# Include any dependencies generated for this target.
include CMakeFiles/HW1.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/HW1.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/HW1.dir/flags.make

CMakeFiles/HW1.dir/HW1.cpp.o: CMakeFiles/HW1.dir/flags.make
CMakeFiles/HW1.dir/HW1.cpp.o: ../HW1.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/home/spie/Desktop/ADIP/homework/HW#1_109368008/build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/HW1.dir/HW1.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/HW1.dir/HW1.cpp.o -c "/home/spie/Desktop/ADIP/homework/HW#1_109368008/HW1.cpp"

CMakeFiles/HW1.dir/HW1.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/HW1.dir/HW1.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/home/spie/Desktop/ADIP/homework/HW#1_109368008/HW1.cpp" > CMakeFiles/HW1.dir/HW1.cpp.i

CMakeFiles/HW1.dir/HW1.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/HW1.dir/HW1.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/home/spie/Desktop/ADIP/homework/HW#1_109368008/HW1.cpp" -o CMakeFiles/HW1.dir/HW1.cpp.s

# Object files for target HW1
HW1_OBJECTS = \
"CMakeFiles/HW1.dir/HW1.cpp.o"

# External object files for target HW1
HW1_EXTERNAL_OBJECTS =

HW1: CMakeFiles/HW1.dir/HW1.cpp.o
HW1: CMakeFiles/HW1.dir/build.make
HW1: /usr/local/lib/libopencv_dnn.so.4.5.0
HW1: /usr/local/lib/libopencv_gapi.so.4.5.0
HW1: /usr/local/lib/libopencv_highgui.so.4.5.0
HW1: /usr/local/lib/libopencv_ml.so.4.5.0
HW1: /usr/local/lib/libopencv_objdetect.so.4.5.0
HW1: /usr/local/lib/libopencv_photo.so.4.5.0
HW1: /usr/local/lib/libopencv_stitching.so.4.5.0
HW1: /usr/local/lib/libopencv_video.so.4.5.0
HW1: /usr/local/lib/libopencv_videoio.so.4.5.0
HW1: /usr/local/lib/libopencv_imgcodecs.so.4.5.0
HW1: /usr/local/lib/libopencv_calib3d.so.4.5.0
HW1: /usr/local/lib/libopencv_features2d.so.4.5.0
HW1: /usr/local/lib/libopencv_flann.so.4.5.0
HW1: /usr/local/lib/libopencv_imgproc.so.4.5.0
HW1: /usr/local/lib/libopencv_core.so.4.5.0
HW1: CMakeFiles/HW1.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir="/home/spie/Desktop/ADIP/homework/HW#1_109368008/build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable HW1"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/HW1.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/HW1.dir/build: HW1

.PHONY : CMakeFiles/HW1.dir/build

CMakeFiles/HW1.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/HW1.dir/cmake_clean.cmake
.PHONY : CMakeFiles/HW1.dir/clean

CMakeFiles/HW1.dir/depend:
	cd "/home/spie/Desktop/ADIP/homework/HW#1_109368008/build" && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" "/home/spie/Desktop/ADIP/homework/HW#1_109368008" "/home/spie/Desktop/ADIP/homework/HW#1_109368008" "/home/spie/Desktop/ADIP/homework/HW#1_109368008/build" "/home/spie/Desktop/ADIP/homework/HW#1_109368008/build" "/home/spie/Desktop/ADIP/homework/HW#1_109368008/build/CMakeFiles/HW1.dir/DependInfo.cmake" --color=$(COLOR)
.PHONY : CMakeFiles/HW1.dir/depend

