# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.0

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /net/cremi/nbremond/Projet/fichiers

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /net/cremi/nbremond/Projet/fichiers

# Include any dependencies generated for this target.
include libsotl/CMakeFiles/sotl.dir/depend.make

# Include the progress variables for this target.
include libsotl/CMakeFiles/sotl.dir/progress.make

# Include the compile flags for this target's objects.
include libsotl/CMakeFiles/sotl.dir/flags.make

libsotl/CMakeFiles/sotl.dir/src/atom.c.o: libsotl/CMakeFiles/sotl.dir/flags.make
libsotl/CMakeFiles/sotl.dir/src/atom.c.o: libsotl/src/atom.c
	$(CMAKE_COMMAND) -E cmake_progress_report /net/cremi/nbremond/Projet/fichiers/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building C object libsotl/CMakeFiles/sotl.dir/src/atom.c.o"
	cd /net/cremi/nbremond/Projet/fichiers/libsotl && /usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -o CMakeFiles/sotl.dir/src/atom.c.o   -c /net/cremi/nbremond/Projet/fichiers/libsotl/src/atom.c

libsotl/CMakeFiles/sotl.dir/src/atom.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/sotl.dir/src/atom.c.i"
	cd /net/cremi/nbremond/Projet/fichiers/libsotl && /usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -E /net/cremi/nbremond/Projet/fichiers/libsotl/src/atom.c > CMakeFiles/sotl.dir/src/atom.c.i

libsotl/CMakeFiles/sotl.dir/src/atom.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/sotl.dir/src/atom.c.s"
	cd /net/cremi/nbremond/Projet/fichiers/libsotl && /usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -S /net/cremi/nbremond/Projet/fichiers/libsotl/src/atom.c -o CMakeFiles/sotl.dir/src/atom.c.s

libsotl/CMakeFiles/sotl.dir/src/atom.c.o.requires:
.PHONY : libsotl/CMakeFiles/sotl.dir/src/atom.c.o.requires

libsotl/CMakeFiles/sotl.dir/src/atom.c.o.provides: libsotl/CMakeFiles/sotl.dir/src/atom.c.o.requires
	$(MAKE) -f libsotl/CMakeFiles/sotl.dir/build.make libsotl/CMakeFiles/sotl.dir/src/atom.c.o.provides.build
.PHONY : libsotl/CMakeFiles/sotl.dir/src/atom.c.o.provides

libsotl/CMakeFiles/sotl.dir/src/atom.c.o.provides.build: libsotl/CMakeFiles/sotl.dir/src/atom.c.o

libsotl/CMakeFiles/sotl.dir/src/device.c.o: libsotl/CMakeFiles/sotl.dir/flags.make
libsotl/CMakeFiles/sotl.dir/src/device.c.o: libsotl/src/device.c
	$(CMAKE_COMMAND) -E cmake_progress_report /net/cremi/nbremond/Projet/fichiers/CMakeFiles $(CMAKE_PROGRESS_2)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building C object libsotl/CMakeFiles/sotl.dir/src/device.c.o"
	cd /net/cremi/nbremond/Projet/fichiers/libsotl && /usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -o CMakeFiles/sotl.dir/src/device.c.o   -c /net/cremi/nbremond/Projet/fichiers/libsotl/src/device.c

libsotl/CMakeFiles/sotl.dir/src/device.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/sotl.dir/src/device.c.i"
	cd /net/cremi/nbremond/Projet/fichiers/libsotl && /usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -E /net/cremi/nbremond/Projet/fichiers/libsotl/src/device.c > CMakeFiles/sotl.dir/src/device.c.i

libsotl/CMakeFiles/sotl.dir/src/device.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/sotl.dir/src/device.c.s"
	cd /net/cremi/nbremond/Projet/fichiers/libsotl && /usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -S /net/cremi/nbremond/Projet/fichiers/libsotl/src/device.c -o CMakeFiles/sotl.dir/src/device.c.s

libsotl/CMakeFiles/sotl.dir/src/device.c.o.requires:
.PHONY : libsotl/CMakeFiles/sotl.dir/src/device.c.o.requires

libsotl/CMakeFiles/sotl.dir/src/device.c.o.provides: libsotl/CMakeFiles/sotl.dir/src/device.c.o.requires
	$(MAKE) -f libsotl/CMakeFiles/sotl.dir/build.make libsotl/CMakeFiles/sotl.dir/src/device.c.o.provides.build
.PHONY : libsotl/CMakeFiles/sotl.dir/src/device.c.o.provides

libsotl/CMakeFiles/sotl.dir/src/device.c.o.provides.build: libsotl/CMakeFiles/sotl.dir/src/device.c.o

libsotl/CMakeFiles/sotl.dir/src/domain.c.o: libsotl/CMakeFiles/sotl.dir/flags.make
libsotl/CMakeFiles/sotl.dir/src/domain.c.o: libsotl/src/domain.c
	$(CMAKE_COMMAND) -E cmake_progress_report /net/cremi/nbremond/Projet/fichiers/CMakeFiles $(CMAKE_PROGRESS_3)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building C object libsotl/CMakeFiles/sotl.dir/src/domain.c.o"
	cd /net/cremi/nbremond/Projet/fichiers/libsotl && /usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -o CMakeFiles/sotl.dir/src/domain.c.o   -c /net/cremi/nbremond/Projet/fichiers/libsotl/src/domain.c

libsotl/CMakeFiles/sotl.dir/src/domain.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/sotl.dir/src/domain.c.i"
	cd /net/cremi/nbremond/Projet/fichiers/libsotl && /usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -E /net/cremi/nbremond/Projet/fichiers/libsotl/src/domain.c > CMakeFiles/sotl.dir/src/domain.c.i

libsotl/CMakeFiles/sotl.dir/src/domain.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/sotl.dir/src/domain.c.s"
	cd /net/cremi/nbremond/Projet/fichiers/libsotl && /usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -S /net/cremi/nbremond/Projet/fichiers/libsotl/src/domain.c -o CMakeFiles/sotl.dir/src/domain.c.s

libsotl/CMakeFiles/sotl.dir/src/domain.c.o.requires:
.PHONY : libsotl/CMakeFiles/sotl.dir/src/domain.c.o.requires

libsotl/CMakeFiles/sotl.dir/src/domain.c.o.provides: libsotl/CMakeFiles/sotl.dir/src/domain.c.o.requires
	$(MAKE) -f libsotl/CMakeFiles/sotl.dir/build.make libsotl/CMakeFiles/sotl.dir/src/domain.c.o.provides.build
.PHONY : libsotl/CMakeFiles/sotl.dir/src/domain.c.o.provides

libsotl/CMakeFiles/sotl.dir/src/domain.c.o.provides.build: libsotl/CMakeFiles/sotl.dir/src/domain.c.o

libsotl/CMakeFiles/sotl.dir/src/global_definitions.c.o: libsotl/CMakeFiles/sotl.dir/flags.make
libsotl/CMakeFiles/sotl.dir/src/global_definitions.c.o: libsotl/src/global_definitions.c
	$(CMAKE_COMMAND) -E cmake_progress_report /net/cremi/nbremond/Projet/fichiers/CMakeFiles $(CMAKE_PROGRESS_4)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building C object libsotl/CMakeFiles/sotl.dir/src/global_definitions.c.o"
	cd /net/cremi/nbremond/Projet/fichiers/libsotl && /usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -o CMakeFiles/sotl.dir/src/global_definitions.c.o   -c /net/cremi/nbremond/Projet/fichiers/libsotl/src/global_definitions.c

libsotl/CMakeFiles/sotl.dir/src/global_definitions.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/sotl.dir/src/global_definitions.c.i"
	cd /net/cremi/nbremond/Projet/fichiers/libsotl && /usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -E /net/cremi/nbremond/Projet/fichiers/libsotl/src/global_definitions.c > CMakeFiles/sotl.dir/src/global_definitions.c.i

libsotl/CMakeFiles/sotl.dir/src/global_definitions.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/sotl.dir/src/global_definitions.c.s"
	cd /net/cremi/nbremond/Projet/fichiers/libsotl && /usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -S /net/cremi/nbremond/Projet/fichiers/libsotl/src/global_definitions.c -o CMakeFiles/sotl.dir/src/global_definitions.c.s

libsotl/CMakeFiles/sotl.dir/src/global_definitions.c.o.requires:
.PHONY : libsotl/CMakeFiles/sotl.dir/src/global_definitions.c.o.requires

libsotl/CMakeFiles/sotl.dir/src/global_definitions.c.o.provides: libsotl/CMakeFiles/sotl.dir/src/global_definitions.c.o.requires
	$(MAKE) -f libsotl/CMakeFiles/sotl.dir/build.make libsotl/CMakeFiles/sotl.dir/src/global_definitions.c.o.provides.build
.PHONY : libsotl/CMakeFiles/sotl.dir/src/global_definitions.c.o.provides

libsotl/CMakeFiles/sotl.dir/src/global_definitions.c.o.provides.build: libsotl/CMakeFiles/sotl.dir/src/global_definitions.c.o

libsotl/CMakeFiles/sotl.dir/src/kernel_list.c.o: libsotl/CMakeFiles/sotl.dir/flags.make
libsotl/CMakeFiles/sotl.dir/src/kernel_list.c.o: libsotl/src/kernel_list.c
	$(CMAKE_COMMAND) -E cmake_progress_report /net/cremi/nbremond/Projet/fichiers/CMakeFiles $(CMAKE_PROGRESS_5)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building C object libsotl/CMakeFiles/sotl.dir/src/kernel_list.c.o"
	cd /net/cremi/nbremond/Projet/fichiers/libsotl && /usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -o CMakeFiles/sotl.dir/src/kernel_list.c.o   -c /net/cremi/nbremond/Projet/fichiers/libsotl/src/kernel_list.c

libsotl/CMakeFiles/sotl.dir/src/kernel_list.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/sotl.dir/src/kernel_list.c.i"
	cd /net/cremi/nbremond/Projet/fichiers/libsotl && /usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -E /net/cremi/nbremond/Projet/fichiers/libsotl/src/kernel_list.c > CMakeFiles/sotl.dir/src/kernel_list.c.i

libsotl/CMakeFiles/sotl.dir/src/kernel_list.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/sotl.dir/src/kernel_list.c.s"
	cd /net/cremi/nbremond/Projet/fichiers/libsotl && /usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -S /net/cremi/nbremond/Projet/fichiers/libsotl/src/kernel_list.c -o CMakeFiles/sotl.dir/src/kernel_list.c.s

libsotl/CMakeFiles/sotl.dir/src/kernel_list.c.o.requires:
.PHONY : libsotl/CMakeFiles/sotl.dir/src/kernel_list.c.o.requires

libsotl/CMakeFiles/sotl.dir/src/kernel_list.c.o.provides: libsotl/CMakeFiles/sotl.dir/src/kernel_list.c.o.requires
	$(MAKE) -f libsotl/CMakeFiles/sotl.dir/build.make libsotl/CMakeFiles/sotl.dir/src/kernel_list.c.o.provides.build
.PHONY : libsotl/CMakeFiles/sotl.dir/src/kernel_list.c.o.provides

libsotl/CMakeFiles/sotl.dir/src/kernel_list.c.o.provides.build: libsotl/CMakeFiles/sotl.dir/src/kernel_list.c.o

libsotl/CMakeFiles/sotl.dir/src/ocl.c.o: libsotl/CMakeFiles/sotl.dir/flags.make
libsotl/CMakeFiles/sotl.dir/src/ocl.c.o: libsotl/src/ocl.c
	$(CMAKE_COMMAND) -E cmake_progress_report /net/cremi/nbremond/Projet/fichiers/CMakeFiles $(CMAKE_PROGRESS_6)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building C object libsotl/CMakeFiles/sotl.dir/src/ocl.c.o"
	cd /net/cremi/nbremond/Projet/fichiers/libsotl && /usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -o CMakeFiles/sotl.dir/src/ocl.c.o   -c /net/cremi/nbremond/Projet/fichiers/libsotl/src/ocl.c

libsotl/CMakeFiles/sotl.dir/src/ocl.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/sotl.dir/src/ocl.c.i"
	cd /net/cremi/nbremond/Projet/fichiers/libsotl && /usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -E /net/cremi/nbremond/Projet/fichiers/libsotl/src/ocl.c > CMakeFiles/sotl.dir/src/ocl.c.i

libsotl/CMakeFiles/sotl.dir/src/ocl.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/sotl.dir/src/ocl.c.s"
	cd /net/cremi/nbremond/Projet/fichiers/libsotl && /usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -S /net/cremi/nbremond/Projet/fichiers/libsotl/src/ocl.c -o CMakeFiles/sotl.dir/src/ocl.c.s

libsotl/CMakeFiles/sotl.dir/src/ocl.c.o.requires:
.PHONY : libsotl/CMakeFiles/sotl.dir/src/ocl.c.o.requires

libsotl/CMakeFiles/sotl.dir/src/ocl.c.o.provides: libsotl/CMakeFiles/sotl.dir/src/ocl.c.o.requires
	$(MAKE) -f libsotl/CMakeFiles/sotl.dir/build.make libsotl/CMakeFiles/sotl.dir/src/ocl.c.o.provides.build
.PHONY : libsotl/CMakeFiles/sotl.dir/src/ocl.c.o.provides

libsotl/CMakeFiles/sotl.dir/src/ocl.c.o.provides.build: libsotl/CMakeFiles/sotl.dir/src/ocl.c.o

libsotl/CMakeFiles/sotl.dir/src/ocl_kernels.c.o: libsotl/CMakeFiles/sotl.dir/flags.make
libsotl/CMakeFiles/sotl.dir/src/ocl_kernels.c.o: libsotl/src/ocl_kernels.c
	$(CMAKE_COMMAND) -E cmake_progress_report /net/cremi/nbremond/Projet/fichiers/CMakeFiles $(CMAKE_PROGRESS_7)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building C object libsotl/CMakeFiles/sotl.dir/src/ocl_kernels.c.o"
	cd /net/cremi/nbremond/Projet/fichiers/libsotl && /usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -o CMakeFiles/sotl.dir/src/ocl_kernels.c.o   -c /net/cremi/nbremond/Projet/fichiers/libsotl/src/ocl_kernels.c

libsotl/CMakeFiles/sotl.dir/src/ocl_kernels.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/sotl.dir/src/ocl_kernels.c.i"
	cd /net/cremi/nbremond/Projet/fichiers/libsotl && /usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -E /net/cremi/nbremond/Projet/fichiers/libsotl/src/ocl_kernels.c > CMakeFiles/sotl.dir/src/ocl_kernels.c.i

libsotl/CMakeFiles/sotl.dir/src/ocl_kernels.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/sotl.dir/src/ocl_kernels.c.s"
	cd /net/cremi/nbremond/Projet/fichiers/libsotl && /usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -S /net/cremi/nbremond/Projet/fichiers/libsotl/src/ocl_kernels.c -o CMakeFiles/sotl.dir/src/ocl_kernels.c.s

libsotl/CMakeFiles/sotl.dir/src/ocl_kernels.c.o.requires:
.PHONY : libsotl/CMakeFiles/sotl.dir/src/ocl_kernels.c.o.requires

libsotl/CMakeFiles/sotl.dir/src/ocl_kernels.c.o.provides: libsotl/CMakeFiles/sotl.dir/src/ocl_kernels.c.o.requires
	$(MAKE) -f libsotl/CMakeFiles/sotl.dir/build.make libsotl/CMakeFiles/sotl.dir/src/ocl_kernels.c.o.provides.build
.PHONY : libsotl/CMakeFiles/sotl.dir/src/ocl_kernels.c.o.provides

libsotl/CMakeFiles/sotl.dir/src/ocl_kernels.c.o.provides.build: libsotl/CMakeFiles/sotl.dir/src/ocl_kernels.c.o

libsotl/CMakeFiles/sotl.dir/src/profiling.c.o: libsotl/CMakeFiles/sotl.dir/flags.make
libsotl/CMakeFiles/sotl.dir/src/profiling.c.o: libsotl/src/profiling.c
	$(CMAKE_COMMAND) -E cmake_progress_report /net/cremi/nbremond/Projet/fichiers/CMakeFiles $(CMAKE_PROGRESS_8)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building C object libsotl/CMakeFiles/sotl.dir/src/profiling.c.o"
	cd /net/cremi/nbremond/Projet/fichiers/libsotl && /usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -o CMakeFiles/sotl.dir/src/profiling.c.o   -c /net/cremi/nbremond/Projet/fichiers/libsotl/src/profiling.c

libsotl/CMakeFiles/sotl.dir/src/profiling.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/sotl.dir/src/profiling.c.i"
	cd /net/cremi/nbremond/Projet/fichiers/libsotl && /usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -E /net/cremi/nbremond/Projet/fichiers/libsotl/src/profiling.c > CMakeFiles/sotl.dir/src/profiling.c.i

libsotl/CMakeFiles/sotl.dir/src/profiling.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/sotl.dir/src/profiling.c.s"
	cd /net/cremi/nbremond/Projet/fichiers/libsotl && /usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -S /net/cremi/nbremond/Projet/fichiers/libsotl/src/profiling.c -o CMakeFiles/sotl.dir/src/profiling.c.s

libsotl/CMakeFiles/sotl.dir/src/profiling.c.o.requires:
.PHONY : libsotl/CMakeFiles/sotl.dir/src/profiling.c.o.requires

libsotl/CMakeFiles/sotl.dir/src/profiling.c.o.provides: libsotl/CMakeFiles/sotl.dir/src/profiling.c.o.requires
	$(MAKE) -f libsotl/CMakeFiles/sotl.dir/build.make libsotl/CMakeFiles/sotl.dir/src/profiling.c.o.provides.build
.PHONY : libsotl/CMakeFiles/sotl.dir/src/profiling.c.o.provides

libsotl/CMakeFiles/sotl.dir/src/profiling.c.o.provides.build: libsotl/CMakeFiles/sotl.dir/src/profiling.c.o

libsotl/CMakeFiles/sotl.dir/src/sotl.c.o: libsotl/CMakeFiles/sotl.dir/flags.make
libsotl/CMakeFiles/sotl.dir/src/sotl.c.o: libsotl/src/sotl.c
	$(CMAKE_COMMAND) -E cmake_progress_report /net/cremi/nbremond/Projet/fichiers/CMakeFiles $(CMAKE_PROGRESS_9)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building C object libsotl/CMakeFiles/sotl.dir/src/sotl.c.o"
	cd /net/cremi/nbremond/Projet/fichiers/libsotl && /usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -o CMakeFiles/sotl.dir/src/sotl.c.o   -c /net/cremi/nbremond/Projet/fichiers/libsotl/src/sotl.c

libsotl/CMakeFiles/sotl.dir/src/sotl.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/sotl.dir/src/sotl.c.i"
	cd /net/cremi/nbremond/Projet/fichiers/libsotl && /usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -E /net/cremi/nbremond/Projet/fichiers/libsotl/src/sotl.c > CMakeFiles/sotl.dir/src/sotl.c.i

libsotl/CMakeFiles/sotl.dir/src/sotl.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/sotl.dir/src/sotl.c.s"
	cd /net/cremi/nbremond/Projet/fichiers/libsotl && /usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -S /net/cremi/nbremond/Projet/fichiers/libsotl/src/sotl.c -o CMakeFiles/sotl.dir/src/sotl.c.s

libsotl/CMakeFiles/sotl.dir/src/sotl.c.o.requires:
.PHONY : libsotl/CMakeFiles/sotl.dir/src/sotl.c.o.requires

libsotl/CMakeFiles/sotl.dir/src/sotl.c.o.provides: libsotl/CMakeFiles/sotl.dir/src/sotl.c.o.requires
	$(MAKE) -f libsotl/CMakeFiles/sotl.dir/build.make libsotl/CMakeFiles/sotl.dir/src/sotl.c.o.provides.build
.PHONY : libsotl/CMakeFiles/sotl.dir/src/sotl.c.o.provides

libsotl/CMakeFiles/sotl.dir/src/sotl.c.o.provides.build: libsotl/CMakeFiles/sotl.dir/src/sotl.c.o

libsotl/CMakeFiles/sotl.dir/src/seq.c.o: libsotl/CMakeFiles/sotl.dir/flags.make
libsotl/CMakeFiles/sotl.dir/src/seq.c.o: libsotl/src/seq.c
	$(CMAKE_COMMAND) -E cmake_progress_report /net/cremi/nbremond/Projet/fichiers/CMakeFiles $(CMAKE_PROGRESS_10)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building C object libsotl/CMakeFiles/sotl.dir/src/seq.c.o"
	cd /net/cremi/nbremond/Projet/fichiers/libsotl && /usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -o CMakeFiles/sotl.dir/src/seq.c.o   -c /net/cremi/nbremond/Projet/fichiers/libsotl/src/seq.c

libsotl/CMakeFiles/sotl.dir/src/seq.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/sotl.dir/src/seq.c.i"
	cd /net/cremi/nbremond/Projet/fichiers/libsotl && /usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -E /net/cremi/nbremond/Projet/fichiers/libsotl/src/seq.c > CMakeFiles/sotl.dir/src/seq.c.i

libsotl/CMakeFiles/sotl.dir/src/seq.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/sotl.dir/src/seq.c.s"
	cd /net/cremi/nbremond/Projet/fichiers/libsotl && /usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -S /net/cremi/nbremond/Projet/fichiers/libsotl/src/seq.c -o CMakeFiles/sotl.dir/src/seq.c.s

libsotl/CMakeFiles/sotl.dir/src/seq.c.o.requires:
.PHONY : libsotl/CMakeFiles/sotl.dir/src/seq.c.o.requires

libsotl/CMakeFiles/sotl.dir/src/seq.c.o.provides: libsotl/CMakeFiles/sotl.dir/src/seq.c.o.requires
	$(MAKE) -f libsotl/CMakeFiles/sotl.dir/build.make libsotl/CMakeFiles/sotl.dir/src/seq.c.o.provides.build
.PHONY : libsotl/CMakeFiles/sotl.dir/src/seq.c.o.provides

libsotl/CMakeFiles/sotl.dir/src/seq.c.o.provides.build: libsotl/CMakeFiles/sotl.dir/src/seq.c.o

libsotl/CMakeFiles/sotl.dir/src/util.c.o: libsotl/CMakeFiles/sotl.dir/flags.make
libsotl/CMakeFiles/sotl.dir/src/util.c.o: libsotl/src/util.c
	$(CMAKE_COMMAND) -E cmake_progress_report /net/cremi/nbremond/Projet/fichiers/CMakeFiles $(CMAKE_PROGRESS_11)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building C object libsotl/CMakeFiles/sotl.dir/src/util.c.o"
	cd /net/cremi/nbremond/Projet/fichiers/libsotl && /usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -o CMakeFiles/sotl.dir/src/util.c.o   -c /net/cremi/nbremond/Projet/fichiers/libsotl/src/util.c

libsotl/CMakeFiles/sotl.dir/src/util.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/sotl.dir/src/util.c.i"
	cd /net/cremi/nbremond/Projet/fichiers/libsotl && /usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -E /net/cremi/nbremond/Projet/fichiers/libsotl/src/util.c > CMakeFiles/sotl.dir/src/util.c.i

libsotl/CMakeFiles/sotl.dir/src/util.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/sotl.dir/src/util.c.s"
	cd /net/cremi/nbremond/Projet/fichiers/libsotl && /usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -S /net/cremi/nbremond/Projet/fichiers/libsotl/src/util.c -o CMakeFiles/sotl.dir/src/util.c.s

libsotl/CMakeFiles/sotl.dir/src/util.c.o.requires:
.PHONY : libsotl/CMakeFiles/sotl.dir/src/util.c.o.requires

libsotl/CMakeFiles/sotl.dir/src/util.c.o.provides: libsotl/CMakeFiles/sotl.dir/src/util.c.o.requires
	$(MAKE) -f libsotl/CMakeFiles/sotl.dir/build.make libsotl/CMakeFiles/sotl.dir/src/util.c.o.provides.build
.PHONY : libsotl/CMakeFiles/sotl.dir/src/util.c.o.provides

libsotl/CMakeFiles/sotl.dir/src/util.c.o.provides.build: libsotl/CMakeFiles/sotl.dir/src/util.c.o

libsotl/CMakeFiles/sotl.dir/src/openmp.c.o: libsotl/CMakeFiles/sotl.dir/flags.make
libsotl/CMakeFiles/sotl.dir/src/openmp.c.o: libsotl/src/openmp.c
	$(CMAKE_COMMAND) -E cmake_progress_report /net/cremi/nbremond/Projet/fichiers/CMakeFiles $(CMAKE_PROGRESS_12)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building C object libsotl/CMakeFiles/sotl.dir/src/openmp.c.o"
	cd /net/cremi/nbremond/Projet/fichiers/libsotl && /usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -o CMakeFiles/sotl.dir/src/openmp.c.o   -c /net/cremi/nbremond/Projet/fichiers/libsotl/src/openmp.c

libsotl/CMakeFiles/sotl.dir/src/openmp.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/sotl.dir/src/openmp.c.i"
	cd /net/cremi/nbremond/Projet/fichiers/libsotl && /usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -E /net/cremi/nbremond/Projet/fichiers/libsotl/src/openmp.c > CMakeFiles/sotl.dir/src/openmp.c.i

libsotl/CMakeFiles/sotl.dir/src/openmp.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/sotl.dir/src/openmp.c.s"
	cd /net/cremi/nbremond/Projet/fichiers/libsotl && /usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -S /net/cremi/nbremond/Projet/fichiers/libsotl/src/openmp.c -o CMakeFiles/sotl.dir/src/openmp.c.s

libsotl/CMakeFiles/sotl.dir/src/openmp.c.o.requires:
.PHONY : libsotl/CMakeFiles/sotl.dir/src/openmp.c.o.requires

libsotl/CMakeFiles/sotl.dir/src/openmp.c.o.provides: libsotl/CMakeFiles/sotl.dir/src/openmp.c.o.requires
	$(MAKE) -f libsotl/CMakeFiles/sotl.dir/build.make libsotl/CMakeFiles/sotl.dir/src/openmp.c.o.provides.build
.PHONY : libsotl/CMakeFiles/sotl.dir/src/openmp.c.o.provides

libsotl/CMakeFiles/sotl.dir/src/openmp.c.o.provides.build: libsotl/CMakeFiles/sotl.dir/src/openmp.c.o

libsotl/CMakeFiles/sotl.dir/src/vbo.c.o: libsotl/CMakeFiles/sotl.dir/flags.make
libsotl/CMakeFiles/sotl.dir/src/vbo.c.o: libsotl/src/vbo.c
	$(CMAKE_COMMAND) -E cmake_progress_report /net/cremi/nbremond/Projet/fichiers/CMakeFiles $(CMAKE_PROGRESS_13)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building C object libsotl/CMakeFiles/sotl.dir/src/vbo.c.o"
	cd /net/cremi/nbremond/Projet/fichiers/libsotl && /usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -o CMakeFiles/sotl.dir/src/vbo.c.o   -c /net/cremi/nbremond/Projet/fichiers/libsotl/src/vbo.c

libsotl/CMakeFiles/sotl.dir/src/vbo.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/sotl.dir/src/vbo.c.i"
	cd /net/cremi/nbremond/Projet/fichiers/libsotl && /usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -E /net/cremi/nbremond/Projet/fichiers/libsotl/src/vbo.c > CMakeFiles/sotl.dir/src/vbo.c.i

libsotl/CMakeFiles/sotl.dir/src/vbo.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/sotl.dir/src/vbo.c.s"
	cd /net/cremi/nbremond/Projet/fichiers/libsotl && /usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -S /net/cremi/nbremond/Projet/fichiers/libsotl/src/vbo.c -o CMakeFiles/sotl.dir/src/vbo.c.s

libsotl/CMakeFiles/sotl.dir/src/vbo.c.o.requires:
.PHONY : libsotl/CMakeFiles/sotl.dir/src/vbo.c.o.requires

libsotl/CMakeFiles/sotl.dir/src/vbo.c.o.provides: libsotl/CMakeFiles/sotl.dir/src/vbo.c.o.requires
	$(MAKE) -f libsotl/CMakeFiles/sotl.dir/build.make libsotl/CMakeFiles/sotl.dir/src/vbo.c.o.provides.build
.PHONY : libsotl/CMakeFiles/sotl.dir/src/vbo.c.o.provides

libsotl/CMakeFiles/sotl.dir/src/vbo.c.o.provides.build: libsotl/CMakeFiles/sotl.dir/src/vbo.c.o

libsotl/CMakeFiles/sotl.dir/src/shaders.c.o: libsotl/CMakeFiles/sotl.dir/flags.make
libsotl/CMakeFiles/sotl.dir/src/shaders.c.o: libsotl/src/shaders.c
	$(CMAKE_COMMAND) -E cmake_progress_report /net/cremi/nbremond/Projet/fichiers/CMakeFiles $(CMAKE_PROGRESS_14)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building C object libsotl/CMakeFiles/sotl.dir/src/shaders.c.o"
	cd /net/cremi/nbremond/Projet/fichiers/libsotl && /usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -o CMakeFiles/sotl.dir/src/shaders.c.o   -c /net/cremi/nbremond/Projet/fichiers/libsotl/src/shaders.c

libsotl/CMakeFiles/sotl.dir/src/shaders.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/sotl.dir/src/shaders.c.i"
	cd /net/cremi/nbremond/Projet/fichiers/libsotl && /usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -E /net/cremi/nbremond/Projet/fichiers/libsotl/src/shaders.c > CMakeFiles/sotl.dir/src/shaders.c.i

libsotl/CMakeFiles/sotl.dir/src/shaders.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/sotl.dir/src/shaders.c.s"
	cd /net/cremi/nbremond/Projet/fichiers/libsotl && /usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -S /net/cremi/nbremond/Projet/fichiers/libsotl/src/shaders.c -o CMakeFiles/sotl.dir/src/shaders.c.s

libsotl/CMakeFiles/sotl.dir/src/shaders.c.o.requires:
.PHONY : libsotl/CMakeFiles/sotl.dir/src/shaders.c.o.requires

libsotl/CMakeFiles/sotl.dir/src/shaders.c.o.provides: libsotl/CMakeFiles/sotl.dir/src/shaders.c.o.requires
	$(MAKE) -f libsotl/CMakeFiles/sotl.dir/build.make libsotl/CMakeFiles/sotl.dir/src/shaders.c.o.provides.build
.PHONY : libsotl/CMakeFiles/sotl.dir/src/shaders.c.o.provides

libsotl/CMakeFiles/sotl.dir/src/shaders.c.o.provides.build: libsotl/CMakeFiles/sotl.dir/src/shaders.c.o

libsotl/CMakeFiles/sotl.dir/src/window.c.o: libsotl/CMakeFiles/sotl.dir/flags.make
libsotl/CMakeFiles/sotl.dir/src/window.c.o: libsotl/src/window.c
	$(CMAKE_COMMAND) -E cmake_progress_report /net/cremi/nbremond/Projet/fichiers/CMakeFiles $(CMAKE_PROGRESS_15)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building C object libsotl/CMakeFiles/sotl.dir/src/window.c.o"
	cd /net/cremi/nbremond/Projet/fichiers/libsotl && /usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -o CMakeFiles/sotl.dir/src/window.c.o   -c /net/cremi/nbremond/Projet/fichiers/libsotl/src/window.c

libsotl/CMakeFiles/sotl.dir/src/window.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/sotl.dir/src/window.c.i"
	cd /net/cremi/nbremond/Projet/fichiers/libsotl && /usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -E /net/cremi/nbremond/Projet/fichiers/libsotl/src/window.c > CMakeFiles/sotl.dir/src/window.c.i

libsotl/CMakeFiles/sotl.dir/src/window.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/sotl.dir/src/window.c.s"
	cd /net/cremi/nbremond/Projet/fichiers/libsotl && /usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -S /net/cremi/nbremond/Projet/fichiers/libsotl/src/window.c -o CMakeFiles/sotl.dir/src/window.c.s

libsotl/CMakeFiles/sotl.dir/src/window.c.o.requires:
.PHONY : libsotl/CMakeFiles/sotl.dir/src/window.c.o.requires

libsotl/CMakeFiles/sotl.dir/src/window.c.o.provides: libsotl/CMakeFiles/sotl.dir/src/window.c.o.requires
	$(MAKE) -f libsotl/CMakeFiles/sotl.dir/build.make libsotl/CMakeFiles/sotl.dir/src/window.c.o.provides.build
.PHONY : libsotl/CMakeFiles/sotl.dir/src/window.c.o.provides

libsotl/CMakeFiles/sotl.dir/src/window.c.o.provides.build: libsotl/CMakeFiles/sotl.dir/src/window.c.o

# Object files for target sotl
sotl_OBJECTS = \
"CMakeFiles/sotl.dir/src/atom.c.o" \
"CMakeFiles/sotl.dir/src/device.c.o" \
"CMakeFiles/sotl.dir/src/domain.c.o" \
"CMakeFiles/sotl.dir/src/global_definitions.c.o" \
"CMakeFiles/sotl.dir/src/kernel_list.c.o" \
"CMakeFiles/sotl.dir/src/ocl.c.o" \
"CMakeFiles/sotl.dir/src/ocl_kernels.c.o" \
"CMakeFiles/sotl.dir/src/profiling.c.o" \
"CMakeFiles/sotl.dir/src/sotl.c.o" \
"CMakeFiles/sotl.dir/src/seq.c.o" \
"CMakeFiles/sotl.dir/src/util.c.o" \
"CMakeFiles/sotl.dir/src/openmp.c.o" \
"CMakeFiles/sotl.dir/src/vbo.c.o" \
"CMakeFiles/sotl.dir/src/shaders.c.o" \
"CMakeFiles/sotl.dir/src/window.c.o"

# External object files for target sotl
sotl_EXTERNAL_OBJECTS =

libsotl/libsotl.so: libsotl/CMakeFiles/sotl.dir/src/atom.c.o
libsotl/libsotl.so: libsotl/CMakeFiles/sotl.dir/src/device.c.o
libsotl/libsotl.so: libsotl/CMakeFiles/sotl.dir/src/domain.c.o
libsotl/libsotl.so: libsotl/CMakeFiles/sotl.dir/src/global_definitions.c.o
libsotl/libsotl.so: libsotl/CMakeFiles/sotl.dir/src/kernel_list.c.o
libsotl/libsotl.so: libsotl/CMakeFiles/sotl.dir/src/ocl.c.o
libsotl/libsotl.so: libsotl/CMakeFiles/sotl.dir/src/ocl_kernels.c.o
libsotl/libsotl.so: libsotl/CMakeFiles/sotl.dir/src/profiling.c.o
libsotl/libsotl.so: libsotl/CMakeFiles/sotl.dir/src/sotl.c.o
libsotl/libsotl.so: libsotl/CMakeFiles/sotl.dir/src/seq.c.o
libsotl/libsotl.so: libsotl/CMakeFiles/sotl.dir/src/util.c.o
libsotl/libsotl.so: libsotl/CMakeFiles/sotl.dir/src/openmp.c.o
libsotl/libsotl.so: libsotl/CMakeFiles/sotl.dir/src/vbo.c.o
libsotl/libsotl.so: libsotl/CMakeFiles/sotl.dir/src/shaders.c.o
libsotl/libsotl.so: libsotl/CMakeFiles/sotl.dir/src/window.c.o
libsotl/libsotl.so: libsotl/CMakeFiles/sotl.dir/build.make
libsotl/libsotl.so: /usr/lib/x86_64-linux-gnu/libOpenCL.so
libsotl/libsotl.so: /usr/lib/x86_64-linux-gnu/libm.so
libsotl/libsotl.so: /usr/lib/x86_64-linux-gnu/libGLU.so
libsotl/libsotl.so: /usr/lib/x86_64-linux-gnu/libGL.so
libsotl/libsotl.so: /usr/lib/x86_64-linux-gnu/libSM.so
libsotl/libsotl.so: /usr/lib/x86_64-linux-gnu/libICE.so
libsotl/libsotl.so: /usr/lib/x86_64-linux-gnu/libX11.so
libsotl/libsotl.so: /usr/lib/x86_64-linux-gnu/libXext.so
libsotl/libsotl.so: /usr/lib/x86_64-linux-gnu/libglut.so
libsotl/libsotl.so: /usr/lib/x86_64-linux-gnu/libXmu.so
libsotl/libsotl.so: /usr/lib/x86_64-linux-gnu/libXi.so
libsotl/libsotl.so: libsotl/CMakeFiles/sotl.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking C shared library libsotl.so"
	cd /net/cremi/nbremond/Projet/fichiers/libsotl && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/sotl.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
libsotl/CMakeFiles/sotl.dir/build: libsotl/libsotl.so
.PHONY : libsotl/CMakeFiles/sotl.dir/build

# Object files for target sotl
sotl_OBJECTS = \
"CMakeFiles/sotl.dir/src/atom.c.o" \
"CMakeFiles/sotl.dir/src/device.c.o" \
"CMakeFiles/sotl.dir/src/domain.c.o" \
"CMakeFiles/sotl.dir/src/global_definitions.c.o" \
"CMakeFiles/sotl.dir/src/kernel_list.c.o" \
"CMakeFiles/sotl.dir/src/ocl.c.o" \
"CMakeFiles/sotl.dir/src/ocl_kernels.c.o" \
"CMakeFiles/sotl.dir/src/profiling.c.o" \
"CMakeFiles/sotl.dir/src/sotl.c.o" \
"CMakeFiles/sotl.dir/src/seq.c.o" \
"CMakeFiles/sotl.dir/src/util.c.o" \
"CMakeFiles/sotl.dir/src/openmp.c.o" \
"CMakeFiles/sotl.dir/src/vbo.c.o" \
"CMakeFiles/sotl.dir/src/shaders.c.o" \
"CMakeFiles/sotl.dir/src/window.c.o"

# External object files for target sotl
sotl_EXTERNAL_OBJECTS =

libsotl/CMakeFiles/CMakeRelink.dir/libsotl.so: libsotl/CMakeFiles/sotl.dir/src/atom.c.o
libsotl/CMakeFiles/CMakeRelink.dir/libsotl.so: libsotl/CMakeFiles/sotl.dir/src/device.c.o
libsotl/CMakeFiles/CMakeRelink.dir/libsotl.so: libsotl/CMakeFiles/sotl.dir/src/domain.c.o
libsotl/CMakeFiles/CMakeRelink.dir/libsotl.so: libsotl/CMakeFiles/sotl.dir/src/global_definitions.c.o
libsotl/CMakeFiles/CMakeRelink.dir/libsotl.so: libsotl/CMakeFiles/sotl.dir/src/kernel_list.c.o
libsotl/CMakeFiles/CMakeRelink.dir/libsotl.so: libsotl/CMakeFiles/sotl.dir/src/ocl.c.o
libsotl/CMakeFiles/CMakeRelink.dir/libsotl.so: libsotl/CMakeFiles/sotl.dir/src/ocl_kernels.c.o
libsotl/CMakeFiles/CMakeRelink.dir/libsotl.so: libsotl/CMakeFiles/sotl.dir/src/profiling.c.o
libsotl/CMakeFiles/CMakeRelink.dir/libsotl.so: libsotl/CMakeFiles/sotl.dir/src/sotl.c.o
libsotl/CMakeFiles/CMakeRelink.dir/libsotl.so: libsotl/CMakeFiles/sotl.dir/src/seq.c.o
libsotl/CMakeFiles/CMakeRelink.dir/libsotl.so: libsotl/CMakeFiles/sotl.dir/src/util.c.o
libsotl/CMakeFiles/CMakeRelink.dir/libsotl.so: libsotl/CMakeFiles/sotl.dir/src/openmp.c.o
libsotl/CMakeFiles/CMakeRelink.dir/libsotl.so: libsotl/CMakeFiles/sotl.dir/src/vbo.c.o
libsotl/CMakeFiles/CMakeRelink.dir/libsotl.so: libsotl/CMakeFiles/sotl.dir/src/shaders.c.o
libsotl/CMakeFiles/CMakeRelink.dir/libsotl.so: libsotl/CMakeFiles/sotl.dir/src/window.c.o
libsotl/CMakeFiles/CMakeRelink.dir/libsotl.so: libsotl/CMakeFiles/sotl.dir/build.make
libsotl/CMakeFiles/CMakeRelink.dir/libsotl.so: /usr/lib/x86_64-linux-gnu/libOpenCL.so
libsotl/CMakeFiles/CMakeRelink.dir/libsotl.so: /usr/lib/x86_64-linux-gnu/libm.so
libsotl/CMakeFiles/CMakeRelink.dir/libsotl.so: /usr/lib/x86_64-linux-gnu/libGLU.so
libsotl/CMakeFiles/CMakeRelink.dir/libsotl.so: /usr/lib/x86_64-linux-gnu/libGL.so
libsotl/CMakeFiles/CMakeRelink.dir/libsotl.so: /usr/lib/x86_64-linux-gnu/libSM.so
libsotl/CMakeFiles/CMakeRelink.dir/libsotl.so: /usr/lib/x86_64-linux-gnu/libICE.so
libsotl/CMakeFiles/CMakeRelink.dir/libsotl.so: /usr/lib/x86_64-linux-gnu/libX11.so
libsotl/CMakeFiles/CMakeRelink.dir/libsotl.so: /usr/lib/x86_64-linux-gnu/libXext.so
libsotl/CMakeFiles/CMakeRelink.dir/libsotl.so: /usr/lib/x86_64-linux-gnu/libglut.so
libsotl/CMakeFiles/CMakeRelink.dir/libsotl.so: /usr/lib/x86_64-linux-gnu/libXmu.so
libsotl/CMakeFiles/CMakeRelink.dir/libsotl.so: /usr/lib/x86_64-linux-gnu/libXi.so
libsotl/CMakeFiles/CMakeRelink.dir/libsotl.so: libsotl/CMakeFiles/sotl.dir/relink.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking C shared library CMakeFiles/CMakeRelink.dir/libsotl.so"
	cd /net/cremi/nbremond/Projet/fichiers/libsotl && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/sotl.dir/relink.txt --verbose=$(VERBOSE)

# Rule to relink during preinstall.
libsotl/CMakeFiles/sotl.dir/preinstall: libsotl/CMakeFiles/CMakeRelink.dir/libsotl.so
.PHONY : libsotl/CMakeFiles/sotl.dir/preinstall

libsotl/CMakeFiles/sotl.dir/requires: libsotl/CMakeFiles/sotl.dir/src/atom.c.o.requires
libsotl/CMakeFiles/sotl.dir/requires: libsotl/CMakeFiles/sotl.dir/src/device.c.o.requires
libsotl/CMakeFiles/sotl.dir/requires: libsotl/CMakeFiles/sotl.dir/src/domain.c.o.requires
libsotl/CMakeFiles/sotl.dir/requires: libsotl/CMakeFiles/sotl.dir/src/global_definitions.c.o.requires
libsotl/CMakeFiles/sotl.dir/requires: libsotl/CMakeFiles/sotl.dir/src/kernel_list.c.o.requires
libsotl/CMakeFiles/sotl.dir/requires: libsotl/CMakeFiles/sotl.dir/src/ocl.c.o.requires
libsotl/CMakeFiles/sotl.dir/requires: libsotl/CMakeFiles/sotl.dir/src/ocl_kernels.c.o.requires
libsotl/CMakeFiles/sotl.dir/requires: libsotl/CMakeFiles/sotl.dir/src/profiling.c.o.requires
libsotl/CMakeFiles/sotl.dir/requires: libsotl/CMakeFiles/sotl.dir/src/sotl.c.o.requires
libsotl/CMakeFiles/sotl.dir/requires: libsotl/CMakeFiles/sotl.dir/src/seq.c.o.requires
libsotl/CMakeFiles/sotl.dir/requires: libsotl/CMakeFiles/sotl.dir/src/util.c.o.requires
libsotl/CMakeFiles/sotl.dir/requires: libsotl/CMakeFiles/sotl.dir/src/openmp.c.o.requires
libsotl/CMakeFiles/sotl.dir/requires: libsotl/CMakeFiles/sotl.dir/src/vbo.c.o.requires
libsotl/CMakeFiles/sotl.dir/requires: libsotl/CMakeFiles/sotl.dir/src/shaders.c.o.requires
libsotl/CMakeFiles/sotl.dir/requires: libsotl/CMakeFiles/sotl.dir/src/window.c.o.requires
.PHONY : libsotl/CMakeFiles/sotl.dir/requires

libsotl/CMakeFiles/sotl.dir/clean:
	cd /net/cremi/nbremond/Projet/fichiers/libsotl && $(CMAKE_COMMAND) -P CMakeFiles/sotl.dir/cmake_clean.cmake
.PHONY : libsotl/CMakeFiles/sotl.dir/clean

libsotl/CMakeFiles/sotl.dir/depend:
	cd /net/cremi/nbremond/Projet/fichiers && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /net/cremi/nbremond/Projet/fichiers /net/cremi/nbremond/Projet/fichiers/libsotl /net/cremi/nbremond/Projet/fichiers /net/cremi/nbremond/Projet/fichiers/libsotl /net/cremi/nbremond/Projet/fichiers/libsotl/CMakeFiles/sotl.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : libsotl/CMakeFiles/sotl.dir/depend

