SHELL := /bin/bash
.ONESHELL:

ARCH := -arch=sm_86
EXTRA := -D__NO_MATH_NOEXCEPT --expt-relaxed-constexpr --expt-extended-lambda -Xptxas -warn-spills,-warn-lmem-usage,-warn-double-usage -Xcompiler -Wall,-Wextra,-Wno-unknown-pragmas
JNI_HEADER_NAME ?= JNITest
OUTDIR ?= .
INCLUDE ?= /usr/include
LIB ?= /usr/lib

%-debug: %.cu $(DEPS)
	@echo "Compiling $@ with nvcc..."
	$(eval OUTPUT := $(OUTDIR)/$@)
	nvcc -G -g -O0 -lineinfo -DDEBUG $(ARCH) $(EXTRA) -I$(INCLUDE) -L$(LIB) -o $(OUTPUT) -lzmq $<

%: %.cu $(DEPS)
	@echo "Compiling $@ with nvcc..."
	$(eval OUTPUT := $(OUTDIR)/$@)
	nvcc -O3 $(ARCH) $(EXTRA) -I$(INCLUDE) -L$(LIB) -o $(OUTPUT) -lzmq $<

%-nvshmem-debug: %.cu $(DEPS)
	@echo "Compiling $@ with nvcc and NVSHMEM..."
	$(eval OUTPUT := $(OUTDIR)/$@)
	source $$HOME/hpcx/hpcx-init.sh
	hpcx_load
	nvcc -G -g -O0 -lineinfo -DDEBUG -x cu -rdc=true -ccbin=mpicxx -I/usr/include/nvshmem -L/usr/lib/x86_64-linux-gnu/nvshmem/13 -L$$HOME/hpcx/ompi/lib -o $(OUTPUT) -lnvshmem -lnvshmem_host -lmpi $<

%-nvshmem: %.cu $(DEPS)
	@echo "Compiling $@ with nvcc and NVSHMEM..."
	$(eval OUTPUT := $(OUTDIR)/$@)
	source $$HOME/hpcx/hpcx-init.sh
	hpcx_load
	nvcc -x cu -rdc=true -ccbin=mpicxx -I/usr/include/nvshmem -L/usr/lib/x86_64-linux-gnu/nvshmem/13 -L$$HOME/hpcx/ompi/lib -o $(OUTPUT) -lnvshmem -lnvshmem_host -lmpi $<

%.ii: %.cu
	@echo "Generating preprocessed region-only output $@..."
	rm -f $@
	nvcc $(ARCH) -I$(INCLUDE) -E $< | \
	awk '/#pragma PREPROCESSOR_MARKER_BEGIN/,/#pragma PREPROCESSOR_MARKER_END/' | \
	grep -v '^#' | \
	clang-format > $@

%-jni: %.cu
	@echo "Preparing to compile $* with JNI support..."
	$(eval JAVA_FILE := $(filter %.java,$(MAKECMDGOALS)))
	$(eval JAVA_CLASS := $(basename $(notdir $(JAVA_FILE))))
	$(eval OUTPUT := $(OUTDIR)/$@.so)
	@if [ ! -f *$(JAVA_CLASS).h ] || [ "$(JAVA_FILE)" -nt $$(ls *$(JAVA_CLASS).h 2>/dev/null || echo /dev/null) ]; then \
		echo "Generating JNI header for class $(JAVA_CLASS)..."; \
		javac -h . "$(JAVA_FILE)"; \
	fi
	@JNI_HEADER=$$(ls *$(JAVA_CLASS).h); \
	JNI_HEADER_NAME=$$(basename "$$JNI_HEADER" .h); \
	echo "Using JNI header: $$JNI_HEADER"; \
	cat "$$JNI_HEADER"; \
	nvcc -O3 $(ARCH) $(EXTRA) \
		-Xcompiler -fPIC,-shared \
		-DJAVA_SUPPORT \
		-I"$(JAVA_HOME)/include" -I"$(JAVA_HOME)/include/linux" \
		-I$(INCLUDE) -L$(LIB) \
		-include "$$JNI_HEADER" \
		-DJNI_HEADER_NAME=$$JNI_HEADER_NAME \
		-o $(OUTPUT) -lzmq $<; \
	rm -f *$(JAVA_CLASS).h

%-debug-jni: %.cu
	@echo "Preparing to compile $* with JNI support..."
	$(eval JAVA_FILE := $(filter %.java,$(MAKECMDGOALS)))
	$(eval JAVA_CLASS := $(basename $(notdir $(JAVA_FILE))))
	$(eval OUTPUT := $(OUTDIR)/$@.so)
	@if [ ! -f *$(JAVA_CLASS).h ] || [ "$(JAVA_FILE)" -nt $$(ls *$(JAVA_CLASS).h 2>/dev/null || echo /dev/null) ]; then \
		echo "Generating JNI header for class $(JAVA_CLASS)..."; \
		javac -h . "$(JAVA_FILE)"; \
	fi
	@JNI_HEADER=$$(ls *$(JAVA_CLASS).h); \
	JNI_HEADER_NAME=$$(basename "$$JNI_HEADER" .h); \
	echo "Using JNI header: $$JNI_HEADER"; \
	cat "$$JNI_HEADER"; \
	nvcc -G -g -O0 -lineinfo $(ARCH) $(EXTRA) \
		-Xcompiler -fPIC,-shared \
		-DJAVA_SUPPORT \
		-DDEBUG \
		-I"$(JAVA_HOME)/include" -I"$(JAVA_HOME)/include/linux" \
		-I$(INCLUDE) -L$(LIB) \
		-include "$$JNI_HEADER" \
		-DJNI_HEADER_NAME=$$JNI_HEADER_NAME \
		-o $(OUTPUT) -lzmq $<; \
	rm -f *$(JAVA_CLASS).h

%.java: ;

.PHONY: clean
clean:
	rm -f *.o *~ *.out *.exe *.a *.so *.dSYM core.* *.d *.cu~ *.ptx *.cubin *.i *.ii *_be_net be_net *-eval
