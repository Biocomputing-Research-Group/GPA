################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/gpa.cpp \
../src/gpa_mpi.cpp \
../src/gpa_openmp.cpp \
../src/gpa_type1.cpp 

OBJS += \
./src/gpa.o \
./src/gpa_mpi.o \
./src/gpa_openmp.o \
./src/gpa_type1.o 

CPP_DEPS += \
./src/gpa.d \
./src/gpa_mpi.d \
./src/gpa_openmp.d \
./src/gpa_type1.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: Cross G++ Compiler'
	-$(MCC) $(MOPTS) -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


