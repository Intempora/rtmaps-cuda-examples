/////////////////////////////////////////////////////////////////////////////////
//
//   Copyright 2018-2024 Intempora S.A.S.
//
//   Licensed under the Apache License, Version 2.0 (the "License");
//   you may not use this file except in compliance with the License.
//   You may obtain a copy of the License at
//
//       http://www.apache.org/licenses/LICENSE-2.0
//
//   Unless required by applicable law or agreed to in writing, software
//   distributed under the License is distributed on an "AS IS" BASIS,
//   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//   See the License for the specific language governing permissions and
//   limitations under the License.
//
/////////////////////////////////////////////////////////////////////////////////

#include "maps_cuda_struct_producer.hpp"

#include <cstdint>

#include <maps_io_access.hpp>

#include "kernel1.h"

MAPS_BEGIN_INPUTS_DEFINITION(MyCudaStructProducer)
MAPS_END_INPUTS_DEFINITION

MAPS_BEGIN_OUTPUTS_DEFINITION(MyCudaStructProducer)
    // For the purposes of this simple example, we define a single output that uses a dynamic struct
    MAPS_OUTPUT_USER_DYNAMIC_STRUCTURE("o_dynamic_struct", MyCudaStruct)
    //MAPS_OUTPUT_USER_DYNAMIC_STRUCTURES_VECTOR("o_dynamic_struct", MyCudaStruct, 42)
    //MAPS_OUTPUT_USER_DYNAMIC_STRUCTURE("o_other_dynamic_struct", MyOtherDynamicStruct)
MAPS_END_OUTPUTS_DEFINITION

MAPS_BEGIN_PROPERTIES_DEFINITION(MyCudaStructProducer)
MAPS_END_PROPERTIES_DEFINITION

MAPS_BEGIN_ACTIONS_DEFINITION(MyCudaStructProducer)
MAPS_END_ACTIONS_DEFINITION

MAPS_COMPONENT_DEFINITION(MyCudaStructProducer, "MyCudaStructProducer", "1.0.0", 128, MAPS::Threaded, MAPS::Threaded,
    -1, // inputs
    -1, // outputs
    -1, // properties
    -1) // actions

void MyCudaStructProducer::Dynamic()
{
}

void MyCudaStructProducer::Birth()
{
    // init
    m_meshWidth     = 256;
    m_meshHeight    = 256;
    m_nbPoints      = m_meshWidth * m_meshHeight;
    m_animationTime = 0.0;

    m_appointment = MAPS::CurrentTime();

    initializeCuda();

    try  // wrap in try/catch because MyCudaStruct ctor might throw
    {
        // Allocate the buffers for the dynamic struct (MyCudaStruct)
        // The current component has single output that uses a dynamic struct. Therefore, we pass
        // a single output reference to MAPS_DynamicCustomStructComponent::AllocateDynamicOutputs()
        AllocateDynamicOutputBuffers(
            DynamicOutput<MyCudaStruct>(Output("o_dynamic_struct"),
                [&] { return new MyCudaStruct(m_nbPoints); }  // struct allocation
                //, [&] (MyCudaStruct* p) {  // struct destruction (in FreeBuffers())
                //    cudaFree(p->m_nbPoints);  // not necessary in this example because
                //    delete p;                 // ~MyCudaStruct() already calls cudaFree()
                //}
            )
            //DynamicOutput<MyOtherDynamicStruct>(Output("o_other_dynamic_struct"))  // creates MyOtherDynamicStruct using its default constructor
        );
    }
    catch (...)  // you might want to have a "finer-grained" exception handling here (e.g. `const std::exception&`, `const int`, `const char*` and then `...`
    {
        // prints an error message and causes the component to die
        // RTMaps won't crash
        Error("Failed to allocate the dynamic output buffers");
    }

    // The default FIFO size for RTMaps components is 16. Therefore, you should see 16 calls
    // to MyCudaStruct's constructor in the RTMaps Console
    // (i.e. "new MyCudaStruct" , cf. MyCudaStruct's constructor)
}

void MyCudaStructProducer::Core()
{
    // animation synchronisation logic (irrelevant to the subject of using "dynamic structs")
    m_appointment   += 30000;
	m_animationTime += 0.05f;
	Wait(m_appointment);

    processData();

    Rest(1000000);  // for the purposes of this simple example: wait for 1sec
}

void MyCudaStructProducer::Death()
{
}

void MyCudaStructProducer::FreeBuffers()
{
    // Important: Free the memory that has been allocated for the dynamic struct
    MAPS_DynamicCustomStructComponent::FreeBuffers();

    // As mentioned in Birth(), 16 objects of type MyCudaStruct have been allocated.
    // Therefore, when freeing the memory, 16 objects will be destroyed and 16 calls
    // to MyCudaStruct's destructor should be seen in the RTMaps Console
    // (i.e. "delete MyCudaStruct" , cf. MyCudaStruct's destructor)
}

void MyCudaStructProducer::initializeCuda()
{
    const auto initRc = cuInit(0);
    if (initRc != CUresult::CUDA_SUCCESS)
    {
        switch (initRc)
        {
            case CUresult::CUDA_ERROR_INVALID_VALUE:
                Error("Failed to init CUDA: CUDA_ERROR_INVALID_VALUE");
                break;
            case CUresult::CUDA_ERROR_INVALID_DEVICE:
                Error("Failed to init CUDA: CUDA_ERROR_INVALID_DEVICE");
                break;
            default:
                Error(MAPSStreamedString() << "Failed to set the device: Unknown error " << static_cast<int>(initRc));
                break;
        }
    }

    const auto setDeviceRc = cudaSetDevice(0);
    if (setDeviceRc != cudaError_t::cudaSuccess)
    {
        switch (setDeviceRc)
        {
            case cudaError_t::cudaErrorInvalidDevice:
                Error("Failed to set the device: cudaErrorInvalidDevice");
                break;
            case cudaError_t::cudaErrorDeviceAlreadyInUse:
                Error("Failed to set the device: cudaErrorDeviceAlreadyInUse");
                break;
            default:
                Error(MAPSStreamedString() << "Failed to set the device: Unknown error " << static_cast<int>(setDeviceRc));
                break;
        }
    }
}

void MyCudaStructProducer::processData()
{
    // Open the output in order to write the data
    // Use an RAII wrapper around StartReading/StopReading
    // https://support.intempora.com/hc/en-us/articles/360007881874
    MAPS::OutputGuard<MyCudaStruct> outGuard(this, Output("o_dynamic_struct"));

    // Get a reference to the dynamically-allocated custom struct
    MyCudaStruct& myStruct = outGuard.Data();

    // Fill the the struct
    fillStruct(myStruct);

    // Important: Always set the timestamp (for this example, we will use the current RTMaps virtual time)
    outGuard.Timestamp() = MAPS::CurrentTime();

    // Important: Always set the vector size.
    // Here, VectorSize() = sizeof(MyCudaStruct) is an RTMaps convention to express
    // the fact that there is a single element of type MyCudaStruct in the output vector.
    // If we wanted to output more than one element at a time, then:
    // 1. MAPS_OUTPUT_USER_DYNAMIC_STRUCTURES_VECTOR must be used to define the output
    // 2. The output buffer should be allocated as follows
    //    AllocateDynamicOutputBuffers(
    //        DynamicOutput<MyCudaStruct>(Output("o_dynamic_struct"),
    //            [&] {
    //                MyCudaStruct* outArray = new MyCudaStruct[maxNumberOfElements];  // you probably want to define MyCudaStruct() (i.e. default ctor) for this to work in a sane manner ;)
    //                // logic to init each element of outArray...
    //            },
    //            [&] (MyCudaStruct* p) {  // struct destruction (in FreeBuffers())
    //                // logic to free each element of outArray...
    //                delete[] p;  // call `delete[]` instead of `delete`
    //            }
    //        )
    //    );
    // 3. VectorSize() = numberOfElements * sizeof(MyCudaStruct);  // numberOfElements <= maxNumberOfElements
    outGuard.VectorSize() = sizeof(MyCudaStruct);  // RTMaps convention

    // StopWriting() will be called in the destructor of outGuard
}

void MyCudaStructProducer::fillStruct(MyCudaStruct& myStruct)
{
    launch_kernel1(static_cast<double3*>(myStruct.m_points), m_meshWidth, m_meshHeight, m_animationTime);

    if (cudaDeviceSynchronize() != cudaError_t::cudaSuccess)
    {
        Error("Device Synchronization failed");
    }
}
