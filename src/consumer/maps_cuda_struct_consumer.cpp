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

#include "maps_cuda_struct_consumer.hpp"

#include <maps_io_access.hpp>

MAPS_BEGIN_INPUTS_DEFINITION(MyCudaStructConsumer)
    MAPS_INPUT("i_dynamic_struct", Filter_MyCudaStruct, MAPS::FifoReader)
MAPS_END_INPUTS_DEFINITION

MAPS_BEGIN_OUTPUTS_DEFINITION(MyCudaStructConsumer)
MAPS_END_OUTPUTS_DEFINITION

MAPS_BEGIN_PROPERTIES_DEFINITION(MyCudaStructConsumer)
MAPS_END_PROPERTIES_DEFINITION

MAPS_BEGIN_ACTIONS_DEFINITION(MyCudaStructConsumer)
MAPS_END_ACTIONS_DEFINITION

MAPS_COMPONENT_DEFINITION(MyCudaStructConsumer, "MyCudaStructConsumer", "1.0.1", 128, MAPS::Threaded, MAPS::Threaded,
    -1, // inputs
    -1, // outputs
    -1, // properties
    -1) // actions

void MyCudaStructConsumer::Dynamic()
{
}

void MyCudaStructConsumer::Birth()
{
}

void MyCudaStructConsumer::Core()
{
    // Use an RAII wrapper around StartReading/StopReading
    // https://support.intempora.com/hc/en-us/articles/360007881874
    MAPS::InputGuard<MyCudaStruct> inGuard(this, Input("i_dynamic_struct"));
    if (!inGuard.IsValid())
    {
        return;
    }

    // Get a const reference to the dynamic struct
    const MyCudaStruct& myStruct = inGuard.Data();

    // use the struct
    useStruct(myStruct);
}

void MyCudaStructConsumer::Death()
{
}

void MyCudaStructConsumer::useStruct(const MyCudaStruct& inStruct)
{
    // your code here...
    // use the const reference directly (do NOT alter the content of myStruct)

    std::ostringstream oss;
    oss << "Consumed " << inStruct.toString();
    ReportInfo(oss.str().c_str());
}

