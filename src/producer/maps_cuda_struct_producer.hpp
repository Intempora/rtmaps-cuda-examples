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

#pragma once

#include <maps.hpp>

#include "../common/maps_dynamic_custom_struct_component.hpp"  // MAPS_DynamicCustomStructComponent
#include "../common/my_cuda_struct.h"  // MyCudaStruct

/// A user component that uses a dynamic custom struct for its output
class MyCudaStructProducer:
    // For the purposes of the current simple example, there is only one output
    // that uses a dynamic custom struct of type MyCuda (cf. the .cpp file).
    public MAPS_DynamicCustomStructComponent
{
public:
    // Standard RTMaps component "header" for child components
    MAPS_CHILD_COMPONENT_HEADER_CODE(MyCudaStructProducer, MAPS_DynamicCustomStructComponent)

    void Dynamic() override;
    void FreeBuffers() override;

private:
    unsigned int m_meshWidth;
    unsigned int m_meshHeight;
    unsigned int m_nbPoints;
    float        m_animationTime;

    MAPSTimestamp m_appointment;

    void initializeCuda();
    void processData();
    void fillStruct(MyCudaStruct& outStruct);
};
