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

#include <cstdint>
#include <sstream>

#include <cuda.h>
#include <cuda_runtime.h>

#if defined(MAPS_COMPILING_PCK)
    #include <maps.h>  // MAPS::ReportInfo(), should be removed if used in a 3D Viewer's Plugin
    #define maps_report_callback MAPS::ReportInfo
#elif defined(MAPS_3DV_PLUGIN_BUILDING)

    #include <maps_3dv_plugin_interface.h>
    #define maps_report_callback maps_3dvp_report_info
#endif

#pragma pack(push,1)

struct MyCudaStruct
{
	int   m_nbPoints;
	void* m_points;

    MyCudaStruct(const int nbPoints_)
    : m_nbPoints(nbPoints_)
    , m_points(allocateMemory(nbPoints_))
    {
        std::ostringstream oss;
        oss << "New " << toString();
        maps_report_callback(oss.str().c_str());
    }

    ~MyCudaStruct()
    {
        std::ostringstream oss;
        oss << "Delete " << toString();
        maps_report_callback(oss.str().c_str());

        freeMemory(m_points);
    }

    std::string toString() const
    {
        std::ostringstream oss;
        oss << "MyCudaStruct [this:" << this << "] (nbPoints:" << m_nbPoints << ", points:" << m_points << ")";
        return oss.str();
    }

    static
    void* allocateMemory(const int nbPoints_)
    {
        void* points = nullptr;
        const cudaError_t cudaErr = cudaMalloc(&points, nbPoints_ * sizeof(double3));

        if (cudaErr != cudaError_t::cudaSuccess)
        {
            maps_report_callback("Allocation failed");
            throw std::runtime_error("Allocation failed");
        }

        return points;
    }

    static
    void freeMemory(void* points_)
    {
        if (points_ != nullptr)
        {
            cudaFree(points_);
        }
    }
};

#pragma pack(pop)

// Input type filter. Will be valid if you #include <maps.hpp> or <maps_type_filter.hpp> before this header.
// This is irrelevant if you use `MyCudaStruct` in a 3D Viewer's Plugin
#ifdef MAPS_FILTER_USER_DYNAMIC_STRUCTURE
    const MAPSTypeFilterBase Filter_MyCudaStruct = MAPS_FILTER_USER_DYNAMIC_STRUCTURE(MyCudaStruct);
#endif
