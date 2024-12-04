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

#include "kernel1.h"

#include <cuda.h>
#include <cuda_runtime.h>

__global__ void kernel1(double3 *out, unsigned int width, unsigned int height, float time)
{
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	// calculate uv coordinates
	float u = x / (float)width;
	float v = y / (float)height;
	u = u*2.0f - 1.0f;
	v = v*2.0f - 1.0f;

	// calculate simple sine wave pattern
	float freq = 4.0f;
	float w = sinf(u*freq + time) * cosf(v*freq + time) * 0.5f;

	// write output vertex
	out[y*width + x] = make_double3(u, w, v);
}

extern "C" cudaError_t launch_kernel1(double3 *out, unsigned int mesh_width, unsigned int mesh_height, float time)
{
	dim3 block(8, 8, 1);
	dim3 grid(mesh_width / block.x, mesh_height / block.y, 1);
	kernel1<<<grid,block>>>(out, mesh_width, mesh_height, time);
	cudaError_t err = cudaGetLastError();
	return err;
}
