/*
  A GPU implementation of Andersen's analysis

  Copyright (c) 2012 The University of Texas at Austin

  This library is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License as published by the Free Software Foundation; either
  version 2.1 of the License, or (at your option) any later version.

  This library is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this library; if not, write to the Free Software
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301
  USA, or see <http://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>.

  Author: Mario Mendez-Lojo
*/
#include "andersen.h"

//__device__ uint __worklistIndex0__ = 0;
//__device__ uint __worklistIndex1__ = 1;

extern __device__ uint nextPowerOfTwo(uint v) ;

extern __device__ uint __count(int predicate) ;

extern __device__ uint isFirstThreadOfWarp();

//extern __device__  uint getWarpIdInGrid();
//
//extern __device__  uint isFirstWarpOfGrid();
//
//extern __device__  uint isFirstWarpOfBlock();
//
//extern __device__  uint getThreadIdInBlock();
//
//extern __device__  uint isFirstThreadOfBlock();
//
//extern __device__  uint getThreadIdInGrid();
//
//extern __device__  uint getThreadsPerBlock() ;
//
//extern __device__  uint isLastThreadOfBlock();
//
//extern __device__  uint getWarpsPerBlock() ;
//
//extern __device__  uint getWarpsPerGrid() ;
//
//extern __device__  uint getThreadsPerGrid() ;
//
//extern __device__ inline uint getBlockIdInGrid();
//
//extern __device__ inline uint getBlocksPerGrid();
//
//upper level
//extern __device__ inline uint getAndIncrement(const uint delta);

//extern __device__ inline uint getAndIncrement(uint* counter, uint delta);
