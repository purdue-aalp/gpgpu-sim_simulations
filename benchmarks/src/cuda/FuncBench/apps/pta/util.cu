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
#include "util.h"

//extern __device__ uint nextPowerOfTwo(uint v) {
//  return 1U << (uintSize * 8 - __clz(v - 1));
//}
//
//extern __device__ uint __count(int predicate) {
//  const uint ballot = __ballot_sync(0xFFFFFFFF, predicate);
//  return __popc(ballot);
//}
//
//__device__  uint isFirstThreadOfWarp(){
//  return !threadIdx.x;
//}
//
//__device__  uint getWarpIdInGrid(){
//  return (blockIdx.x * (blockDim.x * blockDim.y / WARP_SIZE) + threadIdx.y);
//}
//
//__device__  uint isFirstWarpOfGrid(){
//  return !(blockIdx.x || threadIdx.y);
//}
//
//__device__  uint isFirstWarpOfBlock(){
//  return !threadIdx.y;
//}
//
//__device__  uint getThreadIdInBlock(){
//  return mul32(threadIdx.y) + threadIdx.x;
//}
//
//__device__  uint isFirstThreadOfBlock(){
//  return !getThreadIdInBlock();
//}
//
//__device__  uint getThreadIdInGrid(){
//  return mul32(getWarpIdInGrid()) + threadIdx.x;
//}
//
//__device__  uint getThreadsPerBlock() {
//  return blockDim.x * blockDim.y;
//}
//
//__device__  uint isLastThreadOfBlock(){
//  return getThreadIdInBlock() == getThreadsPerBlock() - 1;
//}
//
//__device__  uint getWarpsPerBlock() {
//  return blockDim.y;
//}
//
//__device__  uint getWarpsPerGrid() {
//  return blockDim.y * gridDim.x;
//}
//
//__device__  uint getThreadsPerGrid() {
//  return mul32(getWarpsPerGrid());
//}
//
//__device__  uint getBlockIdInGrid(){
//  return blockIdx.x;
//}
//
//__device__  uint getBlocksPerGrid(){
//  return gridDim.x;
//}
