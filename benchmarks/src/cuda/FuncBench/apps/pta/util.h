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

extern __constant__ uint __storeStart__;
extern __constant__ uint __loadInvStart__;
//INLINE utility device functions for copyInv_loadInv_store2storeInv()
//inlined to avoid reg pressure
__device__ INLINE uint nextPowerOfTwo(uint v) {
  return 1U << (uintSize * 8 - __clz(v - 1));
}

__device__ INLINE uint __count(int predicate) {
  const uint ballot = __ballot_sync(0xFFFFFFFF, predicate);
  return __popc(ballot);
}

__device__ INLINE uint isFirstThreadOfWarp(){
  return !threadIdx.x;
}


/**
 * Representative array
 */
extern __constant__ volatile uint* __rep__; // HAS to be volatile

/**
 * array of elements containing all the edges in the graph.
 */
extern __constant__ volatile uint* __edges__; // HAS to be volatile
extern __constant__ uint* __graph__;

extern __constant__  uint* __lock__;

__device__ INLINE uint __graphGet__(const uint row,  const uint col) {
  return __edges__[row + col];
}

__device__ INLINE uint __graphGet__(const uint pos) {
  return __graph__[pos];
}

__device__ INLINE void __graphSet__(const uint row,  const uint col, const uint val) {
  __edges__[row + col] = val;
}

__device__ INLINE void __graphSet__(const uint pos, const uint val) {
  __graph__[pos] = val;
}

__device__ INLINE uint getHeadIndex(uint var, uint rel){
  if (rel == NEXT_DIFF_PTS) {
    return NEXT_DIFF_PTS_START - mul32(var);
  }
  if (rel == COPY_INV) {
    return COPY_INV_START + mul32(var);
  }
  if (rel == CURR_DIFF_PTS) {
    return CURR_DIFF_PTS_START - mul32(var);
  }
  if (rel == PTS) {
    return mul32(var);
  }
  if (rel == STORE) {
    return __storeStart__ + mul32(var);
  }
  // it has to be LOAD_INV, right?
  return __loadInvStart__ + mul32(var);
}

__device__ INLINE uint getNextDiffPtsHeadIndex(uint var){
    return NEXT_DIFF_PTS_START - mul32(var);
}

__device__ INLINE uint getCopyInvHeadIndex(uint var){
    return COPY_INV_START + mul32(var);
}

__device__ INLINE uint getCurrDiffPtsHeadIndex(uint var){
    return CURR_DIFF_PTS_START - mul32(var);
}

__device__ INLINE uint getPtsHeadIndex(uint var){
    return mul32(var);
}

__device__ INLINE uint getStoreHeadIndex(uint var){
    return __storeStart__ + mul32(var);
}

__device__ INLINE uint getLoadInvHeadIndex(uint var){
    return __loadInvStart__ + mul32(var);
}

__device__ INLINE int isEmpty(uint var, uint rel) {
  const uint headIndex = getHeadIndex(var, rel);
  return __graphGet__(headIndex, BASE) == NIL;
}

__device__  INLINE uint unique(volatile uint* const _shared_, uint to) {
  uint startPos = 0;
  uint myMask = (1 << (threadIdx.x + 1)) - 1;
  for (int id = threadIdx.x; id < to; id += WARP_SIZE) {
    uint myVal = _shared_[id];
    uint fresh = __ballot_sync(0xFFFFFFFF, myVal != _shared_[id - 1]);
    // pos = starting position + number of 1's to my right (incl. myself) minus one
    uint pos = startPos + __popc(fresh & myMask) - 1;
    _shared_[pos] = myVal;
    startPos += __popc(fresh);
  }
  return startPos;
}

/**
 * Index of the next free element in the corresponding free list.
 * The index is given in words, not bytes or number of elements.
 */
extern __device__ uint __ptsFreeList__,__nextDiffPtsFreeList__, __currDiffPtsFreeList__, __otherFreeList__;

__device__ INLINE uint mallocPts(uint size = ELEMENT_WIDTH) {
  __shared__ volatile uint _shared_[MAX_WARPS_PER_BLOCK];
  if (isFirstThreadOfWarp()) {
    _shared_[threadIdx.y] = atomicAdd(&__ptsFreeList__, size);
  }
  return _shared_[threadIdx.y];
}

__device__ INLINE uint mallocNextDiffPts() {
  __shared__ volatile uint _shared_[MAX_WARPS_PER_BLOCK];
  if (isFirstThreadOfWarp()) {
    _shared_[threadIdx.y] = atomicSub(&__nextDiffPtsFreeList__, ELEMENT_WIDTH);
  }
  return _shared_[threadIdx.y];
}

__device__ INLINE uint mallocCurrDiffPts() {
  __shared__ volatile uint _shared_[MAX_WARPS_PER_BLOCK];
  if (isFirstThreadOfWarp()) {
    _shared_[threadIdx.y] = atomicSub(&__currDiffPtsFreeList__, ELEMENT_WIDTH);
  }
  return _shared_[threadIdx.y];
}

__device__ INLINE uint mallocOther() {
  __shared__ volatile uint _shared_[MAX_WARPS_PER_BLOCK]; 
  if (isFirstThreadOfWarp()) {
    _shared_[threadIdx.y] = atomicAdd(&__otherFreeList__, ELEMENT_WIDTH);
  }
  return _shared_[threadIdx.y];
}

__device__ INLINE uint mallocIn(uint rel) {
  if (rel == NEXT_DIFF_PTS) {
    return mallocNextDiffPts();
  }
  if (rel >= COPY_INV) {
    return mallocOther();
  }
  if (rel == PTS) {
    return mallocPts();
  }
  if (rel == CURR_DIFF_PTS) {
    return mallocCurrDiffPts();
  }
  //printf("WTF! (%u)", rel);
  return 0;
}

/**
 * Get and increment the current worklist index
 * Granularity: warp
 * @param delta Number of elements to be retrieved at once 
 * @return Worklist index 'i'. All the work items in the [i, i + delta) interval are guaranteed
 * to be assigned to the current warp.
 */
extern __device__ uint __worklistIndex0__;
extern __device__ uint __worklistIndex1__;

__device__ INLINE uint getAndIncrement(const uint delta) {
  __shared__ volatile uint _shared_[MAX_WARPS_PER_BLOCK];
  if (isFirstThreadOfWarp()) {
    _shared_[threadIdx.y] = atomicAdd(&__worklistIndex0__, delta);
  }
  return _shared_[threadIdx.y];
}

__device__ INLINE uint getAndIncrement(uint* counter, uint delta) {
  __shared__ volatile uint _shared_[MAX_WARPS_PER_BLOCK];
  if (isFirstThreadOfWarp()) {
    _shared_[threadIdx.y] = atomicAdd(counter, delta);
  }
  return _shared_[threadIdx.y];
}

/**
 * Lock a given variable 
 * Granularity: warp
 * @param var Id of the variable
 * @return A non-zero value if the operation succeeded
 */
__device__ INLINE uint lock(const uint var) {
  return __any_sync(0xFFFFFFFF,isFirstThreadOfWarp() && (atomicCAS(__lock__ + var, UNLOCKED, LOCKED) 
      == UNLOCKED));
}

/**
 * Unlock a variable
 * Granularity: warp or thread
 * @param var Id of the variable
 */
__device__ INLINE void unlock(const uint var) {
  __lock__[var] = UNLOCKED;
}

__device__ INLINE int isRep(const uint var) {
  return __rep__[var] == var;
}

__device__ INLINE void setRep(const uint var, const uint rep) {
  __rep__[var] = rep;
}

__device__ INLINE uint getRep(const uint var) {
  return __rep__[var];
}

__device__ INLINE uint getRepRec(const uint var) {
  uint rep = var;
  uint repRep = __rep__[rep];
  while (repRep != rep) {
    rep = repRep;
    repRep = __rep__[rep];
  } 
  return rep;
}

/**
 * Mask that tells whether the pts-to of an element changed.
 * the BASE and NEXT words are always equal to 0
 * stored in compressed format
 */
extern __constant__ uint* __diffPtsMask__;

__device__ INLINE uint __diffPtsMaskGet__(const uint base, const uint col) {
  return __diffPtsMask__[mul32(base) + col];
}

__device__ INLINE void __diffPtsMaskSet__(const uint base, const uint col, const uint val) {
  __diffPtsMask__[mul32(base) + col] = val;
}

__device__ INLINE uint mul960(uint num) {
  // 960 = 1024 - 64
  return (num << 10) - (num << 6);
}
//INLINE device functions for copyInv_loadInv_store2storeInv()
extern __device__ /*INLINE*/ void swap(volatile uint* const keyA, volatile uint* const keyB, const uint dir);

//INLINE2 device functions for copyInv_loadInv_store2storeInv()
extern __device__ INLINE2 uint removeDuplicates(volatile uint* const _shared_, const uint to);
extern __device__ INLINE2 uint getValAtThread(const uint myVal, const uint i);
extern __device__ INLINE2 void unionToCopyInv(const uint to, const uint fromIndex, uint* const _shared_, 
    bool applyCopy = true);
extern __device__ INLINE2 void store2storeInv(const uint src, uint* const _shared_);

//extern template<uint firstRel, uint secondRel, uint thirdRel>
extern __constant__ uint* __key__;
extern __constant__ uint* __val__;
extern __device__ uint __numKeysCounter__;
extern __constant__ uint* __currPtsHead__;

//__device__ INLINE2 void apply(const uint src, uint* const _shared_);
extern __device__ INLINE2 void apply_3_2_1(const uint src, uint* const _shared_);
extern __device__ INLINE2 void apply_4_2_3(const uint src, uint* const _shared_);
extern __device__ INLINE2 void insertAll(const uint src, uint* const _shared_, uint numFrom, const bool sort);
extern __device__ INLINE2 void clone(uint toIndex, uint fromBits, uint fromNext, const uint toRel); 
