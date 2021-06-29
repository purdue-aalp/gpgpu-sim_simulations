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

// Functions must be local and extern since local __device__ vars
__constant__ uint __storeStart__;
__constant__ uint __loadInvStart__;

/**
 * Representative array
 */
__constant__ volatile uint* __rep__; // HAS to be volatile

/**
 * array of elements containing all the edges in the graph.
 */
__constant__ volatile uint* __edges__; // HAS to be volatile
__constant__ uint* __graph__;

__constant__  uint* __lock__;

__device__ uint __worklistIndex0__ = 0;
__device__ uint __worklistIndex1__ = 1;

__device__ uint __ptsFreeList__,__nextDiffPtsFreeList__, __currDiffPtsFreeList__, __otherFreeList__;

__constant__ uint* __diffPtsMask__;

__constant__ uint* __key__;
__constant__ uint* __val__;
__device__ uint __numKeysCounter__ = 0;
__constant__ uint* __currPtsHead__;

//INLINE device functions for copyInv_loadInv_store2storeInv()
__device__ /*INLINE*/ void swap(volatile uint* const keyA, volatile uint* const keyB, const uint dir) {
  uint n1 = *keyA;
  uint n2 = *keyB;
  if ((n1 < n2) != dir) {
    *keyA = n2;
    *keyB = n1;
  }
}

// Bitonic Sort, in ascending order using one WARP
// precondition: size of _shared_ has to be a power of 2
__device__ INLINE void bitonicSort(volatile uint* const _shared_, const uint to) {
  for (int size = 2; size <= to; size <<= 1) {
    for (int stride = size / 2; stride > 0; stride >>= 1) {
      for (int id = threadIdx.x; id < (to / 2); id += WARP_SIZE) {
        const uint myDir = ((id & (size / 2)) == 0);
        uint pos = 2 * id - mod(id, stride);
        volatile uint* start = _shared_  + pos;
        swap(start, start + stride, myDir);
      }
    }
  }
}

__device__ INLINE2 uint getValAtThread(volatile uint* const _shared_, const uint myVal, const uint i) {
  if (threadIdx.x == i) {
    _shared_[threadIdx.y] = myVal;
  }
  return _shared_[threadIdx.y];
}

__device__ INLINE2 uint getValAtThread(const uint myVal, const uint i) {
  __shared__ volatile uint _shared_[MAX_WARPS_PER_BLOCK];
  if (threadIdx.x == i) {
    _shared_[threadIdx.y] = myVal;
  }
  return _shared_[threadIdx.y];
}

//INLINE2 device functions for copyInv_loadInv_store2storeInv()
/*
 * Forward declarations
 */
__device__ INLINE2 void insertAll(const uint storeIndex, uint* _shared_, uint numFrom, bool sort = true);

template<uint toRel, uint fromRel>
__device__ INLINE2 void unionAll(const uint to, uint* _shared_, uint numFrom, bool sort = true);

template<uint toRel, uint fromRel>
__device__ INLINE2  void map(const uint to, const uint base, const uint myBits, uint* _shared_,
    uint& numFrom);

__device__ INLINE2 uint removeDuplicates(volatile uint* const _shared_, const uint to) {
  const uint size = max(nextPowerOfTwo(to), 32);
  for (int i = to + threadIdx.x; i < size; i += WARP_SIZE) {
    _shared_[i] = NIL;
  }
  bitonicSort(_shared_, size);
  uint ret = unique(_shared_, size);
  return (size > to) ? ret - 1 : ret;
}

__device__ INLINE2 void unionToCopyInv(const uint to, const uint fromIndex, uint* const _shared_, 
    bool applyCopy) {
  uint toIndex = getCopyInvHeadIndex(to);
  if (fromIndex == toIndex) {
    return;
  }
  uint fromBits = __graphGet__(fromIndex + threadIdx.x);
  uint fromBase = __graphGet__(fromIndex + BASE);
  if (fromBase == NIL) {
    return;
  }
  uint fromNext = __graphGet__(fromIndex + NEXT);
  uint toBits = __graphGet__(toIndex + threadIdx.x);
  uint toBase = __graphGet__(toIndex + BASE);
  uint toNext = __graphGet__(toIndex + NEXT);
  uint numFrom = 0;
  uint newVal;
  while (1) {
    if (toBase > fromBase) {
      if (toBase == NIL) {
        newVal = fromNext == NIL ? NIL : mallocOther();
      } else {
        newVal = mallocOther();
        __graphSet__(newVal + threadIdx.x, toBits);
      }
      fromBits = threadIdx.x == NEXT ? newVal : fromBits;
      __graphSet__(toIndex + threadIdx.x, fromBits);
      if (applyCopy) {
        map<NEXT_DIFF_PTS, PTS>(to, fromBase, fromBits, _shared_, numFrom);
      }
      if (fromNext == NIL) {
        break;
      }
      toIndex = newVal;
      fromBits = __graphGet__(fromNext + threadIdx.x);
      fromBase = __graphGet__(fromNext + BASE);
      fromNext = __graphGet__(fromNext + NEXT);      
    } else if (toBase == fromBase) {
      uint orBits = fromBits | toBits;
      uint diffs = __any_sync(0xFFFFFFFF,orBits != toBits && threadIdx.x < NEXT);
      bool nextWasNil = false;
      if (toNext == NIL && fromNext != NIL) {
        toNext = mallocOther();
        nextWasNil = true;
      }
      uint newBits = threadIdx.x == NEXT ? toNext : orBits;
      if (newBits != toBits) {
        __graphSet__(toIndex + threadIdx.x, newBits);
      }
      // if there was any element added to COPY_INV, apply COPY_INV rule
      if (applyCopy && diffs) {
        uint diffBits = fromBits & ~toBits;
        map<NEXT_DIFF_PTS, PTS > (to, fromBase, diffBits, _shared_, numFrom);
      }
      //advance `to` and `from`
      if (fromNext == NIL) {
        break;
      }
      toIndex = toNext;
      if (nextWasNil) {
        toBits = NIL;
        toBase = NIL;
        toNext = NIL;
      } else {
        toBits = __graphGet__(toIndex + threadIdx.x);
        toBase = __graphGet__(toIndex + BASE);
        toNext = __graphGet__(toIndex + NEXT);
      }
      fromBits = __graphGet__(fromNext + threadIdx.x);
      fromBase = __graphGet__(fromNext + BASE);
      fromNext = __graphGet__(fromNext + NEXT);      
    } else { //toBase < fromBase
      if (toNext == NIL) {
        uint newNext = mallocOther();
        __graphSet__(toIndex + NEXT, newNext);
        toIndex = newNext;
        toBits = NIL;
        toBase = NIL;
      } else {
        toIndex = toNext;
        toBits = __graphGet__(toNext + threadIdx.x);
        toBase = __graphGet__(toIndex + BASE);
        toNext = __graphGet__(toNext + NEXT);        
      }
    }
  }
  if (applyCopy && numFrom) {
    // flush pending unions
    unionAll<NEXT_DIFF_PTS, PTS> (to, _shared_, numFrom);
  }
}

__device__ INLINE2 void clone(uint toIndex, uint fromBits, uint fromNext, const uint toRel) {  
  while (1) {
    uint newIndex = fromNext == NIL ? NIL : mallocIn(toRel);    
    uint val = threadIdx.x == NEXT ? newIndex : fromBits;
    __graphSet__(toIndex + threadIdx.x, val);
    if (fromNext == NIL) {
      break;
    }
    toIndex = newIndex;
    fromBits = __graphGet__(fromNext + threadIdx.x);
    fromNext = __graphGet__(fromNext + NEXT);        
  } 
}
// toRel = any non-static relationship
__device__ INLINE2 void unionG2G(const uint to, const uint toRel, const uint fromIndex) {
  uint toIndex = getHeadIndex(to, toRel);
  uint fromBits = __graphGet__(fromIndex + threadIdx.x); 
  uint fromBase = __graphGet__(fromIndex + BASE);
  if (fromBase == NIL) {
    return;
  }
  uint fromNext = __graphGet__(fromIndex + NEXT);
  uint toBits = __graphGet__(toIndex + threadIdx.x);
  uint toBase = __graphGet__(toIndex + BASE);
  if (toBase == NIL) {
    clone(toIndex, fromBits, fromNext, toRel);
    return;
  }
  uint toNext = __graphGet__(toIndex + NEXT);
  while (1) {
    if (toBase > fromBase) {
      uint newIndex = mallocIn(toRel);
      __graphSet__(newIndex + threadIdx.x, toBits);      
      uint val = threadIdx.x == NEXT ? newIndex : fromBits;
      __graphSet__(toIndex + threadIdx.x, val);
      // advance 'from'
      if (fromNext == NIL) {
        return;
      }
      toIndex = newIndex;
      fromBits = __graphGet__(fromNext + threadIdx.x);
      fromBase = __graphGet__(fromNext + BASE);
      fromNext = __graphGet__(fromNext + NEXT);        
    } else if (toBase == fromBase) {
      uint newToNext = (toNext == NIL && fromNext != NIL) ? mallocIn(toRel) : toNext;
      uint orBits = fromBits | toBits;
      uint newBits = threadIdx.x == NEXT ? newToNext : orBits;
      if (newBits != toBits) {
        __graphSet__(toIndex + threadIdx.x, newBits);
      }
      //advance `to` and `from`
      if (fromNext == NIL) {
        return;
      }
      fromBits = __graphGet__(fromNext + threadIdx.x);
      fromBase = __graphGet__(fromNext + BASE);
      fromNext = __graphGet__(fromNext + NEXT);      
      if (toNext == NIL) {
        clone(newToNext, fromBits, fromNext, toRel);
        return;
      } 
      toIndex = newToNext;
      toBits = __graphGet__(toNext + threadIdx.x);
      toBase = __graphGet__(toNext + BASE);
      toNext = __graphGet__(toNext + NEXT);
    } else { // toBase < fromBase
      if (toNext == NIL) {
        toNext = mallocIn(toRel);
        __graphSet__(toIndex + NEXT, toNext);
        clone(toNext, fromBits, fromNext, toRel);
        return;
      } 
      toIndex = toNext;
      toBits = __graphGet__(toNext + threadIdx.x);
      toBase = __graphGet__(toNext + BASE);
      toNext = __graphGet__(toNext + NEXT);      
    }
  } 
}

template<uint toRel, uint fromRel>
__device__  INLINE2 void unionAll(const uint to, uint* const _shared_, uint numFrom, bool sort) {
  if (numFrom > 1 && sort) {
    numFrom = removeDuplicates(_shared_, numFrom);
  }
  for (int i = 0; i < numFrom; i++) {
    uint fromIndex = _shared_[i];     
    if (fromRel != CURR_DIFF_PTS) {
      fromIndex = getHeadIndex(fromIndex, fromRel);
    }
    if (toRel == COPY_INV) {
      unionToCopyInv(to, fromIndex, _shared_ + DECODE_VECTOR_SIZE + 1);
    } else {
      unionG2G(to, toRel, fromIndex);
    }
  }
}

template<uint toRel, uint fromRel>
__device__ INLINE2  void map(uint to, const uint base, const uint myBits, uint* const _shared_, 
    uint& numFrom) {
  uint nonEmpty = __ballot_sync(0xFFFFFFFF, myBits) & LT_BASE;
  const uint threadMask = 1 << threadIdx.x;
  const uint myMask = threadMask - 1;
  const uint mul960base = mul960(base);
  while (nonEmpty) {
    uint pos = __ffs(nonEmpty) - 1;
    nonEmpty &= (nonEmpty - 1);
    uint bits = getValAtThread(myBits, pos);
    uint var =  getRep(mul960base + mul32(pos) + threadIdx.x); //coalesced
    uint bitActive = (var != I2P) && (bits & threadMask);
    bits = __ballot_sync(0xFFFFFFFF, bitActive);
    uint numOnes = __popc(bits);
    if (numFrom + numOnes > DECODE_VECTOR_SIZE) {
      numFrom = removeDuplicates(_shared_, numFrom);
      if (numFrom + numOnes > DECODE_VECTOR_SIZE) {
        if (toRel == STORE) {
          insertAll(to, _shared_, numFrom, false);
        } else {                
          unionAll<toRel, fromRel>(to, _shared_, numFrom, false); 
        }
        numFrom = 0;
      }
    }
    pos = numFrom + __popc(bits & myMask);
    if (bitActive) {      
      _shared_[pos] = (fromRel == CURR_DIFF_PTS) ? __currPtsHead__[var] : var;
    }
    numFrom += numOnes;
  }
}

//template<uint firstRel, uint secondRel, uint thirdRel>
//__device__ INLINE2 void apply(const uint src, uint* const _shared_) {
//  uint numFrom = 0;
//  uint index = getHeadIndex(src, firstRel);
//  do {
//    uint myBits = __graphGet__(index + threadIdx.x);
//    uint base = __graphGet__(index + BASE);
//    if (base == NIL) {
//      break;
//    }
//    index = __graphGet__(index + NEXT);
//    if (secondRel == CURR_DIFF_PTS) {
//      myBits &= __diffPtsMaskGet__(base, threadIdx.x);
//    } 
//    map<thirdRel, secondRel>(src, base, myBits, _shared_, numFrom);
//  } while (index != NIL);
//  if (numFrom) {
//    unionAll<thirdRel, secondRel>(src, _shared_, numFrom);
//  }
//}

__device__ INLINE2 void apply_3_2_1(const uint src, uint* const _shared_) {
  uint numFrom = 0;
  uint index = getHeadIndex(src, 3);
  do {
    uint myBits = __graphGet__(index + threadIdx.x);
    uint base = __graphGet__(index + BASE);
    if (base == NIL) {
      break;
    }
    index = __graphGet__(index + NEXT);
    if (2 == CURR_DIFF_PTS) {
      myBits &= __diffPtsMaskGet__(base, threadIdx.x);
    } 
    map<1, 2>(src, base, myBits, _shared_, numFrom);
  } while (index != NIL);
  if (numFrom) {
    unionAll<1, 2>(src, _shared_, numFrom);
  }
}

__device__ INLINE2 void apply_4_2_3(const uint src, uint* const _shared_) {
  uint numFrom = 0;
  uint index = getHeadIndex(src, 4);
  do {
    uint myBits = __graphGet__(index + threadIdx.x);
    uint base = __graphGet__(index + BASE);
    if (base == NIL) {
      break;
    }
    index = __graphGet__(index + NEXT);
    if (2 == CURR_DIFF_PTS) {
      myBits &= __diffPtsMaskGet__(base, threadIdx.x);
    } 
    map<3, 2>(src, base, myBits, _shared_, numFrom);
  } while (index != NIL);
  if (numFrom) {
    unionAll<3, 2>(src, _shared_, numFrom);
  }
}

__device__ INLINE2 void insertAll(const uint src, uint* const _shared_, uint numFrom, const bool sort) {
  if (numFrom > 1 && sort) {
    numFrom = removeDuplicates(_shared_, numFrom);
  }
  const uint storeIndex = getStoreHeadIndex(src);
  for (int i = 0; i < numFrom; i += WARP_SIZE) {
    uint size = min(numFrom - i, WARP_SIZE);
    uint next = getAndIncrement(&__numKeysCounter__, size);
    // TODO: we need to make sure that (next + threadIdx.x < MAX_HASH_SIZE)
    if (threadIdx.x < size) {
      __key__[next + threadIdx.x] = _shared_[i + threadIdx.x]; // at most 2 transactions
      __val__[next + threadIdx.x] = storeIndex;    
    }
  }
}

__device__ INLINE2 void store2storeInv(const uint src, uint* const _shared_) {
  uint currDiffPtsIndex = getCurrDiffPtsHeadIndex(src);
  uint numFrom = 0;
  do {
    uint myBits = __graphGet__(currDiffPtsIndex + threadIdx.x);
    uint base = __graphGet__(currDiffPtsIndex + BASE);
    if (base == NIL) {
      break;
    }
    currDiffPtsIndex = __graphGet__(currDiffPtsIndex + NEXT);
    map<STORE, STORE>(src, base, myBits, _shared_, numFrom);
  } while (currDiffPtsIndex != NIL);
  if (numFrom) {
    insertAll(src, _shared_, numFrom);
  }
}
