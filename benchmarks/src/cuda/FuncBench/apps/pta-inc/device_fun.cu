#include "device_fun.h"

//Note: Section 1 inline functions

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
  
__device__ INLINE uint getWarpIdInGrid(){
    return (blockIdx.x * (blockDim.x * blockDim.y / WARP_SIZE) + threadIdx.y);
}
  
__device__ INLINE uint isFirstWarpOfGrid(){
    return !(blockIdx.x || threadIdx.y);
}
  
__device__ INLINE uint isFirstWarpOfBlock(){
    return !threadIdx.y;
}
  
__device__ INLINE uint getThreadIdInBlock(){
    return mul32(threadIdx.y) + threadIdx.x;
}
  
__device__ INLINE uint isFirstThreadOfBlock(){
    return !getThreadIdInBlock();
}
  
__device__ INLINE uint getThreadIdInGrid(){
    return mul32(getWarpIdInGrid()) + threadIdx.x;
}
  
__device__ INLINE uint getThreadsPerBlock() {
    return blockDim.x * blockDim.y;
}
  
__device__ INLINE uint isLastThreadOfBlock(){
    return getThreadIdInBlock() == getThreadsPerBlock() - 1;
}
  
__device__ INLINE uint getWarpsPerBlock() {
    return blockDim.y;
}
  
__device__ INLINE uint getWarpsPerGrid() {
    return blockDim.y * gridDim.x;
}
  
__device__ INLINE uint getThreadsPerGrid() {
    return mul32(getWarpsPerGrid());
}
  
__device__ INLINE uint getBlockIdInGrid(){
    return blockIdx.x;
}
  
__device__ INLINE uint getBlocksPerGrid(){
    return gridDim.x;
}


//Note: Section 2 inline functions

__constant__ volatile uint* __edges__; // HAS to be volatile
__constant__ uint* __graph__;

__constant__ uint __storeStart__;
__constant__ uint __loadInvStart__;

__device__ INLINE uint mul960(uint num) {
    // 960 = 1024 - 64
    return (num << 10) - (num << 6);
}
  
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
  
__device__ INLINE uint _sharedGet_(volatile uint* _shared_, uint index, uint offset) {
    return _shared_[index + offset];
}
  
__device__ INLINE void _sharedSet_(volatile uint* _shared_, uint index, uint offset, uint val) {
    _shared_[index + offset] = val;
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

//Note: Section 3 inline functions
/**
 * Mask that tells whether the variables contained in an element have size > offset
 * There is one such mask per offset.
 * stored in compressed format
 */
__constant__ uint* __offsetMask__;

/**
 * Number of rows needed to represent the mask of ONE offset.
 * = ceil(numObjectVars / DST_PER_ELEMENT), since non-object pointers have size 1.
 */
__constant__ uint __offsetMaskRowsPerOffset__; 

__device__ INLINE uint __offsetMaskGet__(const uint base, const uint col, const uint offset) {
  return __offsetMask__[mul32((offset - 1) * __offsetMaskRowsPerOffset__ + base) + col];
}

__device__ INLINE void __offsetMaskSet__(const uint base, const uint col, const uint offset,
    const uint val) {
  __offsetMask__[mul32((offset - 1) * __offsetMaskRowsPerOffset__ + base) + col] = val;
}

/**
 * Mask that tells whether the pts-to of an element changed.
 * the BASE and NEXT words are always equal to 0
 * stored in compressed format
 */
__constant__ uint* __diffPtsMask__;

__device__ INLINE uint __diffPtsMaskGet__(const uint base, const uint col) {
  return __diffPtsMask__[mul32(base) + col];
}

__device__ INLINE void __diffPtsMaskSet__(const uint base, const uint col, const uint val) {
  __diffPtsMask__[mul32(base) + col] = val;
}

__device__ uint __ptsFreeList__,__nextDiffPtsFreeList__, __currDiffPtsFreeList__, __otherFreeList__;

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

//Note: Section 4 inline functions

/**
 * Get and increment the current worklist index
 * Granularity: warp
 * @param delta Number of elements to be retrieved at once 
 * @return Worklist index 'i'. All the work items in the [i, i + delta) interval are guaranteed
 * to be assigned to the current warp.
 */
__device__ uint __worklistIndex0__ = 0;
__device__ uint __worklistIndex1__ = 1;
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

__constant__  uint* __lock__;
__constant__ volatile uint* __rep__; // HAS to be volatile
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

__device__ INLINE uint decodeWord(const uint base, const uint word, const uint bits) {
  uint ret = mul960(base) + mul32(word);
  return (isBitActive(bits, threadIdx.x)) ? __rep__[ret + threadIdx.x] : NIL;
}

__device__ INLINE void swap(volatile uint* const keyA, volatile uint* const keyB, const uint dir) {
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

/**
 * Remove duplicates on a sorted sequence, equivalent to Thrust 'unique' function but uses one warp.
 * If there are NILS, they are treated like any other number
 * precondition: the input list is sorted
 * precondition: to >= 32
 * precondition: shared_[-1] exists and is equal to NIL
 * Granularity: warp
 *
 * @param _shared_ list of integers
 * @param to size of the sublist we want to process
 * @return number of unique elements in the input.
 */
__device__ INLINE uint unique(volatile uint* const _shared_, uint to) {
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

__device__ uint __counter__ = 0;
__device__ INLINE uint resetWorklistIndex() {
  __syncthreads();
  uint numBlocks = getBlocksPerGrid();
  if (isFirstThreadOfBlock() && atomicInc(&__counter__, numBlocks - 1) == (numBlocks - 1)) {
    __worklistIndex0__ = 0;
    __counter__ = 0;
    return 1;
  }  
  return 0;
}

///Note: Section 5 inline2 functions

__device__ INLINE2 void syncAllThreads() {
  __syncthreads();
  uint to = getBlocksPerGrid() - 1;
  if (isFirstThreadOfBlock()) {      
    volatile uint* counter = &__counter__;
    if (atomicInc((uint*) counter, to) < to) {       
      while (*counter); // spinning...
    }
  }
  __syncthreads();
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

__device__ INLINE2 void blockBitonicSort(volatile uint* _shared_, uint to) {
  uint idInBlock = getThreadIdInBlock();
  for (int size = 2; size <= to; size <<= 1) {
    for (int stride = size / 2; stride > 0; stride >>= 1) {
      __syncthreads();
      for (int id = idInBlock; id < (to / 2); id += getThreadsPerBlock()) {
        const uint myDir = ((id & (size / 2)) == 0);
        uint pos = 2 * id - mod(id, stride);
        volatile uint* start = _shared_ + pos;
        swap(start, start + stride, myDir);
      }
    }
  }
}

/**
 * Sort an array in ascending order.
 * Granularity: block
 * @param _shared_ list of integers
 * @param to size of the sublist we want to process
 */
 __device__ INLINE2 void blockSort(volatile uint* _shared_, uint to) {
  uint size = max(nextPowerOfTwo(to), 32);
  uint id = getThreadIdInBlock();
  for (int i = to + id; i < size; i += getThreadsPerBlock()) {
    _shared_[i] = NIL;
  }
  blockBitonicSort(_shared_, size);  
  __syncthreads();
}

__device__ INLINE2 uint removeDuplicates(volatile uint* const _shared_, const uint to) {
  const uint size = max(nextPowerOfTwo(to), 32);
  for (int i = to + threadIdx.x; i < size; i += WARP_SIZE) {
    _shared_[i] = NIL;
  }
  bitonicSort(_shared_, size);
  uint ret = unique(_shared_, size);
  return (size > to) ? ret - 1 : ret;
}

__device__ INLINE2 uint hashCode(uint index) {
  __shared__ uint _sh_[DEF_THREADS_PER_BLOCK];
  volatile uint* _shared_ = &_sh_[threadIdx.y * WARP_SIZE];
  uint myRet = 0;
  uint bits = __graphGet__(index + threadIdx.x);
  uint base = __graphGet__(index + BASE);
  if (base == NIL) {
    return 0;
  }
  while (1) {
    uint elementHash = base * (30 + threadIdx.x) ^ bits;
    if (bits) {
      myRet ^= elementHash;      
    }
    index = __graphGet__(index + NEXT);
    if (index == NIL) {
      break;
    }
    bits = __graphGet__(index + threadIdx.x);
    base = __graphGet__(index + BASE);
  } 
  _shared_[threadIdx.x] = myRet;
  if (threadIdx.x < 14) {
    _shared_[threadIdx.x] ^= _shared_[threadIdx.x + WARP_SIZE / 2];
  }
  if (threadIdx.x < 8) {
    _shared_[threadIdx.x] ^= _shared_[threadIdx.x + WARP_SIZE / 4];
  }
  if (threadIdx.x < 4) {
    _shared_[threadIdx.x] ^= _shared_[threadIdx.x + WARP_SIZE / 8];
  }
  return _shared_[0] ^ _shared_[1] ^ _shared_[2] ^ _shared_[3];
}

__device__ INLINE2 uint equal(uint index1, uint index2) {
  uint bits1 = __graphGet__(index1 + threadIdx.x);
  uint bits2 = __graphGet__(index2 + threadIdx.x);
  while (__all_sync(0xFFFFFFFF,(threadIdx.x == NEXT) || (bits1 == bits2))) {
    index1 = __graphGet__(index1 + NEXT);
    index2 = __graphGet__(index2 + NEXT);
    if (index1 == NIL || index2 == NIL) {
      return index1 == index2;
    }
    bits1 = __graphGet__(index1 + threadIdx.x);
    bits2 = __graphGet__(index2 + threadIdx.x);
  }
  return 0;
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