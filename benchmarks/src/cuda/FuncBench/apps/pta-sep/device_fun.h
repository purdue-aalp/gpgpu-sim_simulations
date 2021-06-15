#include "andersen.h"

extern __constant__ uint __storeStart__;
extern __constant__ uint __loadInvStart__;

/**
 *  number of variables of the input program.
 */
extern __constant__ uint __numVars__;

extern __device__ uint __counter__;

extern __device__ uint __worklistIndex0__;
extern __device__ uint __worklistIndex1__;

extern __constant__  uint* __lock__;

extern __constant__ uint* __key__;
extern __constant__ uint* __val__;
extern __constant__ uint* __keyAux__;
extern __device__ uint __numKeysCounter__;
extern __constant__ uint* __currPtsHead__;

extern __constant__ uint* __gepInv__;
extern __constant__ uint __numGepInv__;

extern __constant__ volatile uint* __rep__; // HAS to be volatile

extern __constant__ volatile uint* __edges__; // HAS to be volatile
extern __constant__ uint* __graph__;

extern __constant__ uint* __diffPtsMask__;

extern __device__ uint __ptsFreeList__,__nextDiffPtsFreeList__, __currDiffPtsFreeList__, __otherFreeList__;

extern __constant__ uint* __offsetMask__;
extern __constant__ uint __offsetMaskRowsPerOffset__; 

extern __device__ INLINE2 void printDiffPtsMask();
extern __device__ INLINE2 void printOffsetMasks(uint numObjectsVars, uint maxOffset);
extern __device__ INLINE2 void printEdgesOf(const uint src, int rel);
extern __device__ INLINE2 void printEdges(const uint src, const uint rel, const uint printEmptySets); 
extern __device__ INLINE2 void printEdgesOf(uint src);
extern __device__ INLINE2 void printEdges(int rel);
extern __device__ INLINE2 void printGepEdges();
extern __device__ INLINE uint getWarpsPerGrid();
extern __device__ INLINE uint getWarpIdInGrid();
extern __device__ INLINE2 int checkForErrors(uint var, uint rel);
extern __device__ INLINE uint getAndIncrement(const uint delta);
extern __device__ INLINE uint getAndIncrement(uint* counter, uint delta);
extern __device__ INLINE uint getHeadIndex(uint var, uint rel);
extern __device__ INLINE2 uint insert(const uint index, const uint var, const int rel); 
extern __device__ INLINE uint resetWorklistIndex();
template<uint firstRel, uint secondRel, uint thirdRel> extern __device__ INLINE2 void apply(const uint src, uint* const _shared_);
extern __device__ INLINE2 void store2storeInv(const uint src, uint* const _shared_);
extern __device__ INLINE uint isFirstWarpOfBlock();
extern __device__ INLINE uint isFirstThreadOfWarp();
extern __device__ INLINE2 void warpStoreInv(const uint i, uint* const _pending_, uint* _numPending_);
extern __device__ INLINE2 void blockStoreInv(uint src, uint* const _dummyVars_, volatile uint* _warpInfo_, uint& _numPending_);
extern __device__ INLINE uint getRep(const uint var);
extern __device__ INLINE uint lock(const uint var);
extern __device__ INLINE2 void applyGepInvRule(uint x, const uint y, const uint offset, volatile uint* _shared_);
extern __device__ INLINE void unlock(const uint var);
extern __device__ INLINE2 bool updatePtsAndDiffPts(const uint var);
extern __device__ INLINE uint getCurrDiffPtsHeadIndex(uint var);
extern __device__ INLINE void __graphSet__(const uint pos, const uint val);
extern __device__ INLINE uint mul960(uint num);
extern __device__ INLINE void __offsetMaskSet__(const uint base, const uint col, const uint offset, const uint val);
extern __device__ INLINE uint getThreadIdInBlock();
extern __device__ INLINE uint getThreadsPerBlock();
extern __device__ INLINE uint isFirstThreadOfBlock();
extern __device__ INLINE uint getBlockIdInGrid();
extern __device__ INLINE2 void lockVars(uint* const _currVar_, uint& _currVarSize_, uint* const _nextVar_, uint* _nextVarSize_);
extern __device__ INLINE2 void lockPtrs(uint* const _currPtr_, uint& _currPtrSize_, uint* const _nextPtr_, uint* _nextPtrSize_, uint* const _currVar_, uint* _currVarSize_, uint* const _nextVar_, uint* _nextVarSize_);
extern __device__ INLINE2 void unlockPtrs(const uint* const _list_, const uint _listSize_);
extern __device__ INLINE2 void blockSort(volatile uint* _shared_, uint to);
extern __device__ INLINE2 void mergeCycle(const uint* const _list_, const uint _listSize_);
extern __device__ INLINE uint getThreadsPerGrid();
extern __device__ INLINE uint getThreadIdInGrid();
extern __device__ INLINE uint getRepRec(const uint var);
extern __device__ INLINE void setRep(const uint var, const uint rep);
extern __device__ INLINE int isEmpty(uint var, uint rel);
extern __device__ INLINE uint __diffPtsMaskGet__(const uint base, const uint col);
extern __device__ INLINE void __diffPtsMaskSet__(const uint base, const uint col, const uint val);
extern __device__ INLINE2 void syncAllThreads();
extern __device__ INLINE2 uint hashCode(uint index);
extern __device__ INLINE2 uint equal(uint index1, uint index2);
