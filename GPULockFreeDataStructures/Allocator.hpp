#pragma once

#include "cuda_runtime.h"

/**
* An implementation of the paper Efficient and Reliable Lock-Free Memory Reclamation Based on Reference Counting
*/
namespace NUCARLockFreeDS {
	template<typename T, // For now T needs to be default constructible
		typename ILink_t, // The type of link, needs to implement Node_t* GetNode()
		typename IIndex_t, // The type used to store index information for all internal lists. indexSentinel should fit in this. Should be like an int.
		typename TerminateNode_t, // The functor invoked when a node is about to be deleted by the os. Should free any remaining references.
		typename CleanUpNode_t, // Functor invoked to ensure that the references held by a node are still active. Used to free references to logically deleted nodes.
		typename GetThisThreadID_t, // Should be a functor returning the thread id of the calling thread of type IIndex_t
		// An index representing a null numeric index. If a signed integral type, any number less than 0 will do. For unsigned
		// values, a value orders of magnitude larger than the number of threads, indices per thread and scan threshold will suffice but is risky.
		IIndex_t indexSentinel, 
		IIndex_t numThreads, // Total number of thread running concurrently
		IIndex_t indicesPerThread, // Maximum number of references that a thread can own at one time.
		IIndex_t maxLinks, // Maximum number of links that a node in the dependent data structure can have to other nodes
		IIndex_t maxLinksToDeletedNode, // The maximum number of links that a node can have to logically deleted nodes (depends on data structure using this)
		IIndex_t scanThreshold, //threshold_2 in the paper. Smaller values more aggressively attempt to reclaim memory at the expensive of processing time.
		unsigned long long int nodeAlignment=8> // Node alignment must be a power of 2 greater than or equal to 8
	class Allocator {
	public:
		using Pointer_t = unsigned long long int;
		using This_t = Allocator<T,
			ILink_t,
			IIndex_t,
			TerminateNode_t,
			CleanUpNode_t,
			GetThisThreadID_t,
			indexSentinel,
			numThreads,
			indicesPerThread,
			maxLinks,
			maxLinksToDeletedNode,
			scanThreshold,
			nodeAlignment>;

		/**
		* Allocator node type that extends the type to be allocated T with some metadata
		* required by the allocator. Instances of this type are passed to client code
		* and are expected to be returned to client code via Link_t::GetNode()
		*/
		class alignas(nodeAlignment) Node : public T {
			friend class This_t;
		public:
			__device__ Node() : refCount(0), trace(false), del(false) {
				static_assert(sizeof(refCount) <= 4, "refCount larger than word size, use atomic ops");
				static_assert(sizeof(trace) <= 4, "trace larger than word size, use atomic ops");
				static_assert(sizeof(del) <= 4, "del larger than word size, use atomic ops");

				__threadfence();
			}
		private:
			// Don't need to make operations on these atomic, since they are less than or equal to word size (as of now 32 bits)
			volatile int refCount;
			volatile bool trace;
			volatile bool del;
		};

		using Link_t = ILink_t;
		using Index_t = IIndex_t;
		using Node_t = Node;

		/** 
		* Sync all initial values to global memory before doing anything. Client code should wait on thread
		* that calls this constructor so that it gets a consistent view of the allocator instance after initialization.
		*/
		__device__ Allocator() : HP{ { nullptr } }, DLNodes{ { nullptr } }, DLClaims{ { 0 } }, DLDone{ { false } } {
			__threadfence();
		}

		/** 
		* Fetch the value pointed to by a link. Reserve a hazard pointer and store the node at the end of the link
		* in the hazard pointer table.
		* @param link the link to dereference
		* @return the dereferenced link
		*/
		__device__ Link_t DeRefLink(Link_t* link) {
			const Index_t callingThread = GetThisThreadID();
			const Index_t freeHPIndex = findIndexWhereEqual<numThreads, indicesPerThread>(HP, callingThread, nullptr);
			if (freeHPIndex != indexSentinel) {
				while (true) {
					const Link_t linkVal = atomicRead<Link_t>(link);
					atomicWrite(&HP[callingThread][freeHPIndex], linkVal.GetNode());
					if (atomicRead<Link_t>(link) == linkVal) {
						return linkVal;
					}
				}
			}
			else {
				printf("ERROR: out of hazard pointers on thread %lld\n", callingThread);
			}
			return Link_t();
		}

		/**
		* Free the hazard pointer allocated to the node pointed to by a link.
		* @param link the link to release a reference to
		*/
		__device__ void ReleaseRef(Link_t link) {
			const Index_t callingThread = GetThisThreadID();
			const Index_t occupiedHPIndex = findIndexWhereEqual<numThreads, indicesPerThread>(HP, callingThread, link.GetNode());
			if (occupiedHPIndex != indexSentinel) {
				atomicWrite<Node*>(&HP[callingThread][occupiedHPIndex], nullptr);
			}
			else {
				printf("%lld ERROR: node to be released does not exist in hp list\n", callingThread);
			}
		}

		/**
		* Perform an atomic compare and swap on Link_t by treating Link_t as a Pointer_t integral type
		* @param link a pointer to the link to be conditionally modified
		* @param old if the link to be conditionally modified is equal to this value, its value will be updated
		* @param node the new value to assign to the first argument
		*/
		__device__ static bool CompareAndSwapLinks(Link_t* link, const Link_t& old, const Link_t& node) {
			static_assert(sizeof(Link_t) == sizeof(Pointer_t), "The size of a link must match atomic cas size");
			Pointer_t* convertedLink = reinterpret_cast<Pointer_t*>(link);
			Pointer_t const* oldInt = reinterpret_cast<Pointer_t const*>(&old);
			Pointer_t const* newInt = reinterpret_cast<Pointer_t const*>(&node);
			const Pointer_t result = atomicCAS(convertedLink, *oldInt, *newInt);
			return result == *oldInt;
		}

		/**
		* Hack to simulate an atomic read (TODO replace this once CUDA supports something similar)
		* @param pLink pointer to TypeToRead that is read atomically (avoids torn views of data)
		* @return a local copy of the memory pointed to by the input link
		*/
		template<typename TypeToRead>
		__device__ static TypeToRead atomicRead(TypeToRead* pLink) {
			static_assert(sizeof(TypeToRead) == sizeof(Pointer_t), "The size of the type for atomic read must match atomic add size");
			Pointer_t* convertedLink = reinterpret_cast<Pointer_t*>(pLink);
			const Pointer_t resultOfAtomicWrite = atomicAdd(convertedLink, 0);
			const TypeToRead* reinterpretedAfterWrite = reinterpret_cast<TypeToRead const*>(&resultOfAtomicWrite);
			return *reinterpretedAfterWrite;
		}

		/** 
		* Hack to simulate an atomic write (TODO replace this once CUDA supports something similar)
		* @param pLink the destination in which to store the value
		* @param val the value to write
		*/
		template<typename TypeToWrite>
		__device__ static void atomicWrite(TypeToWrite* pLink, const TypeToWrite& val) {
			static_assert(sizeof(TypeToWrite) == sizeof(Pointer_t), "The size of type to atomic write size must match atomic CAS size");
			bool valWrittenSuccessfully = false;
			Pointer_t* convertedLink = reinterpret_cast<Pointer_t*>(pLink);
			Pointer_t const* pointerVal = reinterpret_cast<Pointer_t const*>(&val);
			const Pointer_t localValCopy = *pointerVal;
			while (!valWrittenSuccessfully) {
				const Pointer_t expectedOld = *convertedLink;
				const Pointer_t prevVal = atomicCAS(convertedLink, expectedOld, localValCopy);
				valWrittenSuccessfully = prevVal == expectedOld;
			}
		}

		/**
		* Performs an atomic compare and swap with the input arguments as they are in CompareAndSwapLinks.
		* If the atomic compare and swap succeeds, the ref count of old is decremented and
		* the ref count of node is incremented.
		* @return whether or not the the atomic compare and swap succeeded
		*/ 
		__device__ bool CompareAndSwapRef(Link_t* link, const Link_t old, const Link_t node) {
			if (This_t::CompareAndSwapLinks(link, old, node)) {
				updateRefCounts(old.GetNode(), node.GetNode());
				return true;
			}
			return false;
		}

		/** 
		* Swap the value of a link such that the ref counts of the linked nodes are consistent with the change.
		* @param link the link to have its value swapped. The previously linked node will have its ref count decremented.
		* @param node the value that will replace the previously linked value of the first arg. The linked node's ref count will be incremented.
		*/
		__device__ void StoreRef(Link_t* link, Link_t node) {
			Node* old = atomicRead<Link_t>(link).GetNode();
			atomicWrite<Link_t>(link, node);
			Node* newNode = node.GetNode();
			updateRefCounts(old, newNode);
		}

		/** 
		* Allocate a new node from the GPU heap. Allocate a hazard pointer
		* for it.
		* @return a link to the newly created node
		*/
		__device__ Link_t NewNode(){
			const Index_t callingThread = GetThisThreadID();
			Node* node = new Node();
			const Index_t emptyIndex = findIndexWhereEqual<numThreads, indicesPerThread>(HP, callingThread, nullptr);
			if (emptyIndex != indexSentinel) {
				atomicWrite(&HP[callingThread][emptyIndex], node);
				return Link_t(node);
			}
			else {
				printf("%lld ERROR: could not get hazard pointer for newly alloced node\n", callingThread);
			}
			return Link_t();
		}

		/** 
		* Free the hazard pointer allocated to the node pointed to by the input link.
		* Add the node pointed to to this thread's deletion list. If the thread's deletion
		* list is too long, start attempting to reclaim the memory of deleted nodes (see Scan)
		* @param link the link to the node to be deleted
		*/
		__device__ void DeleteNode(Link_t link) {
			const Index_t callingThread = GetThisThreadID();
			Node* node = link.GetNode();
			ReleaseRef(link);

			node->del = true;
			__threadfence();

			node->trace = false;
			__threadfence();

			const Index_t freeIndex = findIndexWhereEqual<numThreads, threshold1>(DLNodes, callingThread, nullptr);
			if (freeIndex == indexSentinel) {
				printf("%lld ERROR could not find free index for deleted node\n", callingThread);
			}
			DLDone[callingThread][freeIndex] = false;
			__threadfence();

			atomicWrite<Node*>(&DLNodes[callingThread][freeIndex], node);

			PerThreadState* thisThreadState = &perThreadState[callingThread];
			thisThreadState->DLNexts[freeIndex] = thisThreadState->dlist;
			thisThreadState->dlist = freeIndex;
			++thisThreadState->dcount;

			while (true) {
				if (thisThreadState->dcount == threshold1) {
					CleanUpLocal(callingThread);
				}
				if (thisThreadState->dcount >= scanThreshold) {
					Scan(callingThread);
				}
				if (thisThreadState->dcount == threshold1) {
					CleanUpAll();
				} else {
					break;
				}
			}
		}
		
	private:
		// The maximum number of nodes in any given thread's deletion list
		static constexpr size_t threshold1 = numThreads * (indicesPerThread + maxLinks + maxLinksToDeletedNode + 1);

		// Functor instances
		TerminateNode_t TerminateNode;
		CleanUpNode_t CleanUpNode;
		GetThisThreadID_t GetThisThreadID;

		// PList is just intended for use per thread. Functions are not thread-safe.
		// Stores a set of nodes in a dense array that is allocated in the same space as PList
		class PList {
		public:
			__device__ void addNode(const Node* data) {
				if (position >= numThreads * indicesPerThread) {
					printf("ERROR: out of space in plist\n");
				}
				storage[position] = data;
				++position;
			}
			__device__ bool contains(const Node* elem) {
				for (Index_t i = 0; i < position; i++) {
					if (storage[i] == elem) {
						return true;
					}
				}
				return false;
			}
			__device__ void clear() {
				position = 0;
			}
			__device__ PList() : position(0) {}
		private:
			Node const * storage[numThreads * indicesPerThread];
			Index_t position;
		};

		// Per-thread state
		class PerThreadState {
		public:
			Index_t dlist; // Index into first deleted node (stored in DLNodes)
			Index_t dcount; // Total number of deleted nodes for this thread
			/* Since there are numThreads instances of these per allocator and threshold1 is proportional to numThreads,
			   the space used by the allocator grows as numThreads^2, which is pretty bad.
			*/
			// Indices into next deleted node index, used as follows
			//    index = dlist;
			//    Node* node = DLNodes[callingThread][index];
			//    ...
			//    index = thisThreadState->DLNexts[index];
			Index_t DLNexts[threshold1];

			//Temporary storage used to keep track of nodes in other threads' hp arrays during scan
			PList nodeList;

			__device__ PerThreadState() : dlist(indexSentinel), dcount(0) {
				// To sync dlist and dcount initialization if allocated in global memory
				__threadfence();
			}
		};

		// State in global memory per Allocator instance
		
		// Hazerd pointers per thread
		Node* HP[numThreads][indicesPerThread];
		// Nodes marked for deletion in this thread
		Node* DLNodes[numThreads][threshold1];
		volatile int DLClaims[numThreads][threshold1];
		volatile bool DLDone[numThreads][threshold1];
		PerThreadState perThreadState[numThreads];
		
		template<Index_t height, Index_t width>
		__device__ Index_t findIndexWhereEqual(Node* input[height][width], const Index_t callingThread, const Node* expected) {
			// TODO maybe parallelize this using a sub-kernel
			for (Index_t i = 0; i < width; i++) {
				if (atomicRead<Node*>(&input[callingThread][i]) == expected) {
					return i;
				}
			}
			return indexSentinel;
		}

		__device__ void updateRefCounts(Node* old, Node* node) {
			if (node != nullptr) {
				int* nonvolRefCount = (int*)(&node->refCount);
				atomicAdd(nonvolRefCount, 1);
				node->trace = false;
				__threadfence();
			}
			if (old != nullptr) {
				int* nonvolRefCount = (int*)(&old->refCount);
				atomicSub(nonvolRefCount, 1);
			}
		}

		// Internal functions as described in paper

		// Remove references to redundant deleted nodes from deleted nodes in this thread's deletion list
		__device__ void CleanUpLocal(const Index_t callingThread) {
			PerThreadState* thisThreadState = &perThreadState[callingThread];
			Index_t index = thisThreadState->dlist;
			while (index != indexSentinel) {
				Node* node = atomicRead<Node*>(&DLNodes[callingThread][index]);
				CleanUpNode(this, Link_t(node));
				index = thisThreadState->DLNexts[index];
			}
		}

		// Same as CleanUpLocal but operates on all threads
		__device__ void CleanUpAll() {
			// TODO parallelize these checks with subkernels
			for (Index_t thread = 0; thread < numThreads; ++thread) {
				for (Index_t index = 0; index < threshold1; ++index) {
					Node* node = atomicRead<Node*>(&DLNodes[thread][index]);
					if (node != nullptr && !DLDone[thread][index]) { 
						int* volDLClaims = (int*)(&DLClaims[thread][index]);
						atomicAdd(volDLClaims, 1);
						if (node == atomicRead<Node*>(&DLNodes[thread][index])) {
							CleanUpNode(this, Link_t(node));
						}
						atomicSub(volDLClaims, 1);
					}
				}
			}
		}

		// Try to reclaim nodes that are not in any other threads' hp set or
		// are referenced by other nodes (have positive ref count)
		__device__ void Scan(const Index_t callingThread) {
			PerThreadState* thisThreadState = &perThreadState[callingThread];
			Index_t index = thisThreadState->dlist;
			Node* node;
			while (index != indexSentinel) {
				node = atomicRead<Node*>(&DLNodes[callingThread][index]);
				if (node->refCount == 0) {
					node->trace = true;
					__threadfence();
					if (node->refCount != 0) {
						node->trace = false;
						__threadfence();
					}
				}
				index = thisThreadState->DLNexts[index];
			}
			PList* nodeList = &thisThreadState->nodeList;
			nodeList->clear();
			for (Index_t thread = 0; thread < numThreads; ++thread) {
				for (Index_t index = 0; index < indicesPerThread; ++index) {
					node = atomicRead(&HP[thread][index]);
					if (node != nullptr) {
						nodeList->addNode(node);
					}
				}
			}

			Index_t newDList = indexSentinel;
			Index_t newDCount = 0;

			while (thisThreadState->dlist != indexSentinel) {
				index = thisThreadState->dlist;
				node = atomicRead<Node*>(&DLNodes[callingThread][index]);
				thisThreadState->dlist = thisThreadState->DLNexts[index];
				if (node->refCount == 0 && node->trace && !nodeList->contains(node)) {
					atomicWrite<Node*>(&DLNodes[callingThread][index], nullptr);
					if (DLClaims[callingThread][index] == 0) {
						TerminateNode(this, node, false);
						delete node;
						continue;
					}
					TerminateNode(this, node, true);

					DLDone[callingThread][index] = true;
					__threadfence();

					atomicWrite<Node*>(&DLNodes[callingThread][index], node);
				}
				thisThreadState->DLNexts[index] = newDList;
				newDList = index;
				++newDCount;
			}

			thisThreadState->dlist = newDList;
			thisThreadState->dcount = newDCount;
		}

		static_assert(scanThreshold <= threshold1, "scanThreshold (threshold 2) should be less than or equal to threshold 1.");
	};
}