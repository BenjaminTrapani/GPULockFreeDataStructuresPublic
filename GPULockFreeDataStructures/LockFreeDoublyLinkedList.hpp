#pragma once

#include "cuda_runtime.h"
#include "Allocator.hpp"

namespace NUCARLockFreeDS {

	namespace LockFreeDoublyLinkedListConfig {
		// The index type for all containers in the instantiated Allocator instance
		using DoublyLinkedListIndex_t = long long int;
		// The maximum number of hazard pointers that LockFreeDoublyLinkedList will need to hold at a time
		constexpr DoublyLinkedListIndex_t maxAllocationsPerThread = 10; // This is pretty optimistic, only about 5 is necessary now.

		template<typename Index_t>
		constexpr Index_t GetDefaultScanThreshold(const Index_t numThreads) {
			return (numThreads * LockFreeDoublyLinkedListConfig::maxAllocationsPerThread) / 2;
		}
	};

	template<typename T, // The type to be contained in the doubly linked list
		LockFreeDoublyLinkedListConfig::DoublyLinkedListIndex_t numThreads, // The maximum number of threads that will access the list
		LockFreeDoublyLinkedListConfig::DoublyLinkedListIndex_t scanThreshold = LockFreeDoublyLinkedListConfig::GetDefaultScanThreshold<LockFreeDoublyLinkedListConfig::DoublyLinkedListIndex_t>(numThreads)> // How many nodes can be in a thread's deletion list before the allocator starts reclaiming memory?
		class LockFreeDoublyLinkedList {
		public:
			// Empty public class that wraps a Link_t
			class Cursor {};

			/**
			* Instantiate the doubly linked list. Set up head and tail nodes and links.
			*/
			__device__ LockFreeDoublyLinkedList() {
				head = allocator.NewNode();
				tail = allocator.NewNode();
				allocator.StoreRef(head.GetNode()->getNextP(), tail);
				allocator.StoreRef(tail.GetNode()->getPrevP(), head);
				__threadfence();
				//head.prev and tail.next are left pointing to null
			}

			/**
			* Destroy the doubly linked list instance. Clear links between head and tail, and 
			* then delete both.
			*/
			__device__ ~LockFreeDoublyLinkedList() {
				bool gotAnElement = true;
				while (gotAnElement) {
					T dummy;
					gotAnElement = PopRight(dummy);
				}
				allocator.StoreRef(head.GetNode()->getNextP(), nullptr);
				allocator.StoreRef(tail.GetNode()->getPrevP(), nullptr);
				allocator.DeleteNode(head);
				allocator.DeleteNode(tail);
			}

			/**
			* Append a value to the left of the queue. Blocks calling thread until success.
			* @param valueToAdd the value of the template type to append to the queue
			*/
			__device__ void PushLeft(const T& valueToAdd) {
				Link_t node = CreateNode(valueToAdd);
				Link_t prev = allocator.DeRefLink(&head);
				Link_t next = allocator.DeRefLink(prev.GetNode()->getNextP());
				while (true) {
					allocator.StoreRef(node.GetNode()->getPrevP(), Link_t(prev, false));
					allocator.StoreRef(node.GetNode()->getNextP(), Link_t(next, false));
					if (allocator.CompareAndSwapRef(prev.GetNode()->getNextP(),
						Link_t(next, false),
						Link_t(node, false))) {
						break;
					}
					allocator.ReleaseRef(next);
					next = allocator.DeRefLink(prev.GetNode()->getNextP());
					// TODO back off
				}
				allocator.ReleaseRef(prev);
				PushEnd(node, next);
			}

			/**
			* Append a value to the right side of the queue. Will usually be a little slower than PushLeft.
			* @param valueToAdd the value to append to the queue
			*/
			__device__ void PushRight(const T& valueToAdd) {
				Link_t node = CreateNode(valueToAdd);
				//Tail address is constant throughout this instances lifetime, but careful with reads and writes to that address
				Link_t next = allocator.DeRefLink(&tail);
				Link_t prev = allocator.DeRefLink(next.GetNode()->getPrevP());
				while (true) {
					allocator.StoreRef(node.GetNode()->getPrevP(), Link_t(prev, false));
					allocator.StoreRef(node.GetNode()->getNextP(), Link_t(next, false));
					if (allocator.CompareAndSwapRef(prev.GetNode()->getNextP(),
						Link_t(next, false),
						Link_t(node, false))) {
						break;
					}
					prev = CorrectPrev(prev, next);
					// TODO back off
				}
				allocator.ReleaseRef(prev);
				PushEnd(node, next);
			}

			/**
			* Remove an item from the left side of the queue. Block calling thread until success or an empty queue
			* is found, at which point the false is returned. 
			* @param retVal the variable to be filled with the value popped from the queue
			* @return whether or not retVal was filled (false if the queue is empty)
			*/
			__device__ bool PopLeft(T& retVal) {
				Link_t prev = allocator.DeRefLink(&head);
				Link_t next;
				Link_t node;
				while (true) {
					node = allocator.DeRefLink(prev.GetNode()->getNextP());
					if (node.GetNode() == tail.GetNode()) {
						allocator.ReleaseRef(node);
						allocator.ReleaseRef(prev);
						return false;
					}
					next = allocator.DeRefLink(node.GetNode()->getNextP());
					if (next.d() == true) {
						LinkWithDeleteMark::SetMark(node.GetNode()->getPrevP());
						allocator.CompareAndSwapRef(prev.GetNode()->getNextP(), node, Link_t(next, false));
						allocator.ReleaseRef(next);
						allocator.ReleaseRef(node);
						continue;
					}
					const bool atomicCASSucceeded = allocator.CompareAndSwapRef(node.GetNode()->getNextP(), next, Link_t(next, true));
					if (atomicCASSucceeded) {
						prev = CorrectPrev(prev, next);
						allocator.ReleaseRef(prev);
						retVal = node.GetNode()->value;
						allocator.DeleteNode(node);
						break;
					}
					allocator.ReleaseRef(next);
					if (!atomicCASSucceeded)
						allocator.ReleaseRef(node);

					// TODO back off
				}
				allocator.ReleaseRef(next);
				return true;
			}
			
			/**
			* Identical to PopLeft but operates on the right side of the queue.
			*/
			__device__ bool PopRight(T& retVal) {
				Link_t next = allocator.DeRefLink(&tail);
				Link_t nodeLink = allocator.DeRefLink(next.GetNode()->getPrevP());
				while (true) {
					if (nodeLink.GetNode()->atomicReadNext() != Link_t(next, false)) {
						nodeLink = CorrectPrev(nodeLink, next);
						continue;
					}
					if (nodeLink.GetNode() == head.GetNode()) {
						allocator.ReleaseRef(nodeLink);
						allocator.ReleaseRef(next);
						return false;
					}
					if (allocator.CompareAndSwapRef(nodeLink.GetNode()->getNextP(), 
						Link_t(next, false),
						Link_t(next, true))) {
						Link_t prev = allocator.DeRefLink(nodeLink.GetNode()->getPrevP());
						prev = CorrectPrev(prev, next);
						allocator.ReleaseRef(prev);
						allocator.ReleaseRef(next);
						retVal = nodeLink.GetNode()->value;
						allocator.DeleteNode(nodeLink);
						break;
					}
				}
				return true;
			}

			/**
			UNTESTED! Use at your own risk, probably doesn't compile either
			*/
			__device__ bool Delete(Cursor* cursor, T& result) {
				Link_t* linkAtEnd = static_cast<Link_t*>(cursor);
				Link_t nodeLink = Allocator_t::atomicRead(linkAtEnd);
				if (nodeLink.GetNode() == head.GetNode() || nodeLink.GetNode() == tail.GetNode()) {
					return false;
				}
				while (true) {
					Link_t next = allocator.DeRefLink((Allocator_t::atomicRead(linkAtEnd)).GetNode()->getNextP());
					if (next.d() == true) {
						allocator.ReleaseRef(next);
						return false;
					}
					printf("Will CAS next\n");
					if (Allocator_t::CompareAndSwapLinks(nodeLink.GetNode()->getNextP(), next, Link_t(next.GetNode(), true))) {
						printf("CASd next\n");
						Link_t prev;
						while (true) {
							prev = allocator.DeRefLink(nodeLink.GetNode()->getPrevP());
							printf("Will CAS prev\n");
							if (prev.d() == true ||
								Allocator_t::CompareAndSwapLinks(nodeLink.GetNode()->getPrevP(), prev, Link_t(prev.GetNode(), true))) {
								printf("CASd prev\n");
								break;
							}
						}
						prev = CorrectPrev(prev, next);
						allocator.ReleaseRef(prev);
						allocator.ReleaseRef(next);
						result = nodeLink.GetNode()->value;
						allocator.ReleaseRef(nodeLink);
						allocator.DeleteNode(nodeLink);
						return true;
					}
				}
			}

		private:
			class LinkedListNode;
			class TerminateNodeFunctor;
			class CleanUpNodeFunctor;
			class LinkWithDeleteMark;

			class ThisThreadFunctor {
			public:
				__device__ LockFreeDoublyLinkedListConfig::DoublyLinkedListIndex_t operator()() {
					return threadIdx.x + blockIdx.x * blockDim.x;
				}
			};

			using Allocator_t = Allocator<LinkedListNode,
				LinkWithDeleteMark,
				LockFreeDoublyLinkedListConfig::DoublyLinkedListIndex_t,
				TerminateNodeFunctor,
				CleanUpNodeFunctor,
				ThisThreadFunctor,
				-1, numThreads, LockFreeDoublyLinkedListConfig::maxAllocationsPerThread, 2, 2, scanThreshold, 8>;
			using Link_t = LinkWithDeleteMark;
			using Node_t = Allocator_t::Node_t;

			typename Allocator_t allocator;
			Link_t head;
			Link_t tail;

			class LinkWithDeleteMark : public Cursor {
			public:
				__device__ Node_t* GetNode()const {
					return reinterpret_cast<Node_t*>(link & linkPointerMask);
				}
				__device__ bool d() const {
					return link & linkDeleteFlagMask;
				}
				__device__ Allocator_t::Pointer_t GetValue() const {
					return link;
				}
				__device__ static void SetMark(LinkWithDeleteMark* link) {
					while (true) {
						LinkWithDeleteMark node = *link;
						if (node.d() == true ||
							(atomicCAS(&link->link, node.link, LinkWithDeleteMark(node, true).link) == node.link)) {
							break;
						}
					}
				}

				__device__ LinkWithDeleteMark(Node_t* ilink, const bool deleteMark) {
					const auto pointerILink = reinterpret_cast<typename Allocator_t::Pointer_t>(ilink);
					link = (pointerILink & linkPointerMask) | static_cast<typename Allocator_t::Pointer_t>(deleteMark);
				}
				__device__ LinkWithDeleteMark(Node_t* ilink):link(reinterpret_cast<typename Allocator_t::Pointer_t>(ilink)) {}
				__device__ LinkWithDeleteMark(const LinkWithDeleteMark& other, const bool deleteMark) : 
					link((other.link & linkPointerMask) | static_cast<typename Allocator_t::Pointer_t>(deleteMark)) {}
				__device__ LinkWithDeleteMark(const Allocator_t::Pointer_t& value) : link(value) {}
				__device__ LinkWithDeleteMark() {
					// Doing this is required because some compilers complain about reinterpret_casting nullptr to int.
					static const Node_t* linkToNull = nullptr;
					link = reinterpret_cast<typename Allocator_t::Pointer_t>(linkToNull);
				}

				__device__ bool operator==(const LinkWithDeleteMark& other) const {
					return link == other.link;
				}
				__device__ bool operator!=(const LinkWithDeleteMark& other) const {
					return link != other.link;
				}
			private:
				Allocator_t::Pointer_t link;

				static constexpr Allocator_t::Pointer_t linkDeleteFlagMask = 1;
				static constexpr Allocator_t::Pointer_t linkPointerMask = ~linkDeleteFlagMask;
			};

			class LinkedListNode {
			public:
				T value;

				__device__ LinkedListNode() {}

				// getPrevP and getNextP expose the addresses of prev and next respectively
				// No atomic read is required because the addresses of these are constant throughout
				// the lifetime of a LinkedListNode instance. The memory pointed to is likely in global
				// memory so should be read from and written to carefully.
				__device__ Link_t* getPrevP() {
					return &prev;
				}
				__device__ Link_t* getNextP() {
					return &next;
				}

				// Perform and atomic read on either the prev or next links in global memory respectively. 
				// Prevents client code from seeing corrupt view of this 64 bit data structure
				__device__ Link_t atomicReadPrev() {
					return Allocator_t::atomicRead(&prev);
				}
				__device__ Link_t atomicReadNext() {
					return Allocator_t::atomicRead(&next);
				}
			private:
				// These are private so that client code does not issue non-atomic reads or writes
				Link_t prev;
				Link_t next;
			};

			class TerminateNodeFunctor {
			public:
				__device__ void operator()(Allocator_t* pAllocator, 
					typename Allocator_t::Link_t node, 
					const bool isConcurrent) {
					if (!isConcurrent) {
						pAllocator->StoreRef(node.GetNode()->getPrevP(), Allocator_t::Link_t());
						pAllocator->StoreRef(node.GetNode()->getNextP(), Allocator_t::Link_t());
					}
					else {
						while (true) {
							typename Allocator_t::Link_t node1 = node.GetNode()->atomicReadPrev();
							if (pAllocator->CompareAndSwapRef(node.GetNode()->getPrevP(), node1, Allocator_t::Link_t(nullptr))) {
								break;
							}
						}
						while (true) {
							typename Allocator_t::Link_t node1 = node.GetNode()->atomicReadNext();
							if (pAllocator->CompareAndSwapRef(node.GetNode()->getNextP(), node1, Allocator_t::Link_t(nullptr))) {
								break;
							}
						}
					}
				}
			};

			class CleanUpNodeFunctor {
			public:
				__device__ void operator()(Allocator_t* pAllocator, typename Allocator_t::Link_t node) {
					Link_t prev;
					Link_t next;
					while (true) {
						prev = pAllocator->DeRefLink(node.GetNode()->getPrevP());
						if (prev.GetNode() == nullptr || 
							prev.GetNode()->atomicReadPrev().d() == false) {
							break;
						}
						Link_t prev2 = pAllocator->DeRefLink(prev.GetNode()->getPrevP());
						pAllocator->CompareAndSwapRef(node.GetNode()->getPrevP(),
							Link_t(prev, true),
							Link_t(prev2, true));
						pAllocator->ReleaseRef(prev2);
						pAllocator->ReleaseRef(prev);
					}
					while (true) {
						next = pAllocator->DeRefLink(node.GetNode()->getNextP());
						if (next.GetNode() == nullptr || 
							next.GetNode()->atomicReadNext().d() == false) {
							break;
						}
						Link_t next2 = pAllocator->DeRefLink(next.GetNode()->getNextP());
						pAllocator->CompareAndSwapRef(node.GetNode()->getNextP(),
							Link_t(next, true),
							Link_t(next2, true));
						pAllocator->ReleaseRef(next2);
						pAllocator->ReleaseRef(next);
					}
					pAllocator->ReleaseRef(prev);
					pAllocator->ReleaseRef(next);
				}
			};

			__device__ Link_t CreateNode(const T& value) {
				Link_t newNode = allocator.NewNode();
				newNode.GetNode()->value = value;
				__threadfence();
				return newNode;
			}

			__device__ void PushEnd(Link_t nodeLink, Link_t nextLink) {
				while (true) {
					Link_t link1 = nextLink.GetNode()->atomicReadPrev();
					if (link1.d() == true || nodeLink.GetNode()->atomicReadNext() != Link_t(nextLink, false)) {
						break;
					}
					if (allocator.CompareAndSwapRef(nextLink.GetNode()->getPrevP(), link1, Link_t(nodeLink, false))) {
						if (nodeLink.GetNode()->atomicReadPrev().d() == true) {
							nodeLink = CorrectPrev(nodeLink, nextLink);
						}
						break;
					}
					// TODO backoff
				}
				allocator.ReleaseRef(nextLink);
				allocator.ReleaseRef(nodeLink);
			}

			/**
			* Correct the prev pointer of a linked node given a linked node that is known to be 
			* to its left.
			* @param prevLink the link to a node know to be on the left
			* @param nodeLink the link to have its prev pointer corrected (known to be on the right of the other linked node)
			* @return a link to the updated prev pointer of nodeLink
			*/
			__device__ Link_t CorrectPrev(Link_t prevLink, Link_t nodeLink) {
				// lastLink points to the previous value of prevLink after first iteration
				// prevLink is moved to the right at each step in general
				// Once prevLink->next = nodeLink, correct prev pointer of nodeLink to point at prevLink and try to exit
				Link_t lastLink;
				while (true) {
					Link_t link1 = nodeLink.GetNode()->atomicReadPrev();
					if (link1.d() == true) {
						break;
					}
					Link_t prev2 = allocator.DeRefLink(prevLink.GetNode()->getNextP());
					if (prev2.d() == true) {
						if (lastLink != Link_t()) {
							LinkWithDeleteMark::SetMark(prevLink.GetNode()->getPrevP());
							allocator.CompareAndSwapRef(lastLink.GetNode()->getNextP(), prevLink, Link_t(prev2, false));
							allocator.ReleaseRef(prev2);
							allocator.ReleaseRef(prevLink);
							prevLink = lastLink;
							lastLink = Link_t();
							continue;
						}
						allocator.ReleaseRef(prev2);
						prev2 = allocator.DeRefLink(prevLink.GetNode()->getPrevP());
						allocator.ReleaseRef(prevLink);
						prevLink = prev2;
						continue;
					}
					if (prev2 != nodeLink) {
						if (lastLink != Link_t()) {
							allocator.ReleaseRef(lastLink);
						}
						lastLink = prevLink;
						prevLink = prev2;
						continue;
					}
					allocator.ReleaseRef(prev2);
					if (allocator.CompareAndSwapRef(nodeLink.GetNode()->getPrevP(), link1, Link_t(prevLink, false))) {
						if (prevLink.GetNode()->atomicReadPrev().d() == true) {
							continue;
						}
						break;
					}
					// TODO backoff
				}
				if (lastLink != Link_t()) {
					allocator.ReleaseRef(lastLink);
				}
				return prevLink;
			}
	};
}