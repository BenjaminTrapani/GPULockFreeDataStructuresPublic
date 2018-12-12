#pragma once

#include "LockFreeDoublyLinkedList.hpp"

namespace NUCARLockFreeDS {
	template<typename T,
		LockFreeDoublyLinkedListConfig::DoublyLinkedListIndex_t numThreads,
		LockFreeDoublyLinkedListConfig::DoublyLinkedListIndex_t scanThreshold =
		LockFreeDoublyLinkedListConfig::GetDefaultScanThreshold<LockFreeDoublyLinkedListConfig::DoublyLinkedListIndex_t>(numThreads)>
		class DoubleListBasedQueue {
		public:
			__device__ void enqueue(const T& value) {
				list.PushLeft(value);
			}
			__device__ bool dequeue(T& result) {
				return list.PopRight(result);
			}

		private:
			using List_t = LockFreeDoublyLinkedList<T, numThreads, scanThreshold>;
			List_t list;
	};
}