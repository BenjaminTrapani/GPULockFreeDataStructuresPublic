#pragma once

#include "LockFreeDoublyLinkedList.hpp"

namespace NUCARLockFreeDS {
	template<typename T,
		LockFreeDoublyLinkedListConfig::DoublyLinkedListIndex_t numThreads,
		LockFreeDoublyLinkedListConfig::DoublyLinkedListIndex_t scanThreshold =
		LockFreeDoublyLinkedListConfig::GetDefaultScanThreshold<LockFreeDoublyLinkedListConfig::DoublyLinkedListIndex_t>(numThreads)>
		class DoubleListBasedStack {
		public:
			__device__ void push(const T& value) {
				list.PushLeft(value);
			}
			__device__ bool pop(T& result) {
				return list.PopLeft(result);
			}

		private:
			using List_t = LockFreeDoublyLinkedList<T, numThreads, scanThreshold>;
			List_t list;
	};
}