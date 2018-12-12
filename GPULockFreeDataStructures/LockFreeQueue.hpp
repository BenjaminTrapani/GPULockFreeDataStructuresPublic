//
// Created by Benjamin Trapani on 12/21/17.
//

#ifndef GPULOCKFREEDATASTRUCTURES_LOCKFREEQUEUE_H
#define GPULOCKFREEDATASTRUCTURES_LOCKFREEQUEUE_H

#include "cuda_runtime.h"

// When defined, queue specializations that waste space given alignment configuration below will fail to compile.
#define DISALLOW_MISALIGNED_NODE

/***
 * Lock-free concurrent queue supporting many concurrent readers and writers. This is an implementation
 * of the lock-free queue described here: https://www.research.ibm.com/people/m/michael/podc-1996.pdf
 * The implementation relies on device-side allocations, which may be slow. To avoid leaking memory / get correct
 * results, there must not be any ref count collisions in a log2(queueNodeAlignment) bit counter.
 * This might be unlikely depending on how the queue is accessed. A tradeoff between minimum required queue node size
 * can be adjusted by modifying queueNodeAlignment in the AlignmentConfigs namespace below.
 */
namespace LockFreeQueueGPU {

    namespace AlignmentConfigs {
        constexpr unsigned int queueNodeAlignment = 16;
        //Must always equal log2(queueNodeAlignment). Unfortunately no constexpr log2
        constexpr unsigned int numCountBits = 4;
        constexpr unsigned int ptrMask = queueNodeAlignment - 1;
    }

    template<typename T, T sentinel>
    class LockFreeQueue {
    public:
        __device__ LockFreeQueue() {
            QueueNode *initialNode = new QueueNode(sentinel);
            _head = QueueNodePointer(initialNode);
            _tail = QueueNodePointer(initialNode);
        }

        __device__ void enqueue(const T &value) {
            //Perform device-side memory allocation (likely slow)
            QueueNode *newNode = new QueueNode(value);
            QueueNodePointer tail;
            while (true) {
                tail = _tail;
                const QueueNodePointer next = tail.ptr()->next;
                if (tail == _tail) {
                    if (next.ptr() == nullptr) {
                        if (tail.ptr()->next.performAtomicCAS(next, QueueNodePointer(newNode, next.count() + 1))) {
                            break;
                        }
                    } else {
                        _tail.performAtomicCAS(tail, QueueNodePointer(next.ptr(), tail.count() + 1));
                    }
                }
            }
            _tail.performAtomicCAS(tail, QueueNodePointer(newNode, tail.count() + 1));
        }

        __device__ bool dequeue(T &result) {
            QueueNodePointer head;
            while (true) {
                head = _head;
                const QueueNodePointer tail = _tail;
                const QueueNodePointer next = head.ptr()->next;
                if (head == _head) {
                    if (head.ptr() == tail.ptr()) {
                        if (next.ptr() == nullptr) {
                            return false;
                        }
                        _tail.performAtomicCAS(tail, QueueNodePointer(next.ptr(), tail.count() + 1));
                    } else {
                        result = next.ptr()->value;
                        if (_head.performAtomicCAS(head, QueueNodePointer(next.ptr(), head.count() + 1))) {
                            break;
                        }
                    }
                }
            }
            delete head.ptr();
            return true;
        }

    private:

        //Forward-declare QueueNode so that QueueNodePointer can have a member that is a pointer to it
        class alignas(AlignmentConfigs::queueNodeAlignment) QueueNode;

        class QueueNodePointer {
        public:
            typedef unsigned long long int Pointer_t;

            __device__ QueueNodePointer() : _ptr(nullptr) {}

            __device__ QueueNodePointer(QueueNode *initPtr) :
                    _ptr(initPtr) {}

            __device__ QueueNodePointer(QueueNode *initPtr, const unsigned int initCount) {
                Pointer_t intPtr = reinterpret_cast<Pointer_t>(initPtr);
                intPtr |= initCount & AlignmentConfigs::ptrMask;
                _ptr = reinterpret_cast<QueueNode*>(intPtr);
            }

            __device__ QueueNode *ptr() const {
                const Pointer_t intPointer = reinterpret_cast<Pointer_t>(_ptr);
                const Pointer_t correctRawPtr = intPointer >> AlignmentConfigs::numCountBits << AlignmentConfigs::numCountBits;
                return reinterpret_cast<QueueNode*>(correctRawPtr);
            }

            __device__ unsigned int count() const {
                const Pointer_t intPtr = reinterpret_cast<Pointer_t>(_ptr);
                return (unsigned int) (intPtr & AlignmentConfigs::ptrMask);
            }

            __device__ bool performAtomicCAS(const QueueNodePointer &expectedNode, const QueueNodePointer &newNode) {
                Pointer_t *pointerThis = reinterpret_cast<Pointer_t *>(this);
                //Reinterpret cast to pointers here because the size of QueueNodePointer is not known at this point
                // during compilation, so the compiler cannot allocate the right size for a full copy on the stack.
                Pointer_t const *pointerExpectedNode = reinterpret_cast<Pointer_t const*>(&expectedNode);
                Pointer_t const *pointerNewNode = reinterpret_cast<Pointer_t const*>(&newNode);
                const Pointer_t updatedVal = atomicCAS(pointerThis, *pointerExpectedNode, *pointerNewNode);
                return updatedVal == *pointerExpectedNode;
            }

            __device__ inline bool operator==(const QueueNodePointer& other) const {
                return _ptr == other._ptr;
            }

        private:
            QueueNode *_ptr;
        };

        class alignas(AlignmentConfigs::queueNodeAlignment) QueueNode {
        public:
            QueueNodePointer next;
            const T value;

            __device__ QueueNode(const T &initData) : value(initData) {}
        };

        QueueNodePointer _head;
        QueueNodePointer _tail;

        static_assert(sizeof(QueueNodePointer) <= sizeof(unsigned long long int), "To perform AtomicCAS on queue node pointers, "
                "they must be less than or equal to 8 bytes in size");
#ifdef DISALLOW_MISALIGNED_NODE
        static_assert(sizeof(QueueNode) % AlignmentConfigs::queueNodeAlignment == 0,
                      "QueueNode size must be a multiple of queueNodeAlignment bytes so that space is not "
                              "wasted due to alignment. Make sure that ((sizeof(T) + 8) % queueNodeAlignment) = 0");
#endif
    };
};


#endif //GPULOCKFREEDATASTRUCTURES_LOCKFREEQUEUE_H
