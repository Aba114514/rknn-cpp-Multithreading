#ifndef THREADPOOL_H
#define THREADPOOL_H

#include <cassert>              // 用于断言
#include <condition_variable>   // 用于线程同步的条件变量
#include <functional>           // 用于 std::function
#include <future>               // 用于 std::future 和 std::packaged_task
#include <memory>               // 用于智能指针
#include <mutex>                // 用于互斥锁
#include <queue>                // 用于任务队列
#include <thread>               // 用于线程操作
#include <unordered_map>        // 用于存储线程对象

namespace dpool
{

    class ThreadPool
    {
    public:
        // 类型别名，简化代码
        using MutexGuard = std::lock_guard<std::mutex>;
        using UniqueLock = std::unique_lock<std::mutex>;
        using Thread = std::thread;
        using ThreadID = std::thread::id;
        using Task = std::function<void()>; // 任务被封装成一个无参无返回值的函数

        // 默认构造函数，线程数默认为CPU核心数
        ThreadPool()
            : ThreadPool(Thread::hardware_concurrency())
        {
        }

        // 显式构造函数，指定最大线程数
        explicit ThreadPool(size_t maxThreads)
            : quit_(false),               // 退出标志初始化为 false
              currentThreads_(0),         // 当前线程数初始化为 0
              idleThreads_(0),            // 空闲线程数初始化为 0
              maxThreads_(maxThreads)     // 设置最大线程数
        {
        }

        // 禁用拷贝构造和赋值操作，确保线程池的唯一性
        ThreadPool(const ThreadPool &) = delete;
        ThreadPool &operator=(const ThreadPool &) = delete;

        // 析构函数，用于安全地关闭线程池
        ~ThreadPool()
        {
            {
                MutexGuard guard(mutex_);
                quit_ = true; // 设置退出标志
            }
            cv_.notify_all(); // 唤醒所有等待的线程，以便它们检查退出标志并退出

            // 等待所有线程执行完毕
            for (auto &elem : threads_)
            {
                assert(elem.second.joinable()); // 确保线程是可加入的
                elem.second.join(); // 等待线程结束
            }
        }

        /**
         * @brief 提交一个任务到线程池
         * @tparam Func 函数类型
         * @tparam Ts 参数类型
         * @param func 函数对象
         * @param params 函数参数
         * @return std::future<ReturnType> 一个与任务关联的future对象，用于获取任务的返回值
         */
        template <typename Func, typename... Ts>
        auto submit(Func &&func, Ts &&...params)
            -> std::future<typename std::result_of<Func(Ts...)>::type>
        {
            // 使用 std::bind 将函数和参数绑定成一个可调用对象
            auto execute = std::bind(std::forward<Func>(func), std::forward<Ts>(params)...);

            // 获取任务的返回类型
            using ReturnType = typename std::result_of<Func(Ts...)>::type;
            // 使用 packaged_task 包装任务，以便获取其 future
            using PackagedTask = std::packaged_task<ReturnType()>;

            auto task = std::make_shared<PackagedTask>(std::move(execute));
            auto result = task->get_future(); // 获取与任务关联的 future

            // 加锁以保护任务队列
            MutexGuard guard(mutex_);
            assert(!quit_); // 确保线程池未被关闭

            // 将任务封装成lambda表达式放入任务队列
            tasks_.emplace([task]()
                           { (*task)(); });

            // 决定如何调度线程
            if (idleThreads_ > 0)
            {
                // 如果有空闲线程，则唤醒其中一个来执行任务
                cv_.notify_one();
            }
            else if (currentThreads_ < maxThreads_)
            {
                // 如果没有空闲线程且当前线程数未达上限，则创建一个新线程
                Thread t(&ThreadPool::worker, this);
                assert(threads_.find(t.get_id()) == threads_.end());
                threads_[t.get_id()] = std::move(t); // 将新线程存入map
                ++currentThreads_;
            }

            return result; // 返回 future，调用者可以用它来等待结果
        }

        // 获取当前线程池中的线程数量
        size_t threadsNum() const
        {
            MutexGuard guard(mutex_);
            return currentThreads_;
        }

    private:
        // 工作线程的主函数
        void worker()
        {
            while (true)
            {
                Task task;
                {
                    UniqueLock uniqueLock(mutex_);
                    ++idleThreads_;
                    // 等待任务或超时。wait_for 会自动解锁，等待被唤醒或超时后，重新加锁
                    auto hasTimedout = !cv_.wait_for(uniqueLock,
                                                     std::chrono::seconds(WAIT_SECONDS),
                                                     [this]()
                                                     {
                                                         // 等待条件：线程池退出 或 任务队列不为空
                                                         return quit_ || !tasks_.empty();
                                                     });
                    --idleThreads_;

                    if (tasks_.empty())
                    {
                        if (quit_) // 如果是被唤醒且需要退出
                        {
                            --currentThreads_;
                            return; // 线程退出
                        }
                        if (hasTimedout) // 如果是等待超时
                        {
                            --currentThreads_;
                            joinFinishedThreads(); // 清理已完成的线程资源
                            finishedThreadIDs_.emplace(std::this_thread::get_id()); // 记录自身为待清理线程
                            return; // 线程退出
                        }
                    }

                    // 从任务队列中取出一个任务
                    task = std::move(tasks_.front());
                    tasks_.pop();
                } // 锁在此处释放

                task(); // 执行任务
            }
        }

        // 清理已结束线程的资源
        void joinFinishedThreads()
        {
            while (!finishedThreadIDs_.empty())
            {
                auto id = std::move(finishedThreadIDs_.front());
                finishedThreadIDs_.pop();
                auto iter = threads_.find(id);

                assert(iter != threads_.end());
                assert(iter->second.joinable());

                iter->second.join(); // 等待线程完全结束
                threads_.erase(iter); // 从线程map中移除
            }
        }

        static constexpr size_t WAIT_SECONDS = 2; // 空闲线程等待新任务的超时时间（秒）

        bool quit_;                                  // 线程池退出标志
        size_t currentThreads_;                      // 当前线程数量
        size_t idleThreads_;                         // 空闲线程数量
        size_t maxThreads_;                          // 最大线程数量

        mutable std::mutex mutex_;                   // 互斥锁（mutable允许在const成员函数中修改）
        std::condition_variable cv_;                 // 条件变量，用于线程同步
        std::queue<Task> tasks_;                     // 任务队列
        std::queue<ThreadID> finishedThreadIDs_;     // 已完成（待清理）的线程ID队列
        std::unordered_map<ThreadID, Thread> threads_; // 存储线程ID和线程对象的map
    };

    constexpr size_t ThreadPool::WAIT_SECONDS;

} // namespace dpool

#endif /* THREADPOOL_H */