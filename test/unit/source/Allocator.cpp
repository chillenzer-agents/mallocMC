/*
  mallocMC: Memory Allocator for Many Core Architectures.

  Copyright 2025 Helmholtz-Zentrum Dresden - Rossendorf

  Author(s):  Julian Johannes Lenz, Rene Widera

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in
  all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
  THE SOFTWARE.
*/

#include "mallocMC/allocator.hpp"

#include "mallocMC/alignmentPolicies/Shrink.hpp"
#include "mallocMC/creationPolicies/FlatterScatter.hpp"
#include "mallocMC/creationPolicies/Scatter.hpp"
#include "mallocMC/distributionPolicies/Noop.hpp"
#include "mallocMC/distributionPolicies/XMallocSIMD.hpp"
#include "mallocMC/oOMPolicies/ReturnNull.hpp"
#include "mallocMC/reservePoolPolicies/AlpakaBuf.hpp"

#include <alpaka/acc/AccCpuSerial.hpp>
#include <alpaka/example/ExampleDefaultAcc.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <mallocMC/mallocMC.hpp>

#include <algorithm>
#include <cstdint>
using Dim = alpaka::DimInt<1>;
using Idx = std::size_t;

// Define the device accelerator
using Acc = alpaka::ExampleDefaultAcc<Dim, Idx>;

TEST_CASE("Allocator")
{
    SECTION("can be initialised with 0 memory.")
    {
        auto const platform = alpaka::Platform<Acc>{};
        auto const dev = alpaka::getDevByIdx(platform, 0);
        auto queue = alpaka::Queue<Acc, alpaka::Blocking>{dev};

        mallocMC::Allocator<
            alpaka::AccToTag<Acc>,
            mallocMC::CreationPolicies::FlatterScatter<>,
            mallocMC::DistributionPolicies::Noop,
            mallocMC::OOMPolicies::ReturnNull,
            mallocMC::ReservePoolPolicies::AlpakaBuf<Acc>,
            mallocMC::AlignmentPolicies::Shrink<>>
            allocator{dev, queue, 0};
    }
}

// XMallocSIMD sizes its shared-memory arrays with the warp size and falls back to a single slot when alpaka has no
// compile-time warp size (GPU host compilation and SYCL); CPU accelerators report one. The test still uses
// AccCpuSerial because coalescing is device-only and this suite is CPU-only.
using XMallocAcc = alpaka::AccCpuSerial<Dim, Idx>;

// Both creation policies support XMallocSIMD: both treat a zero-byte request as no allocation and both release
// coalesced sub-chunks only when their last participant leaves. The creation and distribution policy have to agree on
// the page size (see mallocMC_constraints.hpp).
struct XMallocScatterConfig
{
    static constexpr std::uint32_t pagesize = 4096U;
    static constexpr std::uint32_t accessblocksize = 1024U * 1024U;
    static constexpr std::uint32_t regionsize = 16U;
    static constexpr std::uint32_t wastefactor = 2U;
    static constexpr bool resetfreedpages = false;
};

using XMallocScatter = mallocMC::CreationPolicies::Scatter<XMallocScatterConfig>;

// FlatterScatter has to use the same page size as XMallocSIMD (4096 bytes by default). An access block size of 1 MiB
// makes the 4 MiB heap below hold four full access blocks.
using XMallocFlatterScatterConfig
    = mallocMC::CreationPolicies::FlatterScatterAlloc::DefaultHeapConfig<1024U * 1024U, 4096U, 2U>;
using XMallocFlatterScatter = mallocMC::CreationPolicies::FlatterScatter<XMallocFlatterScatterConfig>;

// The distinct names are what makes the four runs distinguishable in Catch2's output.
template<typename T_CreationPolicy, typename T_DistributionPolicy>
struct AllocatorPolicies
{
    using CreationPolicy = T_CreationPolicy;
    using DistributionPolicy = T_DistributionPolicy;
};

struct ScatterNoop : AllocatorPolicies<XMallocScatter, mallocMC::DistributionPolicies::Noop>
{
};

struct ScatterXMallocSIMD : AllocatorPolicies<XMallocScatter, mallocMC::DistributionPolicies::XMallocSIMD<>>
{
};

struct FlatterScatterNoop : AllocatorPolicies<XMallocFlatterScatter, mallocMC::DistributionPolicies::Noop>
{
};

struct FlatterScatterXMallocSIMD
    : AllocatorPolicies<XMallocFlatterScatter, mallocMC::DistributionPolicies::XMallocSIMD<>>
{
};

struct AllocateWriteCheckAndFree
{
    template<typename TAcc, typename TAllocatorHandle>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, TAllocatorHandle allocHandle, std::uint32_t* results) const -> void
    {
        constexpr std::uint32_t numBytes = 32U;
        // XMallocSIMD only coalesces requests smaller than pagesize / warpSize. Bounding by pagesize / 64 keeps
        // 32 bytes coalescible for both a 32-wide and a 64-wide warp. This CPU-only test still cannot coalesce,
        // because coalescing is a device-only code path.
        static_assert(numBytes < mallocMC::DistributionPolicies::XMallocSIMD<>::Properties::pagesize / 64U);

        auto const idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
        void* const pointer = allocHandle.malloc(acc, numBytes);
        results[idx] = pointer != nullptr ? 1U : 0U;
        if(pointer != nullptr)
        {
            auto* const data = static_cast<std::uint32_t*>(pointer);
            std::fill(data, data + numBytes / sizeof(std::uint32_t), static_cast<std::uint32_t>(idx + 1U));
        }

        // The barrier makes writes to overlapping sub-chunks (i.e. broken coalescing) visible before any thread reads
        // back its own data.
        alpaka::syncBlockThreads(acc);
        if(pointer != nullptr)
        {
            auto const* const data = static_cast<std::uint32_t const*>(pointer);
            bool const intact = std::all_of(
                data,
                data + numBytes / sizeof(std::uint32_t),
                [idx](std::uint32_t const value) { return value == idx + 1U; });
            results[idx] = intact ? results[idx] : 0U;
        }

        allocHandle.free(acc, pointer);
    }
};

TEMPLATE_TEST_CASE(
    "Allocator allocates and frees",
    "",
    ScatterNoop,
    ScatterXMallocSIMD,
    FlatterScatterNoop,
    FlatterScatterXMallocSIMD)
{
    using CreationPolicy = typename TestType::CreationPolicy;
    using DistributionPolicy = typename TestType::DistributionPolicy;
    using Allocator = mallocMC::Allocator<
        alpaka::AccToTag<XMallocAcc>,
        CreationPolicy,
        DistributionPolicy,
        mallocMC::OOMPolicies::ReturnNull,
        mallocMC::ReservePoolPolicies::AlpakaBuf<XMallocAcc>,
        mallocMC::AlignmentPolicies::Shrink<>>;

    auto const platform = alpaka::Platform<XMallocAcc>{};
    auto const dev = alpaka::getDevByIdx(platform, 0);
    auto queue = alpaka::Queue<XMallocAcc, alpaka::Blocking>{dev};

    // Both creation policies need a heap that holds at least one full access block and is a multiple of the access
    // block size. With 1 MiB access blocks, this 4 MiB heap holds four of them.
    constexpr std::size_t heapSize = 4U * 1024U * 1024U;
    Allocator allocator{dev, queue, heapSize};

    // Use up to a warp so XMallocSIMD has something to coalesce. CPU accelerators may only allow a single thread.
    auto const devProps = alpaka::getAccDevProps<XMallocAcc>(dev);
    auto const numThreads = std::min<std::size_t>(32U, static_cast<std::size_t>(devProps.m_blockThreadCountMax));

    auto results = alpaka::allocBuf<std::uint32_t, Idx>(dev, Idx{numThreads});
    auto const workDiv = alpaka::WorkDivMembers<Dim, Idx>{Idx{1}, Idx{numThreads}, Idx{1}};
    alpaka::exec<XMallocAcc>(
        queue,
        workDiv,
        AllocateWriteCheckAndFree{},
        allocator.getAllocatorHandle(),
        alpaka::getPtrNative(results));
    alpaka::wait(queue);

    auto const platformHost = alpaka::Platform<alpaka::DevCpu>{};
    auto const devHost = alpaka::getDevByIdx(platformHost, 0);
    auto resultsHost = alpaka::allocBuf<std::uint32_t, Idx>(devHost, Idx{numThreads});
    alpaka::memcpy(queue, resultsHost, results);
    alpaka::wait(queue);

    auto const* const resultsPtr = alpaka::getPtrNative(resultsHost);
    CHECK(std::all_of(resultsPtr, resultsPtr + numThreads, [](std::uint32_t const result) { return result == 1U; }));
}
