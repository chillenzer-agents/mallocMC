/*
  mallocMC: Memory Allocator for Many Core Architectures.
  https://www.hzdr.de/crp

  Copyright 2014-2024 Institute of Radiation Physics,
                 Helmholtz-Zentrum Dresden - Rossendorf

  Author(s):  Carlchristian Eckert - c.eckert ( at ) hzdr.de
              Julian Lenz - j.lenz ( at ) hzdr.de

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

#pragma once

#include "creationPolicies/FlatterScatter.hpp"
#include "creationPolicies/Scatter.hpp"
#include "distributionPolicies/XMallocSIMD.hpp"

#include <cstdint>

namespace mallocMC
{
    /** The default PolicyCheckers (do always succeed)
     */
    template<typename Policy1>
    class PolicyCheck1
    {
    };

    template<typename Policy1, typename Policy2>
    class PolicyCheck2
    {
    };

    template<typename Policy1, typename Policy2, typename Policy3>
    class PolicyCheck3
    {
    };

    template<typename Policy1, typename Policy2, typename Policy3, typename Policy4>
    class PolicyCheck4
    {
    };

    template<typename Policy1, typename Policy2, typename Policy3, typename Policy4, typename Policy5>
    class PolicyCheck5
    {
    };

    /** Enforces constraints on policies or combinations of polices
     *
     * Uses template specialization of PolicyChecker
     */
    template<
        typename T_CreationPolicy,
        typename T_DistributionPolicy,
        typename T_OOMPolicy,
        typename T_GetHeapPolicy,
        typename T_AlignmentPolicy>

    class PolicyConstraints : PolicyCheck2<T_CreationPolicy, T_DistributionPolicy>
    {
    };

    /** Scatter and XMallocSIMD need the same pagesize!
     *
     * This constraint ensures that if the CreationPolicy "Scatter" and the
     * DistributionPolicy "XMallocSIMD" are selected, they are configured to use
     * the same value for their "pagesize"-parameter. The comparison uses the
     * effective `_pagesize` of both policies, so a command-line override such as
     * -DMALLOCMC_CP_SCATTER_PAGESIZE or -DMALLOCMC_DP_XMALLOCSIMD_PAGESIZE is
     * reflected on either side.
     */
    template<typename x, typename y, typename z>
    class PolicyCheck2<typename CreationPolicies::Scatter<x, y>, typename DistributionPolicies::XMallocSIMD<z>>
    {
        static_assert(
            CreationPolicies::ScatterImpl<x, y, void>::_pagesize == DistributionPolicies::XMallocSIMD<z>::_pagesize,
            "Pagesize must be the same when combining Scatter and XMallocSIMD");
    };

    /** FlatterScatter and XMallocSIMD need the same pagesize!
     *
     * This constraint ensures that if the CreationPolicy "FlatterScatter" and the DistributionPolicy "XMallocSIMD"
     * are selected, they are configured to use the same value for their "pagesize"-parameter. XMallocSIMD decides
     * which requests to coalesce and lays out the sub-pointers based on its own page size, so it must match the page
     * size the FlatterScatter heap is actually using. The comparison uses XMallocSIMD's effective `_pagesize`, which
     * also reflects a command-line override such as -DMALLOCMC_DP_XMALLOCSIMD_PAGESIZE.
     */
    template<typename T_HeapConfig, typename T_HashConfig, typename T_AlignmentPolicy, typename T_XMallocSIMDConfig>
    class PolicyCheck2<
        CreationPolicies::FlatterScatter<T_HeapConfig, T_HashConfig, T_AlignmentPolicy>,
        DistributionPolicies::XMallocSIMD<T_XMallocSIMDConfig>>
    {
        static_assert(
            static_cast<std::uint32_t>(T_HeapConfig::pagesize)
                == DistributionPolicies::XMallocSIMD<T_XMallocSIMDConfig>::_pagesize,
            "Pagesize must be the same when combining FlatterScatter and XMallocSIMD");
    };

} // namespace mallocMC
