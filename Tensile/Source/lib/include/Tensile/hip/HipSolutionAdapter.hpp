/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#pragma once

#include <Tensile/AMDGPU.hpp>
#include <Tensile/Tensile.hpp>
#include <hip/hip_runtime.h>
#include <unordered_set>

#include <mutex>

namespace Tensile
{
    namespace hip
    {
        class SolutionAdapter : public Tensile::SolutionAdapter
        {
        public:
            SolutionAdapter();
            SolutionAdapter(bool debug);
            SolutionAdapter(bool debug, std::string const& name);
            ~SolutionAdapter();

            virtual std::string name() const
            {
                return m_name;
            }

            hipError_t loadCodeObjectFile(std::string const& path);

            hipError_t initializeLazyLoading(std::string architecture, std::string codeObjectDir);

            hipError_t loadCodeObject(const void* image);

            hipError_t loadCodeObjectBytes(std::vector<uint8_t> const& bytes);

            void loadEmbeddedCodeObjects();
            void loadEmbeddedCodeObjects(std::string const& key);

            hipError_t launchKernel(KernelInvocation const& kernel);
            hipError_t launchKernel(KernelInvocation const& kernel,
                                    hipStream_t             stream,
                                    hipEvent_t              startEvent,
                                    hipEvent_t              stopEvent,
                                    KernelGraphInvocation* kernel_graph_invocation = nullptr);
            
            hipError_t launchKernels(std::vector<KernelInvocation> const& kernels);

            hipError_t launchKernels(std::vector<KernelInvocation> const& kernels,
                                     hipStream_t                          stream,
                                     hipEvent_t                           startEvent,
                                     hipEvent_t                           stopEvent,
                                     KernelGraphInvocation* kernel_graph_invocation = nullptr);

            hipError_t launchKernels(std::vector<KernelInvocation> const& kernels,
                                     hipStream_t                          stream,
                                     std::vector<hipEvent_t> const&       startEvents,
                                     std::vector<hipEvent_t> const&       stopEvents,
                                     KernelGraphInvocation* kernel_graph_invocation = nullptr);
            

            hipError_t initKernel(std::string const& name);

        private:
            hipError_t getKernel(hipFunction_t& rv, std::string const& name);

            std::mutex m_access;

            std::vector<hipModule_t>                       m_modules;
            std::vector<std::unique_ptr<char[]>>           m_moduleBuffers;
            std::unordered_map<std::string, hipFunction_t> m_kernels;
            bool                                           m_debug           = false;
            bool                                           m_debugSkipLaunch = false;
            std::string                                    m_name            = "HipSolutionAdapter";
            std::string                                    m_codeObjectDirectory;

            std::vector<std::string>        m_loadedModuleNames;
            std::unordered_set<std::string> m_loadedCOFiles;

            friend std::ostream& operator<<(std::ostream& stream, SolutionAdapter const& adapter);
        };
        
        inline void load_scalar(void* KernelGraphInfo , const std::string& name) 
        {
            auto graphInfo = (KernelGraphInvocation*)KernelGraphInfo;
            void* arg = nullptr;
            DataType argType = DataType::None; 

            if(name == "alpha")
            {   
                arg = graphInfo->alpha;
                argType = graphInfo->alphaType;
            }
            else if (name == "beta")
            {
                arg = graphInfo->beta;
                argType = graphInfo->betaType;
            }
             
            if(argType == DataType::Float || argType == DataType::BFloat16 )
            {
                auto value = *static_cast<float*>(arg);
                graphInfo->kArgs->updateValue<float>(name, value);
            }
            else if(argType == DataType::Double)
            {
                auto value = *static_cast<double*>(arg);
                graphInfo->kArgs->updateValue<double>(name, value);
            }
            else if(argType == DataType::ComplexFloat)
            {
                auto value = *static_cast<std::complex<float>*>(arg);
                graphInfo->kArgs->updateValue<std::complex<float>>(name, value);
            }
            else if(argType == DataType::ComplexDouble)
            {
                auto value = *static_cast<std::complex<double>*>(arg);
                graphInfo->kArgs->updateValue<std::complex<double>>(name, value);
            }
            else if(argType == DataType::Int32)
            {
                auto value = *static_cast<int32_t*>(arg);
                graphInfo->kArgs->updateValue<float>(name, value);           
            }
            else if(argType == DataType::Half)
            {
                auto value = *static_cast<float*>(arg);
                graphInfo->kArgs->updateValue<float>(name, value);
                if(!graphInfo->isSourceKernel)
                {
                    std::string name_2 = name + "_2";
                    graphInfo->kArgs->updateValue<float>(name_2, value);
                }
            }
            else 
            {
                throw std::runtime_error( " Type mismatch for Argument : "  + name );            
            }
        }
    
        inline void updateKernelArgsFuncCB(void* KernelGraphInfo)
        {
            auto graphInfo = (KernelGraphInvocation*)KernelGraphInfo;
            load_scalar(KernelGraphInfo, "alpha");
            load_scalar(KernelGraphInfo, "beta");
           // graphInfo->kArgsSize = graphInfo->kArgs->size();
        }

        inline void deleteKernelArgsObjFuncCB(void* kernel_graph_info)
        {
            std::cout <<" RK: Delete Kernel Arguments ... " << std::endl;
            auto graphInfo = (KernelGraphInvocation*)kernel_graph_info;
           // std::cout <<" RK: arg size : " << graphInfo->kArgsSize << std::endl;
            delete graphInfo->kArgs;
        }

        std::ostream& operator<<(std::ostream& stream, SolutionAdapter const& adapter);
        std::ostream& operator<<(std::ostream& stream, std::shared_ptr<SolutionAdapter> const& ptr);
    } // namespace hip
} // namespace Tensile