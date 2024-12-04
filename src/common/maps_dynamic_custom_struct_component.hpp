/////////////////////////////////////////////////////////////////////////////////
//
//   Copyright 2018-2024 Intempora S.A.S.
//
//   Licensed under the Apache License, Version 2.0 (the "License");
//   you may not use this file except in compliance with the License.
//   You may obtain a copy of the License at
//
//       http://www.apache.org/licenses/LICENSE-2.0
//
//   Unless required by applicable law or agreed to in writing, software
//   distributed under the License is distributed on an "AS IS" BASIS,
//   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//   See the License for the specific language governing permissions and
//   limitations under the License.
//
/////////////////////////////////////////////////////////////////////////////////

#pragma once

// A recent C++11-compliant compiler is required in order to use this component parent
#if defined(_MSC_VER)
    #if _MSC_VER < 1800
        #error "In order to use this component parent, please use MSVC++ 2013 or newer"
    #endif
#elif defined(__GNUC__) || defined(__clang__)
    #if __cplusplus < 201103L
        #error "In order to use this component parent, please enable C++11 or a more recent version of the C++ standard by using e.g. the -std=c++11 compiler flag"
    #endif
#else
    #error "Compiler not supported"
#endif

// std
#include <algorithm>    // find
#include <cstddef>      // size_t
#include <exception>    // exception
#include <functional>   // function
#include <memory>       // addressof
#include <new>          // bad_alloc
#include <sstream>      // ostringstream
#include <string>       // string
#include <type_traits>  // is_*
#include <utility>      // forward, tuple_element
#include <vector>       // vector

// rtmaps
#include <maps.hpp>

// version
#define MAPS_DynamicCustomStructComponent_Version_MAJOR 2
#define MAPS_DynamicCustomStructComponent_Version_MINOR 1
#define MAPS_DynamicCustomStructComponent_Version_PATCH 0

/// \brief A component parent that abstracts away memory management when using Dynamic Custom Structs in component outputs
///
/// In order to use this component parent, you should:
/// \li Make sure that you are using a recent C++11-compliant compiler. \n
///     On Windows, you must use Microsoft Visual C++ >= 2013 \n
///     On Linux, you must use GCC >= 4.8 or Clang >= 3.4 and enable the -std=c++11 (or -std=c++14, etc.) flag
/// \li Inherit from this class
/// \li In your component, define and/or create the outputs that use dynamic structs
///     (between MAPS_BEGIN_OUTPUTS_DEFINITION and MAPS_END_OUTPUTS_DEFINITION
///     and/or in your component's Dynamic() method)
/// \li In the Birth() method of your component class,
///     call MAPS_DynamicCustomStructComponent::AllocateDynamicOutputBuffers()
///     in order to allocate the output buffers
/// \li In the FreeBuffers() method of your component class,
///     call MAPS_DynamicCustomStructComponent::FreeBuffers()
///     in order to free the output buffers
///
/// For instance, if your component has (among other outputs) 3 outputs that use dynamic custom structs:
/// \li "o_dynamic_struct_1": Uses MyStruct_X
/// \li "o_dynamic_struct_2": Uses MyStruct_Y
/// \li "o_dynamic_struct_3": Uses MyStruct_X (for the purposes of this example, this output
///                                            uses the same type of struct as "o_dynamic_struct_1")
///
/// Then, you must inherit from MAPS_DynamicCustomStructComponent as follows:
/// \code
///     class MyComponent: public MAPS_DynamicCustomStructComponent
/// \endcode
///
/// Then, you must allocate the output buffers in Birth() as follows:
/// \code
///     // Calls the default constructor of the custom structs
///     // as many times as necessary to fill each output's FIFO
///     MAPS_DynamicCustomStructComponent::AllocateDynamicOutputBuffers(
///         DynamicOutput<MyStruct_X>("o_dynamic_struct_1"),
///         DynamicOutput<MyStruct_Y>("o_dynamic_struct_2"),
///         DynamicOutput<MyStruct_X>("o_dynamic_struct_3")
///     );
/// \endcode
///
/// Finally, you must free the allocated buffers in FreeBuffers() as follows;
/// \code
///     // Calls the destructor of each custom struct that was previously allocated
///     MAPS_DynamicCustomStructComponent::FreeBuffers();
/// \endcode
///
class MAPS_DynamicCustomStructComponent: public MAPSComponent
{
public:
    /// \brief Contains information that is relevant for managing outputs that use dynamic custom structs
    ///
    /// This class is intended for use with the AllocateDynamicOutputBuffers() method.
    ///
    /// In order to instantiate an object of this class, use the DynamicOutput() methods.
    struct OutputWrapper
    {
        MAPSOutput* const                output;
        const std::function<void*()>     ctor;
        const std::function<void(void*)> dtor;
        const std::string                typeName;
        const size_t                     elementByteSize;

        OutputWrapper()                                      = delete;
        ~OutputWrapper()                                     = default;
        OutputWrapper& operator=(const OutputWrapper& other) = delete;
        OutputWrapper& operator=(OutputWrapper&&)            = delete;

        OutputWrapper(const OutputWrapper& other)
        : output(other.output)
        , ctor(other.ctor)
        , dtor(other.dtor)
        , typeName(other.typeName)
        , elementByteSize(other.elementByteSize)
        {}

        OutputWrapper(OutputWrapper&& other)
        : output(other.output)
        , ctor(other.ctor)
        , dtor(other.dtor)
        , typeName(other.typeName)
        , elementByteSize(other.elementByteSize)
        {}

        OutputWrapper(
            MAPSOutput* output_,
            std::function<void*()>     ctor_,
            std::function<void(void*)> dtor_,
            std::string                typeName_,
            const size_t               elementByteSize_)
        : output(output_)
        , ctor(std::move(ctor_))
        , dtor(std::move(dtor_))
        , typeName(std::move(typeName_))
        , elementByteSize(elementByteSize_)
        {}

        std::string outputName() const { return std::string((const char*) output->Name().Tail('.')); }
        bool operator==(const OutputWrapper& other) const { return output == other.output; }
    };

    /// \brief Creates an OutputWrapper object for use in AllocateDynamicOutputBuffers()
    ///
    /// \tparam T            Output data type -- i.e. the dynamic custom struct type
    /// \tparam TConstruct_T A callable object (e.g. lambda, function pointer, functor, etc.)
    ///                      that constructs an object of type T
    /// \tparam TDestroy_T   A callable object (e.g. lambda, function pointer, functor, etc.)
    ///                      that frees the memory allocated for an object of type T
    ///
    /// \param[in] output     A reference to the output that uses T
    /// \param[in] constructT A callable object that constructs an object of type T.
    ///                       This callable must take no arguments and return a pointer to T
    /// \param[in] destroyT   A callable object that destroys an object of type T.
    ///                       This callable must take a pointer to T and return void
    template <typename T, typename TConstruct_T, typename TDestroy_T>
    static OutputWrapper DynamicOutput(MAPSOutput& output, TConstruct_T constructT, TDestroy_T destroyT)
    {
        static_assert(
            std::is_same<decltype(constructT()), T*>::value,
            "\n    When calling DynamicOutput<T>(MAPSOutput& output, TConstruct_T constructT, TDestroy_T destroyT): "
            "\n    * Wrong signature for constructT."
            "\n    * constructT must take no arguments and return [ T* ]. "
            "\n    * The required signature for constructT is [ T* constructT() ]"
        );

        using destroyT_info       = DtorCallableInfo<typename std::decay<TDestroy_T>::type>;
        using destroyT_returnType = typename destroyT_info::ret_type;
        using destroyT_argType    = typename destroyT_info::arg_type;
        static_assert(
            std::is_void<destroyT_returnType>::value
            && std::is_same<destroyT_argType, T*>::value,
            "\n    When calling DynamicOutput<T>(MAPSOutput& output, TConstruct_T constructT, TDestroy_T destroyT): "
            "\n    * Wrong signature for destroyT."
            "\n    * destroyT must take a single argument of type [ T* ] and return [ void ]. "
            "\n    * The required signature for destroyT is [ void destroyT(T*) ]"
        );

        return {
            std::addressof(output),
            [constructT] { return static_cast<void*>(constructT());  },
            [destroyT] (void* p) { destroyT(static_cast<T*>(p)); },
            typeName<T>(),
            sizeof(T)
        };
    }
    /// \brief Creates an OutputWrapper object for use in AllocateDynamicOutputBuffers()
    ///
    /// \tparam T            Output data type -- i.e. the dynamic custom struct type
    /// \tparam TConstruct_T A callable object (e.g. lambda, function pointer, functor, etc.)
    ///                      that constructs an object of type T
    ///
    /// \param[in] output     A reference to the output that uses T
    /// \param[in] constructT A callable object that constructs an object of type T.
    ///                       This callable must take no arguments and return a pointer to T
    template <typename T, typename TConstruct_T>
    static OutputWrapper DynamicOutput(MAPSOutput& output, TConstruct_T constructT)
    {
        static_assert(
            std::is_same<decltype(constructT()), T*>::value,
            "\n    When calling DynamicOutput<T>(MAPSOutput& output, TConstruct_T constructT):"
            "\n    * Wrong signature for constructT."
            "\n    * constructT must take no arguments and return [ T* ]."
            "\n    * The required signature for constructT is [ T* constructT() ]"
        );

        return DynamicOutput<T>(output, constructT, [] (T* p) { delete p; });
    }
    /// \brief Creates an OutputWrapper object for use in AllocateDynamicOutputBuffers()
    ///
    /// \tparam T Output data type -- i.e. the dynamic custom struct type
    ///
    /// \param[in] output A reference to the output that uses T
    template <typename T>
    static OutputWrapper DynamicOutput(MAPSOutput& output)
    {
        return DynamicOutput<T>(output, [] { return new T(); });
    }

protected:
    MAPS_DynamicCustomStructComponent(const char* componentName, MAPSComponentDefinition& md)
    : MAPSComponent(componentName, md)
    , m_outputWrappers()
    {
    }

private:
    std::vector<OutputWrapper> m_outputWrappers;

protected:

    /// \brief Allocates output buffers for the given outputs
    ///
    /// \note Must be called in your component's Birth()
    ///
    /// \tparam TOutputWrapper  OutputWrapper
    /// \tparam TOutputWrappers More OutputWrapper's
    ///
    /// \param[in] outputWrapper  An OutputWrapper object. Use the DynamicOutput<T>(...) method to create one
    /// \param[in] outputWrappers More OutputWrapper objects
    template <typename TOutputWrapper, typename... TOutputWrappers>
    void AllocateDynamicOutputBuffers(TOutputWrapper&& outputWrapper, TOutputWrappers&&... outputWrappers)
    {
        static_assert(AreOutputWrappers<TOutputWrapper, TOutputWrappers...>::value,
            "The arguments of AllocateDynamicOutputBuffers() must all be of type OutputWrapper. "
            "Use DynamicOutput<T>(...) when calling AllocateDynamicOutputBuffers()");

        FreeDynamicOutputs();

        addOutputWrappers(std::forward<TOutputWrapper>(outputWrapper), std::forward<TOutputWrappers>(outputWrappers)...);

        if (!allUniqueOutputs())
        {
            m_outputWrappers.clear();
            Error("AllocateDynamicOutputBuffers: Outputs are not unique. Refer to the previous error messages to know which outputs have been added more than once");
        }

        allocateDynamicOutputs();
    }

    /// \brief Allocates output buffers for the given outputs
    ///
    /// \note Must be called in your component's Birth()
    ///
    /// \tparam TOutputWrapperInputIteratorBegin Iterator pointing to an output wrapper
    ///
    /// \param[in] first, last The range of output wrappers to use
    template <typename TOutputWrapperInputIterator>
    typename std::enable_if<
        !std::is_same<OutputWrapper, typename std::remove_reference<TOutputWrapperInputIterator>::type>::value,
        void
    >::type AllocateDynamicOutputBuffers(TOutputWrapperInputIterator first, TOutputWrapperInputIterator last)
    {
        static_assert(AreOutputWrappers<decltype(*first), decltype(*last)>::value,
            "The [first, last) range that is passed to AllocateDynamicOutputBuffers() must contain elements of type OutputWrapper."
        );

        FreeDynamicOutputs();

        addOutputWrappers(first, last);

        if (!allUniqueOutputs())
        {
            m_outputWrappers.clear();
            Error("AllocateDynamicOutputBuffers: Outputs are not unique. Refer to the previous error messages to know which outputs have been added more than once");
        }

        allocateDynamicOutputs();
    }

    /// \brief Frees the memory that has been allocated by AllocateDynamicOutputBuffers
    ///
    /// \note Must be called in your component's FreeBuffers()
    void FreeBuffers() override
    {
        FreeDynamicOutputs();
        MAPSComponent::FreeBuffers();
    }

    // for internal use only ///////////////////////////////////////////////////////////////////////

private:

    // buffer allocation ///////////////////////////////////////////////////////////////////////////

    void allocateDynamicOutputs()
    {
        for (auto& outputWrapper : m_outputWrappers)
        {
            allocateDynamicOutput(outputWrapper);
        }
    }

    void allocateDynamicOutput(OutputWrapper& outputWrapper)
    {
        {
            std::ostringstream oss;
            oss << "Allocating output [" << outputWrapper.outputName() << "] of type [" << outputWrapper.typeName << "]";
            const std::string infoStr(oss.str());
            ReportInfo(infoStr.c_str());
        }

        forEachFifoElt(outputWrapper, [this] (MAPSIOElt& ioEltOut, OutputWrapper& outputWrapper_, const size_t fifoIdx) {
            allocateDynamicOutputElement(ioEltOut, outputWrapper_, fifoIdx);
        });
    }

    void allocateDynamicOutputElement(MAPSIOElt& ioEltOut, OutputWrapper& outputWrapper, const size_t fifoIdx)
    {
        // convention
        ioEltOut.BufferSize() = static_cast<int>(outputWrapper.elementByteSize);
        ioEltOut.VectorSize() = static_cast<int>(outputWrapper.elementByteSize);

        try
        {
            ioEltOut.Data() = outputWrapper.ctor();
        }
        catch (const std::bad_alloc& ex)
        {
            std::ostringstream oss;
            oss << "Exception [std::bad_alloc] when allocating element [" << fifoIdx << "] of output [" << outputWrapper.outputName() << "]: " << ex.what();
            const std::string errStr(oss.str());
            Error(errStr.c_str());
        }
        catch (const std::exception& ex)
        {
            std::ostringstream oss;
            oss << "Exception [std::exception] when allocating element [" << fifoIdx << "] of output [" << outputWrapper.outputName() << "]: " << ex.what();
            const std::string errStr(oss.str());
            Error(errStr.c_str());
        }
        catch (const char* ex)
        {
            std::ostringstream oss;
            oss << "Exception [const char*] when allocating element [" << fifoIdx << "] of output [" << outputWrapper.outputName() << "]: " << ex;
            const std::string errStr(oss.str());
            Error(errStr.c_str());
        }
        catch (...)
        {
            std::ostringstream oss;
            oss << "Exception [...] when allocating element [" << fifoIdx << "] of output [" << outputWrapper.outputName() << "]";
            const std::string errStr(oss.str());
            Error(errStr.c_str());
        }

        if (ioEltOut.Data() == nullptr)
        {
            std::ostringstream oss;
            oss << "Not enough memory when allocating element [" << fifoIdx << "] of output [" << outputWrapper.outputName() << "]";
            const std::string errStr(oss.str());
            Error(errStr.c_str());
        }
    }

    // buffer freeing //////////////////////////////////////////////////////////////////////////////

    void FreeDynamicOutputs()
    {
        if (!m_outputWrappers.empty())
        {
            freeDynamicOutputs();
            m_outputWrappers.clear();
        }
    }

    void freeDynamicOutputs()
    {
        for (auto& outputWrapper : m_outputWrappers)
        {
            freeDynamicOutput(outputWrapper);
        }
    }

    void freeDynamicOutput(OutputWrapper& outputWrapper)
    {
        {
            std::ostringstream oss;
            oss << "Freeing output [" << outputWrapper.outputName() << "] of type [" << outputWrapper.typeName << "]";
            const std::string infoStr(oss.str());
            ReportInfo(infoStr.c_str());
        }

        forEachFifoElt(outputWrapper, [this] (MAPSIOElt& ioEltOut, OutputWrapper& outputWrapper_, const size_t fifoIdx) {
            freeDynamicOutputElement(ioEltOut, outputWrapper_, fifoIdx);
        });
    }

    void freeDynamicOutputElement(MAPSIOElt& ioEltOut, OutputWrapper& outputWrapper, const size_t /*fifoIdx*/)
    {
        outputWrapper.dtor(ioEltOut.Data());
        ioEltOut.Data() = nullptr;
    }

    // type name extraction ////////////////////////////////////////////////////////////////////////

    #ifndef MAPS_FUNCTION_SIGNATURE
        #define MAPS_DynamicCustomStructComponent_UNDEF_MAPS_FUNCTION_SIGNATURE

        #if defined(_MSC_VER)
            #define MAPS_FUNCTION_SIGNATURE __FUNCSIG__
        #elif defined(__GNUC__) || defined(__clang__)
            #define MAPS_FUNCTION_SIGNATURE __PRETTY_FUNCTION__
        #else
            #warning "MAPS_FUNCTION_SIGNATURE is not supported by your compiler"
            #define MAPS_FUNCTION_SIGNATURE ""
        #endif
    #endif

    #ifdef char
        #define MAPS_DynamicCustomStructComponent_REDEF_CHAR char
        #undef  char
    #endif
    #ifdef double
        #define MAPS_DynamicCustomStructComponent_REDEF_DOUBLE double
        #undef  double
    #endif

    template <typename T>
    static std::string TypeNamE()
    {
        return std::string(MAPS_FUNCTION_SIGNATURE);
    }

    static size_t typeNameStartIdx()
    {
        const std::string s1 = TypeNamE<char>();
        const std::string s2 = TypeNamE<double>();

        size_t idx = 0;
        while (s1[idx] == s2[idx])
        {
            ++idx;
        }

        return idx;
    }

    static size_t typeNameSuffixLen()
    {
        const std::string s1 = TypeNamE<char>();
        const std::string s2 = TypeNamE<double>();

        size_t idx1 = s1.length() - 1;
        size_t idx2 = s2.length() - 1;
        while (s1[idx1] == s2[idx2])
        {
            --idx1;
            --idx2;
        }

        const size_t suffLen = s1.length() - 1 - idx1;
        return suffLen;
    }


    static std::string extractTypeName(const std::string& mangledTypeName)
    {
        const size_t startIdx  = typeNameStartIdx();
        const size_t charCount = mangledTypeName.length() - startIdx - typeNameSuffixLen();

        return mangledTypeName.substr(startIdx, charCount);
    }

    template <typename T>
    static std::string typeName()
    {
        const std::string funcSig(MAPS_FUNCTION_SIGNATURE);
        return funcSig.empty() ? std::string("") : extractTypeName(funcSig);
    }

    #ifdef MAPS_DynamicCustomStructComponent_REDEF_CHAR
        #define char MAPS_DynamicCustomStructComponent_REDEF_CHAR
        #undef  MAPS_DynamicCustomStructComponent_REDEF_CHAR
    #endif
    #ifdef MAPS_DynamicCustomStructComponent_REDEF_DOUBLE
        #define double MAPS_DynamicCustomStructComponent_REDEF_DOUBLE
        #undef  MAPS_DynamicCustomStructComponent_REDEF_DOUBLE
    #endif

    #ifdef MAPS_DynamicCustomStructComponent_UNDEF_MAPS_FUNCTION_SIGNATURE
        #undef MAPS_DynamicCustomStructComponent_UNDEF_MAPS_FUNCTION_SIGNATURE
        #undef MAPS_FUNCTION_SIGNATURE
    #endif

    // callable signature validation ///////////////////////////////////////////////////////////////

    template <typename TCallable>
    struct DtorCallableInfo : public DtorCallableInfo<decltype(&TCallable::operator())>
    {
    };

    template <typename TClass, typename TReturn, typename TArg>
    struct DtorCallableInfo<TReturn(TClass::*)(TArg) const>
    {
        using ret_type = TReturn;
        using arg_type = TArg;
    };

    template <typename TReturn, typename TArg>
    struct DtorCallableInfo<TReturn(*)(TArg)>
    {
        using ret_type = TReturn;
        using arg_type = TArg;
    };

    // misc ////////////////////////////////////////////////////////////////////////////////////////

    template <bool...>
    struct BoolPack
    {};

    template <bool... bs>
    using AreAllTrue = std::is_same<BoolPack<true, bs...>, BoolPack<bs..., true> >;

    template <typename... Ts>
    using AreOutputWrappers = AreAllTrue<
        std::is_same<
            OutputWrapper,
            typename std::remove_cv<typename std::remove_reference<Ts>::type>::type
        >::value...>;

    template <typename... TOutputWrapper>
    void addOutputWrappers(TOutputWrapper&&... outputWrappers)
    {
        const int arr[] = {0, (m_outputWrappers.emplace_back(std::forward<TOutputWrapper>(outputWrappers)), 0)...};
        (void) arr;
    }

    template <typename TOutputWrapperInputIterator>
    typename std::enable_if<
        !std::is_same<OutputWrapper, typename std::remove_reference<TOutputWrapperInputIterator>::type>::value,
        void
    >::type addOutputWrappers(TOutputWrapperInputIterator first, TOutputWrapperInputIterator last)
    {
        std::vector<OutputWrapper>(first, last).swap(m_outputWrappers);
    }

    bool allUniqueOutputs()
    {
        bool allUnique = true;
        for (size_t idx = 1; idx < m_outputWrappers.size(); ++idx)
        {
            const auto& ow           = m_outputWrappers[idx];
            const auto  subVecBegin  = m_outputWrappers.cbegin();
            const auto  subVecEnd    = subVecBegin + idx;
            const bool  alreadyAdded = std::find(subVecBegin, subVecEnd, ow) != subVecEnd;

            if (alreadyAdded)
            {
                allUnique = false;

                std::ostringstream oss;
                oss << "AllocateDynamicOutputBuffers: Output [" << ow.outputName() << "] has been added more than once [argument position = " << idx << "] (argument positions start at 0)";
                const std::string errMsg(oss.str());
                ReportError(errMsg.c_str());
            }
        }
        return allUnique;
    }

    template <typename Op>
    void forEachFifoElt(OutputWrapper& outputWrapper, const Op op)
    {
        MAPSIOMonitor&   outputMonitor = outputWrapper.output->Monitor();
        MAPSFastIOHandle fifoIterator  = outputMonitor.InitBegin();
        size_t           fifoIdx       = 0;

        while (fifoIterator)
        {
            MAPSIOElt& ioEltOut = outputMonitor[fifoIterator];

            op(ioEltOut, outputWrapper, fifoIdx);

            outputMonitor.InitNext(fifoIterator);
            ++fifoIdx;
        }
    }
};
