#pragma once

#include <CL/cl.hpp>
#include <memory>
#include <map>
#include <mutex>
#include "clTaskFormat.h"



bool cmpf(const float A,const float B,const float epsilon = 0.005f)
{
    return (fabs(A - B) < epsilon);
}

namespace clArgInfo
{

	size_t cnt_items(const std::string& type)
	{
		if (type.find("2") != std::string::npos)return 2;
		else if (type.find("3") != std::string::npos)return 3;
		else if (type.find("4") != std::string::npos)return 4;
		else if (type.find("8") != std::string::npos)return 8;
		else if (type.find("16") != std::string::npos)return 16;
		return 1;
	}

	template<typename T1, typename T2>
	void cast_to_bytes(std::vector<int8_t>& bytes,
		const size_t items,
		const double init_value)
	{
		const T1 v = { { static_cast<T2>(init_value),
			static_cast<T2>(init_value),
			static_cast<T2>(init_value),
			static_cast<T2>(init_value),
			static_cast<T2>(init_value),
			static_cast<T2>(init_value),
			static_cast<T2>(init_value),
			static_cast<T2>(init_value),
			static_cast<T2>(init_value),
			static_cast<T2>(init_value),
			static_cast<T2>(init_value),
			static_cast<T2>(init_value),
			static_cast<T2>(init_value),
			static_cast<T2>(init_value),
			static_cast<T2>(init_value),
			static_cast<T2>(init_value) } };

		bytes.resize(sizeof(T2)*items, 0);
		memcpy(bytes.data(), &v, bytes.size());
	}

	std::vector<int8_t>
		get_value(const std::string &type, const double init_value)
	{
		std::vector<int8_t> bytes;
		if (type.empty())return bytes;

		const auto items = cnt_items(type);

		if (type.find("char") != std::string::npos)
		{
			cast_to_bytes<cl_char16, cl_char>(bytes, items, init_value);
		}
		else if (type.find("uchar") != std::string::npos)
		{
			cast_to_bytes<cl_uchar16, cl_uchar>(bytes, items, init_value);
		}
		else if (type.find("short") != std::string::npos)
		{
			cast_to_bytes<cl_short16, cl_short>(bytes, items, init_value);
		}
		else if (type.find("ushort") != std::string::npos)
		{
			cast_to_bytes<cl_ushort16, cl_ushort>(bytes, items, init_value);
		}
		else if (type.find("int") != std::string::npos)
		{
			cast_to_bytes<cl_int16, cl_int>(bytes, items, init_value);
		}
		else if (type.find("uint") != std::string::npos)
		{
			cast_to_bytes<cl_uint16, cl_uint>(bytes, items, init_value);
		}
		else if (type.find("long") != std::string::npos)
		{
			cast_to_bytes<cl_long16, cl_long>(bytes, items, init_value);
		}
		else if (type.find("ulong") != std::string::npos)
		{
			cast_to_bytes<cl_ulong16, cl_ulong>(bytes, items, init_value);
		}
		else if (type.find("half") != std::string::npos)
		{
			//cast_to_bytes<cl_half,cl_long>(bytes,items,init_value);
			cast_to_bytes<cl_uint16, cl_uint>(bytes, items, init_value);
		}
		else if (type.find("float") != std::string::npos)
		{
			cast_to_bytes<cl_float16, cl_float>(bytes, items, init_value);
		}
		else if (type.find("double") != std::string::npos)
		{
			cast_to_bytes<cl_double16, cl_double>(bytes, items, init_value);
		}

		return bytes;
	}

	size_t get_size(const std::string &type)
	{
		if (type.empty())return 0;

		const auto items = cnt_items(type);

		if (type.find("char") != std::string::npos)
		{
			return sizeof(cl_char)*items;
		}
		else if (type.find("uchar") != std::string::npos)
		{
			return sizeof(cl_uchar)*items;
		}
		else if (type.find("short") != std::string::npos)
		{
			return sizeof(cl_short)*items;
		}
		else if (type.find("ushort") != std::string::npos)
		{
			return sizeof(cl_ushort)*items;
		}
		else if (type.find("int") != std::string::npos)
		{
			return sizeof(cl_int)*items;
		}
		else if (type.find("uint") != std::string::npos)
		{
			return sizeof(cl_uint)*items;
		}
		else if (type.find("long") != std::string::npos)
		{
			return sizeof(cl_long)*items;
		}
		else if (type.find("ulong") != std::string::npos)
		{
			return sizeof(cl_ulong)*items;
		}
		else if (type.find("half") != std::string::npos)
		{
			return sizeof(cl_half)*items;
		}
		else if (type.find("float") != std::string::npos)
		{
			return sizeof(cl_float)*items;
		}
		else if (type.find("double") != std::string::npos)
		{
			return sizeof(cl_double)*items;
		}

		return 1;
	}

	bool isFloat(const std::string& type)
	{
		if (type.find("half") != std::string::npos)
			return true;
		else if (type.find("float") != std::string::npos)
			return true;
		else if (type.find("double") != std::string::npos)
			return true;

		return false;
	}

}

namespace coopcl
{
	#define _PROFILE_
	//#define _OPT_OFFLOAD

	static auto check_svm_support = [](int iflag, cl_device_id dev)->std::string
        {       
		if (dev == nullptr)return "";

		cl_device_svm_capabilities caps;

		cl_int err = clGetDeviceInfo(
			dev,
			CL_DEVICE_SVM_CAPABILITIES,
			sizeof(cl_device_svm_capabilities),
			&caps,
			0
		);

		if (err != CL_SUCCESS)return"";
		
		std::string sflag="";
		switch (iflag)
		{
            case CL_DEVICE_SVM_FINE_GRAIN_BUFFER:
			sflag = "CL_DEVICE_SVM_FINE_GRAIN_BUFFER";            
			break;
        case CL_DEVICE_SVM_COARSE_GRAIN_BUFFER:
			sflag = "CL_DEVICE_SVM_COARSE_GRAIN_BUFFER";
			break;
		case CL_DEVICE_SVM_ATOMICS:
			sflag = "CL_DEVICE_SVM_ATOMICS";
			break;
		case CL_DEVICE_SVM_FINE_GRAIN_SYSTEM:
			sflag = "CL_DEVICE_SVM_FINE_GRAIN_SYSTEM";
            break;
		}		

        if (sflag.empty())return sflag;
        return ((caps & iflag) == iflag) ? sflag : "";

        return "";
	};

	static std::string err_msg(const int err)
	{
		switch (err)
		{
		case CL_DEVICE_NOT_FOUND:
			return  "CL_DEVICE_NOT_FOUND";
		case CL_DEVICE_NOT_AVAILABLE:
			return "CL_DEVICE_NOT_AVAILABLE";
		case CL_COMPILER_NOT_AVAILABLE:
			return "CL_COMPILER_NOT_AVAILABLE";
		case CL_MEM_OBJECT_ALLOCATION_FAILURE:
			return  "CL_MEM_OBJECT_ALLOCATION_FAILURE";
		case CL_OUT_OF_RESOURCES:
			return  "CL_OUT_OF_RESOURCES";
		case CL_OUT_OF_HOST_MEMORY:
			return  "CL_OUT_OF_HOST_MEMORY";
		case CL_PROFILING_INFO_NOT_AVAILABLE:
			return  "CL_PROFILING_INFO_NOT_AVAILABLE";
		case CL_MEM_COPY_OVERLAP:
			return  "CL_MEM_COPY_OVERLAP";
		case CL_IMAGE_FORMAT_MISMATCH:
			return  "CL_IMAGE_FORMAT_MISMATCH";
		case CL_IMAGE_FORMAT_NOT_SUPPORTED:
			return  "CL_IMAGE_FORMAT_NOT_SUPPORTED";
		case CL_BUILD_PROGRAM_FAILURE:
			return  "CL_BUILD_PROGRAM_FAILURE";
		case CL_MAP_FAILURE:
			return  "CL_MAP_FAILURE";
		case CL_MISALIGNED_SUB_BUFFER_OFFSET:
			return  "CL_MISALIGNED_SUB_BUFFER_OFFSET";
		case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:
			return  "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
		case CL_COMPILE_PROGRAM_FAILURE:
			return  "CL_COMPILE_PROGRAM_FAILURE";
		case CL_LINKER_NOT_AVAILABLE:
			return  "CL_LINKER_NOT_AVAILABLE";
		case CL_LINK_PROGRAM_FAILURE:
			return  "CL_LINK_PROGRAM_FAILURE";
		case CL_DEVICE_PARTITION_FAILED:
			return  "CL_DEVICE_PARTITION_FAILED";
		case CL_KERNEL_ARG_INFO_NOT_AVAILABLE:
			return  "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";
		case CL_INVALID_VALUE:
			return  "CL_INVALID_VALUE";
		case CL_INVALID_DEVICE_TYPE:
			return  "CL_INVALID_DEVICE_TYPE";
		case CL_INVALID_PLATFORM:
			return  "CL_INVALID_PLATFORM";
		case CL_INVALID_DEVICE:
			return  "CL_INVALID_DEVICE";
		case CL_INVALID_CONTEXT:
			return  "CL_INVALID_CONTEXT";
		case CL_INVALID_QUEUE_PROPERTIES:
			return  "CL_INVALID_QUEUE_PROPERTIES";
		case CL_INVALID_COMMAND_QUEUE:
			return  "CL_INVALID_COMMAND_QUEUE";
		case CL_INVALID_HOST_PTR:
			return  "CL_INVALID_HOST_PTR";
		case CL_INVALID_MEM_OBJECT:
			return  "CL_INVALID_MEM_OBJECT";
		case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
			return  "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
		case CL_INVALID_IMAGE_SIZE:
			return  "CL_INVALID_IMAGE_SIZE";
		case CL_INVALID_SAMPLER:
			return  "CL_INVALID_SAMPLER";
		case CL_INVALID_BINARY:
			return  "CL_INVALID_BINARY";
		case CL_INVALID_BUILD_OPTIONS:
			return  "CL_INVALID_BUILD_OPTIONS";
		case CL_INVALID_PROGRAM:
			return  "CL_INVALID_PROGRAM";
		case CL_INVALID_PROGRAM_EXECUTABLE:
			return  "CL_INVALID_PROGRAM_EXECUTABLE";
		case CL_INVALID_KERNEL_NAME:
			return  "CL_INVALID_KERNEL_NAME";
		case CL_INVALID_KERNEL_DEFINITION:
			return  "CL_INVALID_KERNEL_DEFINITION";
		case CL_INVALID_KERNEL:
			return  "CL_INVALID_KERNEL";
		case CL_INVALID_ARG_INDEX:
			return  "CL_INVALID_ARG_INDEX";
		case CL_INVALID_ARG_VALUE:
			return  "CL_INVALID_ARG_VALUE";
		case CL_INVALID_ARG_SIZE:
			return  "CL_INVALID_ARG_SIZE";
		case CL_INVALID_KERNEL_ARGS:
			return  "CL_INVALID_KERNEL_ARGS";
		case CL_INVALID_WORK_DIMENSION:
			return  "CL_INVALID_WORK_DIMENSION";
		case CL_INVALID_WORK_GROUP_SIZE:
			return  "CL_INVALID_WORK_GROUP_SIZE";
		case CL_INVALID_WORK_ITEM_SIZE:
			return  "CL_INVALID_WORK_ITEM_SIZE";
		case CL_INVALID_GLOBAL_OFFSET:
			return  "CL_INVALID_GLOBAL_OFFSET";
		case CL_INVALID_EVENT_WAIT_LIST:
			return  "CL_INVALID_EVENT_WAIT_LIST";
		case CL_INVALID_EVENT:
			return  "CL_INVALID_EVENT";
		case CL_INVALID_OPERATION:
			return  "CL_INVALID_OPERATION";
		case CL_INVALID_GL_OBJECT:
			return  "CL_INVALID_GL_OBJECT";
		case CL_INVALID_BUFFER_SIZE:
			return  "CL_INVALID_BUFFER_SIZE";
		case CL_INVALID_MIP_LEVEL:
			return  "CL_INVALID_MIP_LEVEL";
		case CL_INVALID_GLOBAL_WORK_SIZE:
			return  "CL_INVALID_GLOBAL_WORK_SIZE";
		case CL_INVALID_PROPERTY:
			return  "CL_INVALID_PROPERTY";
		case CL_INVALID_IMAGE_DESCRIPTOR:
			return  "CL_INVALID_IMAGE_DESCRIPTOR";
		case CL_INVALID_COMPILER_OPTIONS:
			return  "CL_INVALID_COMPILER_OPTIONS";
		case CL_INVALID_LINKER_OPTIONS:
			return  "CL_INVALID_LINKER_OPTIONS";
		case CL_INVALID_DEVICE_PARTITION_COUNT:
			return  "CL_INVALID_DEVICE_PARTITION_COUNT";

		}

		return "Unknown err_msg";
	}

	static void on_cl_error(const int err)
	{
		if (err != CL_SUCCESS)
		{
			std::cerr << "Some error:\t" << err_msg(err) << std::endl;
			std::exit(err);
		}
	}

	class clTask
	{
		cl::Context _ctx_cpu;
		cl::Context _ctx_gpu;

		cl::Kernel _kernel_cpu;
		cl::Kernel _kernel_gpu;

		std::mutex _user_event_mutex;
		std::mutex _observation_mutex;
		
		//event in cpu context
		cl::Event _cpu_ready;
		
		//_gpu_ready associated via callback with ctx_cpu
		cl::UserEvent _gpu_ready_cpu_ctx;
		
		//event in gpu context
		cl::Event _gpu_ready;
		//_cpu_ready associated via callback with ctx_gpu
		cl::UserEvent _cpu_ready_gpu_ctx;
		
		static constexpr auto _log_depth = 10;
		size_t _counter_log{ 0 };

		//std::tuple 1: offload, 2: cpu_duration, 3: gpu_duration
		std::vector<std::tuple<float, float,float>> _previous_observation;
		std::vector<clTask*> _dependence_list;
				
		double _execution_time_cpu_msec{ 0 };
		double _execution_time_gpu_msec{ 0 };
		float _last_offload{ 0.0 };

		struct arg_info
		{
			std::string _type_name; // type_name(string,float ..)
			size_t _type_size; //size in byte
			size_t _CL_KERNEL_ARG_ADDRESS_QUALIFIER; //global,local,private
			size_t _CL_KERNEL_ARG_TYPE_QUALIFIER; //const,volatile
		};

		std::vector<arg_info> _arg_infos;

		std::string _body{ "" };
		std::string _name{ "" };
		std::string _jit_flags{ " " };

		int build_args_info(const cl::Kernel* k)
		{
			if (k == nullptr)return -1;

			int err = 0;
			std::string err_log;
			size_t args = k->getInfo<CL_KERNEL_NUM_ARGS>(&err);
			if (err != CL_SUCCESS) {
				err_log = "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";
				std::cerr << err_log << std::endl;
				return err;

			}

			for (size_t id = 0;id < args;id++)
			{
				const size_t aq = k->getArgInfo<CL_KERNEL_ARG_ADDRESS_QUALIFIER>(id, &err);
				if (err != CL_SUCCESS) {
					err_log = "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";
					std::cerr << err_log << std::endl;
					return err;
				}

				cl_kernel_arg_type_qualifier tq;
				err = k->getArgInfo(id, CL_KERNEL_ARG_TYPE_QUALIFIER, &tq);
				if (err != CL_SUCCESS) {
					err_log = "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";
					std::cerr << err_log << std::endl;
					return err;

				}

				std::string type_name = k->getArgInfo<CL_KERNEL_ARG_TYPE_NAME>(id, &err);
				if (err != CL_SUCCESS) {
					err_log = "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";
					std::cerr << err_log << std::endl;
					return err;
				}

				const auto pos = type_name.find('\000');
				if (pos != std::string::npos)
					type_name.replace(pos, 4, "");

				size_t type_size = clArgInfo::get_size(type_name);

				_arg_infos.push_back(arg_info{ type_name,type_size,aq,tq });
			}

			return 0;
		}
		
	public:		

		clTask(const clTask&) = delete;
		
		clTask(){}
		
		int build(const cl::Kernel& kernel_cpu,
			const cl::Kernel& kernel_gpu,
			const std::string& task_body,
			const std::string& task_name,
			const std::string& task_jit_flags = "")
		{
			_body = task_body;
			_name = task_name;
			_jit_flags = task_jit_flags;

			_kernel_cpu = kernel_cpu;
			int err = 0;
			_ctx_cpu = _kernel_cpu.getInfo<CL_KERNEL_CONTEXT>(&err);
			if (err != 0)return err;

			_kernel_gpu = kernel_gpu;
			_ctx_gpu = _kernel_gpu.getInfo<CL_KERNEL_CONTEXT>(&err);
			if (err != 0)return err;
#ifdef _OPT_OFFLOAD
			_previous_observation.resize(_log_depth);
#endif

			return build_args_info(&_kernel_cpu);
		}
		
		~clTask() { wait(); }

		std::string body()const { return _body; }
		std::string name()const { return _name; }
		std::string jit_flags()const { return _jit_flags; }

		const cl::Kernel* kernel_cpu()const { return &_kernel_cpu; }
		const cl::Kernel* kernel_gpu()const { return &_kernel_gpu; }

		cl::Event* cpu_ready(){ return &_cpu_ready; }
		cl::Event* gpu_ready(){ return &_gpu_ready; }

		const cl::Event* cpu_ready()const { return &_cpu_ready; }
		const cl::Event* gpu_ready()const { return &_gpu_ready; }

		const cl::UserEvent* cpu_user_ready_ctx_gpu()const { return &_cpu_ready_gpu_ctx; }
		const cl::UserEvent* gpu_user_ready_ctx_cpu()const { return &_gpu_ready_cpu_ctx; }

		int create_cpu_user_ready_ctx_gpu(const float offload)
		{
			int err = 0;
			if (_cpu_ready_gpu_ctx() != nullptr)
			{				
				err = _cpu_ready_gpu_ctx.wait();
				if (err != 0)return err;
			}
			_cpu_ready_gpu_ctx = cl::UserEvent(_ctx_gpu, &err);
			if (err != 0)return err;
			_last_offload = offload;
			return err;
		}
		
		int create_gpu_user_ready_ctx_cpu(const float offload)
		{
			int err = 0;
			if (_gpu_ready_cpu_ctx() != nullptr)
			{						
				err = _gpu_ready_cpu_ctx.wait();
				if (err != 0)return err;
			}
			_gpu_ready_cpu_ctx = cl::UserEvent(_ctx_cpu, &err);
			if (err != 0)return err;
			_last_offload = offload;
			return err;
		}			
		
		float update_offload()const
		{
			if (_previous_observation.empty())return 0.5f;

			const auto tuple = _previous_observation.back();
			const auto last_offload = std::get<0>(tuple);

			const auto cpu_duration_n_1 = std::get<1>(tuple);
			if (cpu_duration_n_1 == 0)return 0.5f;

			const auto gpu_duration_n_1 = std::get<2>(tuple);
			if (gpu_duration_n_1 == 0)return 0.5f;

			float updated_offload = 0;
			const auto ratio = cpu_duration_n_1 / gpu_duration_n_1;
			if (std::fabs(1.0 - ratio) < 0.1)
				updated_offload = last_offload;
			else
				updated_offload = ratio < 1.0 ? last_offload - 0.0125f*ratio : ratio > 1.0 ? last_offload + 0.0125f*ratio : last_offload;

			if (updated_offload < 0) updated_offload = 0;
			if (updated_offload > 1) updated_offload = 1;			

			return updated_offload;
		}

		void set_async_task_duration(cl_event ev, const double duartion)
		{
			std::lock_guard<std::mutex> guard(_observation_mutex);

			if (ev == _cpu_ready())							
				_execution_time_cpu_msec=duartion;							
			else if (ev == _gpu_ready())			
				_execution_time_gpu_msec = duartion;						
										
#ifdef _OPT_OFFLOAD
			if (_counter_log > _log_depth-1)_counter_log = 0;

			//std::tuple 1: offload, 2: cpu_duration, 3: gpu_duration
			_previous_observation[_counter_log++] = { _last_offload,_execution_time_cpu_msec,_execution_time_gpu_msec };			
#else
			//std::tuple 1: offload, 2: cpu_duration, 3: gpu_duration			
			_previous_observation.push_back({ _last_offload,_execution_time_cpu_msec,_execution_time_gpu_msec });
#endif
			
		}

		int set_async_event_user_complete(cl_event ev)
		{
			std::lock_guard<std::mutex> guard(_user_event_mutex);
			
			if (ev == _cpu_ready()) 
				return _cpu_ready_gpu_ctx.setStatus(CL_COMPLETE);
							
			if (ev == _gpu_ready())
				return _gpu_ready_cpu_ctx.setStatus(CL_COMPLETE);				
				
			return CL_INVALID_EVENT;
		}

		std::vector<clTask*>& dependence_list() { return _dependence_list; }

		size_t get_arg_type_size(const size_t id)const
		{
			return clArgInfo::get_size(_arg_infos[id]._type_name);
		}

		size_t get_arg_count()const
		{
			return _arg_infos.size();
		}

		bool is_arg_buffer(const size_t id)const
		{
			switch (_arg_infos[id]._CL_KERNEL_ARG_ADDRESS_QUALIFIER)
			{
			case CL_KERNEL_ARG_ADDRESS_PRIVATE:
				return false;
			}
			return true;
		}

		bool is_arg_read_only(const size_t id)const
		{
			switch (_arg_infos[id]._CL_KERNEL_ARG_TYPE_QUALIFIER)
			{
			case CL_KERNEL_ARG_TYPE_CONST:
				return true;
			}
			return false;
		}

		bool is_arg_LocalMem(const size_t id)const
		{
			switch (_arg_infos[id]._CL_KERNEL_ARG_ADDRESS_QUALIFIER)
			{
			case CL_KERNEL_ARG_ADDRESS_LOCAL:
				return true;
			}
			return false;
		}

		bool is_arg_type_Float(const size_t id)const
		{
			return clArgInfo::isFloat(_arg_infos[id]._type_name);
		}

		std::vector<std::int8_t> get_init_arg_value(const size_t id, const float val)const
		{
			return clArgInfo::get_value(_arg_infos[id]._type_name, val);
		}

		int wait()const 
		{
			int err = 0;	
			
			if (_cpu_ready_gpu_ctx() != nullptr)
			{
				err = _cpu_ready_gpu_ctx.wait();
				if (err != 0)return err;
			}

			if (_gpu_ready_cpu_ctx() != nullptr)
			{
				err = _gpu_ready_cpu_ctx.wait();
				if (err != 0)return err;
			}

			if (_gpu_ready() != nullptr)
			{
				err = _gpu_ready.wait();
				if (err != 0)return err;				
			}

			if (_cpu_ready() != nullptr)
			{
				err = _cpu_ready.wait();
				if (err != 0)return err;
			}		

			return err;
		}
		
		void write_records_to_stream(
			std::stringstream& ofs_offload,
			std::stringstream& ofs_cpu_time,
			std::stringstream& ofs_gpu_time)const
		{
			size_t it = 1;
			ofs_offload << "iteration\toffload\n";
			ofs_cpu_time << "iteration\tcpu_time\n";
			ofs_gpu_time << "iteration\tgpu_time\n";
			for (const auto& observation : _previous_observation)
			{
				ofs_offload << it << "\t";
				ofs_offload << std::get<0>(observation) << "\n";
				
				ofs_cpu_time << it << "\t";
				ofs_cpu_time << std::get<1>(observation) << "\n";
				
				ofs_gpu_time << it++ << "\t";
				ofs_gpu_time << std::get<2>(observation) << "\n";
			}
		}
	};

	static void CL_CALLBACK user_ev_handler(cl_event ev, cl_int stat, void* user_data)
	{		
		auto ptr_Task = (clTask*)(user_data);
		if (ptr_Task == nullptr) {
			std::cerr << "Async_callback: couldn't read clTask, fixme!" << std::endl;
			return;
		}

		auto err = ptr_Task->set_async_event_user_complete(ev);
		if (err != 0) {
			on_cl_error(err);
			std::cerr << "Async_callback: couldn't set user_event status, fixme!" << std::endl;
		}

#ifdef _PROFILE_
			cl_ulong start = 0, end = 0;
			err = clGetEventProfilingInfo(ev , CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start,0);
			on_cl_error(err);
			err = clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, 0);
			on_cl_error(err);
			const auto duration = (cl_double)(end - start)*(cl_double)(1e-06);
			ptr_Task->set_async_task_duration(ev, duration);
#endif
	}

	class clMemory
	{
	private:		
		
		void* _data{ nullptr };
		size_t _items{ 0 };
		size_t _size{ 0 };		
		bool _read_only{ false };		
		int _flag{ CL_MEM_READ_WRITE };

		const cl::Context* p_ctx_cpu{ nullptr };
		const cl::Context* p_ctx_gpu{ nullptr };

		std::unique_ptr<cl::Buffer> _buff_cpu{ nullptr };		
		std::unique_ptr<cl::Buffer> _buff_gpu{ nullptr };		
		
		void _clalloc(const cl::Context& ctx_cpu,
					  const cl::Context& ctx_gpu)
		{			
			p_ctx_cpu = &ctx_cpu;
			p_ctx_gpu = &ctx_gpu;

			auto flag = CL_MEM_READ_WRITE;

			if (_read_only) flag = CL_MEM_READ_ONLY;				
									
            int err = 0;
            _buff_cpu = std::unique_ptr<cl::Buffer>(new cl::Buffer(ctx_cpu, flag | CL_MEM_USE_HOST_PTR, _size, (void*)_data, &err));
            on_cl_error(err);
			
            _buff_gpu = std::unique_ptr<cl::Buffer>(new cl::Buffer(ctx_gpu, flag | CL_MEM_USE_HOST_PTR, _size, (void*)_data, &err));						
			//_buff_gpu = std::unique_ptr<cl::Buffer>(new cl::Buffer(ctx_gpu, flag , _size, nullptr, &err));
            on_cl_error(err);
		}

		template<typename T>
		void* _appalloc(const size_t items, const cl::Context& ctx_gpu)
		{
			_items = items;
			_size = items * sizeof(T);
			_flag = CL_MEM_READ_WRITE | CL_MEM_SVM_FINE_GRAIN_BUFFER;
			if (_read_only) _flag = CL_MEM_READ_ONLY | CL_MEM_SVM_FINE_GRAIN_BUFFER;
			_data = clSVMAlloc(ctx_gpu(), _flag, _size, 0);
            if(_data==nullptr)throw std::runtime_error(" memory==nullptr -->Check driver !!");
			return _data;
		}
		
	public:		

		clMemory(const clMemory&) = delete;

		//allocate only
		template<typename T>
		clMemory(	const cl::Context& ctx_cpu, 
					const cl::Context& ctx_gpu,
					const size_t items, 
					const bool read_only = false) 
		{
			
			if (items == 0)return;
			_read_only = read_only;
			_appalloc<T>(items, ctx_gpu);
            _clalloc(ctx_cpu,ctx_gpu);
		}

		//allocate and initialize with val
		template<typename T>
		clMemory(
			const cl::Context& ctx_cpu,
			const cl::Context& ctx_gpu,
			const size_t items, 
			const T val, const bool read_only = false) 
		{
			
			if (items == 0)return;						
			_read_only = read_only;
			_appalloc<T>(items, ctx_gpu);

			for (size_t i = 0; i < items; i++)
				static_cast<T*>(_data)[i] = val;
			
			_clalloc(ctx_cpu,ctx_gpu);
		}

		//allocate and copy from src
		template<typename T>
		clMemory(
			const cl::Context& ctx_cpu,
			const cl::Context& ctx_gpu,
			const size_t items, const T* src, const bool read_only = false) 
		{
			if (items == 0)return;
			if (src == nullptr)return;			
			_read_only = read_only;
			_appalloc<T>(items, ctx_gpu);
			std::memcpy(_data, src, items * sizeof(T));
			_clalloc(ctx_cpu, ctx_gpu);
		}

		template<typename T>
		clMemory(
			const cl::Context& ctx_cpu,
			const cl::Context& ctx_gpu,
			const size_t items, const void* src, const bool read_only = false) 
		{			
			if (items == 0)return;
			if (src == nullptr)return;
			_read_only = read_only;
			_appalloc<T>(items, ctx_gpu);
			std::memcpy(_data, src, items * sizeof(T));
			_clalloc(ctx_cpu, ctx_gpu);
		}

		//allocate and copy from src
		template<typename T>
		clMemory(
			const cl::Context& ctx_cpu,
			const cl::Context& ctx_gpu,
			const std::vector<T>& src, const bool read_only = false) 
		{	
			if (src.empty())return;
			_read_only = read_only;
			const auto bytes = src.size() * sizeof(T);
			_appalloc<T>(src.size(), ctx_gpu);
			std::memcpy(_data, src.data(), bytes);
			_clalloc(ctx_cpu, ctx_gpu);
		}

		cl_mem get_mem(const cl::Context& ctx)const
		{
			if (&ctx == p_ctx_cpu)
				return (*_buff_cpu)();

			return (*_buff_gpu)();
		}
		
		~clMemory() 
		{
			clSVMFree((*p_ctx_gpu)(), _data);
		}

		template<typename T>
		const T at(const size_t id)const
		{			
			return *(static_cast<const T*>(_data) + id);
		}

//#define _DEBUG
#ifdef _DEBUG
		template<typename T>
		void get_val(cl::Buffer* buff,std::vector<T>&values,const size_t size)const
		{
			int err = 0;
			auto ctx = buff->getInfo<CL_MEM_CONTEXT>(&err);
			auto devs = ctx.getInfo<CL_CONTEXT_DEVICES>(&err);

            cl::CommandQueue cq(ctx, devs[0], 0, &err);
			auto ptr = cq.enqueueMapBuffer(*buff, true, CL_MAP_READ, 0,_size, nullptr, nullptr, &err);            
			const auto items = size / sizeof(T);

			values.resize(items);
			std::memcpy(values.data(), ptr, size);					
			err = cq.enqueueUnmapMemObject(*buff, ptr, nullptr, nullptr);
			return;
		}

		template<typename T>
        void val(std::vector<T>& cpu_val,
                 std::vector<T>& gpu_val,
                 std::vector<T>& app_val)const
		{                        

            get_val(_buff_cpu.get(), cpu_val, _size);
            app_val.resize(_items); std::memcpy(app_val.data(),_data,_size);
            get_val(_buff_gpu.get(), gpu_val, _size);

            /*app_val.clear();
            app_val.resize(_items); std::memcpy(app_val.data(),_data,_size);
            cpu_val.clear();
            get_val(_buff_cpu.get(), cpu_val, _size);
			*/

		}
#endif				

		void* data()const{ 			
			return _data; }

		size_t items()const { 
			return _items; }

		size_t item_size()const { 
			return _size / _items; }

		size_t size()const {
			return _size;
		}

		bool isRead_only()const { 
			return _read_only; }	

	};

	struct generic_arg
	{
		std::unique_ptr<coopcl::clMemory> _clmem{ nullptr };
		std::vector<std::uint8_t> _arg_val;
		bool _isReadOnly;
		bool _isLocalMem;

		generic_arg(
			std::unique_ptr<coopcl::clMemory> clmem,
			std::vector<std::uint8_t> arg_val,
			bool isReadOnly,
			bool isLocalMem)
		{
			_clmem = std::move(clmem);
			_arg_val = arg_val;
			_isReadOnly = isReadOnly;
			_isLocalMem = isLocalMem;
		}

	};

	class ocl_device
	{
	private:
		cl::Context _ctx;
		cl::Device _device;
		std::vector<cl::CommandQueue> _queues;

		size_t _qid{ 0 };
		std::map<std::string, std::unique_ptr<cl::Program>> _bin_cache_programs;

		size_t				_max_work_group_size{ 0 };
		std::vector<size_t> _maximum_work_items_sizes;

		cl_device_type _dev_type;		
		
		
		bool _hasUnified_mem{ false };
		bool _support_svm_fine_grain{ false };

		int SetArg(
			const cl::Context& ctx,
			const cl::Kernel &task,
			std::uint8_t &id,
			std::vector<generic_arg>& first)const
		{
			int err = 0;
			cl_kernel k = task();			

			for (auto& arg : first) 
			{
				if (arg._clmem != nullptr)
				{
					cl_mem  app_cl_mem = arg._clmem->get_mem(ctx);
					err = clSetKernelArg(k, id++, sizeof(cl_mem), &app_cl_mem);
					if (err != 0)return err;					
				}
				else
				{
					err = clSetKernelArg(k, id++, arg._arg_val.size(), arg._arg_val.data());
					if (err != 0)return err;
				}			
			}
			return 0;
		}
		
		int SetArg(
			const cl::Context& ctx,
			const cl::Kernel &task,
			std::uint8_t &id,
			clMemory& first)const
		{
			cl_kernel k = task();
			cl_mem app_cl_mem = first.get_mem(ctx);
			return clSetKernelArg(k, id, sizeof(cl_mem), &app_cl_mem);
		}

		int SetArg(
			const cl::Context& ctx,
			const cl::Kernel &task,
			std::uint8_t &id,
			std::unique_ptr<clMemory>& first)const
		{
			cl_kernel k = task();
			cl_mem app_cl_mem = first->get_mem(ctx);
			return clSetKernelArg(k, id, sizeof(cl_mem), &app_cl_mem);
		}

		template <typename T>
		int SetArg(
			const cl::Context& ctx,
			const cl::Kernel &task,
			std::uint8_t &id,
			T &arg)const
		{
			cl_kernel k = task();
			return clSetKernelArg(k, id, sizeof(T), &arg);
		}

		int SetArgs(
			const cl::Context& ctx,
			const cl::Kernel &task,
			std::uint8_t &id)const
		{
			return 0;
		}

		template <typename T, typename... Args>
		int SetArgs(
			const cl::Context& ctx,
			const cl::Kernel &task,
			std::uint8_t &id,
			T &first,
			Args &... rest)const
		{
			int err = 0;
			err = SetArg(ctx, task, id, first);
			if (err != 0) 
				return err;
			id++;
			return SetArgs(ctx, task, id, rest...);
		}

		cl::Kernel build(const std::string task_name,
			const std::string task_name_cache,
			int& err)
		{
			err = 0;
			if (_bin_cache_programs.empty())
			{
				err = -1;
				return cl::Kernel();
			}
			return cl::Kernel(clCreateKernel((*_bin_cache_programs.at(task_name_cache))(), task_name.c_str(), &err));
		}

		std::string calc_cache_name(const std::string& task_name, const std::array<size_t, 3>& global_size)const
		{
			std::stringstream task_cache_name;

			task_cache_name << task_name << "_"
				<< global_size[0] << "_"
				<< global_size[1] << "_"
				<< global_size[2];

			return task_cache_name.str();
		}

		int build_program(
			const std::string& rewriten_task,
			const std::string& task_cache_name,
			const std::string& options)
		{
			int err = 0;
			
			_bin_cache_programs[task_cache_name] = std::unique_ptr<cl::Program>(new cl::Program(_ctx, rewriten_task, false, &err));
			on_cl_error(err);

			std::vector<cl::Device> devs{ _device };
			err = _bin_cache_programs[task_cache_name]->build(devs, options.c_str());
			if (err != CL_SUCCESS)
			{
				std::cerr << _bin_cache_programs[task_cache_name]->getBuildInfo<CL_PROGRAM_BUILD_LOG>(devs[0], &err) << std::endl;
				on_cl_error(err);
				return CL_BUILD_PROGRAM_FAILURE;
			}
			return err;
		}

	public:
		ocl_device(const ocl_device&) = delete;

		ocl_device(const cl::Platform& platform, cl::Device& device)
		{
			int err = 0;
			_device = device;

			_dev_type = _device.getInfo<CL_DEVICE_TYPE>(&err);
			on_cl_error(err);

			_maximum_work_items_sizes = _device.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>(&err);
			on_cl_error(err);

			_max_work_group_size = _device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>(&err);
			on_cl_error(err);

			_hasUnified_mem = _device.getInfo<CL_DEVICE_HOST_UNIFIED_MEMORY>(&err);
			on_cl_error(err);			

			if (_dev_type == CL_DEVICE_TYPE_GPU)
			{
				const auto ret = check_svm_support(CL_DEVICE_SVM_FINE_GRAIN_BUFFER, _device());
				if (!ret.empty()) _support_svm_fine_grain = true;
			}

			_ctx = cl::Context(_device, 0, 0, 0, &err);
			on_cl_error(err);

			const auto cnt_cu = _device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>(&err);
			for (size_t i = 0; i < cnt_cu; i++) {
#ifdef _PROFILE_
				_queues.push_back(cl::CommandQueue(_ctx, _device, CL_QUEUE_PROFILING_ENABLE, &err));
#else
				_queues.push_back(cl::CommandQueue(_ctx, _device, 0, &err));
#endif		
				on_cl_error(err);
			}

		}	

		template <typename... Args>
		int execute_tmp(
			const float offload,
			cl::Event& ev,
			const std::vector<cl::Event>* wait_list,
			const cl::Kernel& task,
			const cl::NDRange global,
			const cl::NDRange local,
			const cl::NDRange offset,
			Args&... rest)
		{
			int err = 0;
			std::uint8_t id = 0;

			err = SetArgs(_ctx, task, id, rest ...);
			on_cl_error(err);

			if (_qid >= _queues.size()) _qid = 0;
			err = _queues[_qid].enqueueNDRangeKernel(task, offset, global, local, wait_list, &ev);
			on_cl_error(err);

			return _queues[_qid++].flush();

		}
		
		template <typename... Args>
		int execute_tmp(
			const float offload,
			cl::Event& ev,
			const cl::Kernel& task,
			const cl::NDRange global,
			const cl::NDRange local,
			const cl::NDRange offset,
			Args&... rest)
		{
			int err = 0;
			std::uint8_t id = 0;

			err = SetArgs(_ctx, task, id, rest ...);
			on_cl_error(err);

			if (_qid >= _queues.size()) _qid = 0;
            err = _queues[_qid].enqueueNDRangeKernel(task, offset, global, local, nullptr, &ev);
			on_cl_error(err);

			return _queues[_qid++].flush();
		}

		template <typename... Args>
		int execute(
			const float offload,
			clTask& task,
			const cl::NDRange global,
			const cl::NDRange local,
			const cl::NDRange offset,
			Args&... rest)
		{
			int err = 0;
			std::uint8_t id = 0;			

			auto kernel = _dev_type == CL_DEVICE_TYPE_CPU ? task.kernel_cpu() : task.kernel_gpu();
			cl::Event* event_wait = _dev_type == CL_DEVICE_TYPE_CPU ? task.cpu_ready() : task.gpu_ready();

			err = SetArgs(_ctx, *kernel, id, rest ...);
			on_cl_error(err);
			
			std::vector<cl::Event> wait_list;
			if (!task.dependence_list().empty())
			{
				for (auto t : task.dependence_list())
				{
					// wait for bot CPU and GPU, if someone is not busy than the wait_ops. is approx. nop
					// dependent on the _dev_type get an event from  CPU and GPU that are from different context
					// an event_user_wait is an event from different context, it means if the _dev_type is a CPU than get an event_GPU from other ctx.
					
					//if a dev_type is CPU than the event_user_wait is from GPU, else opposite					
					const cl::UserEvent* event_user_wait = _dev_type == CL_DEVICE_TYPE_CPU ? t->gpu_user_ready_ctx_cpu() : t->cpu_user_ready_ctx_gpu();
					//if a dev_type is CPU than the event_wait is from CPU, for GPU same
					const cl::Event* event_wait = _dev_type == CL_DEVICE_TYPE_CPU ? t->cpu_ready() : t->gpu_ready();
					
					//Than always wait for CPU+GPU if any event exist!
					if((*event_user_wait)()!=nullptr) 
						wait_list.push_back(*event_user_wait);
					
					if ((*event_wait)() != nullptr) 
						wait_list.push_back(*event_wait);
					
					//t->wait();
				}
			}

			if (_qid >= _queues.size())_qid = 0;            			
			err = _queues[_qid].enqueueNDRangeKernel(*kernel, offset, global, local, &wait_list, event_wait);
			//err = _queues[_qid].enqueueNDRangeKernel(*kernel, offset, global, local, nullptr, event_wait);
			on_cl_error(err);
						
			if (_dev_type == CL_DEVICE_TYPE_CPU)
			{
				err = task.create_cpu_user_ready_ctx_gpu(offload);
				if(err!=0)return err;				
			}
			else
			{
				err = task.create_gpu_user_ready_ctx_cpu(offload);
				if (err != 0)return err;				
			}
			
			//Set host_async_event_callback			
			event_wait->setCallback(CL_COMPLETE, &user_ev_handler, &task);	 
			
			return _queues[_qid++].flush();

		}

		int wait()const
		{
			int err = 0;
			for (auto& q : _queues) {
				err = q.finish();
				if (err != 0)return err;
			}
			return err;
		}
		
		bool has_svm()const { return _support_svm_fine_grain; }

		size_t maximum_work_group_size()const { return _max_work_group_size; }

		std::vector<size_t> maximum_work_items_sizes()const { return _maximum_work_items_sizes; }

		const cl::Context* ctx()const { return &_ctx; }

		cl::Kernel build_task(const std::array<size_t, 3>& global_size,
			const std::string body,
			const std::string name,
			const std::string jit_flags = "")
		{
			int err = 0;

			const auto task_cache_name = calc_cache_name(name, global_size);
			const auto rewriten_task = rewrite::add_execution_guard_to_kernels(body, global_size);		

			if (_bin_cache_programs.empty())
			{
				err = build_program(rewriten_task, task_cache_name, jit_flags);
				on_cl_error(err);

				auto task = cl::Kernel(clCreateKernel((*_bin_cache_programs.at(task_cache_name))(), name.c_str(), &err));
				on_cl_error(err);
				return task;
			}
			else
			{
				for (auto& item : _bin_cache_programs)
				{
					if (item.first == task_cache_name)
					{
						auto task = cl::Kernel(clCreateKernel((*item.second)(), name.c_str(), &err));
						on_cl_error(err);
						return task;
					}
				}

				err = build_program(rewriten_task, task_cache_name, jit_flags);
				on_cl_error(err);
			}

			auto task = cl::Kernel(clCreateKernel((*_bin_cache_programs.at(task_cache_name))(), name.c_str(), &err));
			on_cl_error(err);
			return task;
		}

	};

	class virtual_device
	{

	private:		
		
		//different context cpu, gpu		
		const cl::Context* _ctx_cpu{ nullptr };
		const cl::Context* _ctx_gpu{ nullptr };

		std::unique_ptr<ocl_device> _dGPU{ nullptr };
		std::unique_ptr<ocl_device> _dCPU{ nullptr };

		std::string _platform_name;

		static int divide_ndranges(
			const float offload,
			const cl::NDRange& global,
			const cl::NDRange& local,
			cl::NDRange& global_cpu,
			cl::NDRange& global_gpu,
			cl::NDRange& global_offset,
			cl::NDRange& local_cpu,
			cl::NDRange& local_gpu)
		{
			const std::uint8_t dim_ndr = static_cast<std::uint8_t>(global.dimensions());

			size_t items = 1;
			for (size_t dim = 0; dim < dim_ndr; dim++)
				items *= global[dim];

			if (items < 1) return -1;

			const size_t dim_split = global[0];

			const float one_item = (float)dim_split / 100.0f;
			const auto procent = offload * 100.0f; //offload range--> (0:1>

			size_t items_gpu = static_cast<size_t>((procent * one_item) < dim_split ? ceil(procent * one_item) : dim_split);
			size_t items_cpu = dim_split - items_gpu;

			if (items_cpu < 8) { return 1; }//no offload/data-split, workload too small
			if (items_gpu < 8) { return 1; }//no offload/data-split, workload too small

			//------------------------------------------
			//group_sizes + global_sizes extend/pad
			//------------------------------------------
			if (local == cl::NullRange || local[0] == 0)
			{
				const size_t items_cpu_pad_wg = 32;
				const size_t items_gpu_pad_wg = 64;

				const size_t wg_mul_cpu = items_cpu % items_cpu_pad_wg;
				const size_t wg_mul_gpu = items_gpu % items_gpu_pad_wg;

				const size_t gx_pad_cpu = (wg_mul_cpu == 0 ? items_cpu / items_cpu_pad_wg : (items_cpu / items_cpu_pad_wg) + 1)*items_cpu_pad_wg;
				const size_t gx_pad_gpu = (wg_mul_gpu == 0 ? items_gpu / items_gpu_pad_wg : (items_gpu / items_gpu_pad_wg) + 1)*items_gpu_pad_wg;

				switch (dim_ndr)
				{
				case 1:
					local_cpu = { items_cpu_pad_wg };
					local_gpu = { items_gpu_pad_wg };

					global_cpu = { gx_pad_cpu };
					global_gpu = { gx_pad_gpu };
					global_offset = { items_cpu };

					break;
				case 2:
					local_cpu = { items_cpu_pad_wg,1 };
					local_gpu = { items_gpu_pad_wg,1 };

					global_cpu = { gx_pad_cpu,global[1] };
					global_gpu = { gx_pad_gpu,global[1] };
					global_offset = { items_cpu,0 };

					break;
				case 3:
					local_cpu = { items_cpu_pad_wg,1,1 };
					local_gpu = { items_gpu_pad_wg,1,1 };

					global_cpu = { gx_pad_cpu,global[1],global[2] };
					global_gpu = { gx_pad_gpu,global[1],global[2] };
					global_offset = { items_cpu,0,0 };
					break;
				}
			}
			else
			{
				const size_t wg_mul_cpu = items_cpu % local[0];
				const size_t wg_mul_gpu = items_gpu % local[0];

				const size_t gx_pad_cpu = (wg_mul_cpu == 0 ? items_cpu / local[0] : (items_cpu / local[0]) + 1)*local[0];
				const size_t gx_pad_gpu = (wg_mul_gpu == 0 ? items_gpu / local[0] : (items_gpu / local[0]) + 1)*local[0];

				switch (dim_ndr)
				{
				case 1:
					local_cpu = { local[0] };
					local_gpu = { local[0] };

					global_cpu = { gx_pad_cpu };
					global_gpu = { gx_pad_gpu };
					global_offset = { items_cpu };

					break;
				case 2:
					local_cpu = { local[0],local[1] };
					local_gpu = { local[0],local[1] };

					global_cpu = { gx_pad_cpu,global[1] };
					global_gpu = { gx_pad_gpu,global[1] };
					global_offset = { items_cpu,0 };

					break;
				case 3:
					local_cpu = { local[0],local[1],local[2] };
					local_gpu = { local[0],local[1],local[2] };

					global_cpu = { gx_pad_cpu,global[1],global[2] };
					global_gpu = { gx_pad_gpu,global[1],global[2] };
					global_offset = { items_cpu,0,0 };
					break;
				}
			}
			
			return 0;
		}

		template <typename... Args>
		int execute_async_tmp(
			clTask& task,
			const float offload,
			const cl::NDRange& global,
			const cl::NDRange& local,
			const cl::NDRange& offset,
			Args&... rest)
		{
			int err = 0;

			if ((int)offload < 0)return-1;
			if ((int)offload > 1)return-1;


			if (cmpf(offload, 1.0f))
			{
				return _dGPU->execute_tmp(offload, *task.gpu_ready(), *task.kernel_gpu(), global, local, offset, rest ...);
			}
			else if (cmpf(offload, 0.0f))
			{
				return _dCPU->execute_tmp(offload, *task.cpu_ready(), *task.kernel_cpu(), global, local, offset, rest ...);
			}
			else
			{
				cl::NDRange gcpu, ggpu, offset_split, loc_cpu, loc_gpu;
				const auto res = divide_ndranges(offload, global, local, gcpu, ggpu, offset_split, loc_cpu, loc_gpu);

				if (res == -1) { return CL_INVALID_OPERATION; }

				else if (res == 1) //workload remainder is to small to execute on both devices
				{
					if (offload > 0.5f)
					{
						return _dGPU->execute_tmp(offload, *task.gpu_ready(), *task.kernel_gpu(), global, local, offset, rest ...);
					}
					else
					{
						return _dCPU->execute_tmp(offload, *task.cpu_ready(), *task.kernel_cpu(), global, local, offset, rest ...);
					}
				}
				else
				{
					// Need to wait for both devices,
					// because previous call could be processed via both CPU+GPU
					if (!task.dependence_list().empty())
					{
						for (auto t : task.dependence_list())
						{
							err = t->wait();
							on_cl_error(err);
						}
					}

					_dGPU->execute_tmp(offload, *task.gpu_ready(), *task.kernel_gpu(), ggpu, loc_gpu, offset_split, rest ...);
					on_cl_error(err);

					_dCPU->execute_tmp(offload, *task.cpu_ready(), *task.kernel_cpu(), gcpu, loc_cpu, cl::NullRange, rest ...);
					on_cl_error(err);
				}
			}

			return err;
		}
		
	public:
		
		virtual_device(const virtual_device&) = delete;

		virtual_device()
		{			
			int err = 0;
			std::vector<cl::Platform> platforms;
			err = cl::Platform::get(&platforms);
			on_cl_error(err);

			//check if GPU or CPU is available			
			std::vector<cl::Device> devs;
			for (const auto& p : platforms)
			{	
				const std::string pname = p.getInfo<CL_PLATFORM_NAME>(&err);
				//std::cout << pname<< std::endl;
				
				p.getDevices(CL_DEVICE_TYPE_CPU|CL_DEVICE_TYPE_GPU, &devs);
				if (!devs.empty())
				{
					for (auto& d : devs)
					{
						auto dt = d.getInfo<CL_DEVICE_TYPE>(&err);
						on_cl_error(err);
						if (dt == CL_DEVICE_TYPE_CPU && _dCPU == nullptr)
						{
							_dCPU = std::unique_ptr<ocl_device>(new ocl_device(p, d));
							if (!_platform_name.empty())_platform_name.append("+");
							_platform_name.append(pname);
						}
						else if (dt == CL_DEVICE_TYPE_GPU && _dGPU == nullptr)
						{
							_dGPU = std::unique_ptr<ocl_device>(new ocl_device(p, d));
							if (!_platform_name.empty())_platform_name.append("+");
							_platform_name.append(pname);
						}
						
						
					}					
				}			
			}				

			if (_dCPU == nullptr) throw std::runtime_error("Minimal requirement: CPU with OpenCL installed!  exit ...");
			if (_dGPU == nullptr) throw std::runtime_error("Minimal requirement: CPU+GPU with OpenCL installed!  exit ...");
			
			_ctx_gpu = _dGPU->ctx();
			_ctx_cpu = _dCPU->ctx();

			if(!_dGPU->has_svm())throw std::runtime_error("Minimal requirement: GPU with support for OpenCL2.x and SVM_FINE_GRAIN_BUFFER installed!  exit ...");

		}

		virtual_device(	const std::string& cpu_platform_name,
						const std::string& gpu_platform_name)
		{
			int err = 0;
			std::vector<cl::Platform> platforms;
			err = cl::Platform::get(&platforms);
			on_cl_error(err);

			//check if GPU or CPU is available			
			std::vector<cl::Device> devs;
			for (const auto& p : platforms)
			{
				const std::string pname = p.getInfo<CL_PLATFORM_NAME>(&err);
				//std::cout << pname << std::endl;				
				p.getDevices(CL_DEVICE_TYPE_CPU | CL_DEVICE_TYPE_GPU, &devs);
				if (!devs.empty())
				{
					for (auto& d : devs)
					{
						auto dt = d.getInfo<CL_DEVICE_TYPE>(&err);
						on_cl_error(err);
						if (dt == CL_DEVICE_TYPE_CPU && _dCPU == nullptr) {
							if (pname.find(cpu_platform_name) != std::string::npos)
							{
								_dCPU = std::unique_ptr<ocl_device>(new ocl_device(p, d));

								if (!_platform_name.empty())_platform_name.append("+");
								_platform_name.append(pname);

							}
						}
						else if (dt == CL_DEVICE_TYPE_GPU && _dGPU == nullptr) {
							if (pname.find(gpu_platform_name) != std::string::npos)
							{
								_dGPU = std::unique_ptr<ocl_device>(new ocl_device(p, d));

								if (!_platform_name.empty())_platform_name.append("+");
								_platform_name.append(pname);
							}
						}
					}
				}
			}

			if (_dCPU == nullptr) throw std::runtime_error("Minimal requirement: CPU with OpenCL installed!  exit ...");
			if (_dGPU == nullptr) throw std::runtime_error("Minimal requirement: CPU+GPU with OpenCL installed!  exit ...");

			_ctx_gpu = _dGPU->ctx();
			_ctx_cpu = _dCPU->ctx();

			if (!_dGPU->has_svm())throw std::runtime_error("Minimal requirement: GPU with support for OpenCL2.x and SVM_FINE_GRAIN_BUFFER installed!  exit ...");

		}

		int build_task(
			clTask& task,
			const std::array<size_t, 3>& global_size,
			const std::string body,
			const std::string name,
			const std::string jit_flags = "")
		{
			const auto kcpu = _dCPU->build_task(global_size, body, name, jit_flags);
			const auto kgpu = _dGPU->build_task(global_size, body, name, jit_flags);
			return task.build(kcpu,kgpu,body, name, jit_flags);			
		}

		int wait()const
		{
			int err = _dGPU->wait();
			if(err!=0)return err;
			
			return _dCPU->wait();
		}
		
		template <typename... Args>
		int execute_async(
			clTask& task,
			const float offload_,
			const std::array<size_t, 3> global,
			const std::array<size_t, 3> local,
			Args&... rest)
		{
			int err = 0;
			
			float offload = offload_;

			if(offload_==-1) offload = task.update_offload();

			if ((int)offload < 0)return-1;
			if ((int)offload > 1)return-1;

			cl::NDRange global_in{
				global[0] == 0 ? 1 : global[0],
				global[1] == 0 ? 1 : global[1],
				global[2] == 0 ? 1 : global[2]
			};

			cl::NDRange local_in = {
				local[0],
				local[1],
				local[2] };

			if (local[0] == 0 && local[1] == 0 && local[2] == 0)
				local_in = cl::NullRange;

			
			if (cmpf(offload, 1.0f))
			{				
				return _dGPU->execute(offload, task, global_in, local_in, cl::NullRange, rest ...);
			}
			else if (cmpf(offload, 0.0f))
			{				
				return _dCPU->execute(offload,task, global_in, local_in, cl::NullRange, rest ...);
			}
			else
			{
				cl::NDRange gcpu, ggpu, offset, loc_cpu, loc_gpu;
				const auto res = divide_ndranges(offload, global_in, local_in, gcpu, ggpu, offset, loc_cpu, loc_gpu);

				if (res == -1) { return CL_INVALID_OPERATION; }

				else if (res == 1) //workload remainder is to small to execute on both devices
				{
					if (offload > 0.5f)
						return _dGPU->execute(offload,task, global_in, local_in, cl::NullRange, rest ...);
					else					
						return _dCPU->execute(offload,task, global_in, local_in, cl::NullRange, rest ...);					
				}
				else
				{	                   
					err = _dGPU->execute(offload, task, ggpu, loc_gpu, offset, rest ...);
					on_cl_error(err);

					err = _dCPU->execute(offload, task, gcpu, loc_cpu, cl::NullRange, rest ...);
					on_cl_error(err);					
				}
			}

			return err;
		}
	
		template <typename... Args>
		int execute(
			clTask& task,
			const float offload,
			const std::array<size_t, 3> global,
			const std::array<size_t, 3> local,
			Args&... rest)
		{
			auto err = execute_async(task, offload, global, local, rest ...);
			on_cl_error(err);
			err = task.wait();
			return err;
		}

		template <typename... Args>
		int execute_tmp(
			clTask& task,
			const float offload,
			const cl::NDRange& global,
			const cl::NDRange& local,
			const cl::NDRange& offset,
			Args&... rest)
		{
			auto err = execute_async_tmp(task, offload, global, local, offset, rest ...);
			on_cl_error(err);			
			//err = task.wait();
			return err;
		}
		

		std::unique_ptr<clMemory>
		alloc(const size_t items, const bool read_only = false)
		{
			return std::unique_ptr<clMemory>(new clMemory(*_ctx_cpu, *_ctx_gpu, items, 0, read_only));
		}	
		
		//allocate and initialize with 0
		template<typename T>
		std::unique_ptr<clMemory>
		alloc(const size_t items, const bool read_only = false) 
		{									
			T dummy_zero; 
			std::memset(&dummy_zero, 0, sizeof(T));
			return std::unique_ptr<clMemory>(new clMemory(*_ctx_cpu,*_ctx_gpu, items, dummy_zero, read_only));
		}
		//allocate and initialize with val
		template<typename T>
		std::unique_ptr<clMemory>
		alloc(const size_t items, const T val, const bool read_only = false)
		{
			return std::unique_ptr<clMemory>(new clMemory(*_ctx_cpu, *_ctx_gpu, items, val, read_only));
		}
		//allocate and copy from src
		template<typename T>
		std::unique_ptr<clMemory>
		alloc( const size_t items, const T* src, const bool read_only = false) 
		{			
			return std::unique_ptr<clMemory>(new clMemory(*_ctx_cpu, *_ctx_gpu, items, src, read_only));
		}

		//allocate and copy from src
		template<typename T>
		std::unique_ptr<clMemory>
		alloc(const std::vector<T>& src, const bool read_only = false) 
		{			
			return std::unique_ptr<clMemory>(new clMemory(*_ctx_cpu, *_ctx_gpu, src, read_only));
		}

		size_t maximum_work_group_size()const 
		{							
			const auto mwgs1 = _dCPU->maximum_work_group_size();
			if (_dGPU == nullptr)return mwgs1;

			const auto mwgs2 = _dGPU->maximum_work_group_size();
			//return smaller one
			return mwgs1>mwgs2?mwgs2:mwgs1;			
		}

		std::vector<size_t>
		maximum_work_items_sizes()const 
		{
			const auto wi1 = _dCPU->maximum_work_items_sizes();
			if (_dGPU == nullptr)return wi1;
			const auto wi2 = _dGPU->maximum_work_items_sizes();
			
			std::vector<size_t> max_wi;
			//return smaller one
			for (int i = 0;i < wi1.size();i++)
			{
				if (wi1[i] >= wi2[i])
					max_wi.push_back(wi2[i]);
				else
					max_wi.push_back(wi1[i]);
			}
			return max_wi;
		}

		std::string platform_name()const
		{
			return _platform_name;
		}
	};

}
