#include "common.h"
#include "clDriver.h"
#include "CImg.h"
#include "assert.h"
#include <cstdlib>
#include <thread>
#include <fstream>

//#define _DUMP_

namespace SIFT_tasks 
{	

#define _SIFT_SIGMA 1.6f
#define _SIFT_CONTR_THR 0.0066
#define _SIFT_CURV_THR 10

#define _SIFT_INTVLS  3
#define _SIFT_MAX_INTERP_STEPS  5
#define _SIFT_CONTR_THR_1  (0.5 * (_SIFT_CONTR_THR/_SIFT_INTVLS))
#define _SIFT_IMG_BORDER  5
#define _MAX_SCALES (_SIFT_INTVLS+3)
	
constexpr auto task_defines = R"(

#define MIN(a,b)    ((a)>(b)?(b):(a))
#define MAXV(A, B) 	((A>=B)?(A):(B))
#define MINV(A, B) 	((A<=B)?(A):(B))
#define ABS(A) 		(((A)>=0)?(A):(-A))
#ifndef M_PI
#define M_PI		3.14159265358979323846
#endif
#define M_PI2       (2.0F * M_PI)
#define _RPI (4.0/ M_PI)	// bins/rad

)";

constexpr auto task_Color = R"(
__kernel
void kColor_interleaved_ch( const global char3* restrict inputImage,
             global float* restrict outputImage)
{
    const int x = get_global_id(0);	
    const int y = get_global_id(1);   	
	const int w = get_global_size(0);
    const int pix = mad24(y,w,x);
	
    const float3 coefBGR = {0.114,0.587,0.299};
    const float val_scale = 0.0039215686; // 1/255
	const char3 vpix = inputImage[pix];  
	float3 fpix;
    fpix.x = (float)vpix.x;
    fpix.y = (float)vpix.y;
    fpix.z = (float)vpix.z;  			
	outputImage[pix] = dot(fpix,coefBGR)*val_scale;			
}

__kernel
void kColor( const global uchar* restrict inputImage,
             global float* restrict outputImage, const int w)
{
    const int x = get_global_id(0);	
    const int y = get_global_id(1);   		

    const int pix_R = mad24(y,w,x);
	const int pix_G = mad24(y,w,x)+w;
	const int pix_B = mad24(y,w,x)+2*w;
	
    const float3 coefBGR = {0.114,0.587,0.299};
    const float val_scale = 0.0039215686; // 1/255
	const uchar3 vpix = {inputImage[pix_B],inputImage[pix_G],inputImage[pix_R]};
	float3 fpix;
    fpix.x = (float)vpix.x;
    fpix.y = (float)vpix.y;
    fpix.z = (float)vpix.z;  			
	outputImage[pix_R] = dot(fpix,coefBGR)*val_scale;	
}
)";
	constexpr auto task_Down = R"(
__kernel
void kDown( global float* restrict dst,
            const global float* restrict src,
            const int wo,
            const int d,
			const int w,const int h)
{	
    const int x = get_global_id(0);
    const int y = get_global_id(1);    

    if (x < w && y < h)
    {
        const int pix_dst = mad24(y,w,x);
        const int pix_src = mad24(y*d,wo,x*d);
        dst[pix_dst] = src[pix_src];
    }
}
)";

	constexpr auto task_Blur = R"(
__kernel
void kBlurV( const global float* restrict inputImage,
			const global float* restrict filter,
            global float* restrict outputImage,
			const int w,const int h,const int fr)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);  	
    float prod = 0;		
	for( int i=-fr;i<fr;i++){	
		const float pixv = y+i>h-1?0:y+i<0?0:inputImage[(y+i)*w+x];
		prod += filter[fr+i]*pixv;
	}
    outputImage[y*w+x] = prod;		
}
__kernel
void kBlurH( const global float* restrict inputImage,
			const global float* restrict filter,
            global float* restrict outputImage,
			const int w,const int h,const int fr)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);    
	float prod = 0;		
	for( int i=-fr;i<fr;i++){	
		const float pixv = x+i>w-1?0:x+i<0?0:inputImage[y*w+(x+i)];
		prod += filter[fr+i]*pixv;
	}
    outputImage[y*w+x] = prod;		
}
)";
	constexpr auto task_Diff = R"(
__kernel
void kDiff( const global float* restrict inputImage_next,
			const global float* restrict inputImage_prev,
            global float* restrict outputImage,
			const int w)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);    	

    const int pix = mad24(y,w,x);
    outputImage[pix] = inputImage_next[pix]-inputImage_prev[pix];	
}
)";

	constexpr auto task_Detector = R"(
__kernel
void kDetector( 
global float4* restrict features,
const global float* restrict dog_p,
const global float* restrict dog_c,
const global float* restrict dog_n,
global uint* restrict count,
const int w,const int h,
const int s,const int o,
const uint max_feat)
{
	const int x = get_global_id(0);
	const int y = get_global_id(1);

	float curr_pix;

	if(y <= _SIFT_IMG_BORDER)return;
	if(y >= h-_SIFT_IMG_BORDER)return;
	if(x <= _SIFT_IMG_BORDER)return;
	if(x >= w - _SIFT_IMG_BORDER)return;

	curr_pix = dog_c[mad24(y,w,x)];

	if(fabs(curr_pix) <= _SIFT_CONTR_THR_1)return;

	#define at(pt, y, x) (*((pt) + (y)*w + (x)))

	#define LOC_EXTREMUM(PXL, IMG, CMP, SGN)			\
	( PXL CMP at(IMG ## _p, y, x-1)      &&		\
	PXL CMP at(IMG ## _p, y, x)    &&		\
	PXL CMP at(IMG ## _p, y, x+1)    &&		\
	PXL CMP at(IMG ## _p, y+1, x-1)    &&		\
	PXL CMP at(IMG ## _p, y+1, x)  &&		\
	PXL CMP at(IMG ## _p, y+1, x+1)  &&		\
	PXL CMP at(IMG ## _p, y-1, x-1)    &&		\
	PXL CMP at(IMG ## _p, y-1, x)  &&		\
	PXL CMP at(IMG ## _p, y-1, x-1)  &&		\
	\
	PXL CMP at(IMG ## _c, y, x-1)   &&		\
	PXL CMP at(IMG ## _c, y, x+1)   &&		\
	PXL CMP at(IMG ## _c, y+1, x-1)   &&		\
	PXL CMP at(IMG ## _c, y+1, x) &&		\
	PXL CMP at(IMG ## _c, y+1, x+1) &&		\
	PXL CMP at(IMG ## _c, y-1, x-1)   &&		\
	PXL CMP at(IMG ## _c, y-1, x) &&		\
	PXL CMP at(IMG ## _c, y-1, x+1) &&		\
	\
	PXL CMP at(IMG ## _n, y, x-1)      &&		\
	PXL CMP at(IMG ## _n, y, x)    &&		\
	PXL CMP at(IMG ## _n, y, x+1)    &&		\
	PXL CMP at(IMG ## _n, y+1, x-1)    &&		\
	PXL CMP at(IMG ## _n, y+1, x)  &&		\
	PXL CMP at(IMG ## _n, y+1, x+1)  &&		\
	PXL CMP at(IMG ## _n, y-1, x-1)    &&		\
	PXL CMP at(IMG ## _n, y-1, x)  &&		\
	PXL CMP at(IMG ## _n, y-1, x+1))

	if (LOC_EXTREMUM(curr_pix,dog,>,+) || LOC_EXTREMUM(curr_pix,dog,<,-))
	{
		float Dxx,Dyy,Dss,Dxy,Dxs,Dys,trH,detH;

		//float3 Hrow0,Hrow1,Hrow2,grad;
		float3 grad;

		float3 offset3 = {0.0f,0.0f,0.0f};

		int ty,tx,ts,iter;

		ty = y;//row transition
		tx = x;//col transition
		ts = s;//scale transition

		for(iter=0;iter<_SIFT_MAX_INTERP_STEPS;iter++)
		{
			// compute gradient
			grad.x = 0.5f*(dog_c[mad24(ty,w,tx+1)]  - dog_c[mad24(ty,w,tx-1)]);
			grad.y = 0.5f*(dog_c[mad24(ty+1,w,tx)] - dog_c[mad24(ty-1,w,tx)]);
			grad.z = 0.5f*(dog_n[mad24(ty,w,tx)] - dog_p[mad24(ty,w,tx)]);

			// compute Hessian
			Dxx = dog_c[mad24(ty,w,tx+1)] - 2.0f * dog_c[mad24(ty,w,tx)] + dog_c[mad24(ty,w,tx-1)];
			Dyy = dog_c[mad24(ty+1,w,tx)]- 2.0f * dog_c[mad24(ty,w,tx)] + dog_c[mad24(ty-1,w,tx)];
			Dss = dog_n[mad24(ty,w,tx)] - 2.0f * dog_c[mad24(ty,w,tx)] + dog_p[mad24(ty,w,tx)];

			// compute partial derivative
			Dxy = 0.25f * (dog_c[mad24(ty+1,w,tx+1)] + dog_c[mad24(ty-1,w, tx-1)]	- dog_c[mad24(ty+1,w,tx-1)] - dog_c[mad24(ty-1,w,tx+1)]);
			Dxs = 0.25f * (dog_n[mad24(ty,w,tx+1)]	+ dog_p[mad24(ty,w,tx-1)]		- dog_n[mad24(ty,w,tx-1)]	- dog_p[mad24(ty,w,tx+1)]);
			Dys = 0.25f * (dog_n[mad24(ty+1,w,tx)]	+ dog_p[mad24( ty-1,w,tx)]		- dog_p[mad24(ty+1,w,tx)]	- dog_n[mad24(ty-1,w,tx)]);


			float3 Hrow0,Hrow1,Hrow2;
		
			Hrow0.x = Dyy*Dss - Dys*Dys;//A		
			Hrow1.x = Dys*Dxs - Dxy*Dss;//B		
			Hrow2.x = Dxy*Dys - Dyy*Dxs;//C	
			detH = Dxx*Hrow0.x+Dxy*Hrow1.x+Dxs*Hrow2.x;
			if(detH == 0.0f)return; // non invertible !

		
			Hrow0.y = Dxs*Dys - Dxy*Dss;//D		
			Hrow0.z = Dxy*Dys - Dxs*Dyy;//G		
			Hrow1.y = Dxx*Dss - Dxs*Dxs;//E		
			Hrow1.z = Dxs*Dxy - Dxx*Dys;//H		
			Hrow2.y = Dxy*Dxs - Dxx*Dys;//F		
			Hrow2.z = Dxx*Dyy - Dxy*Dxy;//I

			float dv = native_divide(-1.0f,detH);

			Hrow0*= dv;
			Hrow1*= dv;
			Hrow2*= dv;

			offset3.x = dot(Hrow0,grad);
			offset3.y = dot(Hrow1,grad);
			offset3.z = dot(Hrow2,grad);
		
			if(fabs(offset3.x)<0.5f && fabs(offset3.y)<0.5f && fabs(offset3.z)<0.5f)break;

		
			tx += round(offset3.x);
			if(tx<_SIFT_IMG_BORDER || tx>=w-_SIFT_IMG_BORDER)return;

			ty += round(offset3.y);
			if(ty<_SIFT_IMG_BORDER || ty>=h-_SIFT_IMG_BORDER)return;

			ts += round(offset3.z);
			if(ts<1 || ts>_SIFT_INTVLS)return;
		}

		if(iter>=_SIFT_MAX_INTERP_STEPS)return; // if extremum not sub localized then reject this sample !		
		trH  = Dxx + Dyy;
		detH = Dxx * Dyy - Dxy * Dxy;

		if(detH<=0.0f)return;
		if( (trH*trH)/detH >= ((_SIFT_CURV_THR+1)*(_SIFT_CURV_THR+1)/_SIFT_CURV_THR))return;		
		
		curr_pix = ts==1?dog_p[mad24(ty,w,tx)]:ts==2?dog_c[mad24(ty,w,tx)]:dog_n[mad24(ty,w,tx)];

		// reuse detH as contrast value
		detH = curr_pix + 0.5f * (grad.x * offset3.x + grad.y * offset3.y + grad.z * offset3.z);

		if( fabs(detH) > (float)(_SIFT_CONTR_THR/_SIFT_INTVLS))
		{
			uint fid = atomic_inc(count);
			if(fid<max_feat)
			{
				features[fid].x = ( (float)tx + offset3.x ) * native_powr( 2.0f, (float)(o) );
				features[fid].y = ( (float)ty + offset3.y ) * native_powr( 2.0f, (float)(o) );
				features[fid].z = _SIFT_SIGMA * native_powr(2.0f, (float)(((float)ts+offset3.z)/(float)_SIFT_INTVLS));
				features[fid].w = _SIFT_SIGMA * native_powr(2.0f,(float)((float)o+((float)(ts+offset3.z)/(float)_SIFT_INTVLS)));				
			}
		}
	}
}
)";

constexpr auto task_Reset = R"(
__kernel 
void kReset(global float4* restrict features)
{
	const int tid = get_global_id(0);
	features[tid] = 0;
}
)";
}

class sift_octave
{
	private:
	std::unique_ptr<coopcl::clMemory> _imgColor{ nullptr };
	std::unique_ptr<coopcl::clMemory> _imgGray{ nullptr };
	std::unique_ptr<coopcl::clMemory> _imgFilter{ nullptr };

	coopcl::clTask _task_Reset;
	coopcl::clTask _task_Color;
	coopcl::clTask _task_Down;

	std::vector<std::unique_ptr<coopcl::clMemory>> _scale_images_intermidiate;
	std::vector<std::unique_ptr<coopcl::clMemory>> _scale_images;
	std::vector<std::unique_ptr<coopcl::clMemory>> _diff_scale_images;
	std::vector<std::unique_ptr<coopcl::clMemory>> _counter_kp;
	std::unique_ptr<coopcl::clMemory> _kp_detector{ nullptr };

	std::vector<std::unique_ptr<coopcl::clTask>> _task_BlurH;
	std::vector<std::unique_ptr<coopcl::clTask>> _task_BlurV;
	std::vector<std::unique_ptr<coopcl::clTask>> _task_Diff;
	std::vector<std::unique_ptr<coopcl::clTask>> _task_Detector;

	std::uint32_t _max_features{ 0 };
	size_t _octave_id{ 0 };
	size_t _scale_image_width{ 0 };
	size_t _scale_image_height{ 0 };

	std::vector<std::vector<float>> _scale_filters;
	std::vector<float> _filter_coef_sigmas;
	std::vector<std::uint32_t> _filter_coef_widths;

	int Build_separable_filter_cof(const float _sigma)
	{
		int status = 0;
		std::uint32_t s = 0;
		auto fcval = 0.f;
		auto sum = 0.f;
		auto sigma = _sigma <= 0 ? 1.6 : _sigma;
		_scale_filters.resize(_MAX_SCALES);
		_filter_coef_sigmas.resize(_MAX_SCALES);
		_filter_coef_widths.resize(_MAX_SCALES);

		_filter_coef_sigmas[0] = sigma;
		float k = powf(2.0f, 1.0f / (_SIFT_INTVLS));// k=2^1/S

													// pre-compute Gaussian pyramid sigmas
													// This method calculates the Gaussian coefficients filters 
													// It calculates delta_sigmas = sqrt(sigma_next_lvl^2-sigma_prev_lvl^2)
													//----------------------------------------------------------------------------------
													// Implementation targets the semi-group property of Gauss_kernel
													//----------------------------------------------------------------------------------
		for (s = 1; s < _MAX_SCALES; s++)
		{
			float lvl_sig_prev = sigma *powf(k, (float)(s - 1));
			float lvl_sig_next = lvl_sig_prev*k;
			float delta_sig = (float)sqrt(lvl_sig_next*lvl_sig_next - lvl_sig_prev*lvl_sig_prev);
			_filter_coef_sigmas[s] = delta_sig;
		}

		//pre-compute Gaussian pyramid sigmas
		//This is non-incremental method !! (longer filter widths/radius)
		//sig0,ksig,k^2*sig,k^3*sig,k^4*sig,k^5*sig ...
		//----------------------------------------------------------------------------------
		// Implementation omits the semi-group property of Gauss_kernel
		//----------------------------------------------------------------------------------
		/*for (s = 0; s < MAX_SCALES; s++)
		_filter_coef_sigmas[s] = sigma*pow(k, s);*/

		size_t fc = 0;
		// compute Gauss kernels
		for (s = 0; s < _MAX_SCALES; s++)
		{
			sum = 0.0f;

			auto& s_sigma = _filter_coef_sigmas[s];
			auto& filter_coef = _scale_filters[s];

			// Filter radius / discretized Gauss function with [-3.5sig:3,5sig]
			size_t frad = (size_t)(ceil(3.5f * s_sigma));
			auto fil_width = 2 * frad + 1;

			_filter_coef_widths[s] = fil_width;
			filter_coef.resize(fil_width, 0.0f);

			for (fc = 0; fc < frad; fc++)
			{
				// Gauss filter is symmetric and separable
				filter_coef[fc] = filter_coef[2 * frad - fc] = fcval = float(exp(float(-0.5 *((fc - frad) * (fc - frad) / (_filter_coef_sigmas[s] * _filter_coef_sigmas[s])))));
				sum += 2 * fcval;
			}

			filter_coef[fc] = 1;
			sum += 1;

			// normalize
			fcval = 1.0f / sum;

			for (fc = 0; fc < 2 * frad + 1; fc++)
				filter_coef[fc] *= fcval;
		}
		return status;
	}

	void show_gray()const {
		_task_Color.wait();
		std::cout << "visualize ..." << std::endl;
		cimg_library::CImg<float> img_((float*)_imgGray->data(), _scale_image_width, _scale_image_height, 1, 1, true);
		cimg_library::CImgDisplay disp(img_, "img_Gray");
		disp.display(img_);
		while (!disp.is_closed() && !disp.is_keyESC()) disp.wait();
	}

	void show_blur(const size_t id)const {
		std::string img_name;
		std::cout << "visualize ..." << std::endl;
		const float* pimg = nullptr;

		if (id >= _MAX_SCALES)return;

		pimg = (float*)_scale_images[id]->data();
		img_name = "Blur";
		img_name.append(std::to_string(id + 1));
		
		if (pimg != nullptr)
		{
			cimg_library::CImg<float> img_(pimg, _scale_image_width, _scale_image_height, 1, 1, true);
			cimg_library::CImgDisplay disp(img_, img_name.c_str());
			disp.display(img_);
			while (!disp.is_closed() && !disp.is_keyESC()) disp.wait();
		}
	}

	void show_diff_blur(const size_t id)const {
		std::string img_name;
		std::cout << "visualize ..." << std::endl;
		float* pimg = nullptr;
		
		if (id >= _MAX_SCALES-1)return;

		pimg = (float*)_diff_scale_images[id]->data();
		img_name = "BlurDiff";
		img_name.append(std::to_string(id + 1));

		if (pimg != nullptr)
		{
			cimg_library::CImg<float> img_(pimg, _scale_image_width, _scale_image_height, 1, 1, true);
			cimg_library::CImgDisplay disp(img_, img_name.c_str());
			disp.display(img_);
			while (!disp.is_closed() && !disp.is_keyESC()) disp.wait();
		}
	}

	void show_features()const
	{
		int err = 0;

		for (int i = 0; i < _SIFT_INTVLS; i++)
		{
			err = _task_Detector[i]->wait();
			if (err != 0)return ;
		}

		size_t  features = 0;

		_task_Color.wait();

		
		
		std::cout << "visualize ..." << std::endl;
		std::vector < std::uint8_t> img8b;

		for (size_t i = 0; i < _imgGray->items(); i++)
			img8b.push_back((std::uint8_t)255.0f*_imgGray->at<float>(i));

		cimg_library::CImg <std::uint8_t> img_(img8b.data(), _scale_image_width, _scale_image_height, 1, 1, true);
		cimg_library::CImgDisplay disp(img_, "img_Gray");

		const unsigned char color_kp[] = { 0, 0, 255 };
		const unsigned char color_forgr[] = { 0,0,0 };
		const unsigned char color_backgr[] = { 255,255,255 };
		for (size_t i = 0; i < _max_features; i++)
		{
			const auto feature = _kp_detector->at<cl_float4>(i);

			if (feature.x != 0 && feature.y != 0)
			{
				const int x0 = feature.x;
				const int y0 = feature.y;
				const auto r = feature.z;
				//std::cout << "{x,y,r}\t" << x0 << "," << y0 << "," << r << "\n";
				img_.draw_circle(x0, y0, r, color_kp, 1.0f);
				features++;
			}
		}
		std::string txt = "Features found: "; txt.append(std::to_string(features));
		img_.draw_text(8, 8, txt.c_str(), color_forgr, color_backgr, 1.0f, 25);
		disp.display(img_);
		while (!disp.is_closed() && !disp.is_keyESC()) disp.wait();

		return;

	}
	
	int reset(coopcl::virtual_device& device,const float offload)
	{
		int err;
		for (int i = 0; i < _SIFT_INTVLS; i++)
		{
			auto ptr = (std::uint32_t*)_counter_kp[i]->data();
			*ptr = 0;
		}		

		err = device.execute_async(_task_Reset, offload, { _max_features,1,1 }, { 1,1,1 }, _kp_detector);
		if (err != 0)return err;
	}
	
	std::string 
	store_task_in_file(const coopcl::clTask* task,
		const std::string& path,
		const size_t octave_id,
		const size_t scale_id)const
	{
		std::string file_name = path;
		file_name.append(task->name());
		file_name.append(std::to_string(octave_id));
		file_name.append(std::to_string(scale_id));

		std::string file_name_off = file_name;
		file_name_off.append("_off.dat");

		std::string file_name_cpu = file_name;
		file_name_cpu.append("_cpu.dat");

		std::string file_name_gpu = file_name;
		file_name_gpu.append("_gpu.dat");

		std::ofstream ofs_off(file_name_off);
		std::ofstream ofs_cpu(file_name_cpu);
		std::ofstream ofs_gpu(file_name_gpu);

		std::stringstream obs_offload;
		std::stringstream obs_cpu;
		std::stringstream obs_gpu;
		task->write_records_to_stream(obs_offload,obs_cpu,obs_gpu);
		
		ofs_off << obs_offload.str();
		ofs_cpu << obs_cpu.str();
		ofs_gpu << obs_gpu.str();

		return file_name;
	}

	public:

		sift_octave(const sift_octave&) = delete;

	sift_octave(
		const size_t id,
		const size_t w, const size_t h,
		const std::string& tasks,
		coopcl::virtual_device& device,
		const std::uint8_t* input_image = nullptr)
	{				
		_octave_id = id;
		_scale_image_height = h;
		_scale_image_width = w;

		_scale_image_height = h / std::pow(2,_octave_id);
		_scale_image_width = w / std::pow(2,_octave_id);
#ifdef _DUMP_
		std::cout << "Build octave " << _octave_id + 1 <<", scale_image_size {"<<_scale_image_width<<","<<_scale_image_height<<"} pixels ..."<<std::endl;
#endif
		const int items = _scale_image_width*_scale_image_height;		
		
		int err = 0;
		err = Build_separable_filter_cof(_SIFT_SIGMA);
		if (err != 0)throw std::runtime_error("FilterCoeficients fail -> fixme!!");
		
		_imgFilter = device.alloc<float>(_scale_filters[id],true);	
		
		if (_octave_id == 0)
		{
			if (input_image == nullptr)throw std::runtime_error("Color_Image empty ? fixme!!");
			_imgColor = device.alloc<std::uint8_t>(3 * items, input_image);
		}
		
		_imgGray = device.alloc<float>(items);
		
		//Detector_memory		
		_max_features = items / 100;
		_kp_detector = device.alloc<cl_float4>(_max_features);

		for (int i = 0; i < _SIFT_INTVLS; i++)
			_counter_kp.push_back(device.alloc<std::uint32_t>(1));
		
		std::stringstream jit;		
		jit << "-D _SIFT_SIGMA=" << _SIFT_SIGMA;		
		jit << " -D _SIFT_IMG_BORDER=" << _SIFT_IMG_BORDER;
		jit << " -D _SIFT_CONTR_THR_1=" << _SIFT_CONTR_THR_1;
		jit << " -D _SIFT_MAX_INTERP_STEPS=" << _SIFT_MAX_INTERP_STEPS;
		jit << " -D _SIFT_INTVLS=" << _SIFT_INTVLS;
		jit << " -D _SIFT_CURV_THR=" << _SIFT_CURV_THR;
		jit << " -D _SIFT_CONTR_THR=" << _SIFT_CONTR_THR;		

		if (_octave_id == 0)
		{
			err = device.build_task(_task_Color, { _scale_image_width,_scale_image_height,1 }, tasks, "kColor", jit.str()); 
			if (err != 0)throw std::runtime_error("kColor fail -> fixme!!");
		}
		else
		{
			err = device.build_task(_task_Down, { _scale_image_width,_scale_image_height,1 }, tasks, "kDown", jit.str());
			if (err != 0)throw std::runtime_error("kDown fail -> fixme!!");
		}
		
		for (int scale_id = 0; scale_id < _MAX_SCALES; scale_id++)
		{
			_task_BlurH.push_back(std::make_unique<coopcl::clTask>());
			err = device.build_task(*_task_BlurH[scale_id], { _scale_image_width,_scale_image_height,1 }, tasks, "kBlurH", jit.str());
			if (err != 0)if (err != 0)if (err != 0)throw std::runtime_error("kBlur fail -> fixme!!");

			_task_BlurV.push_back(std::make_unique<coopcl::clTask>());
			err = device.build_task(*_task_BlurV[scale_id], { _scale_image_width,_scale_image_height,1 }, tasks, "kBlurV", jit.str());
			if (err != 0)if (err != 0)if (err != 0)throw std::runtime_error("kBlur fail -> fixme!!");

			// now set task_dependencies			
			if (scale_id > 0)
			{
				_task_BlurH[scale_id]->dependence_list().push_back(_task_BlurV[scale_id - 1].get());
				_task_BlurV[scale_id]->dependence_list().push_back(_task_BlurH[scale_id].get());
			}
			else
			{
				if (_octave_id == 0)
					_task_BlurH[scale_id]->dependence_list().push_back(&_task_Color);
				else
					_task_BlurH[scale_id]->dependence_list().push_back(&_task_Down);

				_task_BlurV[scale_id]->dependence_list().push_back(_task_BlurH[scale_id].get());
			}

			//now memory
			_scale_images_intermidiate.push_back(device.alloc<float>(items));
			_scale_images.push_back(device.alloc<float>(items));
		}

		for (int i = 0; i < _MAX_SCALES-1; i++)
		{
			_task_Diff.push_back(std::make_unique<coopcl::clTask>());
			err = device.build_task(*_task_Diff[i], { _scale_image_width,_scale_image_height,1 }, tasks, "kDiff", jit.str());
			if (err != 0)throw std::runtime_error("kDiff fail -> fixme!!");

			//set task_dependencies
			_task_Diff[i]->dependence_list().push_back(_task_BlurV[i + 1].get());

			//now memory
			_diff_scale_images.push_back(device.alloc<float>(items));

		}		

		//SIFT task_graph_single_octave
		/*
		taskBlur_x = taskBlurH+taskBlurV
																[task_Detector1][task_Detector2][task_Detector3]
																/		|	  \  /	  |    \  /     |     \
															   /		|	   \/     |     \/      |	   \
															  /		    |	   /\     |     /\      |		\
													[taskDiff_21] [taskDiff_32] [taskDiff_43] [taskDiff_54] [taskDiff_65]
													|			  |		        |			  |		        |
													|			  |			    |			  |  		    |
		  [taskColor/Down]->[taskBlur_1]->[taskBlur_2]->[taskBlur_3]->[taskBlur_4]->[taskBlur_5]->[taskBlur_6]
		  
		<BEGIN>------------------------------------------------------------------------------------------------------------> TIME
		*/
		
		for (int i = 0; i < _SIFT_INTVLS; i++)
		{
			_task_Detector.push_back(std::make_unique<coopcl::clTask>());
			err = device.build_task(*_task_Detector[i], { _scale_image_width,_scale_image_height,1 }, tasks, "kDetector", jit.str());
			if (err != 0)throw std::runtime_error("kDetector fail -> fixme!!");			
		}
		//set dependencies
		for (int i = 1; i <= _SIFT_INTVLS; i++)
		{
			_task_Detector[i-1]->dependence_list().push_back(_task_Diff[i - 1].get());
			_task_Detector[i-1]->dependence_list().push_back(_task_Diff[i].get());
			_task_Detector[i-1]->dependence_list().push_back(_task_Diff[i + 1].get());
		}	

		err = device.build_task(_task_Reset, { _scale_image_width,_scale_image_height,1 }, tasks, "kReset", jit.str());
		if (err != 0)throw std::runtime_error("kReset fail -> fixme!!");


		for (int i = 0; i < _SIFT_INTVLS; i++)
			_task_Reset.dependence_list().push_back(_task_Detector[i].get());
	}

	int call_async( const float offload,
		coopcl::virtual_device& device,
		std::unique_ptr<coopcl::clMemory>* scale_image_prev_octave=nullptr,
		coopcl::clTask* wait_task_prev_octave=nullptr)
	{
		const size_t gsx = 16; const size_t gsy = 16;
		int err = 0;

		err = reset(device,offload);
		if (err != 0)return err;

		const int i32w = _scale_image_width;
		const int i32h = _scale_image_height;

		if (_octave_id == 0)
		{
			//__kernel void kColor(const global uchar* restrict inputImage, global float* restrict outputImage, const int w)
			err = device.execute_async(_task_Color, offload, { _scale_image_width,_scale_image_height,1 }, { gsx,gsy,1 }, _imgColor, _imgGray, i32w);
			if (err != 0)return err;								
		}
		else
		{			
			if (scale_image_prev_octave == nullptr)return -111;
			if (wait_task_prev_octave == nullptr)return -222;

			_task_Down.dependence_list().push_back(wait_task_prev_octave);
			
			
			const int wo = (int)(_scale_image_width * 2);
			const int d = (int)2;
			// __kernel void kDown(global float* restrict dst, const global float* restrict src, const int wo, const int d,const int w, const int h)
			err = device.execute_async(_task_Down, offload, { _scale_image_width,_scale_image_height,1 }, { gsx,gsy,1 }, _imgGray, *scale_image_prev_octave, wo ,d,i32w,i32h);
			if (err != 0)return err;
		}		
		
		//Blur 
		for (int i = 0; i < _MAX_SCALES; i++)
		{
			const int i32fr = _filter_coef_widths[i] / 2;
			//void kBlur(const global float* restrict inputImage, const global float* restrict filter, global float* restrict outputImage,const int w, const int h,const int fr)
			if (i == 0)
			{
				err = device.execute_async(*_task_BlurH[i], offload, { _scale_image_width,_scale_image_height,1 }, { gsx,gsy,1 }, _imgGray, _imgFilter, _scale_images_intermidiate[i], i32w, i32h, i32fr);
				if (err != 0)return err;
			}
			else
			{ 
				err = device.execute_async(*_task_BlurH[i], offload, { _scale_image_width,_scale_image_height,1 }, { gsx,gsy,1 }, _scale_images[i-1], _imgFilter, _scale_images_intermidiate[i], i32w, i32h, i32fr);
				if (err != 0)return err;
			}

			err = device.execute_async(*_task_BlurV[i], offload, { _scale_image_width,_scale_image_height,1 }, { gsx,gsy,1 }, _scale_images_intermidiate[i], _imgFilter, _scale_images[i], i32w, i32h, i32fr);
			if (err != 0)return err;
		}		
		// Blur-differences	
		for (int i = 0; i < _MAX_SCALES-1; i++)
		{							
			//__kernel void kDiff(const global float* restrict inputImage_next, const global float* restrict inputImage_prev, global float* restrict outputImage,const int w)
			err = device.execute_async(*_task_Diff[i], offload, { _scale_image_width,_scale_image_height,1 }, { gsx,gsy,1 }, _scale_images[i+1], _scale_images[i], _diff_scale_images[i],i32w);
			if (err != 0)return err;
		}	
		
		//Feature-detector
		//__kernel void kDetector( global float4* restrict features,
		//const global float* restrict dog_p,const global float* restrict dog_c,const global float* restrict dog_n,
		//global uint* restrict count,const int w,const int h, const int s,const int o, const uint max_feat)		
		const int i32ocatve_id = _octave_id;
		for (int i32scale_id = 1; i32scale_id <= _SIFT_INTVLS; i32scale_id++)
		{
			err = device.execute_async(*_task_Detector[i32scale_id-1], offload, { _scale_image_width,_scale_image_height,1 }, { gsx,gsy,1 },
				_kp_detector, _diff_scale_images[i32scale_id -1], _diff_scale_images[i32scale_id], _diff_scale_images[i32scale_id + 1],
				_counter_kp[i32scale_id-1], i32w, i32h, i32scale_id, i32ocatve_id, _max_features);
			if (err != 0)return err;
		}		

		return err;
	}

	int wait()const 
	{	
		int err = 0;
		
		for (int i = 0; i < _SIFT_INTVLS; i++)
		{
			err = _task_Detector[i]->wait();
			if (err != 0)return err;
		}

		return err;
	}
	
	void show()const
	{		
		show_gray();

		for(int i=0;i<_MAX_SCALES;i++) 
			show_blur(i);
		
		for (int i = 0; i<_MAX_SCALES-1; i++) 
			show_diff_blur(i);
		
		show_features();
	}
	
	void dump_cnt_features()const 
	{
		for (int i = 0; i < _SIFT_INTVLS; i++)
		{
			_task_Detector[i]->wait();
			std::cout << "kp_cnt" << i << ":\t" << _counter_kp[i]->at<std::uint32_t>(0) << std::endl;
		}
	}

	std::unique_ptr<coopcl::clMemory>* imgBlur() { return &_scale_images[1]; }

	coopcl::clTask* task_Blur() { return _task_BlurV[1].get(); }

	std::vector<std::string>
		store_task_records_in_files(const std::string& path, const size_t oid)const
	{
		std::vector<std::string> log_files;
		size_t sid = 0;

		log_files.push_back(store_task_in_file(&_task_Reset, path, oid, sid));
		log_files.push_back(store_task_in_file(&_task_Color, path, oid, sid));
		log_files.push_back(store_task_in_file(&_task_Down, path, oid, sid));

		for (const auto& t : _task_BlurH)
			log_files.push_back(store_task_in_file(t.get(), path, oid, sid++));

		sid = 0;
		for (const auto& t : _task_BlurV)
			log_files.push_back(store_task_in_file(t.get(), path, oid, sid++));

		sid = 0;
		for (const auto& t : _task_Diff)
			log_files.push_back(store_task_in_file(t.get(), path, oid, sid++));

		sid = 0;
		for (const auto& t : _task_Detector)
			log_files.push_back(store_task_in_file(t.get(), path, oid, sid++));

		return log_files;
	}
};

int main(int argc,char** argv)
{
	if (argc != 2)
	{
		std::cerr << "App expects following input_format: ./app.exe offload(0.0:1.0f range)<float>" << std::endl;
		std::cerr << "Example:\t ./app.exe 0.0f" << std::endl;
		std::exit(-1);
	}

	const float offload = std::atof(argv[1]);
#ifndef _DUMP_
	std::cout << "Init_time_ms," << std::flush;
#endif
	const auto app_init_start = std::chrono::system_clock::now();
	
	using namespace cimg_library;
	const auto img = "c:/Users/morkon/Pictures/2k.bmp";
	CImg<unsigned char> image(img);
	const int width = image.width();
	const int height = image.height();	
	const auto items = width*height;	
	
	const unsigned char* input_color_image = image.data();	
	
	if (input_color_image == nullptr) {
		std::cerr<< "Load image:\t" << img << " failed, fixme!"<< std::endl;
		exit(-1);
	}
	
#ifdef _DUMP_
	std::cout << "Initialize ..." << std::endl;
#endif
	
	coopcl::virtual_device device;

	static std::stringstream tasks;
	tasks << SIFT_tasks::task_defines;
	tasks << SIFT_tasks::task_Color;		//kColor
	tasks << SIFT_tasks::task_Down;			//kDown
	tasks << SIFT_tasks::task_Blur;			//kBlur
	tasks << SIFT_tasks::task_Diff;			//kDiff
	tasks << SIFT_tasks::task_Detector;		//kDetector
	tasks << SIFT_tasks::task_Reset;		//kReset

	std::vector<std::unique_ptr<sift_octave>> octaves;

	octaves.push_back(std::make_unique<sift_octave>(0, width, height, tasks.str(), device, input_color_image));	
	octaves.push_back(std::make_unique<sift_octave>(1, width, height, tasks.str(), device));
	octaves.push_back(std::make_unique<sift_octave>(2, width, height, tasks.str(), device));
	octaves.push_back(std::make_unique<sift_octave>(3, width, height, tasks.str(), device));
	octaves.push_back(std::make_unique<sift_octave>(4, width, height, tasks.str(), device));
		
	//Call task_graph
#ifdef _DUMP_
	std::cout << "Execute ..." << std::endl;
#endif

	const int iter = 100;					

	//std::vector<float> offloads;
	//generate_offload_range(offloads, 0.25f);
	//offloads.insert(offloads.begin(),-1.0f);	
	//pause for GPU-driver to increase pwr_measure accuracy!
	std::this_thread::sleep_for(std::chrono::seconds(2));

	const auto app_init_end = std::chrono::system_clock::now();
	const auto et = std::chrono::duration_cast<std::chrono::milliseconds>(app_init_end - app_init_start).count();

#ifndef _DUMP_
	std::cout<< et << std::endl;
#endif

	//for (const auto offload : offloads)
	{
		long avg_et = 0; int err = 0;
		for (int i = 0; i < iter; i++)
		{
			const auto start = std::chrono::system_clock::now();
			
			for (size_t oct_id = 0;oct_id<octaves.size();oct_id++)
			{
				if (oct_id == 0)
				{
					err = octaves[oct_id]->call_async(offload, device);
					if (err != 0) return err;
				}
				else
				{
					err = octaves[oct_id]->call_async(offload, device, octaves[oct_id - 1]->imgBlur(), octaves[oct_id - 1]->task_Blur());
					if (err != 0) return err;
				}				
			}			

			for (auto const& octave : octaves)
			{
				err = octave->wait();
				if (err != 0) return err;
				//octave->show();
			}			

			const auto end = std::chrono::system_clock::now();
			const auto et = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

			if(i>0) avg_et += et;
			//std::cout << "Elapsed time:\t" << et << " msec " << std::endl;			
		}
#ifdef _DUMP_
		std::cout << "--------------------------------------------------------" << std::endl;
		std::cout << "Elapsed mean time:\t" << avg_et / iter << " msec \t offload: " << offload << std::endl;
		std::cout << "Elapsed_time_ms," << avg_et << std::endl;
		std::cout << "--------------------------------------------------------" << std::endl;
#else
		std::cout << "Elapsed_time_ms," << avg_et << std::endl;
		std::cout << "Offload," << offload << std::endl;

		const std::string path = "c:/Users/morkon/Sift/";
		size_t ocatve_id = 0;
		for (auto const& octave : octaves)
		{
			auto ret = octave->store_task_records_in_files(path,ocatve_id++);
			for (auto f : ret)
				std::cout << "Stored:\t" << f << std::endl;
		}
#endif

	}	
	return 0;
}
