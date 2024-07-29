/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include "forward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;
#include <cmath>


 __device__ float re_exp(float power) 
{
   float re_power = power  * powf(2.0f,  20) ; 
   int   int_temp = (int)re_power;

  // printf("re_power:%f\n",re_power);
  // printf("int_re_power:%d\n",int_re_power);
   float   re_power2 = int_temp * 1.4375f / powf(2.0f,  20);//硬件真实值    //1.4375 1.44140625 1.442695041
   int   int_re_power2 = (int)re_power2;//硬件真实整数值
   float dec_re_power = re_power2 - int_re_power2;//硬件真实小数值

   float dec_out;
   float int_out;
   //printf("dec_re_power:%f\n",dec_re_power);
   //decimalPart
   if (dec_re_power>-0.102f)
   {
         dec_out = dec_re_power * (0.6694407239556313f) + 0.9997041821479797f; //0.669439  0.999703
   }
   else if(dec_re_power>-0.208f&&dec_re_power<-0.102f)
   {
		 dec_out = dec_re_power * (0.6228911653161049f) + 0.9949778467416763f; //0.622890 0.994976
   }
   else if(dec_re_power>-0.318f&&dec_re_power<-0.208f)
   {
		 dec_out = dec_re_power * (0.5779740959405899f) + 0.9856577143073082f;//0.577973 0.985657
   }
   else if(dec_re_power>-0.432f&&dec_re_power<-0.318f)
   {
		 dec_out = dec_re_power * (0.5348114967346191f) + 0.9719554036855698f;//
   }
   else if(dec_re_power>-0.551f&&dec_re_power<-0.432f)
   {
		 dec_out = dec_re_power * (0.49333369731903076f) + 0.9540561139583588f;//
   }
   else if(dec_re_power>-0.675f&&dec_re_power<-0.551f)
   {
		 dec_out = dec_re_power * (0.4534987658262253f)+ 0.9321274757385254f;//
   }
	else if(dec_re_power>-0.805f&&dec_re_power<-0.675f)
   {
		 dec_out = dec_re_power * (0.41529666632413864f) + 0.9063581079244614f;//
   }
	 else if(dec_re_power>-0.941f&&dec_re_power<-0.805f)
   {
		 dec_out = dec_re_power * (0.3787348121404648f) + 0.8769446089863777f;//
   }
    else if(dec_re_power>-1.0f&&dec_re_power<-0.941f)
   {
		 dec_out = dec_re_power * (0.3537578657269478f) + 0.853704534471035f;//0.353756 0.8537034988
   }
     int_out = powf(2.0f,  int_re_power2) * dec_out;
    return  int_out;
}  

 
/*
__device__ float re_exp(float power) 
{
   float re_power = power  * powf(2.0f,  27) ; 
   int   int_temp = (int)re_power;

  // printf("re_power:%f\n",re_power);
  // printf("int_re_power:%d\n",int_re_power);
   float   re_power2 = int_temp * 1.44140625f / powf(2.0f,  27);//硬件真实值    //1.4375 1.44140625 1.442695041
   int   int_re_power2 = (int)re_power2;//硬件真实整数值
   float dec_re_power = re_power2 - int_re_power2;//硬件真实小数值

   float dec_out;
   float int_out;
   //printf("dec_re_power:%f\n",dec_re_power);
   //decimalPart
   if (dec_re_power>-0.083f)
   {
         dec_out = dec_re_power * (0.673816573204804f) + 0.999803733741375f; //0.669439  0.999703
   }
   else if(dec_re_power>-0.169f&&dec_re_power<-0.083f)
   {
		 dec_out = dec_re_power * (0.635490422326511f) + 0.996639235099418f; //0.622890 0.994976
   }
   else if(dec_re_power>-0.257f&&dec_re_power<-0.169f)
   {
		 dec_out = dec_re_power * (0.598304667005414f) + 0.990376315120686f;//0.577973 0.985657
   }
   else if(dec_re_power>-0.348f&&dec_re_power<-0.257f)
   {
		 dec_out = dec_re_power * (0.562321672700664f) + 0.981145606650053f;//
   }
   else if(dec_re_power>-0.442f&&dec_re_power<-0.348f)
   {
		 dec_out = dec_re_power * (0.527405094186453f) + 0.969011893486804f;//
   }
   else if(dec_re_power>-0.539f&&dec_re_power<-0.442f)
   {
		 dec_out = dec_re_power * (0.493629248880420f)+ 0.954100494419588f;//
   }
	else if(dec_re_power>-0.640f&&dec_re_power<-0.539f)
   {
		 dec_out = dec_re_power * (0.460898963006358f) + 0.936472692578689f;//
   }
	 else if(dec_re_power>-0.744f&&dec_re_power<-0.640f)
   {
		 dec_out = dec_re_power * (0.429294624787000f) + 0.916264221632452f;//
   }
    else if(dec_re_power>-0.852f&&dec_re_power<-0.774f)
   {
		 dec_out = dec_re_power * (0.398890390053833f) + 0.893658215137066f;//0.353756 0.8537034988
   }
    else if(dec_re_power>-0.964f&&dec_re_power<-0.852f)
   {
		 dec_out = dec_re_power * (0.369613524899934f) + 0.868729627323664f;//
   }
    else if(dec_re_power>-1.0f&&dec_re_power<-0.964f)
   {
		 dec_out = dec_re_power * (0.350933859563803f) + 0.850914156665352f;//0.353756 0.8537034988
   }
     int_out = powf(2.0f,  int_re_power2) * dec_out;
    return  int_out;
} 

*/


// Forward method for converting the input spherical harmonics
// coefficients of each Gaussian to a simple RGB color.
__device__ glm::vec3 computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, bool* clamped)
{
	// The implementation is loosely based on code for 
	// "Differentiable Point-Based Radiance Fields for 
	// Efficient View Synthesis" by Zhang et al. (2022)
	glm::vec3 pos = means[idx];
	glm::vec3 dir = pos - campos;
	dir = dir / glm::length(dir);//view direction

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;
	glm::vec3 result = SH_C0 * sh[0];

	if (deg > 0)
	{
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
		result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
			result = result +
				SH_C2[0] * xy * sh[4] +
				SH_C2[1] * yz * sh[5] +
				SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
				SH_C2[3] * xz * sh[7] +
				SH_C2[4] * (xx - yy) * sh[8];

			if (deg > 2)
			{
				result = result +
					SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
					SH_C3[1] * xy * z * sh[10] +
					SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
					SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
					SH_C3[5] * z * (xx - yy) * sh[14] +
					SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
			}
		}
	}
	result += 0.5f;

	// RGB colors are clamped to positive values. If values are
	// clamped, we need to keep track of this for the backward pass.
	clamped[3 * idx + 0] = (result.x < 0);
	clamped[3 * idx + 1] = (result.y < 0);
	clamped[3 * idx + 2] = (result.z < 0);
	return glm::max(result, 0.0f);
}

// Forward version of 2D covariance matrix computation
__device__ float3 computeCov2D(const float3& mean, float focal_x, float focal_y, float tan_fovx, float tan_fovy, const float* cov3D, const float* viewmatrix)
{
	// The following models the steps outlined by equations 29
	// and 31 in "EWA Splatting" (Zwicker et al., 2002). 
	// Additionally considers aspect / scaling of viewport.
	// Transposes used to account for row-/column-major conventions.
	float3 t = transformPoint4x3(mean, viewmatrix);

	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;

	glm::mat3 J = glm::mat3(
		focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
		0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
		0, 0, 0);

	glm::mat3 W = glm::mat3(
		viewmatrix[0], viewmatrix[4], viewmatrix[8],
		viewmatrix[1], viewmatrix[5], viewmatrix[9],
		viewmatrix[2], viewmatrix[6], viewmatrix[10]);

	glm::mat3 T = W * J;

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;

	// Apply low-pass filter: every Gaussian should be at least
	// one pixel wide/high. Discard 3rd row and column.
	cov[0][0] += 0.3f;
	cov[1][1] += 0.3f;
	return { float(cov[0][0]), float(cov[0][1]), float(cov[1][1]) };
}

// Forward method for converting scale and rotation properties of each
// Gaussian to a 3D covariance matrix in world space. Also takes care
// of quaternion normalization.
__device__ void computeCov3D(const glm::vec3 scale, float mod, const glm::vec4 rot, float* cov3D)
{
	// Create scaling matrix//尺度变换
	glm::mat3 S = glm::mat3(1.0f);
	S[0][0] = mod * scale.x;
	S[1][1] = mod * scale.y;
	S[2][2] = mod * scale.z;

	// Normalize quaternion to get valid rotation
	glm::vec4 q = rot;// / glm::length(rot);//旋转矩阵
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	// Compute rotation matrix from quaternion//从四元数计算旋转矩阵
	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 M = S * R;//m是变化矩阵

	// Compute 3D world covariance matrix Sigma
	glm::mat3 Sigma = glm::transpose(M) * M;

	// Covariance is symmetric, only store upper right协方差矩阵是上三角矩阵
	cov3D[0] = Sigma[0][0];
	cov3D[1] = Sigma[0][1];
	cov3D[2] = Sigma[0][2];
	cov3D[3] = Sigma[1][1];
	cov3D[4] = Sigma[1][2];
	cov3D[5] = Sigma[2][2];
}

// Perform initial steps for each Gaussian prior to rasterization.
template<int C>
__global__ void preprocessCUDA(int P, int D, int M,
	const float* orig_points,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* cov3D_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float tan_fovx, float tan_fovy,
	const float focal_x, float focal_y,
	int* radii,
	float2* points_xy_image,
	float* depths,
	float* cov3Ds,
	float* rgb,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Initialize radius and touched tiles to 0. If this isn't changed,
	// this Gaussian will not be processed further.
	radii[idx] = 0;
	tiles_touched[idx] = 0;

	// Perform near culling, quit if outside.
	float3 p_view;
	if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view))
		return;

	// Transform point by projecting
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };//高斯中心位置
	float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	float p_w = 1.0f / (p_hom.w + 0.0000001f);
	float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };

	// If 3D covariance matrix is precomputed, use it, otherwise compute
	// from scaling and rotation parameters. 
	const float* cov3D;
	if (cov3D_precomp != nullptr)
	{
		cov3D = cov3D_precomp + idx * 6;
	}
	else
	{
		computeCov3D(scales[idx], scale_modifier, rotations[idx], cov3Ds + idx * 6);//计算椭球的3个轴
		cov3D = cov3Ds + idx * 6;
	}

	// Compute 2D screen-space covariance matrix
	float3 cov = computeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix);

	// Invert covariance (EWA algorithm)
	float det = (cov.x * cov.z - cov.y * cov.y);
	if (det == 0.0f)
		return;
	float det_inv = 1.f / det;
	float3 conic = { cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv };

	// Compute extent in screen space (by finding eigenvalues of
	// 2D covariance matrix). Use extent to compute a bounding rectangle
	// of screen-space tiles that this Gaussian overlaps with. Quit if
	// rectangle covers 0 tiles. 
	float mid = 0.5f * (cov.x + cov.z);
	float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));//椭圆长轴半径
	float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));//椭圆短轴半径
	float my_radius = ceil(3.f * sqrt(max(lambda1, lambda2)));//用椭圆长轴代替，近似成一个圆//3.f是3个标准差范围，3个标准差覆盖99.7%，尽可能覆盖
	float2 point_image = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) };
	uint2 rect_min, rect_max;
	getRect(point_image, my_radius, rect_min, rect_max, grid);//计算近似的圆覆盖的tile大小
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
		return;

	// If colors have been precomputed, use them, otherwise convert
	// spherical harmonics coefficients to RGB color.
	if (colors_precomp == nullptr)
	{
		glm::vec3 result = computeColorFromSH(idx, D, M, (glm::vec3*)orig_points, *cam_pos, shs, clamped);
		rgb[idx * C + 0] = result.x;
		rgb[idx * C + 1] = result.y;
		rgb[idx * C + 2] = result.z;
	}

	// Store some useful helper data for the next steps.
	depths[idx] = p_view.z;
	radii[idx] = my_radius;
	points_xy_image[idx] = point_image;
	// Inverse 2D covariance and opacity neatly pack into one float4
	conic_opacity[idx] = { conic.x, conic.y, conic.z, opacities[idx] };
	tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
}

// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching 
// and rasterizing data.
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float2* __restrict__ points_xy_image,
	const float* __restrict__ features,
	const float4* __restrict__ conic_opacity,
	float* __restrict__ final_T,
	uint32_t* __restrict__ n_contrib,
	const float* __restrict__ bg_color,
	float* __restrict__ out_color)
{
	//printf("features:%f\n",features);
	// Identify current tile and associated min/max pixel range.//识别目前的tile和相关的最小最大像素范围
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;//计算水平方向上线程块的数量， 偏移量确保整个个图象被覆盖到
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };//线程块在网格中的 x 方向索引
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };//不超出图像边界，所以是最大像素范围，防止造成无效的像素访问
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };//block长度+其中的thread索引对应的长度，确定具体的像素处理范围
	uint32_t pix_id = W * pix.y + pix.x;//表示了像素在一维像素数组中的位置

	float2 pixf = { (float)pix.x, (float)pix.y };//pix的浮点数版本


	// Check if this thread is associated with a valid pixel or outside.//看看我负责的像素有没有跑到图像外面去
	bool inside = pix.x < W&& pix.y < H;
	//printf("H:%d\n",H);
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;//负责的像素还在图像里的话那就还没做完，没有done

	// Load start/end range of IDs to process in bit sorted list.
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];//range是一个数组
    //这段代码的目的是根据当前线程块的索引查找 ranges 数组中相应的像素范围，以便线程块可以处理正确的像素。
	//像素处理范围
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	
	// 我要把任务分成rounds批，每批处理BLOCK_SIZE个Gaussians
	// 每一批，每个线程负责读取一个Gaussian的信息，
	// 所以该block的256个线程每一批就可以读取256个Gaussian的信息
	

	int toDo = range.y - range.x;
	// 我要处理的Gaussian个数
    
	// Allocate storage for batches of collectively fetched data.
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];

	// Initialize helper variables
	float T = 1.0f;//T是透过率，穿过越多gaussian就越小，小到一定程度就提前终止，contributor记录经过了多少gaussian

	uint32_t contributor = 0;//多少个Gaussian对该像素的颜色有贡献，经过了多少高斯点
	uint32_t last_contributor = 0; // 比contributor慢半拍的变量
	float C[CHANNELS] = { 0 };

// Iterate over batches until all done or range is complete对批处理进行迭代，直到全部完成或范围完成
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)//rounds每增加1，当前处理的高斯个数递减一个BLOCK_SIZE
	{
		// End if entire block votes that it is done rasterizing
		int num_done = __syncthreads_count(done);//同步线程块内的所有线程，并统计那些标记为 done 的线程数量。
		if (num_done == BLOCK_SIZE)//等全部线程完成后
			break;

		// Collectively fetch per-Gaussian data from global to shared
		int progress = i * BLOCK_SIZE + block.thread_rank();//当前线程在整个数据处理中的位置 第i轮次的高斯点数量加线程数
		if (range.x + progress < range.y)//如果我要处理的高斯点数量大于累计处理完的高斯点数量，那么就继续
		{
			int coll_id = point_list[range.x + progress];//包含所有高斯点的列表。
			collected_id[block.thread_rank()] = coll_id;//当前线程块内线程收集的高斯点的 ID 数组
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];//所有高斯点在图像中的坐标
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];//所有高斯点的透明度数组。
		}//全部放入共享内存中
		block.sync();//同步当前线程块中的所有线程。

		// Iterate over current batch
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current position in range
			contributor++;

			// Resample using conic matrix (cf. "Surface 
			// Splatting" by Zwicker et al., 2001)
			float2 xy = collected_xy[j];
			float2 d = { xy.x - pixf.x, xy.y - pixf.y };// 该像素到Gaussian中心的位移向量
			float4 con_o = collected_conic_opacity[j];
			float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;//高斯点在高斯分布中的几率
			if (power > 0.0f)
				continue;

			// Eq. (2) from 3D Gaussian splatting paper.
			// Obtain alpha by multiplying with Gaussian opacity
			// and its exponential falloff from mean.
			// Avoid numerical instabilities (see paper appendix). 
			//float alpha = min(0.99f, con_o.w * exp(power));
            float alpha = min(0.99f, con_o.w * re_exp(power));//（高斯分布中心点最大值）alpha值与高斯点的分布几率，得到衰减后的高斯alpha值//如果高斯分布几率越小，则不透明度越低
			// Gaussian对于这个像素点来说的不透明度
				// 注意con_o.w是”opacity“，是Gaussian整体的不透明度
            //简而言之，像素离高斯中心越远就越透明
            //printf("alpha:%f\n",alpha);

			if (alpha < 1.0f / 255.0f)//太小就当成透明
				continue;
			float test_T = T * (1 - alpha);
			float test_T_1 = T * (1 - alpha);
            printf("test_T_1:%f\n",test_T_1);


			if (test_T < 0.0001f)// 当透光率很低的时候，就不继续渲染了（反正渲染了也看不见）
			{
				done = true;
				continue;
			}
            
			// Eq. (3) from 3D Gaussian splatting paper.其中c𝑖是每个点的颜色，
			//𝛼𝑖是通过评估具有协方差的二维高斯Σ而给出的[Yifan et al.2019]乘以习得的逐点不透明度。
			for (int ch = 0; ch < CHANNELS; ch++)
				C[ch] += features[collected_id[j] * CHANNELS + ch] * alpha * T;// 计算当前Gaussian对像素颜色的贡献
			T = test_T;

			// Keep track of last range entry to update this
			// pixel.
			last_contributor = contributor;
         

		}
	}

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)
	{
		final_T[pix_id] = T;
		n_contrib[pix_id] = last_contributor;
		for (int ch = 0; ch < CHANNELS; ch++)
			out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];// 把渲染出来的像素值写进out_color
	}
}



























void FORWARD::render(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float2* means2D,
	const float* colors,
	const float4* conic_opacity,
	float* final_T,
	uint32_t* n_contrib,
	const float* bg_color,
	float* out_color)
{
	renderCUDA<NUM_CHANNELS> << <grid, block >> > (
		ranges,
		point_list,
		W, H,
		means2D,
		colors,
		conic_opacity,
		final_T,
		n_contrib,
		bg_color,
		out_color);
}

void FORWARD::preprocess(
		int P, //高斯点数量
		int D,//高斯分布的维度
		int M,//点云数量
	const float* means3D,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* cov3D_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	int* radii,
	float2* means2D,
	float* depths,
	float* cov3Ds,
	float* rgb,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered)
{
	preprocessCUDA<NUM_CHANNELS> << <(P + 255) / 256, 256 >> > (
		P, D, M,
		means3D,
		scales,
		scale_modifier,
		rotations,
		opacities,
		shs,
		clamped,
		cov3D_precomp,
		colors_precomp,
		viewmatrix, 
		projmatrix,
		cam_pos,
		W, H,
		tan_fovx, tan_fovy,
		focal_x, focal_y,
		radii,
		means2D,
		depths,
		cov3Ds,
		rgb,
		conic_opacity,
		grid,
		tiles_touched,
		prefiltered
		);
}