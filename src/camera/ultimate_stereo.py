"""
ULTIMATE OAK-D Stereo Depth System - ADVANCED ZERO-NOISE v5.3.0
Integrated into SLAM system.
"""

import cv2
import numpy as np
import time
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass
from ..utils.cupy_utils import cupy_manager
from ..utils.logger import get_logger

try:
    import cupy as cp
    HAS_CUPY = True
    from cupyx.scipy import ndimage as ndi
    from cupyx.scipy.ndimage import binary_opening, grey_dilation
except ImportError:
    HAS_CUPY = False
    cp = np
    ndi = None
    binary_opening = None
    grey_dilation = None

# xp remains managed by cupy_manager for general SLAM compatibility
xp = cupy_manager.get_array_module()

logger = get_logger(__name__)

# ===========================
# GPU-ACCELERATED CUDA Kernels
# ===========================

if HAS_CUPY:
    # Census Transform Kernel
    CENSUS_TRANSFORM_KERNEL = cp.RawKernel(r'''
    extern "C" __global__
    void census_transform_kernel(
        const float* img, unsigned long long* census,
        int height, int width, int half_window
    ) {
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        
        if (y >= height || x >= width) return;
        
        float center = img[y * width + x];
        unsigned long long census_val = 0;
        int bit_pos = 0;
        
        for (int dy = -half_window; dy <= half_window; dy++) {
            for (int dx = -half_window; dx <= half_window; dx++) {
                if (dx == 0 && dy == 0) continue;
                
                int ny = y + dy;
                int nx = x + dx;
                
                if (ny >= 0 && ny < height && nx >= 0 && nx < width) {
                    float neighbor = img[ny * width + nx];
                    if (neighbor >= center) {
                        census_val |= (1ULL << bit_pos);
                    }
                }
                bit_pos++;
                if (bit_pos >= 64) break;
            }
            if (bit_pos >= 64) break;
        }
        
        census[y * width + x] = census_val;
    }
    ''', 'census_transform_kernel')

    # Enhanced Fused Kernel with CUBIC SPLINE
    ENHANCED_FUSED_KERNEL_SPLINE = cp.RawKernel(r'''
    extern "C" __global__
    void enhanced_fused_patchmatch_lrc_spline_kernel(
        const float* left, const float* right,
        const float* grad_x_l, const float* grad_y_l,
        const float* grad_x_r, const float* grad_y_r,
        const float* grad_magnitude,
        const unsigned long long* census_l, const unsigned long long* census_r,
        float* disp_l, float* disp_r, float* confidence,
        float* costs_buffer,
        int height, int width, int num_disp, int patch_size,
        float lr_threshold_near, float lr_threshold_far,
        float census_weight, int use_census, int use_spline
    ) {
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        
        if (y >= height || x >= width) return;
        
        int half_patch = patch_size / 2;
        int idx = y * width + x;
        
        float costs[64];
        float best_cost = 1e10f;
        int best_disp = 0;
        
        for (int d = 0; d < num_disp && d < 64; d++) {
            int x_r = x - d;
            if (x_r < 0 || x_r >= width) {
                costs[d] = 1e10f;
                continue;
            }
            
            float cost_intensity = 0.0f;
            float cost_gradient = 0.0f;
            float cost_census = 0.0f;
            int valid_pixels = 0;
            
            for (int dy = -half_patch; dy <= half_patch; dy++) {
                for (int dx = -half_patch; dx <= half_patch; dx++) {
                    int py = y + dy;
                    int px_l = x + dx;
                    int px_r = x_r + dx;
                    
                    if (py < 0 || py >= height || px_l < 0 || px_l >= width || 
                        px_r < 0 || px_r >= width) continue;
                    
                    int idx_l = py * width + px_l;
                    int idx_r = py * width + px_r;
                    
                    float diff_i = left[idx_l] - right[idx_r];
                    cost_intensity += fabsf(diff_i);
                    
                    float diff_gx = grad_x_l[idx_l] - grad_x_r[idx_r];
                    float diff_gy = grad_y_l[idx_l] - grad_y_r[idx_r];
                    cost_gradient += sqrtf(diff_gx * diff_gx + diff_gy * diff_gy);
                    
                    if (use_census) {
                        unsigned long long xor_val = census_l[idx_l] ^ census_r[idx_r];
                        int hamming = __popcll(xor_val);
                        cost_census += hamming;
                    }
                    
                    valid_pixels++;
                }
            }
            
            if (valid_pixels > 0) {
                float total_cost = (cost_intensity + cost_gradient) / valid_pixels;
                if (use_census) {
                    total_cost += (cost_census / valid_pixels) * census_weight;
                }
                costs[d] = total_cost;
                costs_buffer[idx * 64 + d] = total_cost;
                
                if (total_cost < best_cost) {
                    best_cost = total_cost;
                    best_disp = d;
                }
            } else {
                costs[d] = 1e10f;
                costs_buffer[idx * 64 + d] = 1e10f;
            }
        }
        
        float refined_disp = (float)best_disp;
        if (use_spline && best_disp > 1 && best_disp < num_disp - 2 && best_disp < 62) {
            float c1 = costs[best_disp - 1];
            float c2 = costs[best_disp];
            float c3 = costs[best_disp + 1];
            float denom = 2.0f * (c1 - 2.0f * c2 + c3);
            if (fabsf(denom) > 1e-6f) {
                float offset = (c1 - c3) / denom;
                refined_disp = best_disp + fmaxf(-0.5f, fminf(0.5f, offset));
            }
        }
        
        float initial_conf = 1.0f / (1.0f + best_cost);
        int x_r = x - (int)(refined_disp + 0.5f);
        float right_disp = 0.0f;
        
        if (x_r >= 0 && x_r < width) {
            float best_r_cost = 1e10f;
            int best_r_disp = 0;
            for (int d = 0; d < num_disp && d < 64; d++) {
                int x_l_check = x_r + d;
                if (x_l_check < 0 || x_l_check >= width) continue;
                float cost = 0.0f;
                int count = 0;
                for (int dy = -half_patch; dy <= half_patch; dy++) {
                    for (int dx = -half_patch; dx <= half_patch; dx++) {
                        int py = y + dy;
                        int px_r_check = x_r + dx;
                        int px_l_check = x_l_check + dx;
                        if (py >= 0 && py < height && px_r_check >= 0 && px_r_check < width &&
                            px_l_check >= 0 && px_l_check < width) {
                            float diff = right[py * width + px_r_check] - left[py * width + px_l_check];
                            cost += fabsf(diff);
                            count++;
                        }
                    }
                }
                if (count > 0) {
                    cost /= count;
                    if (cost < best_r_cost) {
                        best_r_cost = cost;
                        best_r_disp = d;
                    }
                }
            }
            right_disp = (float)best_r_disp;
        }
        
        float norm_disp = refined_disp / (float)num_disp;
        float lr_threshold = lr_threshold_near + (lr_threshold_far - lr_threshold_near) * (1.0f - norm_disp);
        float consistency_error = fabsf(refined_disp - right_disp);
        
        if (consistency_error > lr_threshold || refined_disp < 0.5f) {
            disp_l[idx] = 0.0f;
            confidence[idx] = 0.0f;
        } else {
            disp_l[idx] = refined_disp;
            confidence[idx] = initial_conf * expf(-consistency_error / lr_threshold);
        }
        
        if (x_r >= 0 && x_r < width) {
            atomicExch(&disp_r[y * width + x_r], right_disp);
        }
    }
    ''', 'enhanced_fused_patchmatch_lrc_spline_kernel')

    # Bidirectional LRC Check Kernels
    LR_CHECK_LEFT_REF_KERNEL = cp.RawKernel(r'''
    extern "C" __global__
    void lr_check_left_ref_kernel(
        const float* disp_l, const float* disp_r,
        float* confidence, float* output_disp,
        unsigned char* validity_mask,
        int height, int width, float threshold_near, float threshold_far, 
        float max_disp, int adaptive_enabled
    ) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= width || y >= height) return;
        int idx = y * width + x;
        float d_l = disp_l[idx];
        validity_mask[idx] = 0;
        if (d_l < 0.1f) { output_disp[idx] = d_l; return; }
        float adaptive_threshold = adaptive_enabled ? (threshold_far * (1.0f - d_l/max_disp) + threshold_near * (d_l/max_disp)) : threshold_near;
        int x_r = (int)(x - d_l + 0.5f);
        if (x_r >= 0 && x_r < width) {
            float d_r = disp_r[y * width + x_r];
            if (d_r > 0.1f) {
                int x_l_check = (int)(x_r + d_r + 0.5f);
                if (x_l_check >= 0 && x_l_check < width) {
                    float diff = fabsf((float)x - (float)x_l_check);
                    if (diff <= adaptive_threshold) {
                        output_disp[idx] = d_l;
                        validity_mask[idx] = 1;
                        confidence[idx] *= expf(-(diff*diff)/(2.0f*adaptive_threshold*adaptive_threshold));
                    } else { output_disp[idx] = 0.0f; confidence[idx] = 0.0f; }
                } else { output_disp[idx] = d_l; confidence[idx] *= 0.8f; }
            } else { output_disp[idx] = d_l; confidence[idx] *= 0.7f; }
        } else { output_disp[idx] = d_l; confidence[idx] *= 0.7f; }
    }
    ''', 'lr_check_left_ref_kernel')

    LR_CHECK_RIGHT_REF_KERNEL = cp.RawKernel(r'''
    extern "C" __global__
    void lr_check_right_ref_kernel(
        const float* disp_l, const float* disp_r,
        float* confidence, float* output_disp,
        unsigned char* validity_mask,
        int height, int width, float threshold_near, float threshold_far,
        float max_disp, int adaptive_enabled
    ) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= width || y >= height) return;
        int idx = y * width + x;
        float d_r = disp_r[idx];
        validity_mask[idx] = 0;
        if (d_r < 0.1f) { output_disp[idx] = d_r; return; }
        float adaptive_threshold = adaptive_enabled ? (threshold_far * (1.0f - d_r/max_disp) + threshold_near * (d_r/max_disp)) : threshold_near;
        int x_l = (int)(x + d_r + 0.5f);
        if (x_l >= 0 && x_l < width) {
            float d_l = disp_l[y * width + x_l];
            if (d_l > 0.1f) {
                int x_r_check = (int)(x_l - d_l + 0.5f);
                if (x_r_check >= 0 && x_r_check < width) {
                    float diff = fabsf((float)x - (float)x_r_check);
                    if (diff <= adaptive_threshold) {
                        output_disp[idx] = d_r;
                        validity_mask[idx] = 1;
                        confidence[idx] *= expf(-(diff*diff)/(2.0f*adaptive_threshold*adaptive_threshold));
                    } else { output_disp[idx] = 0.0f; confidence[idx] = 0.0f; }
                } else { output_disp[idx] = d_r; confidence[idx] *= 0.8f; }
            } else { output_disp[idx] = d_r; confidence[idx] *= 0.7f; }
        } else { output_disp[idx] = d_r; confidence[idx] *= 0.7f; }
    }
    ''', 'lr_check_right_ref_kernel')

    MERGE_LR_RESULTS_KERNEL = cp.RawKernel(r'''
    extern "C" __global__
    void merge_lr_results_confidence_kernel(
        const float* disp_lr_left, const unsigned char* valid_left,
        const float* disp_lr_right, const unsigned char* valid_right,
        const float* conf_left, const float* conf_right,
        float* merged_disp, float* merged_conf,
        int height, int width, int use_confidence_blending
    ) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= width || y >= height) return;
        int idx = y * width + x;
        unsigned char v_l = valid_left[idx], v_r = valid_right[idx];
        float c_l = conf_left[idx], c_r = conf_right[idx];
        if (v_l && v_r) {
            float tc = c_l + c_r;
            merged_disp[idx] = tc > 1e-6f ? (disp_lr_left[idx]*c_l + disp_lr_right[idx]*c_r)/tc : disp_lr_left[idx];
            merged_conf[idx] = tc * 0.5f;
        } else if (v_l) {
            merged_disp[idx] = disp_lr_left[idx]; merged_conf[idx] = c_l;
        } else if (v_r) {
            float d_r = disp_lr_right[idx];
            if (d_r > 0.1f) {
                int x_l = (int)(x + d_r + 0.5f);
                if (x_l >= 0 && x_l < width) {
                    atomicMax((int*)&merged_disp[y * width + x_l], __float_as_int(d_r));
                    atomicMax((int*)&merged_conf[y * width + x_l], __float_as_int(c_r));
                }
            }
        } else { merged_disp[idx] = 0.0f; merged_conf[idx] = 0.0f; }
    }
    ''', 'merge_lr_results_confidence_kernel')

    BILATERAL_FILTER_SMEM_KERNEL = cp.RawKernel(r'''
    extern "C" __global__
    void bilateral_filter_smem_kernel(
        const float* disp_in, const float* guide_img, 
        const float* grad_magnitude,
        float* disp_out,
        int height, int width, int radius, 
        float sigma_color, float sigma_space,
        float edge_threshold, int use_gradient_weighting,
        int use_depth_sigma_scaling, float sigma_near_scale, float sigma_far_scale,
        float max_disp
    ) {
        extern __shared__ float smem[];
        int r = radius, sw = blockDim.x + 2*r, sh = blockDim.y + 2*r;
        float* g_p = smem;
        float* d_p = &smem[sw*sh];
        float* gr_p = &smem[2*sw*sh];
        int x = blockIdx.x*blockDim.x + threadIdx.x, y = blockIdx.y*blockDim.y + threadIdx.y;
        int lx = threadIdx.x + r, ly = threadIdx.y + r;
        for(int dy=-r; dy<=r; dy+=blockDim.y) {
            for(int dx=-r; dx<=r; dx+=blockDim.x) {
                int sy = threadIdx.y+dy+r, sx = threadIdx.x+dx+r;
                if(sy>=0 && sy<sh && sx>=0 && sx<sw) {
                    int cy = max(0, min(height-1, (int)(blockIdx.y*blockDim.y)+sy-r));
                    int cx = max(0, min(width-1, (int)(blockIdx.x*blockDim.x)+sx-r));
                    g_p[sy*sw+sx] = guide_img[cy*width+cx];
                    d_p[sy*sw+sx] = disp_in[cy*width+cx];
                    if(use_gradient_weighting) gr_p[sy*sw+sx] = grad_magnitude[cy*width+cx];
                }
            }
        }
        __syncthreads();
        if(y>=height || x>=width) return;
        float cd = d_p[ly*sw+lx];
        if(cd < 0.1f) { disp_out[y*width+x] = 0.0f; return; }
        float ci = g_p[ly*sw+lx], cg = use_gradient_weighting ? gr_p[ly*sw+lx] : 0.0f;
        float sss = sigma_space * (use_depth_sigma_scaling ? (sigma_near_scale + (sigma_far_scale-sigma_near_scale)*(1.0f-fminf(1.0f, cd/max_disp))) : 1.0f);
        float swd = 0.0f, tw = 0.0f, cc = -1.0f/(2.0f*sigma_color*sigma_color), sc = -1.0f/(2.0f*sss*sss);
        for(int dy=-r; dy<=r; dy++) {
            for(int dx=-r; dx<=r; dx++) {
                float nd = d_p[(ly+dy)*sw+(lx+dx)];
                if(nd < 0.1f) continue;
                float ni = g_p[(ly+dy)*sw+(lx+dx)], ng = use_gradient_weighting ? gr_p[(ly+dy)*sw+(lx+dx)] : 0.0f;
                float w = expf((float)(dx*dx+dy*dy)*sc) * expf((ni-ci)*(ni-ci)*cc);
                if(use_gradient_weighting && fmaxf(cg, ng) > edge_threshold) w *= expf(-(fmaxf(cg, ng)-edge_threshold)/edge_threshold);
                swd += nd*w; tw += w;
            }
        }
        disp_out[y*width+x] = tw > 0.001f ? swd/tw : cd;
    }
    ''', 'bilateral_filter_smem_kernel')

    TEXTURELESS_DETECTION_KERNEL = cp.RawKernel(r'''
    extern "C" __global__
    void textureless_detection_kernel(
        const float* grad_magnitude, const float* confidence,
        unsigned char* texture_mask,
        int height, int width, float grad_threshold, float conf_threshold
    ) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= width || y >= height) return;
        int idx = y * width + x;
        if (grad_magnitude[idx] < grad_threshold && confidence[idx] > conf_threshold) {
            int tc = 0, tt = 0;
            for(int dy=-2; dy<=2; dy++) for(int dx=-2; dx<=2; dx++) {
                int nx=x+dx, ny=y+dy;
                if(nx>=0 && nx<width && ny>=0 && ny<height) { if(grad_magnitude[ny*width+nx] < grad_threshold) tc++; tt++; }
            }
            texture_mask[idx] = (tc >= tt*0.7f) ? 1 : 0;
        } else texture_mask[idx] = 0;
    }
    ''', 'textureless_detection_kernel')

    WEIGHTED_RANSAC_PLANE_FITTING_KERNEL = cp.RawKernel(r'''
    extern "C" __global__
    void weighted_ransac_plane_fitting_kernel(
        const float* depth, const unsigned char* texture_mask,
        const float* confidence, const int* seeds,
        float* plane_params, float* plane_scores,
        int height, int width, int num_iterations,
        float inlier_threshold, float focal, float baseline,
        float cx, float cy,
        int plane_idx
    ) {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid >= num_iterations) return;
        int s = tid * 6;
        int idx1 = seeds[s+0]*width + seeds[s+1], idx2 = seeds[s+2]*width + seeds[s+3], idx3 = seeds[s+4]*width + seeds[s+5];
        if (!texture_mask[idx1] || !texture_mask[idx2] || !texture_mask[idx3]) return;
        float d1=depth[idx1], d2=depth[idx2], d3=depth[idx3];
        if (d1<0.2f || d1>8.0f || d2<0.2f || d2>8.0f || d3<0.2f || d3>8.0f) return;
        float p1x=(seeds[s+1]-cx)*d1/focal, p1y=(seeds[s+0]-cy)*d1/focal, p1z=d1;
        float v1x=(seeds[s+3]-cx)*d2/focal-p1x, v1y=(seeds[s+2]-cy)*d2/focal-p1y, v1z=d2-p1z;
        float v2x=(seeds[s+5]-cx)*d3/focal-p1x, v2y=(seeds[s+4]-cy)*d3/focal-p1y, v2z=d3-p1z;
        float nx=v1y*v2z-v1z*v2y, ny=v1z*v2x-v1x*v2z, nz=v1x*v2y-v1y*v2x;
        float nl=sqrtf(nx*nx+ny*ny+nz*nz); if(nl<1e-6f) return;
        nx/=nl; ny/=nl; nz/=nl; float pd=-(nx*p1x+ny*p1y+nz*p1z);
        int ic=0; float ws=0.0f;
        for(int y=0; y<height; y+=2) for(int x=0; x<width; x+=2) {
            int i=y*width+x; if(!texture_mask[i]) continue;
            float d=depth[i], c=confidence[i]; if(d<0.2f || d>8.0f || c<0.2f) continue;
            float dist = fabsf(nx*(x-cx)*d/focal + ny*(y-cy)*d/focal + nz*d + pd);
            if(dist < inlier_threshold) { ic++; ws += c*c*(1.0f - dist/inlier_threshold); }
        }
        float score = ws + (float)ic*0.5f;
        float os = atomicExch(&plane_scores[plane_idx], score);
        if(score > os) {
            atomicExch(&plane_params[plane_idx*4+0], nx); atomicExch(&plane_params[plane_idx*4+1], ny);
            atomicExch(&plane_params[plane_idx*4+2], nz); atomicExch(&plane_params[plane_idx*4+3], pd);
        }
    }
    ''', 'weighted_ransac_plane_fitting_kernel')

    PLANE_LABEL_ASSIGNMENT_KERNEL = cp.RawKernel(r'''
    extern "C" __global__
    void plane_label_assignment_kernel(
        const float* depth, const unsigned char* texture_mask,
        const float* confidence, const float* plane_params,
        const float* plane_scores, int* plane_labels,
        int height, int width, int num_planes,
        float inlier_threshold, float focal, float cx, float cy,
        int min_plane_size
    ) {
        int x = blockIdx.x*blockDim.x+threadIdx.x, y = blockIdx.y*blockDim.y+threadIdx.y;
        if(x>=width || y>=height) return;
        int i=y*width+x; if(!texture_mask[i] || confidence[i]<0.2f || depth[i]<0.2f || depth[i]>8.0f) { plane_labels[i]=-1; return; }
        float px=(x-cx)*depth[i]/focal, py=(y-cy)*depth[i]/focal, pz=depth[i];
        int bp=-1; float bd=1e10f;
        for(int p=0; p<num_planes; p++) {
            if(plane_scores[p] < (float)min_plane_size*0.5f) continue;
            float dist = fabsf(plane_params[p*4+0]*px + plane_params[p*4+1]*py + plane_params[p*4+2]*pz + plane_params[p*4+3]);
            if(dist < inlier_threshold && dist < bd) { bd=dist; bp=p; }
        }
        plane_labels[i]=bp;
    }
    ''', 'plane_label_assignment_kernel')

    PLANE_DEPTH_REFINEMENT_KERNEL = cp.RawKernel(r'''
    extern "C" __global__
    void plane_depth_refinement_kernel(
        const float* input_depth, const int* plane_labels,
        const float* plane_params, float* output_depth,
        const float* grad_magnitude, const float* confidence,
        int height, int width, float focal, float cx, float cy,
        float blend_alpha
    ) {
        int x=blockIdx.x*blockDim.x+threadIdx.x, y=blockIdx.y*blockDim.y+threadIdx.y;
        if(x>=width || y>=height) return;
        int i=y*width+x, pid=plane_labels[i];
        if(pid < 0) { output_depth[i]=input_depth[i]; return; }
        float nx=plane_params[pid*4+0], ny=plane_params[pid*4+1], nz=plane_params[pid*4+2], pd=plane_params[pid*4+3];
        float denom = nx*(x-cx)/focal + ny*(y-cy)/focal + nz;
        if(fabsf(denom) < 1e-6f) { output_depth[i]=input_depth[i]; return; }
        float pz = -pd/denom;
        float alpha = fmaxf(0.4f, 0.99f * (1.0f-fminf(grad_magnitude[i]/25.0f,1.0f)) * confidence[i]);
        output_depth[i] = alpha * fmaxf(0.2f, fminf(8.0f, pz)) + (1.0f-alpha)*input_depth[i];
    }
    ''', 'plane_depth_refinement_kernel')

# ===========================
# Configuration
# ===========================

@dataclass
class UltimateConfig:
    """Configuration for Ultimate Zero-Noise Stereo System"""
    camera_width: int = 640
    camera_height: int = 400
    baseline: float = 0.075
    scale_factor: float = 1.0
    num_disparities: int = 64
    patch_size: int = 5
    census_window: int = 5
    use_census: bool = True
    census_weight: float = 0.5
    lr_check_threshold_near: float = 0.8
    lr_check_threshold_far: float = 2.0
    adaptive_lr_enabled: bool = True
    temporal_alpha: float = 0.90
    use_bilateral_filter: bool = True
    bilateral_sigma_color: float = 3.0
    bilateral_sigma_space: float = 5.0
    bilateral_radius: int = 15
    bilateral_iterations: int = 3
    use_ransac_refinement: bool = True
    ransac_max_planes: int = 4
    ransac_iterations: int = 2048
    ransac_inlier_threshold: float = 0.03
    ransac_min_plane_size: int = 800
    ransac_texture_threshold: float = 1.5
    ransac_confidence_threshold: float = 0.35
    min_depth: float = 0.2
    max_depth: float = 8.0

class UltimateStereoProcessor:
    def __init__(self, cam_manager, config: dict):
        self.config = UltimateConfig()
        # Merge config from dict
        for k, v in config.items():
            if hasattr(self.config, k):
                setattr(self.config, k, v)
        
        self.cam_manager = cam_manager
        self.focal = cam_manager.P1_rect[0, 0]
        self.cx = cam_manager.P1_rect[0, 2]
        self.cy = cam_manager.P1_rect[1, 2]
        self.baseline = cam_manager.baseline_m
        self.f_b_term = self.focal * self.baseline
        
        self.h, self.w = 400, 640
        self.allocated = False
        self.buffers = {}
        
        # Temporal tracking
        self.prev_plane_params = None
        self.frame_count = 0
        
    def _allocate(self, h, w):
        if self.allocated and self.h == h and self.w == w:
            return
        self.h, self.w = h, w
        self.buffers = {
            'img_l': xp.empty((h, w), dtype=xp.float32),
            'img_r': xp.empty((h, w), dtype=xp.float32),
            'grad_x_l': xp.empty((h, w), dtype=xp.float32),
            'grad_y_l': xp.empty((h, w), dtype=xp.float32),
            'grad_x_r': xp.empty((h, w), dtype=xp.float32),
            'grad_y_r': xp.empty((h, w), dtype=xp.float32),
            'grad_mag': xp.empty((h, w), dtype=xp.float32),
            'census_l': xp.empty((h, w), dtype=xp.uint64),
            'census_r': xp.empty((h, w), dtype=xp.uint64),
            'disp_l': xp.empty((h, w), dtype=xp.float32),
            'disp_r': xp.empty((h, w), dtype=xp.float32),
            'confidence': xp.empty((h, w), dtype=xp.float32),
            'costs': xp.empty((h, w, 64), dtype=xp.float32),
            'validity_l': xp.empty((h, w), dtype=xp.uint8),
            'validity_r': xp.empty((h, w), dtype=xp.uint8),
            'disp_lr_l': xp.empty((h, w), dtype=xp.float32),
            'disp_lr_r': xp.empty((h, w), dtype=xp.float32),
            'texture_mask': xp.empty((h, w), dtype=xp.uint8),
            'plane_params': xp.empty((self.config.ransac_max_planes, 4), dtype=xp.float32),
            'plane_scores': xp.empty((self.config.ransac_max_planes,), dtype=xp.float32),
            'plane_labels': xp.empty((h, w), dtype=xp.int32),
            'refined_depth': xp.empty((h, w), dtype=xp.float32),
        }
        self.allocated = True

    def process(self, left: np.ndarray, right: np.ndarray) -> Tuple[xp.ndarray, xp.ndarray]:
        """Process stereo pair and return rectified left image and depth map."""
        # Rectification
        left_rect = cv2.remap(left, self.cam_manager.map1_left, self.cam_manager.map2_left, cv2.INTER_LINEAR)
        right_rect = cv2.remap(right, self.cam_manager.map1_right, self.cam_manager.map2_right, cv2.INTER_LINEAR)
        
        h, w = left_rect.shape
        self._allocate(h, w)
        
        # Upload
        img_l = self.buffers['img_l']; img_l[:] = xp.asarray(left_rect.astype(np.float32))
        img_r = self.buffers['img_r']; img_r[:] = xp.asarray(right_rect.astype(np.float32))
        
        # Gradients
        self.buffers['grad_x_l'][:] = ndi.sobel(img_l, axis=1)
        self.buffers['grad_y_l'][:] = ndi.sobel(img_l, axis=0)
        self.buffers['grad_x_r'][:] = ndi.sobel(img_r, axis=1)
        self.buffers['grad_y_r'][:] = ndi.sobel(img_r, axis=0)
        self.buffers['grad_mag'][:] = xp.sqrt(self.buffers['grad_x_l']**2 + self.buffers['grad_y_l']**2)
        
        # Census
        grid = ((w + 15)//16, (h + 15)//16); block = (16, 16)
        CENSUS_TRANSFORM_KERNEL(grid, block, (img_l, self.buffers['census_l'], h, w, 2))
        CENSUS_TRANSFORM_KERNEL(grid, block, (img_r, self.buffers['census_r'], h, w, 2))
        
        # PatchMatch
        self.buffers['disp_l'].fill(0); self.buffers['disp_r'].fill(0); self.buffers['confidence'].fill(0)
        ENHANCED_FUSED_KERNEL_SPLINE(grid, block, (
            img_l, img_r, self.buffers['grad_x_l'], self.buffers['grad_y_l'],
            self.buffers['grad_x_r'], self.buffers['grad_y_r'], self.buffers['grad_mag'],
            self.buffers['census_l'], self.buffers['census_r'],
            self.buffers['disp_l'], self.buffers['disp_r'], self.buffers['confidence'], self.buffers['costs'],
            h, w, self.config.num_disparities, self.config.patch_size, 
            self.config.lr_check_threshold_near, self.config.lr_check_threshold_far, 
            self.config.census_weight, int(self.config.use_census), 1
        ))
        
        # Bilateral
        if self.config.use_bilateral_filter:
            out = xp.empty_like(self.buffers['disp_l'])
            smem = 3 * (block[0] + 2*self.config.bilateral_radius)**2 * 4
            for _ in range(self.config.bilateral_iterations):
                bilateral_filter_smem_kernel(grid, block, (
                    self.buffers['disp_l'], img_l, self.buffers['grad_mag'], out,
                    h, w, self.config.bilateral_radius, self.config.bilateral_sigma_color, 
                    self.config.bilateral_sigma_space, 15.0, 1, 1, 0.8, 2.5, float(self.config.num_disparities)
                ), shared_mem=smem)
                self.buffers['disp_l'][:] = out
        
        # RANSAC
        if self.config.use_ransac_refinement:
            self._apply_ransac(h, w, grid, block)
            # Final depth from refined disparity or refined depth buffer
            depth = self.f_b_term / xp.clip(self.buffers['disp_l'], 0.5, 64.0)
            depth[self.buffers['disp_l'] < 0.5] = 0
        else:
            depth = self.f_b_term / xp.clip(self.buffers['disp_l'], 0.5, 64.0)
            depth[self.buffers['disp_l'] < 0.5] = 0

        self.frame_count += 1
        return left_rect, depth

    def _apply_ransac(self, h, w, grid, block):
        texture_mask = self.buffers['texture_mask']
        TEXTURELESS_DETECTION_KERNEL(grid, block, (self.buffers['grad_mag'], self.buffers['confidence'], texture_mask, h, w, 2.0, 0.4))
        
        depth = self.f_b_term / xp.clip(self.buffers['disp_l'], 0.5, 64.0)
        
        # Random seeds on GPU
        valid_indices = xp.where(texture_mask.ravel() > 0)[0]
        if len(valid_indices) < 100: return
        
        seeds = xp.random.choice(valid_indices, (self.config.ransac_iterations, 3)).astype(xp.int32)
        y_seeds = seeds // w; x_seeds = seeds % w
        seeds_6 = xp.column_stack([y_seeds[:,0], x_seeds[:,0], y_seeds[:,1], x_seeds[:,1], y_seeds[:,2], x_seeds[:,2]]).astype(xp.int32)
        
        params = self.buffers['plane_params']; scores = self.buffers['plane_scores']
        params.fill(0); scores.fill(0)
        
        for p_idx in range(self.config.ransac_max_planes):
            WEIGHTED_RANSAC_PLANE_FITTING_KERNEL((64,), (16,), (
                depth, texture_mask, self.buffers['confidence'], seeds_6,
                params, scores, h, w, self.config.ransac_iterations,
                self.config.ransac_inlier_threshold, self.focal, self.baseline, self.cx, self.cy, p_idx
            ))
            
        PLANE_LABEL_ASSIGNMENT_KERNEL(grid, block, (
            depth, texture_mask, self.buffers['confidence'], params, scores,
            self.buffers['plane_labels'], h, w, self.config.ransac_max_planes,
            self.config.ransac_inlier_threshold, self.focal, self.cx, self.cy, self.config.ransac_min_plane_size
        ))
        
        PLANE_DEPTH_REFINEMENT_KERNEL(grid, block, (
            depth, self.buffers['plane_labels'], params, self.buffers['refined_depth'],
            self.buffers['grad_mag'], self.buffers['confidence'],
            h, w, self.focal, self.cx, self.cy, 0.7
        ))
        
        # Convert back to disparity
        valid = self.buffers['refined_depth'] > 0.2
        self.buffers['disp_l'][valid] = self.f_b_term / self.buffers['refined_depth'][valid]
        self.buffers['disp_l'][~valid] = 0
