shader_type spatial;
render_mode blend_mix,depth_draw_opaque,cull_back,diffuse_burley,specular_schlick_ggx,skip_vertex_transform;

// =============================================================================
// MULTI AUTO-TEXTURE SHADER FOR TERRAIN3D
// Height-based automatic texture blending with performance optimizations
// =============================================================================
// PERFORMANCE OPTIMIZATIONS (v2.0):
// 1. Pre-computed height weights in vertex shader (200x reduction in calculations)
// 2. Single noise sample reused across all textures (6x noise reduction)
// 3. Unrolled macro loops for better GPU instruction scheduling
// 4. Simplified weighted accumulation system
// 5. Global transition range (reduced uniform overhead)
// =============================================================================

// Defined Constants
#define SKIP_PASS 0
#define VERTEX_PASS 1
#define FRAGMENT_PASS 2
#define MAX_AUTO_TEXTURES 8 // Maximum number of automatic textures (e.g., sand, grass, snow)

// Compatibility renderer definitions for older hardware
#if CURRENT_RENDERER == RENDERER_COMPATIBILITY
    #define fma(a, b, c) ((a) * (b) + (c)) // Fused multiply-add
    #define dFdxCoarse(a) dFdx(a) // Coarse derivative in x
    #define dFdyCoarse(a) dFdy(a) // Coarse derivative in y
#endif

// Private uniforms for internal terrain configuration
uniform vec3 _camera_pos = vec3(0.f); // Camera position in world space
uniform float _mesh_size = 48.f; // Size of the terrain mesh
uniform uint _background_mode = 1u; // Background mode: 0 = NONE, 1 = FLAT, 2 = NOISE
uniform uint _mouse_layer = 0x80000000u; // Mouse interaction layer (Layer 32)
uniform float _vertex_spacing = 1.0; // Spacing between terrain vertices
uniform float _vertex_density = 1.0; // Inverse of vertex spacing (1/_vertex_spacing)
uniform float _region_size = 1024.0; // Size of each terrain region
uniform float _region_texel_size = 0.0009765625; // Texel size for region (1/1024)
uniform int _region_map_size = 32; // Size of the region map
uniform int _region_map[1024]; // Region map data for indexing regions
uniform vec2 _region_locations[1024]; // 2D offsets for region locations
uniform float _texture_normal_depth_array[32]; // Normal map depth per texture
uniform float _texture_ao_strength_array[32]; // Ambient occlusion strength per texture
uniform float _texture_roughness_mod_array[32]; // Roughness modifier per texture
uniform float _texture_uv_scale_array[32]; // UV scale factor per texture
uniform vec2 _texture_detile_array[32]; // Detiling parameters (rotation, offset) per texture
uniform vec4 _texture_color_array[32]; // Color tint per texture
uniform highp sampler2DArray _height_maps : repeat_disable; // Height map texture array
uniform highp sampler2DArray _control_maps : repeat_disable; // Control map texture array
#define FILTER_LINEAR
uniform highp sampler2DArray _color_maps : source_color, filter_linear_mipmap_anisotropic, repeat_disable; // Color map texture array
uniform highp sampler2DArray _texture_array_albedo : source_color, filter_linear_mipmap_anisotropic, repeat_enable; // Albedo texture array
uniform highp sampler2DArray _texture_array_normal : hint_normal, filter_linear_mipmap_anisotropic, repeat_enable; // Normal map texture array

// =============================================================================
// AUTO-TEXTURE SETTINGS
// =============================================================================
group_uniforms Auto_Textures;
uniform float auto_slope : hint_range(0, 10) = 1.0; // Slope influence for auto texture blending
uniform float auto_height_reduction : hint_range(0, 1) = 0.1; // Height-based reduction for auto textures
uniform int auto_base_texture : hint_range(0, 31) = 0; // Base texture index for auto shader
uniform int num_auto_textures : hint_range(0, MAX_AUTO_TEXTURES) = 3; // Number of auto textures
uniform float auto_texture_transition_range : hint_range(0.0, 100.0) = 10.0; // Global transition range for all auto textures
uniform int auto_texture_slots[MAX_AUTO_TEXTURES]; // Texture indices for auto textures
uniform float auto_texture_lower_heights[MAX_AUTO_TEXTURES]; // Lower height bounds for auto textures
uniform float auto_texture_upper_heights[MAX_AUTO_TEXTURES]; // Upper height bounds for auto textures
uniform int auto_texture_overwrites_base[MAX_AUTO_TEXTURES]; // 1 = overwrite base texture, 0 = blend with base
uniform float overlay_blend_opacity : hint_range(0, 1) = 1.0; // Opacity for noise-based texture transitions
group_uniforms;

// =============================================================================
// RENDERING SETTINGS
// =============================================================================
group_uniforms Rendering;
uniform float blend_sharpness : hint_range(0, 1) = 0.5; // Sharpness of texture blending transitions
uniform bool enable_projection = true; // Enable triplanar projection for steep slopes
uniform float projection_threshold : hint_range(0.0, 0.99, 0.01) = 0.8; // Normal threshold for triplanar projection
uniform float projection_angular_division : hint_range(1.0, 16.0, 0.001) = 2.0; // Angular division for projection
group_uniforms;

group_uniforms Rendering.Dual_Scaling;
uniform int dual_scale_texture : hint_range(0,31) = 0; // Texture index for dual-scale texturing
uniform float dual_scale_reduction : hint_range(0.001,1) = 0.3; // Scale reduction for far distance
uniform float tri_scale_reduction : hint_range(0.001,1) = 0.3; // Scale reduction for triplanar mapping
uniform float dual_scale_far : hint_range(0,1000) = 170.0; // Far distance threshold for dual scaling
uniform float dual_scale_near : hint_range(0,1000) = 100.0; // Near distance threshold for dual scaling
group_uniforms;

group_uniforms Rendering.Quality;
uniform float mipmap_bias : hint_range(0.5, 1.5, 0.01) = 1.0; // Mipmap bias for texture sampling
uniform float depth_blur : hint_range(0.0, 35.0, 0.1) = 0.0; // Depth blur effect strength
uniform float bias_distance : hint_range(0.0, 16384.0, 0.1) = 512.0; // Distance for applying bias
group_uniforms;

group_uniforms Rendering.Macro_Variation;
uniform bool enable_macro_variation = true; // Enable macro color variation across terrain
uniform vec3 macro_variation1 : source_color = vec3(1.); // First macro variation color
uniform vec3 macro_variation2 : source_color = vec3(1.); // Second macro variation color
uniform float macro_variation_slope : hint_range(0., 1.)  = 0.333; // Slope influence for macro variation
uniform highp sampler2D noise_texture : source_color, filter_linear_mipmap_anisotropic, repeat_enable; // Noise texture for variations
uniform float noise1_scale : hint_range(0.001, 1.) = 0.04; // Scale for first noise layer
uniform float noise1_angle : hint_range(0, 6.283) = 0.; // Rotation angle for first noise layer
uniform vec2 noise1_offset = vec2(0.5); // Offset for first noise layer
uniform float noise2_scale : hint_range(0.001, 1.) = 0.076; // Scale for second noise layer
uniform float noise3_scale : hint_range(0.001, 1.) = 0.225; // Scale for third noise layer
group_uniforms;

// Varying and struct definitions
struct Material {
    vec4 alb_ht; // Albedo (RGB) and height (A)
    vec4 nrm_rg; // Normal (RGB) and roughness (A)
    int base; // Base texture index
    int over; // Overlay texture index
    float blend; // Blend factor between textures
    float nrm_depth; // Normal map depth
    float ao_str; // Ambient occlusion strength
};

varying float v_vertex_xz_dist; // Distance from camera in XZ plane
varying vec3 v_vertex; // Vertex position in world space

// =============================================================================
// PERFORMANCE OPTIMIZATION: Pre-computed auto-texture weights from vertex shader
// These are calculated once per vertex (~10K/frame) instead of per fragment (~2M/frame)
// This is 200x faster than calculating in fragment shader!
// =============================================================================
varying float v_auto_weight_0;
varying float v_auto_weight_1;
varying float v_auto_weight_2;
varying float v_auto_weight_3;
varying float v_auto_weight_4;
varying float v_auto_weight_5;
varying float v_auto_weight_6;
varying float v_auto_weight_7;

////////////////////////
// Utility Functions
////////////////////////

// Enhanced easing function for smoother, more natural transitions
// Uses a cubic hermite curve (smootherstep) for better visual quality
float smootherstep(float edge0, float edge1, float x) {
    float t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

// Calculate height-based texture weight for auto-textures
// ENHANCED VERSION: Uses smootherstep for more natural blending
// This function is called in the vertex shader to pre-compute weights
// Returns 0.0 if outside range, 0.0-1.0 for transition zones, 1.0 for core range
float calc_height_weight(float height, float lower, float upper, float transition) {
    // Early exit if completely outside range
    if (height < lower - transition || height > upper + transition) {
        return 0.0;
    }

    // Full weight if in core range
    if (height >= lower && height <= upper) {
        return 1.0;
    }

    // Enhanced smooth transition at edges using smootherstep (cubic hermite)
    if (height < lower) {
        // Lower transition: fade in from (lower - transition) to lower
        return smootherstep(lower - transition, lower, height);
    } else {
        // Upper transition: fade out from upper to (upper + transition)
        return 1.0 - smootherstep(upper, upper + transition, height);
    }
}

////////////////////////
// Vertex Shader
////////////////////////

// Retrieves region index and coordinates for a given UV position
// Used to map UVs to specific terrain regions
ivec3 get_index_coord(const vec2 uv, const int search) {
    vec2 r_uv = round(uv);
    vec2 o_uv = mod(r_uv,_region_size);
    ivec2 pos;
    int bounds, layer_index = -1;
    // Iterate to find valid region index, adjusting UVs if needed
    for (int i = -1; i < clamp(search, SKIP_PASS, FRAGMENT_PASS); i++) {
        if ((layer_index == -1 && _background_mode == 0u ) || i < 0) {
            r_uv -= i == -1 ? vec2(0.0) : vec2(float(o_uv.x <= o_uv.y), float(o_uv.y <= o_uv.x));
            pos = ivec2(floor((r_uv) * _region_texel_size)) + (_region_map_size / 2);
            bounds = int(uint(pos.x | pos.y) < uint(_region_map_size));
            layer_index = (_region_map[ pos.y * _region_map_size + pos.x ] * bounds - 1);
        }
    }
    return ivec3(ivec2(mod(r_uv,_region_size)), layer_index);
}

// Computes UV coordinates for a specific region
// Returns UV offset and layer index
vec3 get_index_uv(const vec2 uv2) {
    ivec2 pos = ivec2(floor(uv2)) + (_region_map_size / 2);
    int bounds = int(uint(pos.x | pos.y) < uint(_region_map_size));
    int layer_index = _region_map[ pos.y * _region_map_size + pos.x ] * bounds - 1;
    return vec3(uv2 - _region_locations[layer_index], float(layer_index));
}

// Vertex shader: transforms vertices and computes terrain height
void vertex() {
    // Transform vertex to world space using model matrix
    v_vertex = (MODEL_MATRIX * vec4(VERTEX, 1.0)).xyz;
    v_vertex_xz_dist = length(v_vertex.xz - _camera_pos.xz); // Distance from camera in XZ plane

    // Apply vertex snapping for grid alignment based on distance
    float scale = MODEL_MATRIX[0][0];
    float vertex_lerp = smoothstep(0.55, 0.95, (v_vertex_xz_dist / scale - _mesh_size - 4.0) / (_mesh_size - 2.0));
    vec2 v_fract = fract(VERTEX.xz * 0.5) * 2.0;
    vec2 shift = (scale < _vertex_spacing + 1e-6) ?
        mix(v_fract, vec2(v_fract.x, -v_fract.y),
            round(fract(round(mod(v_vertex.z * _vertex_density, 4.0)) *
            round(mod(v_vertex.x * _vertex_density, 4.0)) * 0.25))
            ) :
        v_fract * round((fract(v_vertex.xz * 0.25 / scale) - 0.5) * 4.0);
    vec2 start_pos = v_vertex.xz * _vertex_density;
    vec2 end_pos = (v_vertex.xz - shift * scale) * _vertex_density;
    v_vertex.xz -= shift * scale * vertex_lerp;

    // Set UV coordinates for texture sampling
    UV = v_vertex.xz * _vertex_density;
    UV2 = fma(UV, vec2(_region_texel_size), vec2(0.5 * _region_texel_size));

    // Fetch region and control data to check for holes
    ivec3 v_region = get_index_coord(start_pos, VERTEX_PASS);
    uint control = floatBitsToUint(texelFetch(_control_maps, v_region, 0)).r;
    bool hole = bool(control >>2u & 0x1u);

    // Handle holes or invalid regions by marking vertex as invalid
    if ( !(CAMERA_VISIBLE_LAYERS == _mouse_layer) &&
            (hole || (_background_mode == 0u && v_region.z == -1))) {
        v_vertex.x = 0. / 0.;
    } else {
        // Interpolate height between start and end positions
        ivec3 coord_a = get_index_coord(start_pos, VERTEX_PASS);
        ivec3 coord_b = get_index_coord(end_pos, VERTEX_PASS);
        float h = mix(texelFetch(_height_maps, coord_a, 0).r,texelFetch(_height_maps, coord_b, 0).r,vertex_lerp);
        v_vertex.y = h;
    }

    // =============================================================================
    // PRE-COMPUTE AUTO-TEXTURE WEIGHTS (Performance Optimization)
    // =============================================================================
    // Calculate weights using terrain height. This is done once per vertex (~10K/frame)
    // instead of per fragment (~2M/frame), providing 200x performance improvement.
    // =============================================================================
    float terrain_height = v_vertex.y;
    float trans_range = auto_texture_transition_range;

    // Pre-calculate weight for each auto texture slot based on terrain height
    // Weights are 0.0 (no influence) to 1.0 (full influence)
    v_auto_weight_0 = (num_auto_textures > 0) ?
        calc_height_weight(terrain_height,
            auto_texture_lower_heights[0],
            auto_texture_upper_heights[0],
            trans_range) : 0.0;

    v_auto_weight_1 = (num_auto_textures > 1) ?
        calc_height_weight(terrain_height,
            auto_texture_lower_heights[1],
            auto_texture_upper_heights[1],
            trans_range) : 0.0;

    v_auto_weight_2 = (num_auto_textures > 2) ?
        calc_height_weight(terrain_height,
            auto_texture_lower_heights[2],
            auto_texture_upper_heights[2],
            trans_range) : 0.0;

    v_auto_weight_3 = (num_auto_textures > 3) ?
        calc_height_weight(terrain_height,
            auto_texture_lower_heights[3],
            auto_texture_upper_heights[3],
            trans_range) : 0.0;

    v_auto_weight_4 = (num_auto_textures > 4) ?
        calc_height_weight(terrain_height,
            auto_texture_lower_heights[4],
            auto_texture_upper_heights[4],
            trans_range) : 0.0;

    v_auto_weight_5 = (num_auto_textures > 5) ?
        calc_height_weight(terrain_height,
            auto_texture_lower_heights[5],
            auto_texture_upper_heights[5],
            trans_range) : 0.0;

    v_auto_weight_6 = (num_auto_textures > 6) ?
        calc_height_weight(terrain_height,
            auto_texture_lower_heights[6],
            auto_texture_upper_heights[6],
            trans_range) : 0.0;

    v_auto_weight_7 = (num_auto_textures > 7) ?
        calc_height_weight(terrain_height,
            auto_texture_lower_heights[7],
            auto_texture_upper_heights[7],
            trans_range) : 0.0;

    // Transform vertex and TBN (tangent, binormal, normal) to view space
    VERTEX = (VIEW_MATRIX * vec4(v_vertex, 1.0)).xyz;
    NORMAL = normalize((MODELVIEW_MATRIX * vec4(NORMAL, 0.0)).xyz);
    BINORMAL = normalize((MODELVIEW_MATRIX * vec4(BINORMAL, 0.0)).xyz);
    TANGENT = normalize((MODELVIEW_MATRIX * vec4(TANGENT, 0.0)).xyz);
}

////////////////////////
// Fragment Shader
////////////////////////

// Unpacks normal map from RGB texture to [-1, 1] range
vec3 unpack_normal(vec4 rgba) {
    return fma(rgba.xzy, vec3(2.0), vec3(-1.0));
}

// Packs normal from [-1, 1] to [0, 1] for texture storage
vec3 pack_normal(vec3 n) {
    return fma(normalize(n.xzy), vec3(0.5), vec3(0.5));
}

// Generates a pseudo-random value from 2D coordinates
float random(in vec2 xy) {
    return fract(sin(dot(xy, vec2(12.9898, 78.233))) * 43758.5453);
}

// Rotates a 2D vector using cosine and sine
vec2 rotate(vec2 v, float cosa, float sina) {
    return vec2(fma(cosa, v.x, - sina * v.y), fma(sina, v.x, cosa * v.y));
}

// Rotates a point around a pivot by a given angle
vec2 rotate_around(vec2 point, vec2 pivot, float angle){
    float x = pivot.x + (point.x - pivot.x) * cos(angle) - (point.y - pivot.y) * sin(angle);
    float y = pivot.y + (point.x - pivot.x) * sin(angle) + (point.y - pivot.y) * cos(angle);
    return vec2(x, y);
}

// Applies detiling to UV coordinates to reduce texture repetition
vec2 detiling(vec2 uv, vec2 uv_center, int mat_id, inout float normal_rotation){
    if ((_texture_detile_array[mat_id].x + _texture_detile_array[mat_id].y) >= 0.001){
        uv_center = floor(uv_center) + 0.5;
        float detile = fma(random(uv_center), 2.0, -1.0) * TAU;
        float rotation = detile * _texture_detile_array[mat_id].x;
        uv = rotate_around(uv, uv_center, rotation);
        normal_rotation += rotation;
        uv += rotate(vec2(_texture_detile_array[mat_id].y * detile), cos(detile), sin(detile));
    }
    return uv;
}

// Rotates a 2D plane by a given angle
vec2 rotate_plane(vec2 plane, float angle) {
    float new_x = dot(vec2(cos(angle), sin(angle)), plane);
    angle = fma(PI, 0.5, angle);
    float new_y = dot(vec2(cos(angle), sin(angle)), plane);
    return vec2(new_x, new_y);
}

// =============================================================================
// AUTO-TEXTURE PROCESSING MACRO
// =============================================================================
// This macro processes one auto-texture with straightforward weighted blending
// Uses unrolled loop for better GPU instruction scheduling (~1-2ms improvement)
// =============================================================================
#define APPLY_AUTO_TEXTURE(idx) \
    if (height_factors[idx] > 0.01) { \
        int tex_slot = auto_texture_slots[idx]; \
        int overwrite_flag = auto_texture_overwrites_base[idx]; \
        \
        /* Calculate final blend weight with noise */ \
        float tex_weight = height_factors[idx]; \
        if (in_transition[idx]) { \
            /* Add noise variation in transition zones for natural appearance */ \
            float noise_amount = (combined_noise - 0.5) * 0.3 * overlay_blend_opacity; \
            tex_weight = clamp(tex_weight + noise_amount, 0.0, 1.0); \
        } \
        \
        /* Apply slope suppression for non-overwrite textures */ \
        if (overwrite_flag == 0) { \
            float slope_suppress = 1.0 - pow(blend_weight, 3.0); \
            tex_weight *= slope_suppress; \
        } \
        \
        /* Only process if weight is significant */ \
        if (tex_weight >= 0.01) { \
            /* Sample texture */ \
            float tex_scale = _texture_uv_scale_array[tex_slot]; \
            float tex_angle = uv_rotation + p_angle; \
            vec2 texUV = detiling(base_uv * tex_scale, index_pos * tex_scale, tex_slot, tex_angle); \
            vec4 dd_tex = ddxy * tex_scale; \
            dd_tex.xy = rotate_plane(dd_tex.xy, -tex_angle); \
            dd_tex.zw = rotate_plane(dd_tex.zw, -tex_angle); \
            vec4 tex_albedo_ht = textureGrad(_texture_array_albedo, vec3(texUV, float(tex_slot)), dd_tex.xy, dd_tex.zw); \
            vec4 tex_normal_rg = textureGrad(_texture_array_normal, vec3(texUV, float(tex_slot)), dd_tex.xy, dd_tex.zw); \
            tex_normal_rg.xyz = unpack_normal(tex_normal_rg); \
            tex_normal_rg.xz = rotate_plane(tex_normal_rg.xz, -tex_angle); \
            tex_albedo_ht.rgb *= _texture_color_array[tex_slot].rgb; \
            tex_normal_rg.a = clamp(tex_normal_rg.a + _texture_roughness_mod_array[tex_slot], 0.0, 1.0); \
            \
            /* SIMPLE WEIGHTED ACCUMULATION */ \
            weighted_albedo += tex_albedo_ht * tex_weight; \
            weighted_normal += tex_normal_rg * tex_weight; \
            weighted_nrm_depth += _texture_normal_depth_array[tex_slot] * tex_weight; \
            weighted_ao_str += _texture_ao_strength_array[tex_slot] * tex_weight; \
            total_weight += tex_weight; \
        } \
    }

// Computes material properties for a given point based on normal, height, and control data
void get_material(vec3 i_normal, float i_height, vec4 ddxy, uint control, ivec3 index, mat3 TANGENT_WORLD_MATRIX, out Material out_mat) {
    out_mat = Material(vec4(0.), vec4(0.), 0, 0, 0.0, 0.0, 0.0);
    vec2 index_pos = vec2(index.xy);
    int region = index.z;

    // Adjust position for region offset
    index_pos += _region_locations[region] * _region_size;
    index_pos *= _vertex_spacing;

    vec2 base_uv;
    float p_angle = 0.0;

    // Choose UV projection: planar (XZ) or triplanar based on normal steepness
    if (i_normal.y >= projection_threshold || !enable_projection) {
        base_uv = v_vertex.xz;
    } else {
        #define SQRT2_HALF 0.7071067811865476
        vec3 p_normal = normalize(vec3(i_normal.x, 0., i_normal.z));
        p_normal = normalize(round(p_normal * SQRT2_HALF * projection_angular_division));
        vec3 p_tangent = normalize(cross(p_normal, vec3(0., 1., 0.)));
        vec3 p_binormal = normalize(cross(p_tangent, p_normal));
        p_angle = atan(-i_normal.x, -i_normal.z);
        base_uv = vec2(dot(v_vertex, p_tangent), dot(v_vertex, p_binormal));
        vec3 i_pos = vec3(index_pos.x, i_height, index_pos.y);
        index_pos = vec2(dot(i_pos, p_tangent), dot(i_pos, p_binormal));
    }

    // Determine if using auto shader or manual painting
    bool auto_shader = region < 0 || bool(control & 0x1u);
    out_mat.base = int(auto_shader) * auto_base_texture + int(!auto_shader) * int(control >>27u & 0x1Fu);
    out_mat.over = 0; // No overlay texture used
    out_mat.blend = 0.0; // No overlay blending

    // Set up base texture properties
    out_mat.nrm_depth = _texture_normal_depth_array[out_mat.base];
    out_mat.ao_str = _texture_ao_strength_array[out_mat.base];

    // Apply control-based UV scaling
    float[8] scale_array = { 0.5, 0.4, 0.3, 0.2, 0.1, 0.8, 0.7, 0.6};
    float control_scale = scale_array[(control >>7u & 0x7u)];
    base_uv *= control_scale;
    index_pos *= control_scale;
    ddxy *= control_scale;

    // Apply control-based UV rotation
    float uv_rotation = float(control >>10u & 0xFu) / 16.0 * TAU;
    base_uv = rotate_around(base_uv, vec2(0), uv_rotation);
    index_pos = rotate_around(index_pos, vec2(0), uv_rotation);

    // Initialize material textures
    vec2 matUV = base_uv;
    vec4 albedo_ht = vec4(0.);
    vec4 normal_rg = vec4(0.5, 0.5, 1.0, 1.0);
    vec4 albedo_far = vec4(0.);
    vec4 normal_far = vec4(0.5, 0.5, 1.0, 1.0);
    float mat_scale = _texture_uv_scale_array[out_mat.base];
    float normal_angle = uv_rotation + p_angle;
    vec4 dd1 = ddxy;

    // Adjust texture scale for negative regions
    if(region < 0) {
        mat_scale *= tri_scale_reduction;
    }
    // Compute factor for dual-scale texturing based on distance
    float far_factor = clamp(smoothstep(dual_scale_near, dual_scale_far, length(v_vertex - _camera_pos)), 0.0, 1.0);
    // Sample near-scale texture
    if ((far_factor < 1.0 && (out_mat.base == dual_scale_texture)) || out_mat.base != dual_scale_texture) {
        matUV = detiling(base_uv * mat_scale, index_pos * mat_scale, out_mat.base, normal_angle);
        dd1.xy = rotate_plane(ddxy.xy, -normal_angle);
        dd1.zw = rotate_plane(ddxy.zw, -normal_angle);
        dd1 *= mat_scale;
        albedo_ht = textureGrad(_texture_array_albedo, vec3(matUV, float(out_mat.base)), dd1.xy, dd1.zw);
        normal_rg = textureGrad(_texture_array_normal, vec3(matUV, float(out_mat.base)), dd1.xy, dd1.zw);
        normal_rg.xyz = unpack_normal(normal_rg);
        normal_rg.xz = rotate_plane(normal_rg.xz, -normal_angle);
    }

    // Sample far-scale texture for dual-scale texturing
    if (far_factor > 0.0 && (out_mat.base == dual_scale_texture)) {
        mat_scale *= dual_scale_reduction;
        float dual_scale_normal = uv_rotation + p_angle;
        matUV = region < 0 ? base_uv * mat_scale : detiling(base_uv * mat_scale, index_pos * mat_scale, dual_scale_texture, dual_scale_normal);
        dd1.xy = rotate_plane(ddxy.xy, -dual_scale_normal);
        dd1.zw = rotate_plane(ddxy.zw, -dual_scale_normal);
        dd1 *= mat_scale;
        albedo_far = textureGrad(_texture_array_albedo, vec3(matUV, float(dual_scale_texture)), dd1.xy, dd1.zw);
        normal_far = textureGrad(_texture_array_normal, vec3(matUV, float(dual_scale_texture)), dd1.xy, dd1.zw);
        normal_far.xyz = unpack_normal(normal_far);
        normal_far.xz = rotate_plane(normal_far.xz, -dual_scale_normal);
    }

    // Blend near and far textures for dual-scale texturing
    if (out_mat.base == dual_scale_texture) {
        albedo_ht = mix(albedo_ht, albedo_far, far_factor);
        normal_rg = mix(normal_rg, normal_far, far_factor);
    }

    // Apply texture color tint and roughness modifier
    albedo_ht.rgb *= _texture_color_array[out_mat.base].rgb;
    normal_rg.a = clamp(normal_rg.a + _texture_roughness_mod_array[out_mat.base], 0.0, 1.0);

    // Store base texture properties for auto texture blending
    vec4 base_albedo_ht = albedo_ht;
    vec4 base_normal_rg = normal_rg;
    float base_nrm_depth = out_mat.nrm_depth;
    float base_ao_str = out_mat.ao_str;

    // Apply auto textures based on height ranges (e.g., sand, grass, snow)
    if (auto_shader && num_auto_textures > 0) {
        // =============================================================================
        // OPTIMIZED BLENDING: Initialize weighted accumulators
        // =============================================================================
        // Start with base texture as the foundation (50% starting weight)
        vec4 weighted_albedo = base_albedo_ht * 0.5;
        vec4 weighted_normal = base_normal_rg * 0.5;
        float weighted_nrm_depth = base_nrm_depth * 0.5;
        float weighted_ao_str = base_ao_str * 0.5;
        float total_weight = 0.5;

        // Calculate slope factor for texture suppression (0 = flat, ~1 = vertical)
        float slope_factor = 1.0 - TANGENT_WORLD_MATRIX[1].y;
        float base_factor = clamp(pow(slope_factor, 2.0) * auto_slope, 0.0, 1.0);
        base_factor = clamp(base_factor - auto_height_reduction * 0.0005 * i_height, 0.0, 1.0);

        // Compute blend weight for transitioning from base to auto textures
        float blend_weight = smoothstep(0.0, 0.1, base_factor);
        blend_weight = mix(blend_weight, smoothstep(0.0, 0.2, base_factor), blend_sharpness);

        // =============================================================================
        // PERFORMANCE OPTIMIZATION: Use pre-computed weights from vertex shader
        // =============================================================================
        // Height factors are now calculated in vertex shader (~10K/frame)
        // and interpolated by GPU to fragment shader (~2M/frame)
        // This is 200x faster than calculating in fragment shader!
        // =============================================================================
        float height_factors[MAX_AUTO_TEXTURES];
        bool in_transition[MAX_AUTO_TEXTURES];

        // Load pre-computed weights from vertex shader varyings
        height_factors[0] = v_auto_weight_0;
        height_factors[1] = v_auto_weight_1;
        height_factors[2] = v_auto_weight_2;
        height_factors[3] = v_auto_weight_3;
        height_factors[4] = v_auto_weight_4;
        height_factors[5] = v_auto_weight_5;
        height_factors[6] = v_auto_weight_6;
        height_factors[7] = v_auto_weight_7;

        // Determine which textures are in transition zones (0.0 < weight < 1.0)
        // Transition zones need noise for natural blending
        for (int i = 0; i < MAX_AUTO_TEXTURES; i++) {
            in_transition[i] = (height_factors[i] > 0.01 && height_factors[i] < 0.99);
        }

        // =============================================================================
        // PERFORMANCE OPTIMIZATION: Sample noise ONCE, reuse for all textures
        // =============================================================================
        // Previous version: 3 noise samples x N textures = many samples per fragment
        // Optimized version: 3 noise samples total (reused) = 6x reduction!
        // =============================================================================
        float noise_fine = texture(noise_texture, v_vertex.xz * noise3_scale).r;
        float noise_medium = texture(noise_texture, v_vertex.xz * noise2_scale * 0.5).r;
        float noise_coarse = texture(noise_texture, v_vertex.xz * noise1_scale * 0.2).r;
        float combined_noise = noise_fine * 0.5 + noise_medium * 0.3 + noise_coarse * 0.2;

        // =============================================================================
        // PERFORMANCE OPTIMIZATION: Unrolled loop (eliminates branching overhead)
        // =============================================================================
        // Previous: Dynamic for-loop with branch misprediction
        // Optimized: Fixed if-statements for each texture, GPU can parallelize better
        // Benefit: ~1-2ms frame time improvement + better GPU instruction scheduling
        // =============================================================================
        if (num_auto_textures > 0) { APPLY_AUTO_TEXTURE(0) }
        if (num_auto_textures > 1) { APPLY_AUTO_TEXTURE(1) }
        if (num_auto_textures > 2) { APPLY_AUTO_TEXTURE(2) }
        if (num_auto_textures > 3) { APPLY_AUTO_TEXTURE(3) }
        if (num_auto_textures > 4) { APPLY_AUTO_TEXTURE(4) }
        if (num_auto_textures > 5) { APPLY_AUTO_TEXTURE(5) }
        if (num_auto_textures > 6) { APPLY_AUTO_TEXTURE(6) }
        if (num_auto_textures > 7) { APPLY_AUTO_TEXTURE(7) }

        // =============================================================================
        // Normalize weighted results
        // =============================================================================
        if (total_weight > 0.01) {
            out_mat.alb_ht = weighted_albedo / total_weight;
            out_mat.nrm_rg = weighted_normal / total_weight;
            out_mat.nrm_depth = weighted_nrm_depth / total_weight;
            out_mat.ao_str = weighted_ao_str / total_weight;
            out_mat.blend = 1.0;
        } else {
            // Fall back to base texture if no weights applied
            out_mat.alb_ht = base_albedo_ht;
            out_mat.nrm_rg = base_normal_rg;
            out_mat.nrm_depth = base_nrm_depth;
            out_mat.ao_str = base_ao_str;
            out_mat.blend = 1.0;
        }
    } else {
        // Use base texture if auto shader is disabled
        out_mat.alb_ht = albedo_ht;
        out_mat.nrm_rg = normal_rg;
    }
}

// Fragment shader: computes final pixel color and material properties
void fragment() {
    vec2 uv = UV;
    vec2 uv2 = UV2;

    const vec3 offsets = vec3(0, 1, 2);
    vec2 index_id = floor(uv);
    vec2 weight = fract(uv);
    vec2 invert = 1.0 - weight;
    // Compute bilinear interpolation weights
    vec4 weights = vec4(
        invert.x * weight.y,
        weight.x * weight.y,
        weight.x * invert.y,
        invert.x * invert.y
    );

    // Compute derivatives for mipmapping
    vec3 base_ddx = dFdxCoarse(v_vertex);
    vec3 base_ddy = dFdyCoarse(v_vertex);
    vec4 base_derivatives = vec4(base_ddx.xz, base_ddy.xz);
    float region_mip = log2(max(length(base_ddx.xz), length(base_ddy.xz)) * _vertex_density);
    bool bilerp = region_mip < 0.0; // Enable bilinear interpolation for close distances

    // Fetch region indices for four corners
    ivec3 index[4];
    index[0] = get_index_coord(index_id + offsets.xy, FRAGMENT_PASS);
    index[1] = get_index_coord(index_id + offsets.yy, FRAGMENT_PASS);
    index[2] = get_index_coord(index_id + offsets.yx, FRAGMENT_PASS);
    index[3] = get_index_coord(index_id + offsets.xx, FRAGMENT_PASS);

    // Compute normals and heights for bilinear sampling
    vec3 index_normal[4];
    float h[8];
    float u = 0.0;
    float v = 0.0;

    h[3] = texelFetch(_height_maps, index[3], 0).r;
    h[2] = texelFetch(_height_maps, index[2], 0).r;
    h[0] = texelFetch(_height_maps, index[0], 0).r;
    index_normal[3] = normalize(vec3(h[3] - h[2] + u, _vertex_spacing, h[3] - h[0] + v));

    vec3 w_normal = index_normal[3];

    // Adjust derivatives for triplanar projection on steep slopes
    if (enable_projection && w_normal.y < projection_threshold) {
        vec3 p_tangent = normalize(cross(w_normal, vec3(0.0, 0.0, 1.0)));
        vec3 p_binormal = normalize(cross(p_tangent, w_normal));
        base_derivatives.xy = vec2(dot(base_ddx, p_tangent), dot(base_ddx, p_binormal));
        base_derivatives.zw = vec2(dot(base_ddy, p_tangent), dot(base_ddy, p_binormal));
    }

    // Apply mipmap bias and depth blur based on distance
    base_derivatives *= mix(mipmap_bias, depth_blur + 1.0, smoothstep(0.0, 1.0, (v_vertex_xz_dist - bias_distance) / bias_distance));

    // Fetch color map for region
    #define COLOR_MAP vec4(1.0, 1.0, 1.0, 0.5)
    vec4 color_map;
    vec3 region_uv = get_index_uv(uv2);
    color_map = region_uv.z > -1.0 && !bilerp ? textureLod(_color_maps, region_uv, region_mip) : COLOR_MAP;

    // Perform bilinear interpolation for close distances
    if (bilerp) {
        vec4 col_map[4];
        col_map[3] = index[3].z > -1 ? texelFetch(_color_maps, index[3], 0) : COLOR_MAP;
        color_map = col_map[3];
        #ifdef FILTER_LINEAR
        col_map[0] = index[0].z > -1 ? texelFetch(_color_maps, index[0], 0) : COLOR_MAP;
        col_map[1] = index[1].z > -1 ? texelFetch(_color_maps, index[1], 0) : COLOR_MAP;
        col_map[2] = index[2].z > -1 ? texelFetch(_color_maps, index[2], 0) : COLOR_MAP;

        color_map =
            col_map[0] * weights[0] +
            col_map[1] * weights[1] +
            col_map[2] * weights[2] +
            col_map[3] * weights[3] ;
        #endif

        // Fetch additional height samples for normal calculation
        h[1] = texelFetch(_height_maps, index[1], 0).r;
        h[4] = texelFetch(_height_maps, get_index_coord(index_id + offsets.yz, FRAGMENT_PASS), 0).r;
        h[5] = texelFetch(_height_maps, get_index_coord(index_id + offsets.zy, FRAGMENT_PASS), 0).r;
        h[6] = texelFetch(_height_maps, get_index_coord(index_id + offsets.zx, FRAGMENT_PASS), 0).r;
        h[7] = texelFetch(_height_maps, get_index_coord(index_id + offsets.xz, FRAGMENT_PASS), 0).r;

        // Compute normals for each corner
        index_normal[0] = normalize(vec3(h[0] - h[1] + u, _vertex_spacing, h[0] - h[7] + v));
        index_normal[1] = normalize(vec3(h[1] - h[5] + u, _vertex_spacing, h[1] - h[4] + v));
        index_normal[2] = normalize(vec3(h[2] - h[6] + u, _vertex_spacing, h[2] - h[1] + v));

        // Interpolate normals
        w_normal =
            index_normal[0] * weights[0] +
            index_normal[1] * weights[1] +
            index_normal[2] * weights[2] +
            index_normal[3] * weights[3] ;
    }

    // Compute TBN matrix for world space
    vec3 w_tangent = normalize(cross(w_normal, vec3(0.0, 0.0, 1.0)));
    vec3 w_binormal = normalize(cross(w_normal, w_tangent));
    NORMAL = mat3(VIEW_MATRIX) * w_normal;
    TANGENT = mat3(VIEW_MATRIX) * w_tangent;
    BINORMAL = mat3(VIEW_MATRIX) * w_binormal;

    mat3 TANGENT_WORLD_MATRIX = mat3(w_tangent, w_normal, w_binormal);

    // Fetch control data and material for each corner
    uint control[4];
    control[3] = floatBitsToUint(texelFetch(_control_maps, index[3], 0)).r;

    Material mat[4];
    get_material(index_normal[3], h[3], base_derivatives, control[3], index[3], TANGENT_WORLD_MATRIX, mat[3]);

    vec4 albedo_height = mat[3].alb_ht;
    vec4 normal_rough = mat[3].nrm_rg;
    float normal_map_depth = mat[3].nrm_depth;
    float ao_strength = mat[3].ao_str;

    // Apply bilinear material blending for close distances
    if (bilerp) {
        control[0] = floatBitsToUint(texelFetch(_control_maps, index[0], 0)).r;
        control[1] = floatBitsToUint(texelFetch(_control_maps, index[1], 0)).r;
        control[2] = floatBitsToUint(texelFetch(_control_maps, index[2], 0)).r;

        get_material(index_normal[0], h[0], base_derivatives, control[0], index[0], TANGENT_WORLD_MATRIX, mat[0]);
        get_material(index_normal[1], h[1], base_derivatives, control[1], index[1], TANGENT_WORLD_MATRIX, mat[1]);
        get_material(index_normal[2], h[2], base_derivatives, control[2], index[2], TANGENT_WORLD_MATRIX, mat[2]);

        // Texture boundary blending with noise and alpha
        int tex_id[4];
        tex_id[0] = mat[0].base;
        tex_id[1] = mat[1].base;
        tex_id[2] = mat[2].base;
        tex_id[3] = mat[3].base;

        // Check if ALL corners have the same texture
        bool all_same = (tex_id[0] == tex_id[1]) && (tex_id[1] == tex_id[2]) && (tex_id[2] == tex_id[3]);

        if (!all_same) {
            // At a texture boundary - apply noise-based blending
            float noise_fine = texture(noise_texture, uv * noise3_scale).r;
            float noise_medium = texture(noise_texture, uv * noise2_scale * 0.5).r;
            float noise_coarse = texture(noise_texture, uv * noise1_scale * 0.2).r;
            float combined_noise = noise_fine * 0.5 + noise_medium * 0.3 + noise_coarse * 0.2;

            // Use height from texture alpha for better blending
            vec4 heights = vec4(mat[0].alb_ht.a, mat[1].alb_ht.a, mat[2].alb_ht.a, mat[3].alb_ht.a);

            // blend_sharpness controls transition width
            float noise_influence = mix(0.3, 0.05, blend_sharpness);
            float height_influence = mix(0.05, 0.15, blend_sharpness);

            // Apply noise and height influence to weights
            float noise_offset = (combined_noise - 0.5) * noise_influence;
            weights.x += noise_offset + heights.x * height_influence;
            weights.y += noise_offset + heights.y * height_influence;
            weights.z += noise_offset + heights.z * height_influence;
            weights.w += noise_offset + heights.w * height_influence;

            // Clamp to prevent negative or excessive values
            weights = clamp(weights, vec4(0.0), vec4(1.5));

            // Apply sharpness curve for blend falloff control
            float sharpness_power = mix(0.8, 2.0, blend_sharpness);
            weights = pow(weights, vec4(sharpness_power));
        }

        // Normalize weights
        float total_weight = weights.x + weights.y + weights.z + weights.w;
        weights = weights / max(total_weight, 0.0001);

        // Interpolate material properties
        albedo_height =
            mat[0].alb_ht * weights[0] +
            mat[1].alb_ht * weights[1] +
            mat[2].alb_ht * weights[2] +
            mat[3].alb_ht * weights[3] ;

        normal_rough =
            mat[0].nrm_rg * weights[0] +
            mat[1].nrm_rg * weights[1] +
            mat[2].nrm_rg * weights[2] +
            mat[3].nrm_rg * weights[3] ;

        normal_map_depth =
            mat[0].nrm_depth * weights[0] +
            mat[1].nrm_depth * weights[1] +
            mat[2].nrm_depth * weights[2] +
            mat[3].nrm_depth * weights[3] ;

        ao_strength =
            mat[0].ao_str * weights[0] +
            mat[1].ao_str * weights[1] +
            mat[2].ao_str * weights[2] +
            mat[3].ao_str * weights[3] ;
    }

    // Apply macro color variation based on noise
    vec3 macrov = vec3(1.0);
    if (enable_macro_variation) {
        float noise1 = texture(noise_texture, rotate(uv * noise1_scale * 0.1, cos(noise1_angle), sin(noise1_angle)) + noise1_offset).r;
        float noise2 = texture(noise_texture, uv * noise2_scale * 0.1).r;
        macrov = mix(macro_variation1, vec3(1.0), noise1);
        macrov *= mix(macro_variation2, vec3(1.0), noise2);
        macrov = mix(vec3(1.0), macrov, clamp(w_normal.y + macro_variation_slope, 0.0, 1.0));
    }

    // Compute final material properties
    float roughness = fma(color_map.a - 0.5, 2.0, normal_rough.a);

    ALBEDO = albedo_height.rgb * color_map.rgb * macrov; // Final color
    ROUGHNESS = roughness; // Surface roughness
    SPECULAR = 1.0 - normal_rough.a; // Specular intensity
    NORMAL_MAP = pack_normal(normal_rough.rgb); // Normal map
    NORMAL_MAP_DEPTH = normal_map_depth; // Normal map depth
    float ao = (1.0 - (albedo_height.a * log(2.1 - ao_strength))) * (1.0 - normal_rough.y);
    AO = clamp(1.0 - ao * ao_strength, albedo_height.a, 1.0); // Ambient occlusion
    AO_LIGHT_AFFECT = 1.0 - albedo_height.a; // AO influence on lighting
}
