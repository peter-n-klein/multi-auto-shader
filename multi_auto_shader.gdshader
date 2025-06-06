shader_type spatial;
render_mode blend_mix,depth_draw_opaque,cull_back,diffuse_burley,specular_schlick_ggx,skip_vertex_transform;

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

// Public uniforms for terrain customization
uniform float auto_slope : hint_range(0, 10) = 1.0; // Slope influence for auto texture blending
uniform float auto_height_reduction : hint_range(0, 1) = 0.1; // Height-based reduction for auto textures
uniform int auto_base_texture : hint_range(0, 31) = 0; // Base texture index for auto shader
uniform int num_auto_textures : hint_range(0, MAX_AUTO_TEXTURES) = 3; // Number of auto textures (e.g., sand, grass, snow)
uniform int auto_texture_slots[MAX_AUTO_TEXTURES]; // Texture indices for auto textures
uniform float auto_texture_lower_heights[MAX_AUTO_TEXTURES]; // Lower height bounds for auto textures
uniform float auto_texture_upper_heights[MAX_AUTO_TEXTURES]; // Upper height bounds for auto textures
uniform float auto_texture_transition_ranges[MAX_AUTO_TEXTURES]; // Transition range for auto texture blending
uniform float auto_texture_noise_scales[MAX_AUTO_TEXTURES]; // Noise scale for auto texture transitions
uniform int auto_texture_overwrites_base[MAX_AUTO_TEXTURES]; // 1 = overwrite base texture, 0 = blend with base
uniform float overlay_blend_opacity : hint_range(0, 1) = 1.0; // Opacity for noise-based texture transitions
uniform int dual_scale_texture : hint_range(0,31) = 0; // Texture index for dual-scale texturing
uniform float dual_scale_reduction : hint_range(0.001,1) = 0.3; // Scale reduction for far distance
uniform float tri_scale_reduction : hint_range(0.001,1) = 0.3; // Scale reduction for triplanar mapping
uniform float dual_scale_far : hint_range(0,1000) = 170.0; // Far distance threshold for dual scaling
uniform float dual_scale_near : hint_range(0,1000) = 100.0; // Near distance threshold for dual scaling
uniform bool height_blending = true; // Enable height-based texture blending
uniform bool world_space_normal_blend = true; // Use world-space normals for blending
uniform float blend_sharpness : hint_range(0, 1) = 0.5; // Sharpness of texture blending transitions
uniform bool enable_projection = true; // Enable triplanar projection for steep slopes
uniform float projection_threshold : hint_range(0.0, 0.99, 0.01) = 0.8; // Normal threshold for triplanar projection
uniform float projection_angular_division : hint_range(1.0, 16.0, 0.001) = 2.0; // Angular division for projection
uniform float mipmap_bias : hint_range(0.5, 1.5, 0.01) = 1.0; // Mipmap bias for texture sampling
uniform float depth_blur : hint_range(0.0, 35.0, 0.1) = 0.0; // Depth blur effect strength
uniform float bias_distance : hint_range(0.0, 16384.0, 0.1) = 512.0; // Distance for applying bias
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

// Blends two vec4 values (e.g., albedo) based on height and blend factor
vec4 height_blend4(vec4 a_value, float a_height, vec4 b_value, float b_height, float blend) {
    if(height_blending) {
        float ma = max(a_height + (1.0 - blend), b_height + blend) - (1.001 - blend_sharpness);
        float b1 = max(a_height + (1.0 - blend) - ma, 0.0);
        float b2 = max(b_height + blend - ma, 0.0);
        return (a_value * b1 + b_value * b2) / (b1 + b2);
    } else {
        float contrast = 1.0 - blend_sharpness;
        float factor = (blend - contrast) / contrast;
        return mix(a_value, b_value, clamp(factor, 0.0, 1.0));
    }
}

// Blends two float values (e.g., normal depth) based on height and blend factor
float height_blend1(float a_value, float a_height, float b_value, float b_height, float blend) {
    if(height_blending) {
        float ma = max(a_height + (1.0 - blend), b_height + blend) - (1.001 - blend_sharpness);
        float b1 = max(a_height + (1.0 - blend) - ma, 0.0);
        float b2 = max(b_height + blend - ma, 0.0);
        return (a_value * b1 + b_value * b2) / (b1 + b2);
    } else {
        float contrast = 1.0 - blend_sharpness;
        float factor = (blend - contrast) / contrast;
        return mix(a_value, b_value, clamp(factor, 0.0, 1.0));
    }
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
        // Initialize with base texture
        vec4 current_albedo = base_albedo_ht;
        vec4 current_normal = base_normal_rg;
        float current_nrm_depth = base_nrm_depth;
        float current_ao_str = base_ao_str;
        float total_weight = 0.0;

        // Calculate slope factor for texture suppression (0 = flat, ~1 = vertical)
        float slope_factor = 1.0 - TANGENT_WORLD_MATRIX[1].y;
        float base_factor = clamp(pow(slope_factor, 2.0) * auto_slope, 0.0, 1.0);
        base_factor = clamp(base_factor - auto_height_reduction * 0.0005 * i_height, 0.0, 1.0);

        // Compute blend weight for transitioning from base to auto textures
        float blend_weight = smoothstep(0.0, 0.1, base_factor);
        blend_weight = mix(blend_weight, smoothstep(0.0, 0.2, base_factor), blend_sharpness);
        float noise = texture(noise_texture, base_uv * noise3_scale).r;
        blend_weight = clamp(blend_weight + (noise * 0.2 - 0.1) * overlay_blend_opacity, 0.0, 1.0);

        // Compute height factors for all auto textures
        float total_height_factor = 0.0;
        float height_factors[MAX_AUTO_TEXTURES];
        for (int i = 0; i < num_auto_textures && i < MAX_AUTO_TEXTURES; i++) {
            float lower_height = auto_texture_lower_heights[i];
            float upper_height = auto_texture_upper_heights[i];
            float trans_range = abs(auto_texture_transition_ranges[i]);
            height_factors[i] = 0.0;
            // Check if height is within texture's range
            if (i_height >= lower_height - trans_range && i_height <= upper_height + trans_range) {
                if (i_height >= lower_height && i_height <= upper_height) {
                    height_factors[i] = 1.0; // Full influence within core range
                } else {
                    // Smooth transition at edges
                    float blend_edge_lower = lower_height - trans_range;
                    float blend_edge_upper = upper_height + trans_range;
                    if (i_height < lower_height) {
                        height_factors[i] = smoothstep(blend_edge_lower, lower_height, i_height);
                    } else {
                        height_factors[i] = 1.0 - smoothstep(upper_height, blend_edge_upper, i_height);
                    }
                }
            }
            total_height_factor += height_factors[i];
        }

        // Apply auto textures with normalized weights
        for (int i = 0; i < num_auto_textures && i < MAX_AUTO_TEXTURES; i++) {
            int tex_slot = auto_texture_slots[i];
            float lower_height = auto_texture_lower_heights[i];
            float upper_height = auto_texture_upper_heights[i];
            float trans_range = abs(auto_texture_transition_ranges[i]);
            float noise_scale = auto_texture_noise_scales[i];

            // Skip invalid texture slots or height ranges
            if (tex_slot < 0 || tex_slot >= 32 || lower_height > upper_height) continue;

            float height_factor = height_factors[i];
            if (height_factor > 0.0) {
                // Apply noise for smoother transitions at height boundaries
                float noise_factor = 0.0;
                if ((i_height >= lower_height - trans_range && i_height < lower_height) ||
                    (i_height > upper_height && i_height <= upper_height + trans_range)) {
                    float noise = texture(noise_texture, base_uv * noise_scale).r;
                    float noise2 = texture(noise_texture, base_uv * noise_scale * 0.5).r;
                    noise_factor = (noise * 0.5 + noise2 * 0.5 - 0.5) * overlay_blend_opacity * 0.3;
                }
                float tex_blend = clamp(height_factor + noise_factor * (1.0 - height_factor), 0.0, 1.0);

                // Normalize blend factor based on total height influence
                if (total_height_factor > 0.0) {
                    tex_blend = tex_blend / total_height_factor;
                }

                // Suppress snow (slot 2) on steep slopes
                if (i == 2 && blend_weight > 0.8) {
                    tex_blend *= (1.0 - blend_weight);
                }

                if (tex_blend > 0.0) {
                    // Sample auto texture with detiling
                    float tex_scale = _texture_uv_scale_array[tex_slot];
                    float tex_angle = uv_rotation + p_angle;
                    vec2 texUV = detiling(base_uv * tex_scale, index_pos * tex_scale, tex_slot, tex_angle);
                    vec4 dd_tex = ddxy * tex_scale;
                    dd_tex.xy = rotate_plane(dd_tex.xy, -tex_angle);
                    dd_tex.zw = rotate_plane(dd_tex.zw, -tex_angle);
                    vec4 tex_albedo_ht = textureGrad(_texture_array_albedo, vec3(texUV, float(tex_slot)), dd_tex.xy, dd_tex.zw);
                    vec4 tex_normal_rg = textureGrad(_texture_array_normal, vec3(texUV, float(tex_slot)), dd_tex.xy, dd_tex.zw);
                    tex_normal_rg.xyz = unpack_normal(tex_normal_rg);
                    tex_normal_rg.xz = rotate_plane(tex_normal_rg.xz, -tex_angle);

                    // Apply texture color and roughness
                    tex_albedo_ht.rgb *= _texture_color_array[tex_slot].rgb;
                    tex_normal_rg.a = clamp(tex_normal_rg.a + _texture_roughness_mod_array[tex_slot], 0.0, 1.0);

                    // Determine blending mode: overwrite or blend
                    float influence = tex_blend;
                    if (auto_texture_overwrites_base[i] != 0) {
                        influence = tex_blend; // Overwrite base texture
                    } else {
                        float auto_influence = 1.0 - pow(blend_weight, 5.0); // Strong suppression
                        influence = tex_blend * auto_influence; // Blend with base
                    }

                    // Accumulate auto texture contribution
                    if (total_weight == 0.0) {
                        current_albedo = mix(base_albedo_ht, tex_albedo_ht, influence);
                        current_normal = mix(base_normal_rg, tex_normal_rg, influence);
                        current_nrm_depth = mix(base_nrm_depth, _texture_normal_depth_array[tex_slot], influence);
                        current_ao_str = mix(base_ao_str, _texture_ao_strength_array[tex_slot], influence);
                    } else {
                        current_albedo = height_blend4(current_albedo, current_albedo.a, tex_albedo_ht, tex_albedo_ht.a, influence);
                        current_normal = height_blend4(current_normal, current_albedo.a, tex_normal_rg, tex_albedo_ht.a, influence);
                        current_nrm_depth = height_blend1(current_nrm_depth, current_albedo.a, _texture_normal_depth_array[tex_slot], tex_albedo_ht.a, influence);
                        current_ao_str = height_blend1(current_ao_str, current_albedo.a, _texture_ao_strength_array[tex_slot], tex_albedo_ht.a, influence);
                    }
                    total_weight += influence;
                }
            }
        }

        // Finalize material properties
        if (total_weight > 0.0) {
            float norm_factor = 1.0 / max(total_weight, 1.0);
            current_albedo.a *= norm_factor;
            current_normal.a *= norm_factor;
            out_mat.alb_ht = current_albedo;
            out_mat.nrm_rg = current_normal;
            out_mat.nrm_depth = current_nrm_depth;
            out_mat.ao_str = current_ao_str;
            out_mat.blend = total_weight;
        } else {
            // Fall back to base texture if no auto textures applied
            out_mat.alb_ht = base_albedo_ht;
            out_mat.nrm_rg = base_normal_rg;
            out_mat.nrm_depth = base_nrm_depth;
            out_mat.ao_str = base_ao_str;
            out_mat.blend = 1.0;
        }

        // Debug outputs for visualizing specific values (commented out)
        // out_mat.alb_ht.rgb = vec3(total_weight / 2.0); // Red: total_weight
        // out_mat.alb_ht.rgb = vec3(blend_weight); // Green: blend_weight
        // out_mat.alb_ht.rgb = vec3(base_factor); // Blue: base_factor
        // out_mat.alb_ht.rgb = vec3(i_height / 200.0); // Purple: i_height
    } else {
        // Use base texture if auto shader is disabled
        out_mat.alb_ht = albedo_ht;
        out_mat.nrm_rg = normal_rg;
    }
}

// Computes blending weights for bilinear interpolation with noise
float blend_weights(float weight, float detail) {
    weight = smoothstep(0.0, 1.0, weight);
    weight = sqrt(weight * 0.5);
    float result = max(0.1 * weight, fma(10.0, (weight + detail), 1.0f - (detail + 10.0)));
    return result;
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

        // Apply noise to weights for natural blending
        float noise3 = texture(noise_texture, uv * noise3_scale).r * blend_sharpness;
        #define PARABOLA(x) (4.0 * x * (1.0 - x))
        weights = smoothstep(0.0, 1.0, weights);
        weights = vec4(
            blend_weights(weights.x + PARABOLA(weights.x) * noise3, mat[0].alb_ht.a),
            blend_weights(weights.y + PARABOLA(weights.y) * noise3, mat[1].alb_ht.a),
            blend_weights(weights.z + PARABOLA(weights.z) * noise3, mat[2].alb_ht.a),
            blend_weights(weights.w + PARABOLA(weights.w) * noise3, mat[3].alb_ht.a)
        );
        #undef PARABOLA
        weights *= 1.0 / (weights.x + weights.y + weights.z + weights.w);

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
