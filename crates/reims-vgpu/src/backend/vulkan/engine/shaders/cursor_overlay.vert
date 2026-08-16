#version 450
// Draws a quad (triangle strip, 4 verts) at a push-constant NDC rect, emitting
// UVs scaled to the used sub-region of the cursor texture. No vertex buffer.
layout(push_constant) uniform Push {
    vec4 rect;     // (x0, y0, x1, y1) in clip/NDC space
    vec2 uv_scale; // used cursor area as a fraction of the texture (w/tex, h/tex)
} pc;
layout(location = 0) out vec2 v_uv;
void main() {
    vec2 c = vec2(float(gl_VertexIndex & 1), float((gl_VertexIndex >> 1) & 1));
    gl_Position = vec4(mix(pc.rect.xy, pc.rect.zw, c), 0.0, 1.0);
    v_uv = c * pc.uv_scale;
}
