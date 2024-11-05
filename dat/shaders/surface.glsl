#ifdef VERTEX_SHADER
#define VS2FS out
#else
#define VS2FS in
#endif

#ifdef VERTEX_SHADER
layout(location=0) in vec3 position;
layout(location=1) in vec3 normal;
layout(location=2) in vec2 uv;
#endif

uniform mat4 model_to_ndc;
uniform sampler2D albedo_texture;

VS2FS vec2 v_uv;

#ifdef VERTEX_SHADER
void main() {
	v_uv = uv;
	gl_Position = model_to_ndc * vec4(position, 1);
}
#endif

#ifdef FRAGMENT_SHADER

out vec4 fragment_color;

float map(float x, float a, float b, float c, float d) { return (x - a) / (b - a) * (d - c) + c; }
float map_clamped(float x, float a, float b, float c, float d) { return (clamp(x, min(a, b), max(a, b)) - a) / (b - a) * (d - c) + c; }

void main() {
	fragment_color = texture(albedo_texture, v_uv);
}

#endif