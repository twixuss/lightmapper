#ifdef VERTEX_SHADER
#define VS2FS out
#else
#define VS2FS in
#endif

uniform mat4 model_to_ndc;
uniform bool dashed;
uniform vec2 screen_size;

VS2FS vec4 v_color;
VS2FS vec4 v_ndc;
VS2FS vec4 v_ndc_line_start;
VS2FS vec4 v_ndc_line_end;

#ifdef VERTEX_SHADER

struct BufferVertex {
	float[3] position;
	float[4] color;
};

struct Vertex {
	vec3 position;
	vec4 color;
};

layout(std430, binding=0) restrict readonly buffer Vertices { BufferVertex vertices[]; };

vec2 to_vec2(float[2] f) { return vec2(f[0], f[1]); }
vec3 to_vec3(float[3] f) { return vec3(f[0], f[1], f[2]); }
vec4 to_vec4(float[4] f) { return vec4(f[0], f[1], f[2], f[3]); }
Vertex unfuck(BufferVertex b) {
	Vertex v;
	v.position = to_vec3(b.position);
	v.color = to_vec4(b.color);
	return v;
}

void main() {
	
	BufferVertex bv = vertices[gl_VertexID];
	Vertex v = unfuck(bv);
	
	BufferVertex bv0 = vertices[gl_VertexID & -2];
	Vertex v0 = unfuck(bv0);

	BufferVertex bv1 = vertices[gl_VertexID & -2 | 1];
	Vertex v1 = unfuck(bv1);

	v_ndc_line_start = model_to_ndc * vec4(v0.position, 1);
	v_ndc_line_end = model_to_ndc * vec4(v1.position, 1);

	gl_Position = model_to_ndc * vec4(v.position, 1);
	v_ndc = gl_Position;
	v_color = v.color;
}
#endif

#ifdef FRAGMENT_SHADER

out vec4 fragment_color;

#define map(x, a, b, c, d) (((x)-(a))/((b)-(a))*((d)-(c))+(c))

void main() {
	vec2 line_start = map(v_ndc_line_start.xy / v_ndc_line_start.w, vec2(-1,-1), vec2(1,1), vec2(0,0), screen_size);
	vec2 line_end = map(v_ndc_line_end.xy / v_ndc_line_end.w, vec2(-1,-1), vec2(1,1), vec2(0,0), screen_size);
	if (fract(dot(gl_FragCoord.xy - line_start, normalize(line_end - line_start))/10) < 0.5)
		discard;
	fragment_color = v_color;
}

#endif