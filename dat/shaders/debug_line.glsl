#ifdef VERTEX_SHADER
#define VS2FS out
#else
#define VS2FS in
#endif

#ifdef VERTEX_SHADER
layout(location=0) in vec3 position;
layout(location=1) in vec4 color;
#endif

uniform mat4 model_to_ndc;

VS2FS vec4 v_color;

#ifdef VERTEX_SHADER
void main() {
	gl_Position = model_to_ndc * vec4(position, 1);
	v_color = color;
}
#endif

#ifdef FRAGMENT_SHADER

out vec4 fragment_color;

void main() {
	fragment_color = v_color;
}

#endif