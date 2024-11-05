#define TL_IMPL
#define TL_USE_SIMD 1
#include <tl/main.h>
#include <tl/gltf.h>
#include <tl/win32.h>
#include <tl/opengl.h>
#include <tl/math_random.h>
#include <tl/thread.h>
#include <tl/cpu.h>
#include <tl/includer.h>
#include <tl/image_filters.h>

#pragma push_macro("assert")

#include <imgui.h>
#include <backends/imgui_impl_opengl3.h>
#include <backends/imgui_impl_win32.h>
ImVec2 operator-(ImVec2 a, ImVec2 b) { return {a.x - b.x, a.y - b.y}; }

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#pragma pop_macro("assert")

ImVec2 raw_input_mouse_delta;

using namespace tl;

using String = Span<utf8>;

template <class Pixel>
struct Image {
	Pixel *pixels;
	v2u size;

	forceinline constexpr Pixel &at(u32 x, u32 y) { return pixels[y*size.x + x]; }
	forceinline constexpr Pixel &at(v2u v) { return at(v.x, v.y); }
};

struct RasterizeOptions {
	// bool cull_back = true; // TODO: implement
};

void rasterize(triangle<v2f> triangle, aabb<v2s> bounds, auto &&process_pixel, RasterizeOptions options = {}) {
	// NOTE: +x is right, +y is down.

	v2f tip = triangle.a;
	v2f left = triangle.b;
	v2f right = triangle.c;

	if (tip.y > left.y) { Swap(tip, left); }
	if (tip.y > right.y) { Swap(tip, right); }
	f32 left_slope = (tip.x - left.x) / (tip.y - left.y);
	f32 right_slope = (tip.x - right.x) / (tip.y - right.y);
	if (left_slope > right_slope) {
		Swap(left, right);
		Swap(left_slope, right_slope);
	}

	s32 y = max(bounds.min.y, round_to_int(tip.y));

	auto draw_triangle_half = [&y, &tip, &left, &right, &bounds, &process_pixel](s32 end) {
		for (; y <= end; ++y) {
			s32 xmin = max(bounds.min.x, round_to_int(tip.x + (tip.x -  left.x) * (y - tip.y) / (tip.y -  left.y)));
			s32 xmax = min(bounds.max.x, round_to_int(tip.x + (tip.x - right.x) * (y - tip.y) / (tip.y - right.y)));
			for (s32 x = xmin; x < xmax; ++x) {
				process_pixel(x, y);
			}
		}
	};

	if (left_slope > -1e4f && right_slope < 1e4f) {
		draw_triangle_half(min(floor_to_int(min(left.y, right.y)), bounds.max.y - 1));
	}
	
	if (tip.y < left.y) { Swap(tip, left); }
	if (tip.y < right.y) { Swap(tip, right); }
	left_slope = (tip.x - left.x) / (tip.y - left.y);
	right_slope = (tip.x - right.x) / (tip.y - right.y);
	if (left_slope < right_slope) {
		Swap(left, right);
		Swap(left_slope, right_slope);
	}
	
	if (left_slope < 1e4f && right_slope > -1e4f) {
		draw_triangle_half(min(round_to_int(tip.y), bounds.max.y - 1));
	}
}

v2f p0 = {64, 64};
v2f p1 = {192, 64};
v2f p2 = {64, 192};
v2f p3 = {192, 192};

// https://developer.download.nvidia.com/cg/acos.html
f32 acos_approx(f32 x) {
	f32 negate = x < 0;
	x = abs(x);
	f32 ret = -0.0187293f;
	ret = ret * x;
	ret = ret + 0.0742610f;
	ret = ret * x;
	ret = ret - 0.2121144f;
	ret = ret * x;
	ret = ret + 1.5707288f;
	ret = ret * sqrtf(1.0f-x);
	ret = ret - 2 * negate * ret;
	return negate * pi + ret;
}

v3f point_on_spherical_fibonacci_lattice(u32 index, u32 point_count) {
	// std: 51ms
	// bhaskara: 53ms
	// acos_approx: 51ms

	constexpr f32 tau_times_golden_ratio = (f32)10.166407384630519631619018026484L;

	f32 phi = acos(1 - 2 * ((f32)index / point_count)); // polar angle
	f32 theta = tau_times_golden_ratio * index; // azimuthal angle (golden angle)
	//f32 theta = pi * (1 + sqrt(5)) * index; // azimuthal angle (golden angle)

	//return {
	//	sinf(phi) * cosf(theta),
	//	sinf(phi) * sinf(theta),
	//	cosf(phi),
	//};
	
	v2f phi_cs = cos_sin(phi);
	v2f theta_cs = cos_sin(theta);

	return {
		phi_cs.y * theta_cs.x,
		phi_cs.y * theta_cs.y,
		phi_cs.x,
	};
}

List<v3f> generate_subdivided_icosahedron(u32 levels) {
	List<v3f> result;

	result.add({
		{0, 1, golden_ratio},
		{golden_ratio, 0, 1},
		{0, -1, golden_ratio},
		{-golden_ratio, 0, 1},
		{ 0, 1, -golden_ratio },
		{ 1, 0, -golden_ratio },
		{ 0, -1, -golden_ratio },
		{ -1, 0, -golden_ratio },
		{ 1, golden_ratio, 0 },
		{ 1, -golden_ratio, 0 },
		{ -1, golden_ratio, 0 },
		{ -1, -golden_ratio, 0 }
	});

	return result;
}


u32 total_iterations = 0;

TaskQueueThreadPool thread_pool;

// aka BVH
struct RangeBinaryTree {
	static constexpr u32 invalid_index = -1;

	struct Node {
		aabb<v3f> range = {};
		bool is_leaf = false;
		u16 depth;
		union {
			struct {
				u32 left_node_index;
				u32 right_node_index;
			};
			struct {
				u32 start_triangle_index;
				u32 triangle_count;
			};
		};
	};

	// `nodes[0]` is root
	List<Node> nodes;
	List<triangle<v3f>> triangles;
};

aabb<v3f> bounds_of(Span<triangle<v3f>> triangles) {
	aabb<v3f> range = {V3f(max_value<f32>), V3f(min_value<f32>)};
	for (auto triangle : triangles) {
		range = include(range, triangle.a);
		range = include(range, triangle.b);
		range = include(range, triangle.c);
	}
	return range;
}

struct CreateRangeBinaryTreeOptions {
	u32 max_triangles_per_node = 16;
	f32 range_inflation = 1e-3f;
};

RangeBinaryTree create_range_binary_tree(CommonMesh mesh, CreateRangeBinaryTreeOptions options = {}) {
	// I just can't make this show up in debugger.. It fails for unknown reason. It's not debug format, it's not natvis error, it not the names or their hierarchy.
	// It's some stupid vs bug from years ago that makes returned variables not show up. What the fuck...
	// So workaround is to `return {}` when debugging.
	RangeBinaryTree tree;

	assert(mesh.indices.count % 3 == 0);
	tree.triangles.reserve(mesh.indices.count / 3);
	for (umm i = 0; i < mesh.indices.count; i += 3) {
		u32 i0 = mesh.indices.data[i + 0];
		u32 i1 = mesh.indices.data[i + 1];
		u32 i2 = mesh.indices.data[i + 2];
		tree.triangles.add({
			mesh.vertices.data[i0].position,
			mesh.vertices.data[i1].position,
			mesh.vertices.data[i2].position,
		});
	}

	struct StuffToProcess {
		Span<triangle<v3f>> triangles;
		aabb<v3f> range;
		u32 parent_node_index = RangeBinaryTree::invalid_index;
		bool is_left = false;
		u16 depth = 0;
	};

	scoped(temporary_storage_checkpoint);

	List<StuffToProcess, TemporaryAllocator> stuff_stack;

	stuff_stack.add({
		.triangles = tree.triangles,
		.range = bounds_of(tree.triangles),
		.parent_node_index = RangeBinaryTree::invalid_index,
		.depth = 0,
	});

	List<triangle<v3f>, TemporaryAllocator> temp_triangle_buffers[3];

	while (stuff_stack.count) {
		auto stuff = stuff_stack.data[--stuff_stack.count];

		aabb<v3f> range = stuff.range;

		u32 node_index = tree.nodes.count;
		auto &node = tree.nodes.add();
		node.range = extend(range, V3f(options.range_inflation));
		node.depth = stuff.depth;

		if (stuff.parent_node_index != RangeBinaryTree::invalid_index) {
			if (stuff.is_left)
				tree.nodes[stuff.parent_node_index].left_node_index = node_index;
			else
				tree.nodes[stuff.parent_node_index].right_node_index = node_index;
		}

		if (stuff.triangles.count <= options.max_triangles_per_node) {
			node.is_leaf = true;
			node.start_triangle_index = stuff.triangles.data - tree.triangles.data;
			node.triangle_count = stuff.triangles.count;
		} else {
			node.is_leaf = false;

			v3f size = range.size();
			u32 max_size_component_index = 0;
			if (size.y > size.x)
				max_size_component_index = 1;
			if (size.z > size.s[max_size_component_index])
				max_size_component_index = 2;
			
			auto mid = midpoint(stuff.triangles.begin(), stuff.triangles.end());
			auto left_triangles = Span(stuff.triangles.begin(), mid);
			auto right_triangles = Span(mid, stuff.triangles.end());
			struct Trial {
				aabb<v3f> left_bounds;
				aabb<v3f> right_bounds;
				f32 badness = infinity<f32>;
				int component = 0;
			};

			Trial best_trial;

			for (int component = 0; component < 3; ++component) {
				auto &temp_triangle_buffer = temp_triangle_buffers[component];
				temp_triangle_buffer.set(stuff.triangles);
				auto mid = midpoint(temp_triangle_buffer.begin(), temp_triangle_buffer.end());
				auto left_triangles = Span(temp_triangle_buffer.begin(), mid);
				auto right_triangles = Span(mid, temp_triangle_buffer.end());

				quick_sort(temp_triangle_buffer, [=](triangle<v3f> t) {
					return (
						t.a.s[component] +
						t.b.s[component] + 
						t.c.s[component]) * (1.0f/3);
				});

				auto get_badness = [](Span<triangle<v3f>> triangles, aabb<v3f> bounds) {
					#if 0
					return length(stddev(mapped(triangles, [](auto t) { return t.a + t.b + t.c; })));
					#else
					v3f s = bounds.size();
					return (s.x * (s.y + s.z) + s.y * s.z) * 2;
					#endif
				};

				Trial new_trial;
				new_trial.left_bounds = bounds_of(left_triangles);
				new_trial.right_bounds = bounds_of(right_triangles);
				#if 0
				new_trial.badness = volume(new_trial.left_bounds) + volume(new_trial.right_bounds);
				#elif 0
				new_trial.badness = stddev(new_trial.left_bounds.size().s) + stddev(new_trial.right_bounds.size().s);
				#elif 1
				new_trial.badness = get_badness(left_triangles, new_trial.left_bounds) + get_badness(right_triangles, new_trial.right_bounds);
				#endif
				new_trial.component = component;
				if (new_trial.badness < best_trial.badness) {
					best_trial = new_trial;
				}
			}

			stuff.triangles.set(temp_triangle_buffers[best_trial.component]);
			
			stuff_stack.add({
				.triangles = left_triangles,
				.range = best_trial.left_bounds,
				.parent_node_index = node_index,
				.is_left = true,
				.depth = (u16)(stuff.depth + 1),
			});
			stuff_stack.add({
				.triangles = right_triangles,
				.range = best_trial.right_bounds,
				.parent_node_index = node_index,
				.is_left = false,
				.depth = (u16)(stuff.depth + 1),
			});
		}
	}

	int stupid_debugger = 1;

	return tree;
}

u32 get_depth(RangeBinaryTree tree) {
	u32 depth = 0;
	for (auto node : tree.nodes) {
		depth = max(depth, (u32)node.depth);
	}
	return depth;
}

void check_tree(RangeBinaryTree tree) {
	List<bool> triangle_inclusion;
	triangle_inclusion.resize(tree.triangles.count);
	defer { free(triangle_inclusion); };

	for (auto node : tree.nodes) {
		if (node.is_leaf) {
			aabb<v3f> range = {V3f(max_value<f32>), V3f(min_value<f32>)};
			for (auto &triangle : tree.triangles.subspan(node.start_triangle_index, node.triangle_count)) {
				triangle_inclusion[index_of(tree.triangles, &triangle)] = true;
				range = include(range, triangle.a);
				range = include(range, triangle.b);
				range = include(range, triangle.c);
			}

			assert(in_bounds(range, node.range));
		} else {
			aabb<v3f> range = {V3f(max_value<f32>), V3f(min_value<f32>)};
			range = include(range, tree.nodes[node.left_node_index].range);
			range = include(range, tree.nodes[node.right_node_index].range);

			assert(in_bounds(range, node.range));
			assert(node.depth + 1 == tree.nodes[node.left_node_index].depth);
			assert(node.depth + 1 == tree.nodes[node.right_node_index].depth);
		}
	}

	assert(all(triangle_inclusion));
}

forceinline Optional<RaycastHit<v3f>> raycast(ray<v3f> ray, RangeBinaryTree tree, RaycastTriangleOptions options = {}) {
	if (tree.nodes.count == 0)
		return {};

	Optional<RaycastHit<v3f>> result;
	
	StaticList<u32, 32> nodes_to_check;
	nodes_to_check.add(0);
	
	while (nodes_to_check.count) {
		auto node = tree.nodes[nodes_to_check.data[--nodes_to_check.count]];
		auto range_hit = raycast(ray, node.range);
		if (range_hit.count) {
			if (node.is_leaf) {
				for (umm i = 0; i < node.triangle_count; ++i) {
					auto triangle = tree.triangles[node.start_triangle_index + i];
					result = min(result, raycast(ray, triangle, options));
				}
			} else {
				nodes_to_check.add({
					node.left_node_index,
					node.right_node_index,
				});
			}
		}
	}

	constexpr bool debug = false;
	if constexpr (debug) {
		Optional<RaycastHit<v3f>> definitely_valid;
		for (auto triangle : tree.triangles) {
			definitely_valid = min(definitely_valid, raycast(ray, triangle, options));
		}

		if (result || definitely_valid) {
			assert(result && definitely_valid);

			assert(all(result.value().position == definitely_valid.value().position));
			assert(all(result.value().normal == definitely_valid.value().normal));
			assert(all(result.value().distance == definitely_valid.value().distance));
		}
	}

	return result;
}

struct LightmapBaker {
	CommonMesh mesh;
	Image<v4f> image;

	void init(CommonMesh mesh, Image<v4f> image) {
		this->mesh = mesh;
		this->image = image;
	}

	void step(RangeBinaryTree tree, u32 ray_count, u32 max_iters, xorshift32 &rng) {

		//rasterize(p0, p1, p2,
		//	{{}, (v2s)image.size},
		//	[&](umm x, umm y) {
		//		// v4u8 r;
		//		// r.xyz = (v3u8)(255*clamp(barycentric(p0, p1, p2, V2f(x, y)), V3f(0), V3f(1)));
		//		// r.w = 255;
		//		// image.at(x, y) = r;
		//		image.at(x, y) |= v4u8{255, 0, 0, 255};
		//	}
		//);
		//
		//rasterize(p1, p2, p3,
		//	{{}, (v2s)image.size},
		//	[&](umm x, umm y) {
		//		// v4u8 r;
		//		// r.xyz = (v3u8)(255*clamp(barycentric(p0, p1, p2, V2f(x, y)), V3f(0), V3f(1)));
		//		// r.w = 255;
		//		// image.at(x, y) = r;
		//		image.at(x, y) |= v4u8{0, 255, 0, 255};
		//	}
		//);
		//for (umm y = 0; y < image.size.y; ++y) {
		//	for (umm x = 0; x < image.size.x; ++x) {
		//		image.at(x, y) = all(image.at(x, y) == v4u8{255,255,0,255}) ? v4u8{0,0,255,255} : image.at(x, y);
		//	}
		//}

		umm worker_count = thread_pool.threads.count;

		for (umm worker_index = 0; worker_index < worker_count; ++worker_index) {
			thread_pool += [=, &rng] {
				//println("doing task on thread {}", GetCurrentThreadId());
				umm start = mesh.indices.count * (worker_index + 0) / worker_count / 3 * 3;
				umm end   = mesh.indices.count * (worker_index + 1) / worker_count / 3 * 3;
				for (umm i = start; i < end; i += 3) {
					auto v0 = mesh.vertices.data[mesh.indices.data[i + 0]];
					auto v1 = mesh.vertices.data[mesh.indices.data[i + 1]];
					auto v2 = mesh.vertices.data[mesh.indices.data[i + 2]];
					auto uv0 = v0.uv * (v2f)image.size;
					auto uv1 = v1.uv * (v2f)image.size;
					auto uv2 = v2.uv * (v2f)image.size;
					//v3f wn = normalize(cross(v0.position - v1.position, v0.position - v2.position));
					//v0.position += wn * 0.01f;
					//v1.position += wn * 0.01f;
					//v2.position += wn * 0.01f;
	
					//StaticList<u32, 256> neighbor_triangles;
					//
					//for (umm j = 0; j < mesh.indices.count; j += 3) {
					//	StaticSet<v3f, 6> position_set;
					//	position_set.get_or_insert(mesh.vertices.data[mesh.indices.data[i + 0]].position);
					//	position_set.get_or_insert(mesh.vertices.data[mesh.indices.data[i + 1]].position);
					//	position_set.get_or_insert(mesh.vertices.data[mesh.indices.data[i + 2]].position);
					//	position_set.get_or_insert(mesh.vertices.data[mesh.indices.data[j + 0]].position);
					//	position_set.get_or_insert(mesh.vertices.data[mesh.indices.data[j + 1]].position);
					//	position_set.get_or_insert(mesh.vertices.data[mesh.indices.data[j + 2]].position);
					//	if (position_set.count < 6) {
					//		// At least one vertex is shared
					//		neighbor_triangles.add(j);
					//	}
					//}

					rasterize(
						{uv0, uv1, uv2},
						{{}, (v2s)image.size},
						[&](umm x, umm y) {
							v3f bc = barycentric({uv0, uv1, uv2}, V2f(x, y));

							//if (any(bc < 0) || any(bc > 1)) {
							//	println(bc);
							//}

							v3f wp = v0.position * bc.x + v1.position * bc.y + v2.position * bc.z;
							v3f wn = v0.normal * bc.x + v1.normal * bc.y + v2.normal * bc.z;

							u32 hit_count = 0;
							for (umm ray_index = 0; ray_index < ray_count; ++ray_index) {
								v3f ray_direction = point_on_spherical_fibonacci_lattice(((total_iterations * ray_count + ray_index) * next_prime_after_2_to_power_of_31_point_5) % max_iters, max_iters); 
								//v3f ray_direction = random_unit_v3f(rng); 
								ray_direction = face_toward(ray_direction, wn);

								#if 1
								// Raycast BVH
								auto r = raycast({wp, ray_direction}, tree, RaycastTriangleOptions{.min_distance = 1e-2f});
								if (r) {
									++hit_count;
									break;
								}
								#else
								// Raycast all other triangles
								for (umm j = 0; j < mesh.indices.count; j += 3/*(1 + (i == j + 3)) * 3*/) {
									auto nv0 = mesh.vertices.data[mesh.indices.data[j + 0]].position;
									auto nv1 = mesh.vertices.data[mesh.indices.data[j + 1]].position;
									auto nv2 = mesh.vertices.data[mesh.indices.data[j + 2]].position;
									auto r = raycast({wp, ray_direction}, triangle{nv0, nv1, nv2}, RaycastTriangleOptions{.min_distance = 1e-3f});
									if (r) {
										if (find(neighbor_triangles.span(), (u32)j)) {
											bool hit_from_front = dot(r.value().normal, ray_direction) < 0;
											if (hit_from_front) {
												++hit_count;
												break;
											}
										} else {
											++hit_count;
											break;
										}
									}
								}
								#endif
							}

							v4f r = image.at(x, y);
							r.xyz += V3f(map_clamped<f32,f32>(hit_count, 0, ray_count, 1, 0));
							//r.xyz = (v3u8)map_clamped(wp, V3f(-8), V3f(8), V3f(0), V3f(255));
							r.w = 1;
							image.at(x, y) = r;
						}
					);
				}
				//println("stopped task on thread {}", GetCurrentThreadId());
			};
		}
	
		thread_pool.wait_for_completion(WaitForCompletionOption::just_wait);
	}
};

v2u screen_size;
bool screen_size_changed = true;

LRESULT WINAPI wnd_proc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
	extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);
	if (ImGui_ImplWin32_WndProcHandler(hwnd, msg, wp, lp))
		return true;


	switch (msg) {
		case WM_CLOSE: {
			PostQuitMessage(0);
			return 0;
		}
		case WM_SIZE: {
			screen_size = {LOWORD(lp), HIWORD(lp)};
			screen_size_changed = true;
			break;
		}
	}
	return DefWindowProc(hwnd, msg, wp, lp);
}

void bind_texture(GLuint program, u32 slot, char const *name, GLuint texture, GLuint sampler = 0, GLenum target = GL_TEXTURE_2D) {
	glActiveTexture(GL_TEXTURE0 + slot); 
	glBindTexture(target, texture); 
	gl::set_uniform(program, name, (int)slot); 
	glBindSampler(slot, sampler);
}

struct GpuMesh {
	GLuint vb, ib, va;
	u32 vertex_count, index_count;
};

GpuMesh to_gpu_mesh(CommonMesh mesh) {
	struct MyVertex {
		v3f position;
		v3f normal;
		v2f uv;
	};
	
	List<MyVertex> vertices;
	defer { free(vertices); };
	
	for (auto v : mesh.vertices) {
		vertices.add({
			.position = v.position,
			.normal = v.normal,
			.uv = v.uv,
		});
	}
	
	auto &indices = mesh.indices;

	GpuMesh result = {};
	
	glCreateBuffers(1, &result.vb);
	glNamedBufferStorage(result.vb, sizeof(vertices[0]) * vertices.count, vertices.data, 0);

	glCreateBuffers(1, &result.ib);
	glNamedBufferStorage(result.ib, sizeof(indices[0]) * indices.count, indices.data, 0);

	glCreateVertexArrays(1, &result.va);
	glVertexArrayVertexBuffer(result.va, 0, result.vb, 0, sizeof(MyVertex));
	glVertexArrayElementBuffer(result.va, result.ib);
	glEnableVertexArrayAttrib(result.va, 0); glVertexArrayAttribBinding(result.va, 0, 0); glVertexArrayAttribFormat(result.va, 0, 3, GL_FLOAT, false, offsetof(MyVertex, position));
	glEnableVertexArrayAttrib(result.va, 1); glVertexArrayAttribBinding(result.va, 1, 0); glVertexArrayAttribFormat(result.va, 1, 3, GL_FLOAT, false, offsetof(MyVertex, normal));
	glEnableVertexArrayAttrib(result.va, 2); glVertexArrayAttribBinding(result.va, 2, 0); glVertexArrayAttribFormat(result.va, 2, 2, GL_FLOAT, false, offsetof(MyVertex, uv));

	result.vertex_count = vertices.count;
	result.index_count = indices.count;

	return result;
}

struct DebugVertex {
	v3f position;
	v4f color;
};

List<DebugVertex> debug_line_vertices;
List<DebugVertex> debug_dashed_line_vertices;
List<DebugVertex> debug_triangle_vertices;
GLuint debug_lines_vb;
GLuint debug_lines_va;
GLuint debug_dashed_lines_vb;
GLuint debug_dashed_lines_va;
GLuint debug_triangles_vb;
GLuint debug_triangles_va;

void init_debug_draw() {
	glCreateBuffers(1, &debug_lines_vb);
	glCreateVertexArrays(1, &debug_lines_va);
	glVertexArrayVertexBuffer(debug_lines_va, 0, debug_lines_vb, 0, sizeof(DebugVertex));
	glEnableVertexArrayAttrib(debug_lines_va, 0); glVertexArrayAttribBinding(debug_lines_va, 0, 0); glVertexArrayAttribFormat(debug_lines_va, 0, 3, GL_FLOAT, false, offsetof(DebugVertex, position));
	glEnableVertexArrayAttrib(debug_lines_va, 1); glVertexArrayAttribBinding(debug_lines_va, 1, 0); glVertexArrayAttribFormat(debug_lines_va, 1, 4, GL_FLOAT, false, offsetof(DebugVertex, color));

	glCreateBuffers(1, &debug_dashed_lines_vb);
	glCreateVertexArrays(1, &debug_dashed_lines_va);
	glVertexArrayVertexBuffer(debug_dashed_lines_va, 0, debug_dashed_lines_vb, 0, sizeof(DebugVertex));
	glEnableVertexArrayAttrib(debug_dashed_lines_va, 0); glVertexArrayAttribBinding(debug_dashed_lines_va, 0, 0); glVertexArrayAttribFormat(debug_dashed_lines_va, 0, 3, GL_FLOAT, false, offsetof(DebugVertex, position));
	glEnableVertexArrayAttrib(debug_dashed_lines_va, 1); glVertexArrayAttribBinding(debug_dashed_lines_va, 1, 0); glVertexArrayAttribFormat(debug_dashed_lines_va, 1, 4, GL_FLOAT, false, offsetof(DebugVertex, color));

	glCreateBuffers(1, &debug_triangles_vb);
	glCreateVertexArrays(1, &debug_triangles_va);
	glVertexArrayVertexBuffer(debug_triangles_va, 0, debug_triangles_vb, 0, sizeof(DebugVertex));
	glEnableVertexArrayAttrib(debug_triangles_va, 0); glVertexArrayAttribBinding(debug_triangles_va, 0, 0); glVertexArrayAttribFormat(debug_triangles_va, 0, 3, GL_FLOAT, false, offsetof(DebugVertex, position));
	glEnableVertexArrayAttrib(debug_triangles_va, 1); glVertexArrayAttribBinding(debug_triangles_va, 1, 0); glVertexArrayAttribFormat(debug_triangles_va, 1, 4, GL_FLOAT, false, offsetof(DebugVertex, color));
}
void update_debug_draw() {
	glNamedBufferData(debug_lines_vb, sizeof(debug_line_vertices[0]) * debug_line_vertices.count, debug_line_vertices.data, GL_STATIC_DRAW);
	glNamedBufferData(debug_dashed_lines_vb, sizeof(debug_dashed_line_vertices[0]) * debug_dashed_line_vertices.count, debug_dashed_line_vertices.data, GL_STATIC_DRAW);
	glNamedBufferData(debug_triangles_vb, sizeof(debug_triangle_vertices[0]) * debug_triangle_vertices.count, debug_triangle_vertices.data, GL_STATIC_DRAW);
}

struct DebugDrawOptions {
	v4f color = {1,0,1,1};
	bool solid = false;
	bool dashed = false;
};

void debug_line(v3f a, v3f b, DebugDrawOptions options = {}) {
	(options.dashed ? debug_dashed_line_vertices : debug_line_vertices).add({
		{a, options.color},
		{b, options.color},
	});
}

void debug_triangle(triangle<v3f> t, DebugDrawOptions options = {}) {
	if (options.solid) {
		debug_triangle_vertices.add({
			{t.a, options.color},
			{t.b, options.color},
			{t.c, options.color},
		});
	} else {
		debug_line(t.a, t.b, options);
		debug_line(t.b, t.c, options);
		debug_line(t.c, t.a, options);
	}
}
void debug_box(aabb<v3f> box, DebugDrawOptions options = {}) {
	if (options.solid) {
		for (auto triangle : to_triangles(box)) {
			debug_triangle(triangle, options);
		}
	} else {
		debug_line({box.min.x, box.min.y, box.min.z}, {box.min.x, box.min.y, box.max.z}, options);
		debug_line({box.min.x, box.max.y, box.min.z}, {box.min.x, box.max.y, box.max.z}, options);
		debug_line({box.max.x, box.min.y, box.min.z}, {box.max.x, box.min.y, box.max.z}, options);
		debug_line({box.max.x, box.max.y, box.min.z}, {box.max.x, box.max.y, box.max.z}, options);
	
		debug_line({box.min.x, box.min.y, box.min.z}, {box.min.x, box.max.y, box.min.z}, options);
		debug_line({box.min.x, box.min.y, box.max.z}, {box.min.x, box.max.y, box.max.z}, options);
		debug_line({box.max.x, box.min.y, box.min.z}, {box.max.x, box.max.y, box.min.z}, options);
		debug_line({box.max.x, box.min.y, box.max.z}, {box.max.x, box.max.y, box.max.z}, options);
	
		debug_line({box.min.x, box.min.y, box.min.z}, {box.max.x, box.min.y, box.min.z}, options);
		debug_line({box.min.x, box.min.y, box.max.z}, {box.max.x, box.min.y, box.max.z}, options);
		debug_line({box.min.x, box.max.y, box.min.z}, {box.max.x, box.max.y, box.min.z}, options);
		debug_line({box.min.x, box.max.y, box.max.z}, {box.max.x, box.max.y, box.max.z}, options);
	}
}

String program_directory;
String root_directory;

String resource_path(String relative_path) {
	return tformat(u8"{}/../dat/{}"s, program_directory, relative_path);
};

s32 tl_main(Span<String> args) {
	construct(debug_line_vertices);
	construct(debug_dashed_line_vertices);
	construct(debug_triangle_vertices);

	program_directory = parse_path(args[0]).directory;
	root_directory = format(u8"{}\\..", program_directory);

	auto file_content = read_entire_file(tformat(u8"{}\\dat\\models\\cube on floor.glb", root_directory));

	constexpr v2u image_size = {256, 256};
	static v4f pixels_v4f[image_size.x*image_size.y];
	Image<v4f> image = {
		.pixels = pixels_v4f,
		.size = image_size,
	};

	auto reset_image = [&] {
		total_iterations = 0;
		for (umm y = 0; y < image.size.y; ++y) {
			for (umm x = 0; x < image.size.x; ++x) {
				image.at(x, y) = {0,0,0,0};
			}
		}
	};

	reset_image();

	thread_pool.init(get_cpu_info().logical_processor_count);
	defer { thread_pool.deinit(); };

	HWND hwnd = create_class_and_window(u8"lightmapper"s, wnd_proc, u8"Lightmapper"s);
	
	init_rawinput(RawInput_mouse);

	gl::init_opengl((NativeWindowHandle)hwnd);
	
	auto scene = glb::parse_from_memory(file_content);
	auto &mesh = scene.meshes[0];
	auto gpu_mesh = to_gpu_mesh(mesh);

	struct Program {
		struct SourceFile : includer::SourceFileBase {
			FileTime last_write_time = 0;
			
			void init() {
				last_write_time = get_file_write_time(path).value_or(0);;
			}
			bool was_modified() {
				return last_write_time < get_file_write_time(path).value_or(0);
			}
		};

		String path;
		List<utf8> text;

		GLenum vs = 0;
		GLenum fs = 0;
		GLuint program = 0;
		includer::Includer<SourceFile> source = {
		};

		void load() {
			println("Recompiling {}", path);

			source.load(path, &text, includer::LoadOptions{
				.append_location_info = [](StringBuilder &builder, String path, u32 line) {
					append_format(builder, "\n#line {} \"{}\"\n", line, path);
				}
			});

			vs = gl::create_shader(GL_VERTEX_SHADER, 430, true, as_chars(text));
			fs = gl::create_shader(GL_FRAGMENT_SHADER, 430, true, as_chars(text));
			program = gl::create_program({.vertex = vs, .fragment = fs});

			if (!vs || !fs || !program) {
				//println("Preprocessed shader code:\n{}", preprocessed);
			}
		}

		void unload() {
			glDeleteProgram(program);
			glDeleteShader(fs);
			glDeleteShader(vs);
		}
		void free() {
			source.free();
		}
		bool needs_reload() {
			for (auto &source_file : source.source_files) {
				if (source_file.was_modified()) {
					return true;
				}
			}
			return false;
		}
	};

	Program surface_program = {.path = to_list(resource_path(u8"shaders/surface.glsl"s))};
	Program debug_line_program = {.path = to_list(resource_path(u8"shaders/debug_line.glsl"s))};
	Program debug_dashed_line_program = {.path = to_list(resource_path(u8"shaders/debug_dashed_line.glsl"s))};
	Program debug_triangle_program = {.path = to_list(resource_path(u8"shaders/debug_triangle.glsl"s))};

	Program *all_programs[] = { &surface_program, &debug_line_program, &debug_dashed_line_program, &debug_triangle_program };
	
	for (auto program : all_programs) {
		program->load();
	}
	

	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGuiIO &io = ImGui::GetIO();
	io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
	io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;
	io.ConfigDragClickToInputText = true;
	
	auto &style = ImGui::GetStyle();
	style.HoverDelayShort = 2.0f;
	
	ImGui::StyleColorsDark();

	ImGui_ImplWin32_InitForOpenGL(hwnd);
	ImGui_ImplOpenGL3_Init();
	
	init_debug_draw();

	defer{
		ImGui_ImplOpenGL3_Shutdown();
		ImGui_ImplWin32_Shutdown();
	};

	GLuint rasterized_texture;
	glGenTextures(1, &rasterized_texture);
	glBindTexture(GL_TEXTURE_2D, rasterized_texture);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
	glBindTexture(GL_TEXTURE_2D, 0);

	bool rerasterize = true;
	xorshift32 rng = {(u32)__rdtsc()};
	
	RangeBinaryTree tree = create_range_binary_tree(mesh);
	check_tree(tree);

	u32 tree_depth = get_depth(tree);
	println("tree depth: {}", tree_depth);

	LightmapBaker baker;
	baker.init(mesh, image);

	v4u8 pixels_v4u8[image_size.x * image_size.y];
	f32 bake_time = 0;

	v3f camera_position = V3f(-8.313, 7.767, -4.275);
	v3f camera_angles = V3f(0.645, 2.1, 0);
	
	f32 frame_time = 1.0f / 60;
	f64 time = 0;
	PreciseTimer frame_timer = create_precise_timer();

	int show_tree_at_depth = -2;
	int show_node_at_depth = -1;

	MSG msg;
	while (1) {
		static v2s mouse_delta;
		mouse_delta = {};
		while (PeekMessage(&msg, 0, 0, 0, PM_REMOVE)) {
			TranslateMessage(&msg);
			DispatchMessage(&msg);

			process_raw_input_message(msg, 0, 0, &mouse_delta);

			if (msg.message == WM_QUIT) {
				return 0;
			}
		}
		raw_input_mouse_delta = {(float)mouse_delta.x, (float)mouse_delta.y};
		
		for (auto program : all_programs) {
			if (program->needs_reload()) {
				program->unload();
				program->load();
			}
		}
		
		if (screen_size_changed) {
			screen_size_changed = false;
		}
	
		auto show_node = [&] (RangeBinaryTree::Node node, v4f color) {
			debug_box(node.range, {.color = color withx { it.w = 1.0f / 32; }, .solid = true});
			debug_box(node.range, {.color = color withx { it.w = 1.0f / 4; }, .solid = false});

			if (node.is_leaf) {
				for (u32 j = 0; j < node.triangle_count; ++j) {
					auto t = tree.triangles[node.start_triangle_index + j];
					auto normal = normal_of(t);
					t.a += normal * 1e-3f;
					t.b += normal * 1e-3f;
					t.c += normal * 1e-3f;
					debug_triangle(t, {.color = color withx { it.w = 1.0f / 4; }, .solid = true});
					debug_triangle(t, {.color = color withx { it.w = 1.0f / 8; }, .solid = false});
				}
			}
		};

		for (u32 i = 0; i < tree.nodes.count; ++i) {
			auto node = tree.nodes[i];
			if (show_tree_at_depth == -1 || (int)node.depth == show_tree_at_depth) {
				show_node(node, hsv_to_rgb(DefaultRandomizer{}.random<f32>(i), 1, 1, 1));
			}
		}

		if (show_node_at_depth != -1) {
			show_node(tree.nodes[show_node_at_depth], {1,0,0,1});
		}


		//println("frame");
		
		if (ImGui::IsMouseDragging(ImGuiMouseButton_Right) && !ImGui::GetIO().WantCaptureMouse) {
			camera_angles.x += mouse_delta.y * 0.003f;
			camera_angles.y += mouse_delta.x * 0.003f;
		}

		f32 speed = 5;
		if (ImGui::IsKeyDown(ImGuiKey_LeftShift)) speed *= 10;
		if (ImGui::IsKeyDown(ImGuiKey_LeftAlt)) speed /= 10;

		camera_position += m3::rotation_r_zxy(-camera_angles) * (frame_time * speed * v3f {
			(f32)(ImGui::IsKeyDown(ImGuiKey_D) - ImGui::IsKeyDown(ImGuiKey_A)),
			(f32)(ImGui::IsKeyDown(ImGuiKey_E) - ImGui::IsKeyDown(ImGuiKey_Q)),
			(f32)(ImGui::IsKeyDown(ImGuiKey_S) - ImGui::IsKeyDown(ImGuiKey_W)),
		});
		
		//println("v3f camera_position = V3f{};", camera_position);
		//println("v3f camera_angles = V3f{};", camera_angles);

		gl::clear_color(v4f{});
		glDepthMask(GL_TRUE);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		gl::viewport(screen_size);
		
		m4 world_to_ndc = m4::perspective_right_handed((f32)screen_size.x / screen_size.y, pi/2, 0.1f, 100.0f) * m4::rotation_r_yxz(camera_angles) * m4::translation(-camera_position);
		
		m4 model_to_world = m4::identity();
		m4 model_to_ndc = world_to_ndc * model_to_world;

		glUseProgram(surface_program.program);
		gl::set_uniform(surface_program.program, "model_to_ndc", model_to_ndc);
		bind_texture(surface_program.program, 0, "albedo_texture", rasterized_texture);

		glEnable(GL_DEPTH_TEST);
		glDepthMask(GL_TRUE);
		glDisable(GL_BLEND);
		glEnable(GL_CULL_FACE);
		glCullFace(GL_BACK);
		glBindVertexArray(gpu_mesh.va);
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, gpu_mesh.vb);
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, gpu_mesh.ib);
		glDrawElements(GL_TRIANGLES, gpu_mesh.index_count, GL_UNSIGNED_INT, 0);
		

		glEnable(GL_DEPTH_TEST);
		glDepthMask(GL_FALSE);
		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		update_debug_draw();
		
		glUseProgram(debug_triangle_program.program);
		gl::set_uniform(debug_triangle_program.program, "model_to_ndc", model_to_ndc);
		glBindVertexArray(debug_triangles_va);
		glDrawArrays(GL_TRIANGLES, 0, debug_triangle_vertices.count);
		
		glUseProgram(debug_line_program.program);
		gl::set_uniform(debug_line_program.program, "model_to_ndc", model_to_ndc);
		glBindVertexArray(debug_lines_va);
		glDrawArrays(GL_LINES, 0, debug_line_vertices.count);
		
		glUseProgram(debug_dashed_line_program.program);
		gl::set_uniform(debug_dashed_line_program.program, "model_to_ndc", model_to_ndc);
		gl::set_uniform(debug_dashed_line_program.program, "screen_size", (v2f)screen_size);
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, debug_dashed_lines_vb);
		glDrawArrays(GL_LINES, 0, debug_dashed_line_vertices.count);



		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplWin32_NewFrame();
		ImGui::NewFrame();
		
		ImGui::ShowDemoWindow();

		const int MAX_ITERS = 256;

		if (total_iterations < MAX_ITERS) {
			auto timer = create_precise_timer();
			baker.step(tree, 1, MAX_ITERS, rng);
			bake_time = elapsed_time(timer);

			++total_iterations;
		
			if (total_iterations == MAX_ITERS) {
				dilate(image.pixels, image.size, image.size.x);
			}

			for (umm i = 0; i < image_size.x * image_size.y; ++i) {
				pixels_v4u8[i].xyz = (v3u8)V3f(map_clamped<f32,f32>(image.pixels[i].x, 0, total_iterations, 0, 255));
				pixels_v4u8[i].w = (u8)map_clamped<f32,f32>(image.pixels[i].w, 0, 1, 0, 255);
			}
			glBindTexture(GL_TEXTURE_2D, rasterized_texture);
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, image_size.x, image_size.y, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels_v4u8);
			glBindTexture(GL_TEXTURE_2D, 0);
		}

		if (ImGui::Begin("Stuff")) {
			// rerasterize |= ImGui::DragFloat2("p0", p0.s, 0.1f);
			// rerasterize |= ImGui::DragFloat2("p1", p1.s, 0.1f);
			// rerasterize |= ImGui::DragFloat2("p2", p2.s, 0.1f);
			// rerasterize |= ImGui::DragFloat2("p3", p3.s, 0.1f);
			
			
			ImGui::SliderInt("show_tree_at_depth", &show_tree_at_depth, -2, tree_depth);
			ImGui::SliderInt("show_node_at_depth", &show_node_at_depth, -1, tree.nodes.count - 1);
			
			ImGui::Text("total_iterations: %d", total_iterations);
			ImGui::Text("bake_time: %f", bake_time);

			if (ImGui::Button("Reset")) {
				reset_image();
			}
			ImGui::SameLine();
			if (ImGui::Button("Save")) {
				stbi_write_png(tformat("{}\\bake.png\0"s, program_directory).data, image_size.x, image_size.y, 4, pixels_v4u8, image_size.x * sizeof(v4u8));
			}
		}
		ImGui::End();
		
		//if (rerasterize) {
		//	rerasterize = false;
		//	bake_lightmap(mesh, {3, 3, 5}, image);
		//	glBindTexture(GL_TEXTURE_2D, rasterized_texture);
		//	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, image.size.x, image.size.y, 0, GL_RGBA, GL_UNSIGNED_BYTE, image.pixels);
		//	glBindTexture(GL_TEXTURE_2D, 0);
		//}
		ImGui::SetNextWindowSize({256, 256}, ImGuiCond_FirstUseEver);
		ImGui::Begin("Result", 0, ImGuiWindowFlags_NoScrollbar);
		ImGui::Image((ImTextureID)rasterized_texture, ImGui::GetWindowContentRegionMax() - ImGui::GetWindowContentRegionMin());
		ImGui::End();

		ImGui::Render();
		




		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

		gl::present();

		current_temporary_allocator.clear();
		debug_line_vertices.clear();
		debug_dashed_line_vertices.clear();
		debug_triangle_vertices.clear();

		frame_time = reset(frame_timer);
		time += frame_time;
	}

	return 0;
}

