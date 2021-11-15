//
// Implementation for Yocto/PathTrace.
//

//
// LICENSE:
//
// Copyright (c) 2016 -- 2021 Fabio Pellacini
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//

#include "yocto_pathtrace.h"

#include <yocto/yocto_cli.h>
#include <yocto/yocto_geometry.h>
#include <yocto/yocto_parallel.h>
#include <yocto/yocto_sampling.h>
#include <yocto/yocto_shading.h>
#include <yocto/yocto_shape.h>

// -----------------------------------------------------------------------------
// IMPLEMENTATION FOR PATH TRACING
// -----------------------------------------------------------------------------
namespace yocto {

// Convenience functions
[[maybe_unused]] static vec3f eval_position(
    const scene_data& scene, const bvh_intersection& intersection) {
  return eval_position(scene, scene.instances[intersection.instance],
      intersection.element, intersection.uv);
}
[[maybe_unused]] static vec3f eval_normal(
    const scene_data& scene, const bvh_intersection& intersection) {
  return eval_normal(scene, scene.instances[intersection.instance],
      intersection.element, intersection.uv);
}
[[maybe_unused]] static vec3f eval_element_normal(
    const scene_data& scene, const bvh_intersection& intersection) {
  return eval_element_normal(
      scene, scene.instances[intersection.instance], intersection.element);
}
[[maybe_unused]] static vec3f eval_shading_position(const scene_data& scene,
    const bvh_intersection& intersection, const vec3f& outgoing) {
  return eval_shading_position(scene, scene.instances[intersection.instance],
      intersection.element, intersection.uv, outgoing);
}
[[maybe_unused]] static vec3f eval_shading_normal(const scene_data& scene,
    const bvh_intersection& intersection, const vec3f& outgoing) {
  return eval_shading_normal(scene, scene.instances[intersection.instance],
      intersection.element, intersection.uv, outgoing);
}
[[maybe_unused]] static vec2f eval_texcoord(
    const scene_data& scene, const bvh_intersection& intersection) {
  return eval_texcoord(scene, scene.instances[intersection.instance],
      intersection.element, intersection.uv);
}
[[maybe_unused]] static material_point eval_material(
    const scene_data& scene, const bvh_intersection& intersection) {
  return eval_material(scene, scene.instances[intersection.instance],
      intersection.element, intersection.uv);
}
[[maybe_unused]] static bool is_volumetric(
    const scene_data& scene, const bvh_intersection& intersection) {
  return is_volumetric(scene, scene.instances[intersection.instance]);
}

// Evaluates/sample the BRDF scaled by the cosine of the incoming direction.
static vec3f eval_emission(const material_point& material, const vec3f& normal,
    const vec3f& outgoing) {
  return dot(normal, outgoing) >= 0 ? material.emission : vec3f{0, 0, 0};
}

// Evaluates/sample the BRDF scaled by the cosine of the incoming direction.
static vec3f eval_bsdfcos(const material_point& material, const vec3f& normal,
    const vec3f& outgoing, const vec3f& incoming) {
  // YOUR CODE GOES HERE
  if (material.roughness == 0) return zero3f;

  switch (material.type) 
  { 
      case material_type::matte:
        return eval_matte(material.color, normal, outgoing, incoming);
      case material_type::glossy:
        return eval_glossy(material.color, material.ior, material.roughness, normal, outgoing, incoming);
      case material_type::reflective:
        return eval_reflective(material.color, material.roughness, normal, outgoing, incoming);
      case material_type::transparent:
        return eval_transparent(material.color, material.ior, material.roughness, normal, outgoing, incoming);
      default: return zero3f;
  }
  return zero3f;
}

static vec3f eval_delta(const material_point& material, const vec3f& normal,
    const vec3f& outgoing, const vec3f& incoming) {
  // YOUR CODE GOES HERE
  if (material.roughness != 0) return zero3f;

  switch (material.type) {
    case material_type::reflective:
      return eval_reflective(material.color, normal, outgoing, incoming);
    case material_type::transparent:
      return eval_transparent(material.color, material.ior, normal, outgoing, incoming);
    default: return zero3f;
  }

  return zero3f;
}

// Picks a direction based on the BRDF
static vec3f sample_bsdfcos(const material_point& material, const vec3f& normal,
    const vec3f& outgoing, float rnl, const vec2f& rn) {
  // YOUR CODE GOES HERE
  if (material.roughness == 0) return zero3f;
  switch (material.type) 
  { 
  case material_type::matte: 
      return sample_matte(material.color, normal, outgoing, rn);
  case material_type::glossy:
    return sample_glossy(material.color, material.ior, material.roughness, normal, outgoing, rnl, rn);
  case material_type::reflective:
    return sample_reflective(material.color, material.roughness, normal, outgoing, rn);
  case material_type::transparent:
    return sample_transparent(material.color, material.ior, material.roughness, normal, outgoing, rnl, rn);
  default: return zero3f;
  }
  return zero3f;
}

static vec3f sample_delta(const material_point& material, const vec3f& normal,
    const vec3f& outgoing, float rnl) {
  // YOUR CODE GOES HERE
  if (material.roughness != 0) return zero3f;
  switch (material.type) {
    case material_type::reflective:
      return sample_reflective(material.color, normal, outgoing);
    case material_type::transparent:
      return sample_transparent(material.color, material.ior, normal, outgoing, rnl);
    default: return zero3f;
  }
  return zero3f;
}

// Compute the weight for sampling the BRDF
static float sample_bsdfcos_pdf(const material_point& material,
    const vec3f& normal, const vec3f& outgoing, const vec3f& incoming) {
  if (material.roughness == 0) return 0;
  switch (material.type) {
    case material_type::matte:
      return sample_matte_pdf(material.color, normal, outgoing, incoming);
    case material_type::glossy:
      return sample_glossy_pdf(material.color, material.ior, material.roughness, normal, outgoing, incoming);
    case material_type::reflective:
      return sample_reflective_pdf(material.color, material.roughness, normal, outgoing, incoming);
    case material_type::transparent:
      return sample_tranparent_pdf(material.color, material.ior, material.roughness, normal, outgoing, incoming);
    default: 
      return 0;
  }
  return 0;
}

static float sample_delta_pdf(const material_point& material,
    const vec3f& normal, const vec3f& outgoing, const vec3f& incoming) {
  if (material.roughness != 0) return 0;
  switch (material.type) {
    case material_type::reflective:
      return sample_reflective_pdf(material.color, normal, outgoing, incoming);
    case material_type::transparent:
      return sample_tranparent_pdf(
          material.color, material.ior, normal, outgoing, incoming);
    default: return 0;
  }
  return 0;
}

// Sample lights wrt solid angle
static vec3f sample_lights(const scene_data& scene,
    const pathtrace_lights& lights, const vec3f& position, float rl, float rel,
    const vec2f& ruv) {
  // Sample random light
  unsigned int lid = sample_uniform(static_cast<int>(lights.lights.size()), rl);
  const pathtrace_light& light = lights.lights[lid];

  if (light.instance != invalidid) 
  {
    const instance_data& inst = scene.instances[light.instance];
    auto tid = sample_discrete(light.elements_cdf, rel);
    const shape_data& shape = scene.shapes[inst.shape];
    
    auto uv = !shape.triangles.empty() ? sample_triangle(ruv) : ruv;
    
    vec3f p = eval_position(scene, inst, tid, uv);
    return normalize(p - position);
  } 
  else if (light.environment != invalidid) 
  {
    const environment_data& env = scene.environments[light.environment]; 
    if (env.emission_tex == invalidid) return zero3f;

    const texture_data& texture = scene.textures[env.emission_tex];
    int tid  = sample_discrete(light.elements_cdf, rel);
    float u  = ((tid % texture.width) + ruv.x) / static_cast<float>(texture.width);
    float v  = ((tid / texture.width) + ruv.y) / static_cast<float>(texture.height);
    return transform_direction(env.frame, vec3f{cos(u * 2 * pif) * sin(v * pif), cos(v * pif), sin(u * 2 * pif) * sin(v * pif)});
  }
  
  return zero3f;
}

// Sample lights pdf
static float sample_lights_pdf(const scene_data& scene, const bvh_data& bvh,
    const pathtrace_lights& lights, const vec3f& position,
    const vec3f& direction) 
{
  float pdf = 0.f;
  for (const pathtrace_light& l : lights.lights) 
  {
    if (l.instance != invalidid) 
    {
      const instance_data& lightInstance = scene.instances[l.instance];
      float lpdf = 0.f;
      vec3f np   = position;
      for (unsigned int bounce = 0; bounce < 100; ++bounce) 
      {
        auto isec = intersect_bvh(bvh, scene, l.instance, ray3f{np, direction});
        if (!isec.hit) break;

        /*vec3f p = eval_position(scene, lightInstance, isec.element, isec.uv);
        vec3f n = eval_element_normal(scene, lightInstance, isec.element);*/

        vec3f p = eval_position(scene, isec);
        vec3f n = eval_element_normal(scene, isec);

        float A = l.elements_cdf.back();
        lpdf += distance_squared(p, position) / (abs(dot(n, direction)) * A);
        np =  p + direction * 1e-3f;
      }
      pdf += lpdf;
    }
    else if (l.environment != invalidid) 
    {
      const environment_data& env     = scene.environments[l.environment];
      if (env.emission_tex != invalidid) 
      {
        const texture_data& texture = scene.textures[env.emission_tex];
        vec3f wl = transform_direction(inverse(env.frame), direction);
        vec2f texcoord = {atan2(wl.z, wl.x) / (2 * pif), acos(clamp(wl.y, -1.f, 1.f)) / pif};
        if (texcoord.x < 0) texcoord.x += 1;

        int i = clamp(
            static_cast<int>(texcoord.x * texture.width), 0, texture.width - 1);
        int j = clamp(static_cast<int>(texcoord.y * texture.height), 0,
            texture.height - 1);

        float prob = sample_discrete_pdf(l.elements_cdf, j * texture.width + i) / l.elements_cdf.back();
        float angle = (2 * pif / texture.width) *
                      (pif / texture.height) *
                      sin(pif * (j + 0.5f) / texture.height);
        pdf += prob / angle;
      }
    } 
  }
  pdf *= sample_uniform_pdf(static_cast<int>(lights.lights.size()));
  return pdf;
}

// Hair bsdf

// Constants
static const float sqrt_pi_over_8 = 0.626657069f;


struct hair_bsdf
{
  float h = 0;
  float eta = 1.55f;        // Index of refraction inside hair
  vec3f sigma_a = zero3f;   // absorption coefficient of the hair interior
  float beta_m = 0.3f;      // Longitudinal roughness
  float beta_n = 0.3f;      // Azimuthal roughness
  float alpha = 2;          // Scale angle
  int pmax = 3;
};

static float i0(float x) 
{
  float val = 0;
  float x2i = 1;
  long ifact = 1;
  int i4 = 1;
  // I0(x) \approx Sum_i x^(2i) / (4^i (i!)^2)
  for (int i = 0; i < 10; ++i) 
  {
    if (i > 1) ifact *= i;
    val += x2i / (i4 * (ifact * ifact));
    x2i *= x * x;
    i4 *= 4;
  }
  return val;
}

static inline float logi0(float x) 
{
  return x > 12 ? 
    x + 0.5 * (-log(2.f * pif) + log(1.f / x) + 1 / (8 * x)) :
    log(i0(x));
}

static inline float eval_phi(int p, float gamma_o, float gamma_t) { return 2 * p * gamma_t - 2 * gamma_o + p * pif; }

static inline float logistic(float x, float s)
{
  x = abs(x);
  return exp(-x / s) / (s * sqr(1 + exp(-x / s)));
}

static inline float logistic_cdf(float x, float s) { return 1 / (1 + exp(-x / s));}

static inline float logistic(float x, float s, float a, float b)
{
  return logistic(x, s) / (logistic_cdf(b, s) - logistic_cdf(a, s));
}

static inline float np(float phi, int p, float s, float gamma_o, float gamma_t)
{
  float dphi = phi - eval_phi(p, gamma_o, gamma_t);
  // Remap dphi into [-pi, pi]
  while(dphi > pif)   dphi -= 2 * pif;
  while(dphi < -pif)  dphi += 2 * pif;

  return logistic(dphi, s, -pif, pif);
}

// pbrt refers to this function as "Mp"
static inline float longitudinal_scattering(float costheta_i, float costheta_o, float sintheta_i, float sintheta_o, float rv)
{
  float a  = costheta_i * costheta_o / rv;
  float b = sintheta_i * sintheta_o / rv;
  // To limit floating point errors
  return rv <= 0.1f ? 
    (exp(logi0(a) - b - 1 / rv + 0.6931f + log(1.f / (2 * rv)))) :
    (exp(-b) * i0(a)) / (std::sinh(1 / rv) * 2 * rv);
}

static std::vector<vec3f> ap(float costheta_o, float eta, float h, const vec3f& t, int pmax)
{
  std::vector<vec3f> ap(pmax + 1);

  // Compute p = 0 attenuation at initial cylinder intersection
  float cosgamma_o = sqrt(1 - sqr(h));
  float costheta = costheta_o * cosgamma_o;
  float f = fresnel_dielectric(eta, costheta);
  ap.push_back(vec3f{f});

  // p = 1
  ap.push_back(sqr(1.f - f) * t);

  // p >= 2
  for(int p = 2; p < pmax; ++p)
    ap.push_back(ap[p - 1] * t * f);

  // p = pmax
  ap.push_back(ap[pmax - 1] * f * t / (one3f - t * f));
  return ap;
}

static vec3f eval_hair(const vec3f& outgoing, const vec3f& incoming, hair_bsdf& hair)
{
  // Compute hair coordinate system terms related to wo
  float sintheta_o = outgoing.x;
  float costheta_o = sqrt(1 - sqr(sintheta_o));
  float phi_o = atan2(outgoing.z, outgoing.y);

  // Compute hair coordinate system terms related to wi
  float sintheta_i = incoming.x;
  float costheta_i = sqrt(1 - sqr(sintheta_i));
  float phi_i = std::atan2(incoming.z, incoming.y);

  // Compute costheta_t for refracted ray
  float sintheta_t = sintheta_o / hair.eta;
  float costheta_t = sqrt(1 - sqr(sintheta_t));

  // Compute cosgamma_t for refracted ray
  float etap = sqrt(sqr(hair.eta) - sqr(sintheta_o)) / costheta_o;
  float singamma_t = hair.h / etap;
  float cosgamma_t = sqrt(1 - sqr(singamma_t));
  float gamma_t = asin(singamma_t);

  vec3f transmittance = exp(-hair.sigma_a * (2 * cosgamma_t / costheta_t));



  return zero3f;
}

// Recursive path tracing.
static vec4f shade_pathtrace(const scene_data& scene, const bvh_data& bvh,
    const pathtrace_lights& lights, const ray3f& ray_, rng_state& rng,
    const pathtrace_params& params) {
  // Init
  ray3f ray = ray_;
  vec3f l   = zero3f;
  vec3f w   = one3f;

  // For bounce 0 --> max_bounce
  for (unsigned int bounce = 0; bounce < params.bounces; ++bounce) 
  {
    const bvh_intersection& isec = intersect_bvh(bvh, scene, ray);
    if (!isec.hit) 
    {
      l += w * eval_environment(scene, ray.d);
      break;
    }

    const vec3f& outgoing = -ray.d;
    const vec3f& position = eval_shading_position(scene, isec, outgoing);
    const vec3f& normal   = eval_shading_normal(scene, isec, outgoing);
    const material_point& material = eval_material(scene, isec);

    if (rand1f(rng) >= material.opacity) 
    {
      ray = {position + ray.d * 1e-2f, ray.d};
      bounce -= 1;
      continue;
    }

    l += w * eval_emission(material, normal, outgoing);
    vec3f incoming = zero3f;
    if (!is_delta(material)) 
    {
      if (rand1f(rng) < 0.5f) incoming = sample_bsdfcos(material, normal, outgoing, rand1f(rng), rand2f(rng));
      else incoming = sample_lights(scene, lights, position, rand1f(rng), rand1f(rng), rand2f(rng));

      if (incoming == zero3f) break;
      w *= eval_bsdfcos(material, normal, outgoing, incoming) /
           (0.5f * sample_bsdfcos_pdf(material, normal, outgoing, incoming) + 0.5f * sample_lights_pdf(scene, bvh, lights, position, incoming));
    } else 
    {
      incoming = sample_delta(material, normal, outgoing, rand1f(rng));
      if (incoming == zero3f) break;
      w *= eval_delta(material, normal, outgoing, incoming) /
           sample_delta_pdf(material, normal, outgoing, incoming);
    }

    // check weight
    if (w == zero3f || !isfinite(w)) break;

    // Russian roulette
    if (bounce > 3) 
    {
      float rrp = min(1.f, max(w));  // Russian Roulette probability
      if (rand1f(rng) >= rrp) break;
      w *= 1.f / rrp;
    }

    ray = {position, incoming};
  }

  return rgb_to_rgba(l);
}

// Recursive path tracing.
static vec4f shade_naive(const scene_data& scene, const bvh_data& bvh,
    const pathtrace_lights& lights, const ray3f& ray_, rng_state& rng,
    const pathtrace_params& params) {
  // YOUR CODE GOES HERE

  // Init
  ray3f ray = ray_;
  vec3f l   = zero3f;
  vec3f w   = one3f;
 
  // For bounce 0 --> max_bounce
  for (unsigned int bounce = 0; bounce < params.bounces; ++bounce) 
  {
    // Intersect scene
    const bvh_intersection& isec = intersect_bvh(bvh, scene, ray);
    if (!isec.hit) 
    {
      l += w * eval_environment(scene, ray.d);
      break;
    }

    const vec3f& outgoing = -ray.d;
    const vec3f& position = eval_shading_position(scene, isec, outgoing);
    const vec3f& normal   = eval_shading_normal(scene, isec, outgoing);
    const material_point& material = eval_material(scene, isec);
    
    if (rand1f(rng) >= material.opacity) 
    {
      ray = {position + ray.d * 1e-2f, ray.d};
      bounce -= 1;
      continue;
    }

    l += w * eval_emission(material, normal, outgoing);

    vec3f incoming = zero3f;
    if (!is_delta(material)) 
    {
      incoming = sample_bsdfcos(material, normal, outgoing, rand1f(rng), rand2f(rng));
      if (incoming == zero3f) break;
      w *= eval_bsdfcos(material, normal, outgoing, incoming) / sample_bsdfcos_pdf(material, normal, outgoing, incoming);
    } 
    else 
    {
      incoming = sample_delta(material, normal, outgoing, rand1f(rng));
      if (incoming == zero3f) break;
      w *= eval_delta(material, normal, outgoing, incoming) / sample_delta_pdf(material, normal, outgoing, incoming);
    }
    
    // check weight
    if (w == zero3f || !isfinite(w)) break;

    // Russian roulette
    if (bounce > 3) {
      float rrp = min(1.f, max(w));  // Russian Roulette probability
      if (rand1f(rng) >= rrp) break;
      w *= 1.f / rrp;
    }
    
    ray = {position, incoming};
  }

  return rgb_to_rgba(l);
}

// Eyelight for quick previewing.
static vec4f shade_eyelight(const scene_data& scene, const bvh_data& bvh,
    const pathtrace_lights& lights, const ray3f& ray_, rng_state& rng,
    const pathtrace_params& params) {
  // initialize
  auto radiance = vec3f{0, 0, 0};
  auto weight   = vec3f{1, 1, 1};
  auto ray      = ray_;
  auto hit      = false;

  // trace  path
  for (auto bounce = 0; bounce < max(params.bounces, 4); bounce++) {
    // intersect next point
    auto intersection = intersect_bvh(bvh, scene, ray);
    if (!intersection.hit) {
      radiance += weight * eval_environment(scene, ray.d);
      break;
    }

    // prepare shading point
    auto outgoing = -ray.d;
    auto position = eval_shading_position(scene, intersection, outgoing);
    auto normal   = eval_shading_normal(scene, intersection, outgoing);
    auto material = eval_material(scene, intersection);

    // handle opacity
    if (material.opacity < 1 && rand1f(rng) >= material.opacity) {
      ray = {position + ray.d * 1e-2f, ray.d};
      bounce -= 1;
      continue;
    }

    // set hit variables
    if (bounce == 0) hit = true;

    // accumulate emission
    auto incoming = outgoing;
    radiance += weight * eval_emission(material, normal, outgoing);

    // brdf * light
    radiance += weight * pif *
                eval_bsdfcos(material, normal, outgoing, incoming);

    // continue path
    if (!is_delta(material)) break;
    incoming = sample_delta(material, normal, outgoing, rand1f(rng));
    if (incoming == vec3f{0, 0, 0}) break;
    weight *= eval_delta(material, normal, outgoing, incoming) /
              sample_delta_pdf(material, normal, outgoing, incoming);
    if (weight == vec3f{0, 0, 0} || !isfinite(weight)) break;

    // setup next iteration
    ray = {position, incoming};
  }

  return {radiance.x, radiance.y, radiance.z, hit ? 1.0f : 0.0f};
}

// Normal for debugging.
static vec4f shade_normal(const scene_data& scene, const bvh_data& bvh,
    const pathtrace_lights& lights, const ray3f& ray, rng_state& rng,
    const pathtrace_params& params) {
  // intersect next point
  auto intersection = intersect_bvh(bvh, scene, ray);
  if (!intersection.hit) return {0, 0, 0, 0};

  // prepare shading point
  auto outgoing = -ray.d;
  auto normal   = eval_shading_normal(scene, intersection, outgoing);
  return {normal.x, normal.y, normal.z, 1};
}

// Normal for debugging.
static vec4f shade_texcoord(const scene_data& scene, const bvh_data& bvh,
    const pathtrace_lights& lights, const ray3f& ray, rng_state& rng,
    const pathtrace_params& params) {
  // intersect next point
  auto intersection = intersect_bvh(bvh, scene, ray);
  if (!intersection.hit) return {0, 0, 0, 0};

  // prepare shading point
  auto texcoord = eval_texcoord(scene, intersection);
  return {texcoord.x, texcoord.y, 0, 1};
}

// Color for debugging.
static vec4f shade_color(const scene_data& scene, const bvh_data& bvh,
    const pathtrace_lights& lights, const ray3f& ray, rng_state& rng,
    const pathtrace_params& params) {
  // intersect next point
  auto intersection = intersect_bvh(bvh, scene, ray);
  if (!intersection.hit) return {0, 0, 0, 0};

  // prepare shading point
  auto color = eval_material(scene, intersection).color;
  return {color.x, color.y, color.z, 1};
}

// Trace a single ray from the camera using the given algorithm.
using pathtrace_shader_func = vec4f (*)(const scene_data& scene,
    const bvh_scene& bvh, const pathtrace_lights& lights, const ray3f& ray,
    rng_state& rng, const pathtrace_params& params);
static pathtrace_shader_func get_shader(const pathtrace_params& params) {
  switch (params.shader) {
    case pathtrace_shader_type::pathtrace: return shade_pathtrace;
    case pathtrace_shader_type::naive: return shade_naive;
    case pathtrace_shader_type::eyelight: return shade_eyelight;
    case pathtrace_shader_type::normal: return shade_normal;
    case pathtrace_shader_type::texcoord: return shade_texcoord;
    case pathtrace_shader_type::color: return shade_color;
    default: {
      throw std::runtime_error("sampler unknown");
      return nullptr;
    }
  }
}

// Build the bvh acceleration structure.
bvh_scene make_bvh(const scene_data& scene, const pathtrace_params& params) {
  return make_bvh(scene, false, false, params.noparallel);
}

// Init a sequence of random number generators.
pathtrace_state make_state(
    const scene_data& scene, const pathtrace_params& params) {
  auto& camera = scene.cameras[params.camera];
  auto  state  = pathtrace_state{};
  if (camera.aspect >= 1) {
    state.width  = params.resolution;
    state.height = (int)round(params.resolution / camera.aspect);
  } else {
    state.height = params.resolution;
    state.width  = (int)round(params.resolution * camera.aspect);
  }
  state.samples = 0;
  state.image.assign(state.width * state.height, {0, 0, 0, 0});
  state.hits.assign(state.width * state.height, 0);
  state.rngs.assign(state.width * state.height, {});
  auto rng_ = make_rng(1301081);
  for (auto& rng : state.rngs) {
    rng = make_rng(961748941ull, rand1i(rng_, 1 << 31) / 2 + 1);
  }
  return state;
}

// Init trace lights
pathtrace_lights make_lights(
    const scene_data& scene, const pathtrace_params& params) {
  auto lights = pathtrace_lights{};

  for (auto handle = 0; handle < scene.instances.size(); handle++) {
    auto& instance = scene.instances[handle];
    auto& material = scene.materials[instance.material];
    if (material.emission == vec3f{0, 0, 0}) continue;
    auto& shape = scene.shapes[instance.shape];
    if (shape.triangles.empty() && shape.quads.empty()) continue;
    auto& light       = lights.lights.emplace_back();
    light.instance    = handle;
    light.environment = invalidid;
    if (!shape.triangles.empty()) {
      light.elements_cdf = vector<float>(shape.triangles.size());
      for (auto idx = 0; idx < light.elements_cdf.size(); idx++) {
        auto& t                 = shape.triangles[idx];
        light.elements_cdf[idx] = triangle_area(
            shape.positions[t.x], shape.positions[t.y], shape.positions[t.z]);
        if (idx != 0) light.elements_cdf[idx] += light.elements_cdf[idx - 1];
      }
    }
    if (!shape.quads.empty()) {
      light.elements_cdf = vector<float>(shape.quads.size());
      for (auto idx = 0; idx < light.elements_cdf.size(); idx++) {
        auto& t                 = shape.quads[idx];
        light.elements_cdf[idx] = quad_area(shape.positions[t.x],
            shape.positions[t.y], shape.positions[t.z], shape.positions[t.w]);
        if (idx != 0) light.elements_cdf[idx] += light.elements_cdf[idx - 1];
      }
    }
  }
  for (auto handle = 0; handle < scene.environments.size(); handle++) {
    auto& environment = scene.environments[handle];
    if (environment.emission == vec3f{0, 0, 0}) continue;
    auto& light       = lights.lights.emplace_back();
    light.instance    = invalidid;
    light.environment = handle;
    if (environment.emission_tex != invalidid) {
      auto& texture      = scene.textures[environment.emission_tex];
      light.elements_cdf = vector<float>(texture.width * texture.height);
      for (auto idx = 0; idx < light.elements_cdf.size(); idx++) {
        auto ij    = vec2i{idx % texture.width, idx / texture.width};
        auto th    = (ij.y + 0.5f) * pif / texture.height;
        auto value = lookup_texture(texture, ij.x, ij.y);
        light.elements_cdf[idx] = max(value) * sin(th);
        if (idx != 0) light.elements_cdf[idx] += light.elements_cdf[idx - 1];
      }
    }
  }

  // handle progress
  return lights;
}

// Sampling camera by sampling lens
ray3f sample_camera(const camera_data& camera, const vec2i& ij, int img_width, int img_height, const vec2f& puv, const vec2f& luv) 
{
  float u = (ij.x + puv.x) / static_cast<float>(img_width);
  float v = (ij.y + puv.y) / static_cast<float>(img_height);

  return eval_camera(camera, {u, v}, sample_disk(luv));
}

// Progressively compute an image by calling trace_samples multiple times.
void pathtrace_samples(pathtrace_state& state, const scene_data& scene,
    const bvh_scene& bvh, const pathtrace_lights& lights,
    const pathtrace_params& params) {
  if (state.samples >= params.samples) return;
  auto& camera = scene.cameras[params.camera];
  auto  shader = get_shader(params);
  state.samples += 1;
  if (params.samples == 1) {
    for (auto idx = 0; idx < state.width * state.height; idx++) {
      auto i = idx % state.width, j = idx / state.width;
      // auto u = (i + 0.5f) / state.width, v = (j + 0.5f) / state.height;
      // auto ray      = eval_camera(camera, {u, v}, {0, 0});
      auto ray      = sample_camera(camera, {i, j}, state.width, state.height,
                                    vec2f{0.5f, 0.5f}, rand2f(state.rngs[idx]));
      auto radiance = shader(scene, bvh, lights, ray, state.rngs[idx], params);
      if (max(radiance) > 10) radiance = clamp(radiance, 0, 10);
      if (!isfinite(radiance)) radiance = {0, 0, 0};
      state.image[idx] += radiance;
      state.hits[idx] += 1;
    }
  } else if (params.noparallel) {
    for (auto idx = 0; idx < state.width * state.height; idx++) {
      auto i = idx % state.width, j = idx / state.width;
      /*auto u        = (i + rand1f(state.rngs[idx])) / state.width,
           v        = (j + rand1f(state.rngs[idx])) / state.height;
      auto ray      = eval_camera(camera, {u, v}, {0, 0});*/
      auto ray      = sample_camera(camera, {i, j}, state.width, state.height,
                        rand2f(state.rngs[idx]), rand2f(state.rngs[idx]));
      auto radiance = shader(scene, bvh, lights, ray, state.rngs[idx], params);
      if (max(radiance) > 10) radiance = clamp(radiance, 0, 10);
      if (!isfinite(radiance)) radiance = {0, 0, 0};
      state.image[idx] += radiance;
      state.hits[idx] += 1;
    }
  } else {
    parallel_for(state.width * state.height, [&](int idx) {
      auto i = idx % state.width, j = idx / state.width;
     /* auto u        = (i + rand1f(state.rngs[idx])) / state.width,
           v        = (j + rand1f(state.rngs[idx])) / state.height;
      auto ray      = eval_camera(camera, {u, v}, {0, 0});*/
      auto ray      = sample_camera(camera, {i, j}, state.width, state.height,
                                    rand2f(state.rngs[idx]), rand2f(state.rngs[idx]));
      auto radiance = shader(scene, bvh, lights, ray, state.rngs[idx], params);
      if (max(radiance) > 10) radiance = clamp(radiance, 0, 10);
      if (!isfinite(radiance)) radiance = {0, 0, 0};
      state.image[idx] += radiance;
      state.hits[idx] += 1;
    });
  }
}

// Check image type
static void check_image(
    const color_image& image, int width, int height, bool linear) {
  if (image.width != width || image.height != height)
    throw std::invalid_argument{"image should have the same size"};
  if (image.linear != linear)
    throw std::invalid_argument{
        linear ? "expected linear image" : "expected srgb image"};
}

// Get resulting render
color_image get_render(const pathtrace_state& state) {
  auto image = make_image(state.width, state.height, true);
  get_render(image, state);
  return image;
}
void get_render(color_image& image, const pathtrace_state& state) {
  check_image(image, state.width, state.height, true);
  auto scale = 1.0f / (float)state.samples;
  for (auto idx = 0; idx < state.width * state.height; idx++) {
    image.pixels[idx] = state.image[idx] * scale;
  }
}

}  // namespace yocto
