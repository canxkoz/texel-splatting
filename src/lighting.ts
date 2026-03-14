export const SLICE_TARGETS: { dir: [number, number, number]; up: [number, number, number] }[] = [
    { dir: [1, 0, 0], up: [0, -1, 0] },
    { dir: [-1, 0, 0], up: [0, -1, 0] },
    { dir: [0, 1, 0], up: [0, 0, 1] },
    { dir: [0, -1, 0], up: [0, 0, -1] },
    { dir: [0, 0, 1], up: [0, -1, 0] },
    { dir: [0, 0, -1], up: [0, -1, 0] },
];

export function computeFaceMask(fwdX: number, fwdY: number, fwdZ: number, threshold: number): number {
    let mask = 0;
    if (fwdX >= threshold) mask |= 1;
    if (-fwdX >= threshold) mask |= 2;
    if (fwdY >= threshold) mask |= 4;
    if (-fwdY >= threshold) mask |= 8;
    if (fwdZ >= threshold) mask |= 16;
    if (-fwdZ >= threshold) mask |= 32;
    return mask;
}

export function cubePerspective(near: number, far: number): Float32Array {
    const out = new Float32Array(16);
    const f = 1 / Math.tan(Math.PI / 4);
    const nf = 1 / (near - far);
    out[0] = f;
    out[5] = f;
    out[10] = far * nf;
    out[11] = -1;
    out[14] = far * near * nf;
    return out;
}

export const FACE_UNIFORMS_WGSL = /* wgsl */ `
struct FaceUniforms {
    viewProj: mat4x4f,
    origin: vec3f,
    near: f32,
    _pad: vec3f,
    far: f32,
}`;

export type MeshAABB = { min: [number, number, number]; max: [number, number, number] };

export function extractFrustumPlanes(vp: Float32Array): number[][] {
    return [
        [vp[0] + vp[3], vp[4] + vp[7], vp[8] + vp[11], vp[12] + vp[15]],
        [vp[3] - vp[0], vp[7] - vp[4], vp[11] - vp[8], vp[15] - vp[12]],
        [vp[1] + vp[3], vp[5] + vp[7], vp[9] + vp[11], vp[13] + vp[15]],
        [vp[3] - vp[1], vp[7] - vp[5], vp[11] - vp[9], vp[15] - vp[13]],
        [vp[2], vp[6], vp[10], vp[14]],
        [vp[3] - vp[2], vp[7] - vp[6], vp[11] - vp[10], vp[15] - vp[14]],
    ];
}

export function aabbInFrustum(planes: number[][], aabb: MeshAABB): boolean {
    for (const [a, b, c, d] of planes) {
        const px = a >= 0 ? aabb.max[0] : aabb.min[0];
        const py = b >= 0 ? aabb.max[1] : aabb.min[1];
        const pz = c >= 0 ? aabb.max[2] : aabb.min[2];
        if (a * px + b * py + c * pz + d < 0) return false;
    }
    return true;
}

export function computeVertexAABB(vertices: Float32Array, stride: number): MeshAABB {
    let minX = Infinity, minY = Infinity, minZ = Infinity;
    let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
    for (let i = 0; i < vertices.length; i += stride) {
        const x = vertices[i], y = vertices[i + 1], z = vertices[i + 2];
        if (x < minX) minX = x;
        if (y < minY) minY = y;
        if (z < minZ) minZ = z;
        if (x > maxX) maxX = x;
        if (y > maxY) maxY = y;
        if (z > maxZ) maxZ = z;
    }
    return { min: [minX, minY, minZ], max: [maxX, maxY, maxZ] };
}

export interface SceneConfig {
    device: GPUDevice;
    vertexBuffer: GPUBuffer;
    indexBuffer: GPUBuffer;
    indexCount: number;
    stoneVertexBuffer: GPUBuffer;
    stoneIndexBuffer: GPUBuffer;
    stoneIndexCount: number;
    orbVertexBuffer: GPUBuffer;
    orbIndexBuffer: GPUBuffer;
    orbIndexCount: number;
    orbBuffers: GPUBuffer[];
    meshAABBs: { grass: MeshAABB; stones: MeshAABB; orbs: MeshAABB };
    area: number;
    height: number;
    baseR: number;
    baseG: number;
    baseB: number;
    density: number;
    rootL: number;
    tipL: number;
    hueFreq: number;
    hueVar: number;
}

export const UNIFORMS_WGSL = /* wgsl */ `
struct Uniforms {
    viewProj: mat4x4f,
    sunDir: vec3f, shadowFade: f32,
    sunColor: vec3f, _p1: f32,
    ambient: vec3f, _p2: f32,
    cameraPos: vec3f,
}`;

export const POINT_LIGHT_WGSL = /* wgsl */ `
struct PointLight {
    position: vec3f, _p0: f32,
    color: vec3f,
    radius: f32,
}`;

export const LIGHTING_WGSL = /* wgsl */ `
fn computeLighting(worldPos: vec3f, normal: vec3f, albedo: vec3f, sunDir: vec3f, sf: f32, sunColor: vec3f, ambient: vec3f, emissive: f32, pointLightCount: u32) -> vec3f {
    let shadowOrigin = worldPos + normal * 0.02;
    let rawShadow = select(0.0, 1.0, !traceAnyShadow(shadowOrigin, sunDir, 1000.0));
    let sunShadow = mix(1.0, rawShadow, sf);
    let ndotl = max(dot(normal, sunDir), 0.0);

    var pointDiffuse = vec3(0.0);
    for (var i = 0u; i < pointLightCount; i++) {
        let pl = pointLights[i];
        let toLight = pl.position - worldPos;
        let d = length(toLight);
        if (d < pl.radius) {
            let Lp = toLight / d;
            let NdotLp = max(dot(normal, Lp), 0.0);
            let t = 1.0 - d / pl.radius;
            let falloff = t * t;
            let lightShadow = select(0.0, 1.0, !traceAnyShadow(shadowOrigin, Lp, d));
            pointDiffuse += pl.color * NdotLp * falloff * lightShadow;
        }
    }

    return albedo * (sunShadow * ndotl * sunColor + pointDiffuse + ambient) + vec3(emissive);
}`;

export const STONE_COLOR_WGSL = /* wgsl */ `
fn stoneColor(wp: vec3f) -> vec3f {
    let baseColor = vec3f(0.541, 0.537, 0.494);
    let lab = toOKLab(baseColor);

    let _bp = wp * 2.0;
    let _bi = floor(_bp); let _bf = fract(_bp);
    let _bu = _bf * _bf * (3.0 - 2.0 * _bf);
    let _bs = vec3<f32>(127.1, 311.7, 74.7);
    let bias = mix(
        mix(mix(fract(sin(dot(_bi, _bs)) * 43758.5),
                fract(sin(dot(_bi + vec3(1,0,0), _bs)) * 43758.5), _bu.x),
            mix(fract(sin(dot(_bi + vec3(0,1,0), _bs)) * 43758.5),
                fract(sin(dot(_bi + vec3(1,1,0), _bs)) * 43758.5), _bu.x), _bu.y),
        mix(mix(fract(sin(dot(_bi + vec3(0,0,1), _bs)) * 43758.5),
                fract(sin(dot(_bi + vec3(1,0,1), _bs)) * 43758.5), _bu.x),
            mix(fract(sin(dot(_bi + vec3(0,1,1), _bs)) * 43758.5),
                fract(sin(dot(_bi + vec3(1,1,1), _bs)) * 43758.5), _bu.x), _bu.y), _bu.z);

    let thresh = clamp((bias - 0.5) / 0.5 + 0.5, 0.0, 1.0);
    let dither = fract(sin(dot(floor(wp * 20.0), vec3(127.1, 311.7, 74.7))) * 43758.5);
    let L = lab.x + select(-0.03, 0.03, dither > thresh);
    return fromOKLab(vec3(L, lab.y, lab.z));
}`;
