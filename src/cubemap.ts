import { BVH_WGSL } from "./bvh";
import {
    type SceneConfig,
    FACE_UNIFORMS_WGSL,
    LIGHTING_WGSL,
    POINT_LIGHT_WGSL,
    SLICE_TARGETS,
    STONE_COLOR_WGSL,
    computeFaceMask,
    cubePerspective,
    extractFrustumPlanes,
    aabbInFrustum,
} from "./lighting";
import { lookAt, multiply } from "./math";
import { OKLAB_WGSL, PATH_WGSL } from "./oklab";
import {
    SKY_STRUCT_WGSL,
    SKY_SCENE_STRUCT_WGSL,
    NOISE_WGSL,
    STARS_WGSL,
    MOON_WGSL,
    CLOUDS_WGSL,
    SAMPLE_SKY_WGSL,
    HAZE_WGSL,
    COMPUTE_SKY_DIR_WGSL,
} from "./sky";

const PROBE_SIZE = 384;
const NEAR = 0.1;
const FAR = 200;
const EYE_CULL_COS = Math.cos((98 * Math.PI) / 180);

const CUBE_PARAMS_WGSL = /* wgsl */ `
struct CubeParams {
    origin: vec3f,
    near: f32,
    sunDir: vec3f,
    far: f32,
    sunColor: vec3f,
    shadowFade: f32,
    ambient: vec3f,
    faceMask: u32,
    pointLightCount: u32, _cp0: u32, _cp1: u32, _cp2: u32,
}`;

interface CubemapConfig extends SceneConfig {
    nodeBuffer: GPUBuffer;
    triBuffer: GPUBuffer;
    triIdBuffer: GPUBuffer;
    lightBuffer: GPUBuffer;
    skyBuffer: GPUBuffer;
    sceneBuffer: GPUBuffer;
}

export interface CubemapEncoder {
    encode(
        encoder: GPUCommandEncoder,
        params: {
            cameraPos: [number, number, number];
            cameraFwd: [number, number, number];
            sunDir: [number, number, number];
            sunColor: [number, number, number];
            ambient: [number, number, number];
            shadowFade: number;
        },
        colorView: GPUTextureView,
        depthView: GPUTextureView,
    ): void;
    destroy(): void;
}

export function createCubemap(config: CubemapConfig): CubemapEncoder {
    const { device } = config;

    const cubeAlbedo = device.createTexture({
        size: [PROBE_SIZE, PROBE_SIZE, 6],
        format: "rgba8unorm",
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        dimension: "2d",
    });
    const cubeNormal = device.createTexture({
        size: [PROBE_SIZE, PROBE_SIZE, 6],
        format: "rgba8unorm",
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        dimension: "2d",
    });
    const cubeRadial = device.createTexture({
        size: [PROBE_SIZE, PROBE_SIZE, 6],
        format: "r32float",
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        dimension: "2d",
    });
    const cubeDepth = device.createTexture({
        size: [PROBE_SIZE, PROBE_SIZE, 6],
        format: "depth24plus",
        usage: GPUTextureUsage.RENDER_ATTACHMENT,
        dimension: "2d",
    });
    const cubeLit = device.createTexture({
        size: [PROBE_SIZE, PROBE_SIZE, 6],
        format: "rgba8unorm",
        usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
        dimension: "2d",
    });

    const faceViews = {
        albedo: Array.from({ length: 6 }, (_, i) =>
            cubeAlbedo.createView({ dimension: "2d", baseArrayLayer: i, arrayLayerCount: 1 }),
        ),
        normal: Array.from({ length: 6 }, (_, i) =>
            cubeNormal.createView({ dimension: "2d", baseArrayLayer: i, arrayLayerCount: 1 }),
        ),
        radial: Array.from({ length: 6 }, (_, i) =>
            cubeRadial.createView({ dimension: "2d", baseArrayLayer: i, arrayLayerCount: 1 }),
        ),
        depth: Array.from({ length: 6 }, (_, i) =>
            cubeDepth.createView({ dimension: "2d", baseArrayLayer: i, arrayLayerCount: 1 }),
        ),
    };

    const arrayViews = {
        albedo: cubeAlbedo.createView({ dimension: "2d-array" }),
        normal: cubeNormal.createView({ dimension: "2d-array" }),
        radial: cubeRadial.createView({ dimension: "2d-array" }),
        lit: cubeLit.createView({ dimension: "2d-array" }),
    };

    const faceUniformBuffers = Array.from({ length: 6 }, () =>
        device.createBuffer({ size: 96, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }),
    );

    const OCT_ENCODE_WGSL = /* wgsl */ `
fn octEncode(n: vec3<f32>) -> vec2<f32> {
    let p = n.xy / (abs(n.x) + abs(n.y) + abs(n.z));
    let s = select(vec2(-1.0), vec2(1.0), p >= vec2(0.0));
    let q = select((1.0 - abs(p.yx)) * s, p, n.z >= 0.0);
    return q * 0.5 + 0.5;
}`;

    // Stage 1: G-buffer capture
    const grassGbufferShader = device.createShaderModule({
        code: /* wgsl */ `
            ${FACE_UNIFORMS_WGSL}
            @group(0) @binding(0) var<uniform> face: FaceUniforms;

            ${OKLAB_WGSL}
            ${PATH_WGSL}
            ${OCT_ENCODE_WGSL}

            struct VsOut {
                @builtin(position) pos: vec4f,
                @location(0) worldPos: vec3f,
                @location(1) localY: f32,
            }

            @vertex fn vs(@location(0) position: vec3f) -> VsOut {
                let world = vec3(
                    position.x * ${config.area}.0,
                    position.y * ${config.height},
                    position.z * ${config.area}.0,
                );
                var out: VsOut;
                out.pos = face.viewProj * vec4f(world, 1);
                out.worldPos = world;
                out.localY = position.y;
                return out;
            }

            struct GbufferOut {
                @location(0) albedo: vec4f,
                @location(1) normal: vec4f,
                @location(2) radial: f32,
            }

            @fragment fn fs(in: VsOut) -> GbufferOut {
                let t = clamp(in.worldPos.y / ${config.height}, 0.0, 1.0);
                let wp = in.worldPos.xz;
                let h = hash2(floor(wp * ${config.density}.0));
                if (h < t) { discard; }
                if (t > 0.0 && pathGrassDiscard(wp)) { discard; }

                let base = vec3f(${config.baseR}, ${config.baseG}, ${config.baseB});
                let oklab = toOKLab(base);
                let hueVar = hash2(floor(wp * ${config.hueFreq}));
                let l = mix(oklab.x * ${config.rootL}, oklab.x * ${config.tipL}, t) + (hueVar - 0.5) * ${config.hueVar};
                let a = oklab.y + (hueVar - 0.5) * ${config.hueVar};
                let b = mix(oklab.z - 0.01, oklab.z + 0.01, t) + (hueVar - 0.5) * ${config.hueVar};
                var color = fromOKLab(vec3(l, a, b));
                if (t == 0.0) { color = pathGroundColor(wp, color); }

                let dx = in.worldPos.x - face.origin.x;
                let dy = in.worldPos.y - face.origin.y;
                let dz = in.worldPos.z - face.origin.z;
                let chebyshev = max(abs(dx), max(abs(dy), abs(dz)));
                let radial = (chebyshev - face.near) / (face.far - face.near);

                var out: GbufferOut;
                out.albedo = vec4f(color, 1.0);
                out.normal = vec4f(octEncode(vec3f(0.0, 1.0, 0.0)), 0.0, 0.0);
                out.radial = radial;
                return out;
            }
        `,
    });

    const stoneGbufferShader = device.createShaderModule({
        code: /* wgsl */ `
            ${FACE_UNIFORMS_WGSL}
            @group(0) @binding(0) var<uniform> face: FaceUniforms;

            ${OKLAB_WGSL}
            ${STONE_COLOR_WGSL}
            ${OCT_ENCODE_WGSL}

            struct VsOut {
                @builtin(position) pos: vec4f,
                @location(0) normal: vec3f,
                @location(1) worldPos: vec3f,
            }

            @vertex fn vs(@location(0) position: vec3f, @location(1) normal: vec3f) -> VsOut {
                var out: VsOut;
                out.pos = face.viewProj * vec4f(position, 1);
                out.normal = normal;
                out.worldPos = position;
                return out;
            }

            struct GbufferOut {
                @location(0) albedo: vec4f,
                @location(1) normal: vec4f,
                @location(2) radial: f32,
            }

            @fragment fn fs(in: VsOut) -> GbufferOut {
                let color = stoneColor(in.worldPos);
                let n = normalize(in.normal);
                let dx = in.worldPos.x - face.origin.x;
                let dy = in.worldPos.y - face.origin.y;
                let dz = in.worldPos.z - face.origin.z;
                let chebyshev = max(abs(dx), max(abs(dy), abs(dz)));
                let radial = (chebyshev - face.near) / (face.far - face.near);

                var out: GbufferOut;
                out.albedo = vec4f(color, 1.0);
                out.normal = vec4f(octEncode(n), 0.0, 0.0);
                out.radial = radial;
                return out;
            }
        `,
    });

    const gbufferLayout = device.createBindGroupLayout({
        entries: [
            {
                binding: 0,
                visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
                buffer: { type: "uniform" },
            },
        ],
    });
    const gbufferPipelineLayout = device.createPipelineLayout({
        bindGroupLayouts: [gbufferLayout],
    });

    const grassGbufferPipeline = device.createRenderPipeline({
        layout: gbufferPipelineLayout,
        vertex: {
            module: grassGbufferShader,
            buffers: [
                {
                    arrayStride: 12,
                    attributes: [{ shaderLocation: 0, offset: 0, format: "float32x3" }],
                },
            ],
        },
        fragment: {
            module: grassGbufferShader,
            targets: [{ format: "rgba8unorm" }, { format: "rgba8unorm" }, { format: "r32float" }],
        },
        primitive: { topology: "triangle-list" },
        depthStencil: { format: "depth24plus", depthWriteEnabled: true, depthCompare: "less" },
    });

    const stoneGbufferPipeline = device.createRenderPipeline({
        layout: gbufferPipelineLayout,
        vertex: {
            module: stoneGbufferShader,
            buffers: [
                {
                    arrayStride: 24,
                    attributes: [
                        { shaderLocation: 0, offset: 0, format: "float32x3" },
                        { shaderLocation: 1, offset: 12, format: "float32x3" },
                    ],
                },
            ],
        },
        fragment: {
            module: stoneGbufferShader,
            targets: [{ format: "rgba8unorm" }, { format: "rgba8unorm" }, { format: "r32float" }],
        },
        primitive: { topology: "triangle-list", cullMode: "back", frontFace: "cw" },
        depthStencil: { format: "depth24plus", depthWriteEnabled: true, depthCompare: "less" },
    });

    const faceBindGroups = faceUniformBuffers.map((buf) =>
        device.createBindGroup({
            layout: gbufferLayout,
            entries: [{ binding: 0, resource: { buffer: buf } }],
        }),
    );

    // Emissive G-buffer (orb + wisps)
    const orbLayout = device.createBindGroupLayout({
        entries: [
            {
                binding: 0,
                visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
                buffer: { type: "uniform" },
            },
        ],
    });

    const emissiveGbufferShader = device.createShaderModule({
        code: /* wgsl */ `
            ${FACE_UNIFORMS_WGSL}
            @group(0) @binding(0) var<uniform> face: FaceUniforms;

            ${OCT_ENCODE_WGSL}

            struct OrbParams {
                position: vec3f,
                scale: f32,
                color: vec3f,
                _pad: f32,
            }
            @group(1) @binding(0) var<uniform> orb: OrbParams;

            struct VsOut {
                @builtin(position) pos: vec4f,
                @location(0) worldPos: vec3f,
                @location(1) color: vec3f,
            }

            @vertex fn vs(@location(0) position: vec3f) -> VsOut {
                let world = position * orb.scale + orb.position;
                var out: VsOut;
                out.pos = face.viewProj * vec4f(world, 1);
                out.worldPos = world;
                out.color = orb.color;
                return out;
            }

            struct GbufferOut {
                @location(0) albedo: vec4f,
                @location(1) normal: vec4f,
                @location(2) radial: f32,
            }

            @fragment fn fs(in: VsOut) -> GbufferOut {
                let dx = in.worldPos.x - face.origin.x;
                let dy = in.worldPos.y - face.origin.y;
                let dz = in.worldPos.z - face.origin.z;
                let chebyshev = max(abs(dx), max(abs(dy), abs(dz)));
                let radial = (chebyshev - face.near) / (face.far - face.near);

                var out: GbufferOut;
                out.albedo = vec4f(min(in.color, vec3f(1)), 0.0);
                out.normal = vec4f(octEncode(vec3f(0.0, 1.0, 0.0)), 0.0, 1.0);
                out.radial = radial;
                return out;
            }
        `,
    });

    const emissiveGbufferPipeline = device.createRenderPipeline({
        layout: device.createPipelineLayout({
            bindGroupLayouts: [gbufferLayout, orbLayout],
        }),
        vertex: {
            module: emissiveGbufferShader,
            buffers: [
                {
                    arrayStride: 24,
                    attributes: [{ shaderLocation: 0, offset: 0, format: "float32x3" }],
                },
            ],
        },
        fragment: {
            module: emissiveGbufferShader,
            targets: [{ format: "rgba8unorm" }, { format: "rgba8unorm" }, { format: "r32float" }],
        },
        primitive: { topology: "triangle-list" },
        depthStencil: { format: "depth24plus", depthWriteEnabled: true, depthCompare: "less" },
    });

    const orbBindGroups = config.orbBuffers.map((buf) =>
        device.createBindGroup({
            layout: orbLayout,
            entries: [{ binding: 0, resource: { buffer: buf } }],
        }),
    );

    // Stage 2: Lighting compute
    const cubeParamsBuffer = device.createBuffer({
        size: 96,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const lightingShader = device.createShaderModule({
        code: /* wgsl */ `
            @group(0) @binding(0) var cubeAlbedoTex: texture_2d_array<f32>;
            @group(0) @binding(1) var cubeNormalTex: texture_2d_array<f32>;
            @group(0) @binding(2) var cubeRadialTex: texture_2d_array<f32>;
            @group(0) @binding(3) var cubeLitTex: texture_storage_2d_array<rgba8unorm, write>;

            ${CUBE_PARAMS_WGSL}
            @group(0) @binding(4) var<uniform> cubeParams: CubeParams;

            ${POINT_LIGHT_WGSL}
            @group(0) @binding(5) var<uniform> pointLights: array<PointLight, 64>;

            @group(0) @binding(6) var<storage, read> bvhNodes: array<BVHNode>;
            @group(0) @binding(7) var<storage, read> bvhTris: array<BVHTri>;
            @group(0) @binding(8) var<storage, read> bvhTriIds: array<u32>;

            ${SKY_STRUCT_WGSL}
            @group(0) @binding(9) var<uniform> sky: Sky;

            ${SKY_SCENE_STRUCT_WGSL}
            @group(0) @binding(10) var<uniform> scene: SkyScene;

            ${BVH_WGSL}
            ${OKLAB_WGSL}
            ${NOISE_WGSL}
            ${STARS_WGSL}
            ${MOON_WGSL}
            ${CLOUDS_WGSL}
            ${SAMPLE_SKY_WGSL}
            ${LIGHTING_WGSL}

            fn decodeNormal(e: vec2<f32>) -> vec3<f32> {
                let f = e * 2.0 - 1.0;
                var n = vec3(f, 1.0 - abs(f.x) - abs(f.y));
                if (n.z < 0.0) {
                    let s = select(vec2(-1.0), vec2(1.0), n.xy >= vec2(0.0));
                    n = vec3((1.0 - abs(n.yx)) * s, n.z);
                }
                return normalize(n);
            }

            fn faceUVtoDir(face: u32, u: f32, v: f32) -> vec3f {
                let uv = vec2f(u * 2.0 - 1.0, v * 2.0 - 1.0);
                switch (face) {
                    case 0u: { return normalize(vec3f( 1.0, -uv.y, -uv.x)); }
                    case 1u: { return normalize(vec3f(-1.0, -uv.y,  uv.x)); }
                    case 2u: { return normalize(vec3f( uv.x,  1.0,  uv.y)); }
                    case 3u: { return normalize(vec3f( uv.x, -1.0, -uv.y)); }
                    case 4u: { return normalize(vec3f( uv.x, -uv.y,  1.0)); }
                    default: { return normalize(vec3f(-uv.x, -uv.y, -1.0)); }
                }
            }

            @compute @workgroup_size(8, 8, 1)
            fn main(@builtin(global_invocation_id) gid: vec3u) {
                let size = ${PROBE_SIZE}u;
                if (gid.x >= size || gid.y >= size || gid.z >= 6u) { return; }

                let face = gid.z;
                if ((cubeParams.faceMask & (1u << face)) == 0u) {
                    let uv_u = (f32(gid.x) + 0.5) / f32(size);
                    let uv_v = (f32(gid.y) + 0.5) / f32(size);
                    let dir = faceUVtoDir(face, uv_u, uv_v);
                    let skyColor = posterize(sampleSky(dir));
                    textureStore(cubeLitTex, vec2u(gid.x, gid.y), face, vec4f(skyColor, 1.0));
                    return;
                }

                let coords = vec2i(vec2u(gid.x, gid.y));
                let radial = textureLoad(cubeRadialTex, coords, face, 0).r;

                let uv_u = (f32(gid.x) + 0.5) / f32(size);
                let uv_v = (f32(gid.y) + 0.5) / f32(size);
                let dir = faceUVtoDir(face, uv_u, uv_v);

                if (radial >= 0.999) {
                    let skyColor = posterize(sampleSky(dir));
                    textureStore(cubeLitTex, vec2u(gid.x, gid.y), face, vec4f(skyColor, 1.0));
                    return;
                }

                let albedoSample = textureLoad(cubeAlbedoTex, coords, face, 0);
                if (albedoSample.a < 0.5) {
                    textureStore(cubeLitTex, vec2u(gid.x, gid.y), face, vec4f(posterize(albedoSample.rgb), 1.0));
                    return;
                }
                let albedo = albedoSample.rgb;
                let normalSample = textureLoad(cubeNormalTex, coords, face, 0);
                let normal = decodeNormal(normalSample.rg);
                let emissive = normalSample.a;

                let chebyshev = radial * (cubeParams.far - cubeParams.near) + cubeParams.near;
                let absDir = abs(dir);
                let maxComp = max(absDir.x, max(absDir.y, absDir.z));
                let worldPos = cubeParams.origin + dir * (chebyshev / maxComp);

                let lit = computeLighting(worldPos, normal, albedo, cubeParams.sunDir, cubeParams.shadowFade, cubeParams.sunColor, cubeParams.ambient, emissive, cubeParams.pointLightCount);
                let posterized = posterize(lit);
                textureStore(cubeLitTex, vec2u(gid.x, gid.y), face, vec4f(posterized, 1.0));
            }
        `,
    });

    const lightingBindGroupLayout = device.createBindGroupLayout({
        entries: [
            {
                binding: 0,
                visibility: GPUShaderStage.COMPUTE,
                texture: { sampleType: "float", viewDimension: "2d-array" },
            },
            {
                binding: 1,
                visibility: GPUShaderStage.COMPUTE,
                texture: { sampleType: "float", viewDimension: "2d-array" },
            },
            {
                binding: 2,
                visibility: GPUShaderStage.COMPUTE,
                texture: { sampleType: "unfilterable-float", viewDimension: "2d-array" },
            },
            {
                binding: 3,
                visibility: GPUShaderStage.COMPUTE,
                storageTexture: {
                    access: "write-only",
                    format: "rgba8unorm",
                    viewDimension: "2d-array",
                },
            },
            { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
            { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
            {
                binding: 6,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: "read-only-storage" },
            },
            {
                binding: 7,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: "read-only-storage" },
            },
            {
                binding: 8,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: "read-only-storage" },
            },
            { binding: 9, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
            { binding: 10, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        ],
    });

    const lightingPipeline = device.createComputePipeline({
        layout: device.createPipelineLayout({ bindGroupLayouts: [lightingBindGroupLayout] }),
        compute: { module: lightingShader },
    });

    const lightingBindGroup = device.createBindGroup({
        layout: lightingBindGroupLayout,
        entries: [
            { binding: 0, resource: arrayViews.albedo },
            { binding: 1, resource: arrayViews.normal },
            { binding: 2, resource: arrayViews.radial },
            { binding: 3, resource: arrayViews.lit },
            { binding: 4, resource: { buffer: cubeParamsBuffer } },
            { binding: 5, resource: { buffer: config.lightBuffer } },
            { binding: 6, resource: { buffer: config.nodeBuffer } },
            { binding: 7, resource: { buffer: config.triBuffer } },
            { binding: 8, resource: { buffer: config.triIdBuffer } },
            { binding: 9, resource: { buffer: config.skyBuffer } },
            { binding: 10, resource: { buffer: config.sceneBuffer } },
        ],
    });

    // Stage 3: Fullscreen display
    const displayShader = device.createShaderModule({
        code: /* wgsl */ `
            @group(0) @binding(0) var cubeLitTex: texture_2d_array<f32>;
            @group(0) @binding(1) var cubeRadialTex: texture_2d_array<f32>;

            ${CUBE_PARAMS_WGSL}
            @group(0) @binding(2) var<uniform> cubeParams: CubeParams;

            ${SKY_SCENE_STRUCT_WGSL}
            @group(0) @binding(3) var<uniform> scene: SkyScene;

            ${SKY_STRUCT_WGSL}
            @group(0) @binding(4) var<uniform> sky: Sky;

            ${COMPUTE_SKY_DIR_WGSL}

            ${HAZE_WGSL}

            fn dirToFaceUV(dir: vec3f) -> vec3f {
                let absDir = abs(dir);
                var face: f32;
                var u: f32;
                var v: f32;
                if (absDir.x >= absDir.y && absDir.x >= absDir.z) {
                    if (dir.x > 0.0) {
                        face = 0.0; u = -dir.z / absDir.x; v = -dir.y / absDir.x;
                    } else {
                        face = 1.0; u = dir.z / absDir.x; v = -dir.y / absDir.x;
                    }
                } else if (absDir.y >= absDir.x && absDir.y >= absDir.z) {
                    if (dir.y > 0.0) {
                        face = 2.0; u = dir.x / absDir.y; v = dir.z / absDir.y;
                    } else {
                        face = 3.0; u = dir.x / absDir.y; v = -dir.z / absDir.y;
                    }
                } else {
                    if (dir.z > 0.0) {
                        face = 4.0; u = dir.x / absDir.z; v = -dir.y / absDir.z;
                    } else {
                        face = 5.0; u = -dir.x / absDir.z; v = -dir.y / absDir.z;
                    }
                }
                return vec3f(face, u * 0.5 + 0.5, v * 0.5 + 0.5);
            }

            struct VsOut {
                @builtin(position) pos: vec4f,
                @location(0) uv: vec2f,
            }

            @vertex fn vs(@builtin(vertex_index) vi: u32) -> VsOut {
                let positions = array(vec2f(-1, -1), vec2f(3, -1), vec2f(-1, 3));
                let p = positions[vi];
                var out: VsOut;
                out.pos = vec4f(p, 0, 1);
                out.uv = p * vec2f(0.5, -0.5) + 0.5;
                return out;
            }

            struct FsOut {
                @location(0) color: vec4f,
                @builtin(frag_depth) depth: f32,
            }

            @fragment fn fs(in: VsOut) -> FsOut {
                let dir = computeSkyDir(in.uv.x, in.uv.y);
                let fuv = dirToFaceUV(dir);
                let face = u32(fuv.x);
                let size = ${PROBE_SIZE}.0;
                let px = clamp(u32(fuv.y * size), 0u, ${PROBE_SIZE - 1}u);
                let py = clamp(u32(fuv.z * size), 0u, ${PROBE_SIZE - 1}u);
                let coords = vec2i(vec2u(px, py));

                let color = textureLoad(cubeLitTex, coords, face, 0).rgb;
                let radial = textureLoad(cubeRadialTex, coords, face, 0).r;

                if (radial < 0.999) {
                    let chebyshev = radial * (cubeParams.far - cubeParams.near) + cubeParams.near;
                    let absDir = abs(dir);
                    let maxComp = max(absDir.x, max(absDir.y, absDir.z));
                    let dist = chebyshev / maxComp;
                    return FsOut(vec4f(applyHaze(color, dist), 1.0), 0.5);
                }
                return FsOut(vec4f(color, 1.0), 1.0);
            }
        `,
    });

    const displayBindGroupLayout = device.createBindGroupLayout({
        entries: [
            {
                binding: 0,
                visibility: GPUShaderStage.FRAGMENT,
                texture: { sampleType: "float", viewDimension: "2d-array" },
            },
            {
                binding: 1,
                visibility: GPUShaderStage.FRAGMENT,
                texture: { sampleType: "unfilterable-float", viewDimension: "2d-array" },
            },
            { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
            { binding: 3, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
            { binding: 4, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
        ],
    });

    const displayPipeline = device.createRenderPipeline({
        layout: device.createPipelineLayout({ bindGroupLayouts: [displayBindGroupLayout] }),
        vertex: { module: displayShader },
        fragment: {
            module: displayShader,
            targets: [{ format: "rgba8unorm" }],
        },
        primitive: { topology: "triangle-list" },
        depthStencil: { format: "depth32float", depthWriteEnabled: true, depthCompare: "always" },
    });

    const displayBindGroup = device.createBindGroup({
        layout: displayBindGroupLayout,
        entries: [
            { binding: 0, resource: arrayViews.lit },
            { binding: 1, resource: arrayViews.radial },
            { binding: 2, resource: { buffer: cubeParamsBuffer } },
            { binding: 3, resource: { buffer: config.sceneBuffer } },
            { binding: 4, resource: { buffer: config.skyBuffer } },
        ],
    });

    const proj = cubePerspective(NEAR, FAR);
    const faceUniformData = new Float32Array(24);

    return {
        encode(encoder, params, colorView, depthView) {
            const [cx, cy, cz] = params.cameraPos;
            const [fx, fy, fz] = params.cameraFwd;
            const mask = computeFaceMask(fx, fy, fz, EYE_CULL_COS);

            const facePlanes: (number[][] | null)[] = [null, null, null, null, null, null];
            for (let i = 0; i < 6; i++) {
                if (!(mask & (1 << i))) continue;
                const { dir, up } = SLICE_TARGETS[i];
                const view = lookAt(
                    cx,
                    cy,
                    cz,
                    cx + dir[0],
                    cy + dir[1],
                    cz + dir[2],
                    up[0],
                    up[1],
                    up[2],
                );
                const viewProj = multiply(proj, view);
                facePlanes[i] = extractFrustumPlanes(viewProj);
                viewProj[1] *= -1;
                viewProj[5] *= -1;
                viewProj[9] *= -1;
                viewProj[13] *= -1;
                faceUniformData.set(viewProj, 0);
                faceUniformData[16] = cx;
                faceUniformData[17] = cy;
                faceUniformData[18] = cz;
                faceUniformData[19] = NEAR;
                faceUniformData[20] = 0;
                faceUniformData[21] = 0;
                faceUniformData[22] = 0;
                faceUniformData[23] = FAR;
                device.queue.writeBuffer(faceUniformBuffers[i], 0, faceUniformData);
            }

            const { grass, stones, orbs } = config.meshAABBs;

            const cubeParamsData = new Float32Array(20);
            cubeParamsData[0] = cx;
            cubeParamsData[1] = cy;
            cubeParamsData[2] = cz;
            cubeParamsData[3] = NEAR;
            cubeParamsData[4] = params.sunDir[0];
            cubeParamsData[5] = params.sunDir[1];
            cubeParamsData[6] = params.sunDir[2];
            cubeParamsData[7] = FAR;
            cubeParamsData[8] = params.sunColor[0];
            cubeParamsData[9] = params.sunColor[1];
            cubeParamsData[10] = params.sunColor[2];
            cubeParamsData[11] = params.shadowFade;
            cubeParamsData[12] = params.ambient[0];
            cubeParamsData[13] = params.ambient[1];
            cubeParamsData[14] = params.ambient[2];
            new Uint32Array(cubeParamsData.buffer)[15] = mask;
            new Uint32Array(cubeParamsData.buffer)[16] = 1;
            device.queue.writeBuffer(cubeParamsBuffer, 0, cubeParamsData);

            // Stage 1: G-buffer capture per face
            for (let i = 0; i < 6; i++) {
                if (!(mask & (1 << i))) continue;
                const planes = facePlanes[i]!;

                const pass = encoder.beginRenderPass({
                    colorAttachments: [
                        {
                            view: faceViews.albedo[i],
                            clearValue: [0, 0, 0, 0],
                            loadOp: "clear",
                            storeOp: "store",
                        },
                        {
                            view: faceViews.normal[i],
                            clearValue: [0.5, 0.5, 0, 0],
                            loadOp: "clear",
                            storeOp: "store",
                        },
                        {
                            view: faceViews.radial[i],
                            clearValue: [1, 0, 0, 0],
                            loadOp: "clear",
                            storeOp: "store",
                        },
                    ],
                    depthStencilAttachment: {
                        view: faceViews.depth[i],
                        depthClearValue: 1,
                        depthLoadOp: "clear",
                        depthStoreOp: "store",
                    },
                });

                if (aabbInFrustum(planes, grass)) {
                    pass.setPipeline(grassGbufferPipeline);
                    pass.setBindGroup(0, faceBindGroups[i]);
                    pass.setVertexBuffer(0, config.vertexBuffer);
                    pass.setIndexBuffer(config.indexBuffer, "uint16");
                    pass.drawIndexed(config.indexCount);
                }

                if (aabbInFrustum(planes, stones)) {
                    pass.setPipeline(stoneGbufferPipeline);
                    pass.setBindGroup(0, faceBindGroups[i]);
                    pass.setVertexBuffer(0, config.stoneVertexBuffer);
                    pass.setIndexBuffer(config.stoneIndexBuffer, "uint16");
                    pass.drawIndexed(config.stoneIndexCount);
                }

                if (aabbInFrustum(planes, orbs)) {
                    pass.setPipeline(emissiveGbufferPipeline);
                    pass.setBindGroup(0, faceBindGroups[i]);
                    pass.setVertexBuffer(0, config.orbVertexBuffer);
                    pass.setIndexBuffer(config.orbIndexBuffer, "uint16");
                    for (let j = 0; j < 4; j++) {
                        pass.setBindGroup(1, orbBindGroups[j]);
                        pass.drawIndexed(config.orbIndexCount);
                    }
                }

                pass.end();
            }

            // Stage 2: Lighting compute
            const computePass = encoder.beginComputePass();
            computePass.setPipeline(lightingPipeline);
            computePass.setBindGroup(0, lightingBindGroup);
            const wg = Math.ceil(PROBE_SIZE / 8);
            computePass.dispatchWorkgroups(wg, wg, 6);
            computePass.end();

            // Stage 3: Display
            const displayPass = encoder.beginRenderPass({
                colorAttachments: [
                    {
                        view: colorView,
                        clearValue: [0, 0, 0, 1],
                        loadOp: "clear",
                        storeOp: "store",
                    },
                ],
                depthStencilAttachment: {
                    view: depthView,
                    depthClearValue: 1,
                    depthLoadOp: "clear",
                    depthStoreOp: "store",
                },
            });
            displayPass.setPipeline(displayPipeline);
            displayPass.setBindGroup(0, displayBindGroup);
            displayPass.draw(3);
            displayPass.end();
        },

        destroy() {
            cubeAlbedo.destroy();
            cubeNormal.destroy();
            cubeRadial.destroy();
            cubeDepth.destroy();
            cubeLit.destroy();
            for (const buf of faceUniformBuffers) buf.destroy();
            cubeParamsBuffer.destroy();
        },
    };
}
