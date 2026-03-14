const shader = /* wgsl */ `
struct Params {
    sunUV: vec2f,
    intensity: f32,
    samples: f32,
    decay: f32,
    density: f32,
    sunVisibility: f32,
    _pad: f32,
}

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,
}

@group(0) @binding(0) var srcTexture: texture_2d<f32>;
@group(0) @binding(1) var srcSampler: sampler;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var zTexture: texture_depth_2d;

@vertex
fn vertexMain(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
    var positions = array<vec2f, 3>(
        vec2f(-1.0, -1.0),
        vec2f(3.0, -1.0),
        vec2f(-1.0, 3.0)
    );

    let pos = positions[vertexIndex];

    var output: VertexOutput;
    output.position = vec4f(pos, 0.0, 1.0);
    output.uv = (pos + 1.0) * 0.5;
    output.uv.y = 1.0 - output.uv.y;
    return output;
}

@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
    let scene = textureSample(srcTexture, srcSampler, input.uv).rgb;
    let sunUV = params.sunUV;
    let sampleCount = i32(params.samples);
    let delta = (input.uv - sunUV) * params.density / f32(sampleCount);
    let dims = textureDimensions(zTexture);

    var uv = input.uv;
    var weight = 1.0;
    var accum = vec3f(0.0);

    for (var i = 0; i < sampleCount; i++) {
        uv -= delta;
        if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
            break;
        }
        let texCoord = vec2u(u32(uv.x * f32(dims.x)), u32(uv.y * f32(dims.y)));
        let depth = textureLoad(zTexture, texCoord, 0);
        if (depth >= 0.999) {
            let sampleColor = textureSampleLevel(srcTexture, srcSampler, uv, 0.0).rgb;
            accum += sampleColor * weight;
        }
        weight *= params.decay;
    }

    let result = scene + accum * params.intensity * params.sunVisibility / f32(sampleCount);
    return vec4f(result, 1.0);
}
`;

const OUTPUT_FORMAT: GPUTextureFormat = "rgba8unorm";

export interface GodRays {
    encode(
        encoder: GPUCommandEncoder,
        inputView: GPUTextureView,
        depthView: GPUTextureView,
        outputView: GPUTextureView,
        sun: { u: number; v: number; visibility: number },
    ): void;
    resize(width: number, height: number): void;
    destroy(): void;
}

export function createGodRays(device: GPUDevice): GodRays {
    const module = device.createShaderModule({ code: shader });
    const sampler = device.createSampler({ magFilter: "linear", minFilter: "linear" });

    const bindGroupLayout = device.createBindGroupLayout({
        entries: [
            { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
            { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: { type: "filtering" } },
            { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
            { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "depth" } },
        ],
    });

    const pipeline = device.createRenderPipeline({
        layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
        vertex: { module, entryPoint: "vertexMain" },
        fragment: {
            module,
            entryPoint: "fragmentMain",
            targets: [{ format: OUTPUT_FORMAT }],
        },
        primitive: { topology: "triangle-list" },
    });

    const uniformBuffer = device.createBuffer({
        size: 32,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const paramData = new Float32Array(8);

    let cachedInputView: GPUTextureView | null = null;
    let cachedDepthView: GPUTextureView | null = null;
    let bindGroup: GPUBindGroup | null = null;

    return {
        encode(encoder, inputView, depthView, outputView, sun) {
            paramData[0] = sun.u;
            paramData[1] = sun.v;
            paramData[2] = 0.2;
            paramData[3] = 32;
            paramData[4] = 0.97;
            paramData[5] = 1.0;
            paramData[6] = sun.visibility;
            paramData[7] = 0;
            device.queue.writeBuffer(uniformBuffer, 0, paramData);

            if (inputView !== cachedInputView || depthView !== cachedDepthView) {
                bindGroup = device.createBindGroup({
                    layout: bindGroupLayout,
                    entries: [
                        { binding: 0, resource: inputView },
                        { binding: 1, resource: sampler },
                        { binding: 2, resource: { buffer: uniformBuffer } },
                        { binding: 3, resource: depthView },
                    ],
                });
                cachedInputView = inputView;
                cachedDepthView = depthView;
            }

            const pass = encoder.beginRenderPass({
                colorAttachments: [
                    {
                        view: outputView,
                        loadOp: "clear",
                        storeOp: "store",
                        clearValue: { r: 0, g: 0, b: 0, a: 0 },
                    },
                ],
            });
            pass.setPipeline(pipeline);
            pass.setBindGroup(0, bindGroup!);
            pass.draw(3);
            pass.end();
        },

        resize() {},

        destroy() {
            uniformBuffer.destroy();
        },
    };
}
