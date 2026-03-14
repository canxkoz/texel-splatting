const WORKGROUP_SIZE = 256;
const TREE_NODE_SIZE = 32;
const MORTON_QUANTIZATION = 1023;
const MAX_PROPAGATION_ITERS = 64;

const WG_X = 16;
const WG_Y = 16;
const WG_SIZE = WG_X * WG_Y;
const ITEMS_PER_WG = 2 * WG_SIZE;

const initBounds = new Int32Array([
    0x7f7fffff, 0x7f7fffff, 0x7f7fffff, 0, 0x80800000, 0x80800000, 0x80800000, 0,
]);

function dispatchSize(device: GPUDevice, count: number): [number, number] {
    const max = device.limits.maxComputeWorkgroupsPerDimension;
    if (count <= max) return [count, 1];
    const x = Math.ceil(Math.sqrt(count));
    return [x, Math.ceil(count / x)];
}

// --- WGSL shader sources ---

const boundsShader = (count: number) => /* wgsl */ `
struct InstanceAABB {
    minX: f32, minY: f32, minZ: f32, _pad0: u32,
    maxX: f32, maxY: f32, maxZ: f32, _pad1: u32,
}

struct SceneBounds {
    minX: atomic<i32>,
    minY: atomic<i32>,
    minZ: atomic<i32>,
    _pad0: u32,
    maxX: atomic<i32>,
    maxY: atomic<i32>,
    maxZ: atomic<i32>,
    _pad1: u32,
}

const AABB_SENTINEL: f32 = 1e30;

@group(0) @binding(0) var<storage, read> leafAABBs: array<InstanceAABB>;
@group(0) @binding(1) var<storage, read_write> sceneBounds: SceneBounds;

var<workgroup> sharedMin: array<vec3<f32>, ${WORKGROUP_SIZE}>;
var<workgroup> sharedMax: array<vec3<f32>, ${WORKGROUP_SIZE}>;

fn floatToSortableInt(f: f32) -> i32 {
    let bits = bitcast<i32>(f);
    let mask = (bits >> 31) & 0x7FFFFFFF;
    return bits ^ mask;
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let tid = gid.x;
    let localId = lid.x;

    var localMin = vec3<f32>(AABB_SENTINEL, AABB_SENTINEL, AABB_SENTINEL);
    var localMax = vec3<f32>(-AABB_SENTINEL, -AABB_SENTINEL, -AABB_SENTINEL);

    if (tid < ${count}u) {
        let aabb = leafAABBs[tid];
        localMin = vec3<f32>(aabb.minX, aabb.minY, aabb.minZ);
        localMax = vec3<f32>(aabb.maxX, aabb.maxY, aabb.maxZ);
    }

    sharedMin[localId] = localMin;
    sharedMax[localId] = localMax;
    workgroupBarrier();

    for (var stride = ${WORKGROUP_SIZE}u / 2u; stride > 0u; stride >>= 1u) {
        if (localId < stride) {
            sharedMin[localId] = min(sharedMin[localId], sharedMin[localId + stride]);
            sharedMax[localId] = max(sharedMax[localId], sharedMax[localId + stride]);
        }
        workgroupBarrier();
    }

    if (localId == 0u) {
        let wgMin = sharedMin[0];
        let wgMax = sharedMax[0];
        atomicMin(&sceneBounds.minX, floatToSortableInt(wgMin.x));
        atomicMin(&sceneBounds.minY, floatToSortableInt(wgMin.y));
        atomicMin(&sceneBounds.minZ, floatToSortableInt(wgMin.z));
        atomicMax(&sceneBounds.maxX, floatToSortableInt(wgMax.x));
        atomicMax(&sceneBounds.maxY, floatToSortableInt(wgMax.y));
        atomicMax(&sceneBounds.maxZ, floatToSortableInt(wgMax.z));
    }
}
`;

const mortonShader = (count: number) => /* wgsl */ `
struct InstanceAABB {
    minX: f32, minY: f32, minZ: f32, _pad0: u32,
    maxX: f32, maxY: f32, maxZ: f32, _pad1: u32,
}

struct SceneBounds {
    minX: i32, minY: i32, minZ: i32, _pad0: u32,
    maxX: i32, maxY: i32, maxZ: i32, _pad1: u32,
}

@group(0) @binding(0) var<storage, read> leafAABBs: array<InstanceAABB>;
@group(0) @binding(1) var<storage, read> sceneBounds: SceneBounds;
@group(0) @binding(2) var<storage, read_write> mortonCodes: array<u32>;
@group(0) @binding(3) var<storage, read_write> sortedIds: array<u32>;

fn sortableIntToFloat(i: i32) -> f32 {
    let mask = (i >> 31) & 0x7FFFFFFF;
    return bitcast<f32>(i ^ mask);
}

fn expandBits(v: u32) -> u32 {
    var x = v & 0x3ffu;
    x = (x | (x << 16u)) & 0x030000ffu;
    x = (x | (x << 8u)) & 0x0300f00fu;
    x = (x | (x << 4u)) & 0x030c30c3u;
    x = (x | (x << 2u)) & 0x09249249u;
    return x;
}

fn mortonCode(x: u32, y: u32, z: u32) -> u32 {
    return (expandBits(x) << 2u) | (expandBits(y) << 1u) | expandBits(z);
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    if (tid >= ${count}u) {
        return;
    }

    let aabb = leafAABBs[tid];
    let centroid = vec3<f32>(
        (aabb.minX + aabb.maxX) * 0.5,
        (aabb.minY + aabb.maxY) * 0.5,
        (aabb.minZ + aabb.maxZ) * 0.5
    );

    let boundsMin = vec3<f32>(
        sortableIntToFloat(sceneBounds.minX),
        sortableIntToFloat(sceneBounds.minY),
        sortableIntToFloat(sceneBounds.minZ)
    );
    let boundsMax = vec3<f32>(
        sortableIntToFloat(sceneBounds.maxX),
        sortableIntToFloat(sceneBounds.maxY),
        sortableIntToFloat(sceneBounds.maxZ)
    );

    let size = boundsMax - boundsMin;
    let safeSize = max(size, vec3<f32>(1e-6, 1e-6, 1e-6));
    let normalized = (centroid - boundsMin) / safeSize;
    let clamped = clamp(normalized, vec3<f32>(0.0), vec3<f32>(1.0));
    let quantized = vec3<u32>(clamped * ${MORTON_QUANTIZATION}.0);

    mortonCodes[tid] = mortonCode(quantized.x, quantized.y, quantized.z);
    sortedIds[tid] = tid;
}
`;

const treeShader = (count: number, maxTreeDepth: number) => /* wgsl */ `
struct TreeNode {
    minX: f32, minY: f32, minZ: f32, leftChild: u32,
    maxX: f32, maxY: f32, maxZ: f32, rightChild: u32,
}

const LEAF_FLAG: u32 = 0x80000000u;
const AABB_SENTINEL: f32 = 1e30;

@group(0) @binding(0) var<storage, read> mortonCodes: array<u32>;
@group(0) @binding(1) var<storage, read_write> treeNodes: array<TreeNode>;
@group(0) @binding(2) var<storage, read_write> parentIndices: array<u32>;

fn delta(i: i32, j: i32, n: i32) -> i32 {
    if (j < 0 || j >= n) {
        return -1;
    }
    let codeI = mortonCodes[i];
    let codeJ = mortonCodes[j];
    if (codeI == codeJ) {
        return i32(countLeadingZeros(u32(i) ^ u32(j))) + 32;
    }
    return i32(countLeadingZeros(codeI ^ codeJ));
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = ${count}i;
    let i = i32(gid.x);

    if (i >= n - 1) {
        return;
    }

    var first: i32;
    var last: i32;

    if (i == 0) {
        first = 0;
        last = n - 1;
    } else {
        let d = select(-1, 1, delta(i, i + 1, n) > delta(i, i - 1, n));
        let deltaMin = delta(i, i - d, n);

        var lmax = 2;
        for (var iter = 0; iter < ${maxTreeDepth}; iter++) {
            if (delta(i, i + lmax * d, n) <= deltaMin) { break; }
            lmax *= 2;
        }

        var l = 0;
        var t = lmax / 2;
        for (var iter2 = 0; iter2 < ${maxTreeDepth}; iter2++) {
            if (t < 1) { break; }
            if (delta(i, i + (l + t) * d, n) > deltaMin) {
                l += t;
            }
            t /= 2;
        }

        let j = i + l * d;
        first = min(i, j);
        last = max(i, j);
    }

    let deltaNode = delta(first, last, n);

    var split = first;
    var stride = last - first;

    for (var iter3 = 0; iter3 < ${maxTreeDepth}; iter3++) {
        stride = (stride + 1) / 2;
        let middle = split + stride;
        if (middle < last) {
            if (delta(first, middle, n) > deltaNode) {
                split = middle;
            }
        }
        if (stride <= 1) { break; }
    }

    let gamma = split;
    let leftIsLeaf = first == gamma;
    let rightIsLeaf = last == gamma + 1;

    var node: TreeNode;
    node.minX = AABB_SENTINEL;
    node.minY = AABB_SENTINEL;
    node.minZ = AABB_SENTINEL;
    node.maxX = -AABB_SENTINEL;
    node.maxY = -AABB_SENTINEL;
    node.maxZ = -AABB_SENTINEL;

    if (leftIsLeaf) {
        node.leftChild = u32(gamma) | LEAF_FLAG;
        parentIndices[u32(gamma)] = u32(i);
    } else {
        node.leftChild = u32(gamma);
        parentIndices[u32(n) + u32(gamma)] = u32(i);
    }

    if (rightIsLeaf) {
        node.rightChild = u32(gamma + 1) | LEAF_FLAG;
        parentIndices[u32(gamma + 1)] = u32(i);
    } else {
        node.rightChild = u32(gamma + 1);
        parentIndices[u32(n) + u32(gamma + 1)] = u32(i);
    }

    treeNodes[i] = node;
}
`;

const propagateShader = (count: number) => /* wgsl */ `
struct InstanceAABB {
    minX: f32, minY: f32, minZ: f32, _pad0: u32,
    maxX: f32, maxY: f32, maxZ: f32, _pad1: u32,
}

const LEAF_FLAG: u32 = 0x80000000u;

@group(0) @binding(0) var<storage, read> leafAABBs: array<InstanceAABB>;
@group(0) @binding(1) var<storage, read> sortedIds: array<u32>;
@group(0) @binding(2) var<storage, read_write> treeNodesRaw: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> boundsFlags: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read> parentIndices: array<u32>;

fn isLeaf(child: u32) -> bool {
    return (child & LEAF_FLAG) != 0u;
}

fn leafIndex(child: u32) -> u32 {
    return child & ~LEAF_FLAG;
}

fn getLeafBounds(leafIdx: u32) -> array<vec3<f32>, 2> {
    let srcIdx = sortedIds[leafIdx];
    let aabb = leafAABBs[srcIdx];
    return array<vec3<f32>, 2>(
        vec3<f32>(aabb.minX, aabb.minY, aabb.minZ),
        vec3<f32>(aabb.maxX, aabb.maxY, aabb.maxZ)
    );
}

fn getParent(nodeIdx: u32, isLeafNode: bool) -> u32 {
    if (isLeafNode) {
        return parentIndices[nodeIdx];
    } else {
        return parentIndices[${count}u + nodeIdx];
    }
}

fn nodeBase(idx: u32) -> u32 {
    return idx * 8u;
}

fn readChildBounds(childIdx: u32) -> array<vec3<f32>, 2> {
    let base = nodeBase(childIdx);
    let minX = bitcast<f32>(atomicLoad(&treeNodesRaw[base + 0u]));
    let minY = bitcast<f32>(atomicLoad(&treeNodesRaw[base + 1u]));
    let minZ = bitcast<f32>(atomicLoad(&treeNodesRaw[base + 2u]));
    let maxX = bitcast<f32>(atomicLoad(&treeNodesRaw[base + 4u]));
    let maxY = bitcast<f32>(atomicLoad(&treeNodesRaw[base + 5u]));
    let maxZ = bitcast<f32>(atomicLoad(&treeNodesRaw[base + 6u]));
    return array<vec3<f32>, 2>(vec3(minX, minY, minZ), vec3(maxX, maxY, maxZ));
}

fn writeBounds(nodeIdx: u32, minB: vec3<f32>, maxB: vec3<f32>) {
    let base = nodeBase(nodeIdx);
    atomicStore(&treeNodesRaw[base + 0u], bitcast<u32>(minB.x));
    atomicStore(&treeNodesRaw[base + 1u], bitcast<u32>(minB.y));
    atomicStore(&treeNodesRaw[base + 2u], bitcast<u32>(minB.z));
    atomicStore(&treeNodesRaw[base + 4u], bitcast<u32>(maxB.x));
    atomicStore(&treeNodesRaw[base + 5u], bitcast<u32>(maxB.y));
    atomicStore(&treeNodesRaw[base + 6u], bitcast<u32>(maxB.z));
}

fn readLeftChild(nodeIdx: u32) -> u32 {
    return atomicLoad(&treeNodesRaw[nodeBase(nodeIdx) + 3u]);
}

fn readRightChild(nodeIdx: u32) -> u32 {
    return atomicLoad(&treeNodesRaw[nodeBase(nodeIdx) + 7u]);
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = ${count}u;
    let leafIdx = gid.x;

    if (leafIdx >= n) {
        return;
    }

    let bounds = getLeafBounds(leafIdx);
    writeBounds(n - 1u + leafIdx, bounds[0], bounds[1]);

    var current = leafIdx;
    var isLeafNode = true;

    for (var iter = 0u; iter < ${MAX_PROPAGATION_ITERS}u; iter++) {
        let parent = getParent(current, isLeafNode);
        let oldFlag = atomicAdd(&boundsFlags[parent], 1u);

        if (oldFlag == 0u) {
            return;
        }

        let left = readLeftChild(parent);
        let right = readRightChild(parent);

        var leftMin: vec3<f32>;
        var leftMax: vec3<f32>;
        var rightMin: vec3<f32>;
        var rightMax: vec3<f32>;

        if (isLeaf(left)) {
            let leftBounds = getLeafBounds(leafIndex(left));
            leftMin = leftBounds[0];
            leftMax = leftBounds[1];
        } else {
            let leftBounds = readChildBounds(left);
            leftMin = leftBounds[0];
            leftMax = leftBounds[1];
        }

        if (isLeaf(right)) {
            let rightBounds = getLeafBounds(leafIndex(right));
            rightMin = rightBounds[0];
            rightMax = rightBounds[1];
        } else {
            let rightBounds = readChildBounds(right);
            rightMin = rightBounds[0];
            rightMax = rightBounds[1];
        }

        let newMin = min(leftMin, rightMin);
        let newMax = max(leftMax, rightMax);
        writeBounds(parent, newMin, newMax);

        current = parent;
        isLeafNode = false;

        if (parent == 0u) {
            break;
        }
    }
}
`;

// --- Radix sort shaders ---

const histogramShader = /* wgsl */ `
@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> histograms: array<u32>;

override WG_COUNT: u32;
override BIT: u32;
override COUNT: u32;

var<workgroup> bins: array<atomic<u32>, 16>;

@compute @workgroup_size(${WG_X}, ${WG_Y}, 1)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) wdim: vec3<u32>,
    @builtin(local_invocation_index) tid: u32,
) {
    let workgroup = wid.x + wid.y * wdim.x;
    let gid = workgroup * ${WG_SIZE}u + tid;

    if (tid < 16u) {
        atomicStore(&bins[tid], 0u);
    }
    workgroupBarrier();

    if (gid < COUNT && workgroup < WG_COUNT) {
        let digit = (input[gid] >> BIT) & 0xfu;
        atomicAdd(&bins[digit], 1u);
    }
    workgroupBarrier();

    if (tid < 16u) {
        histograms[tid * WG_COUNT + workgroup] = atomicLoad(&bins[tid]);
    }
}
`;

const scatterShader = /* wgsl */ `
@group(0) @binding(0) var<storage, read> inKeys: array<u32>;
@group(0) @binding(1) var<storage, read_write> outKeys: array<u32>;
@group(0) @binding(2) var<storage, read> histograms: array<u32>;
@group(0) @binding(3) var<storage, read> inVals: array<u32>;
@group(0) @binding(4) var<storage, read_write> outVals: array<u32>;

override WG_COUNT: u32;
override BIT: u32;
override COUNT: u32;

var<workgroup> digit_bits: array<atomic<u32>, 128>;

@compute @workgroup_size(${WG_X}, ${WG_Y}, 1)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) wdim: vec3<u32>,
    @builtin(local_invocation_index) tid: u32,
) {
    let workgroup = wid.x + wid.y * wdim.x;
    let gid = workgroup * ${WG_SIZE}u + tid;

    if (tid < 128u) { atomicStore(&digit_bits[tid], 0u); }
    workgroupBarrier();

    var digit = 16u;
    if (gid < COUNT && workgroup < WG_COUNT) {
        digit = (inKeys[gid] >> BIT) & 0xfu;
    }

    if (digit < 16u) {
        atomicOr(&digit_bits[digit * 8u + (tid >> 5u)], 1u << (tid & 31u));
    }
    workgroupBarrier();

    if (digit >= 16u) { return; }

    let word = tid >> 5u;
    var rank = 0u;
    for (var w = 0u; w < word; w++) {
        rank += countOneBits(atomicLoad(&digit_bits[digit * 8u + w]));
    }
    rank += countOneBits(atomicLoad(&digit_bits[digit * 8u + word]) & ((1u << (tid & 31u)) - 1u));

    let dst = histograms[digit * WG_COUNT + workgroup] + rank;
    outKeys[dst] = inKeys[gid];
    outVals[dst] = inVals[gid];
}
`;

const prefixSumShader = /* wgsl */ `
@group(0) @binding(0) var<storage, read_write> data: array<u32>;
@group(0) @binding(1) var<storage, read_write> blockSums: array<u32>;

override COUNT: u32;

var<workgroup> temp: array<u32, ${ITEMS_PER_WG * 2}>;

@compute @workgroup_size(${WG_X}, ${WG_Y}, 1)
fn scan(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) wdim: vec3<u32>,
    @builtin(local_invocation_index) tid: u32,
) {
    let workgroup = wid.x + wid.y * wdim.x;
    let base = workgroup * ${WG_SIZE}u;
    let gid = base + tid;
    let eid = gid * 2;

    temp[tid * 2] = select(data[eid], 0u, eid >= COUNT);
    temp[tid * 2 + 1] = select(data[eid + 1], 0u, eid + 1 >= COUNT);

    var offset = 1u;
    for (var d = ${ITEMS_PER_WG}u >> 1; d > 0; d >>= 1) {
        workgroupBarrier();
        if (tid < d) {
            let ai = offset * (tid * 2 + 1) - 1;
            let bi = offset * (tid * 2 + 2) - 1;
            temp[bi] += temp[ai];
        }
        offset *= 2;
    }

    if (tid == 0) {
        blockSums[workgroup] = temp[${ITEMS_PER_WG}u - 1];
        temp[${ITEMS_PER_WG}u - 1] = 0;
    }

    for (var d = 1u; d < ${ITEMS_PER_WG}u; d *= 2) {
        offset >>= 1;
        workgroupBarrier();
        if (tid < d) {
            let ai = offset * (tid * 2 + 1) - 1;
            let bi = offset * (tid * 2 + 2) - 1;
            let t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    workgroupBarrier();

    if (eid < COUNT) { data[eid] = temp[tid * 2]; }
    if (eid + 1 < COUNT) { data[eid + 1] = temp[tid * 2 + 1]; }
}

@compute @workgroup_size(${WG_X}, ${WG_Y}, 1)
fn addBlocks(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) wdim: vec3<u32>,
    @builtin(local_invocation_index) tid: u32,
) {
    let workgroup = wid.x + wid.y * wdim.x;
    let eid = (workgroup * ${WG_SIZE}u + tid) * 2;

    if (eid >= COUNT) { return; }

    let sum = blockSums[workgroup];
    data[eid] += sum;
    if (eid + 1 < COUNT) { data[eid + 1] += sum; }
}
`;

// --- Prefix sum state ---

interface PrefixPass {
    pipeline: GPUComputePipeline;
    bindGroup: GPUBindGroup;
    dispatch: [number, number];
}

interface PrefixSumState {
    passes: PrefixPass[];
    buffers: GPUBuffer[];
}

async function buildPrefixPasses(
    device: GPUDevice,
    module: GPUShaderModule,
    data: GPUBuffer,
    count: number,
    passes: PrefixPass[],
    buffers: GPUBuffer[],
): Promise<void> {
    const wgCount = Math.ceil(count / ITEMS_PER_WG);
    const dispatch = dispatchSize(device, wgCount);

    const blockSums = device.createBuffer({
        size: Math.max(wgCount * 4, 4),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    buffers.push(blockSums);

    const layout = device.createBindGroupLayout({
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        ],
    });

    const bindGroup = device.createBindGroup({
        layout,
        entries: [
            { binding: 0, resource: { buffer: data } },
            { binding: 1, resource: { buffer: blockSums } },
        ],
    });

    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] });

    passes.push({
        pipeline: await device.createComputePipelineAsync({
            layout: pipelineLayout,
            compute: { module, entryPoint: "scan", constants: { COUNT: count } },
        }),
        bindGroup,
        dispatch,
    });

    if (wgCount > 1) {
        await buildPrefixPasses(device, module, blockSums, wgCount, passes, buffers);

        passes.push({
            pipeline: await device.createComputePipelineAsync({
                layout: pipelineLayout,
                compute: { module, entryPoint: "addBlocks", constants: { COUNT: count } },
            }),
            bindGroup,
            dispatch,
        });
    }
}

// --- Radix sort state ---

interface RadixPass {
    histogram: { pipeline: GPUComputePipeline; bindGroup: GPUBindGroup };
    scatter: { pipeline: GPUComputePipeline; bindGroup: GPUBindGroup };
}

interface RadixSortState {
    passes: RadixPass[];
    prefixSum: PrefixSumState;
    workgroups: [number, number];
    buffers: GPUBuffer[];
}

async function createRadixSort(
    device: GPUDevice,
    keys: GPUBuffer,
    values: GPUBuffer,
    count: number,
): Promise<RadixSortState> {
    const wgCount = Math.ceil(count / WG_SIZE);
    const workgroups = dispatchSize(device, wgCount);

    const tmpKeys = device.createBuffer({
        size: count * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    const tmpVals = device.createBuffer({
        size: count * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    const histograms = device.createBuffer({
        size: 16 * wgCount * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });

    const prefixSumModule = device.createShaderModule({ code: prefixSumShader });
    const prefixPasses: PrefixPass[] = [];
    const prefixBuffers: GPUBuffer[] = [];
    await buildPrefixPasses(device, prefixSumModule, histograms, 16 * wgCount, prefixPasses, prefixBuffers);
    const prefixSum: PrefixSumState = { passes: prefixPasses, buffers: prefixBuffers };

    const histogramModule = device.createShaderModule({ code: histogramShader });
    const scatterModule = device.createShaderModule({ code: scatterShader });

    const histogramLayout = device.createBindGroupLayout({
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        ],
    });

    const scatterLayout = device.createBindGroupLayout({
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
            { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
            { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        ],
    });

    const pipelinePromises: Promise<{
        histogram: GPUComputePipeline;
        scatter: GPUComputePipeline;
    }>[] = [];

    for (let bit = 0; bit < 32; bit += 4) {
        pipelinePromises.push(
            (async () => {
                const [histogramPipeline, scatterPipeline] = await Promise.all([
                    device.createComputePipelineAsync({
                        layout: device.createPipelineLayout({ bindGroupLayouts: [histogramLayout] }),
                        compute: {
                            module: histogramModule,
                            entryPoint: "main",
                            constants: { WG_COUNT: wgCount, BIT: bit, COUNT: count },
                        },
                    }),
                    device.createComputePipelineAsync({
                        layout: device.createPipelineLayout({ bindGroupLayouts: [scatterLayout] }),
                        compute: {
                            module: scatterModule,
                            entryPoint: "main",
                            constants: { WG_COUNT: wgCount, BIT: bit, COUNT: count },
                        },
                    }),
                ]);
                return { histogram: histogramPipeline, scatter: scatterPipeline };
            })(),
        );
    }

    const pipelines = await Promise.all(pipelinePromises);
    const passes: RadixPass[] = [];

    for (let i = 0; i < 8; i++) {
        const even = i % 2 === 0;
        const inK = even ? keys : tmpKeys;
        const inV = even ? values : tmpVals;
        const outK = even ? tmpKeys : keys;
        const outV = even ? tmpVals : values;

        passes.push({
            histogram: {
                pipeline: pipelines[i].histogram,
                bindGroup: device.createBindGroup({
                    layout: histogramLayout,
                    entries: [
                        { binding: 0, resource: { buffer: inK } },
                        { binding: 1, resource: { buffer: histograms } },
                    ],
                }),
            },
            scatter: {
                pipeline: pipelines[i].scatter,
                bindGroup: device.createBindGroup({
                    layout: scatterLayout,
                    entries: [
                        { binding: 0, resource: { buffer: inK } },
                        { binding: 1, resource: { buffer: outK } },
                        { binding: 2, resource: { buffer: histograms } },
                        { binding: 3, resource: { buffer: inV } },
                        { binding: 4, resource: { buffer: outV } },
                    ],
                }),
            },
        });
    }

    return { passes, prefixSum, workgroups, buffers: [tmpKeys, tmpVals, histograms] };
}

// --- LBVH public interface ---

export interface LBVH {
    treeNodes: GPUBuffer;
    sortedIds: GPUBuffer;
    count: number;
    destroy: () => void;

    _mortonCodes: GPUBuffer;
    _sceneBounds: GPUBuffer;
    _parentIndices: GPUBuffer;
    _boundsFlags: GPUBuffer;
    _radixSort: RadixSortState;
    _pipelines: {
        bounds: GPUComputePipeline;
        morton: GPUComputePipeline;
        tree: GPUComputePipeline;
        propagate: GPUComputePipeline;
    };
    _bindGroups: {
        bounds: GPUBindGroup;
        morton: GPUBindGroup;
        tree: GPUBindGroup;
        propagate: GPUBindGroup;
    };
}

export async function createLBVH(
    device: GPUDevice,
    leafAABBs: GPUBuffer,
    count: number,
): Promise<LBVH> {
    const maxTreeDepth = Math.ceil(Math.log2(count)) + 1;

    const treeNodes = device.createBuffer({
        label: "lbvh-treeNodes",
        size: 2 * count * TREE_NODE_SIZE,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    const mortonCodes = device.createBuffer({
        label: "lbvh-mortonCodes",
        size: count * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    const sortedIds = device.createBuffer({
        label: "lbvh-sortedIds",
        size: count * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    const sceneBounds = device.createBuffer({
        label: "lbvh-sceneBounds",
        size: 32,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    const parentIndices = device.createBuffer({
        label: "lbvh-parentIndices",
        size: 2 * count * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    const boundsFlags = device.createBuffer({
        label: "lbvh-boundsFlags",
        size: count * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });

    const [boundsModule, mortonModule, treeModule, propagateModule] = await Promise.all([
        device.createShaderModule({ code: boundsShader(count) }),
        device.createShaderModule({ code: mortonShader(count) }),
        device.createShaderModule({ code: treeShader(count, maxTreeDepth) }),
        device.createShaderModule({ code: propagateShader(count) }),
    ]);

    const [boundsPl, mortonPl, treePl, propagatePl, radixSort] = await Promise.all([
        device.createComputePipelineAsync({
            layout: "auto",
            compute: { module: boundsModule, entryPoint: "main" },
        }),
        device.createComputePipelineAsync({
            layout: "auto",
            compute: { module: mortonModule, entryPoint: "main" },
        }),
        device.createComputePipelineAsync({
            layout: "auto",
            compute: { module: treeModule, entryPoint: "main" },
        }),
        device.createComputePipelineAsync({
            layout: "auto",
            compute: { module: propagateModule, entryPoint: "main" },
        }),
        createRadixSort(device, mortonCodes, sortedIds, count),
    ]);

    const _pipelines = { bounds: boundsPl, morton: mortonPl, tree: treePl, propagate: propagatePl };

    const _bindGroups = {
        bounds: device.createBindGroup({
            layout: _pipelines.bounds.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: leafAABBs } },
                { binding: 1, resource: { buffer: sceneBounds } },
            ],
        }),
        morton: device.createBindGroup({
            layout: _pipelines.morton.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: leafAABBs } },
                { binding: 1, resource: { buffer: sceneBounds } },
                { binding: 2, resource: { buffer: mortonCodes } },
                { binding: 3, resource: { buffer: sortedIds } },
            ],
        }),
        tree: device.createBindGroup({
            layout: _pipelines.tree.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: mortonCodes } },
                { binding: 1, resource: { buffer: treeNodes } },
                { binding: 2, resource: { buffer: parentIndices } },
            ],
        }),
        propagate: device.createBindGroup({
            layout: _pipelines.propagate.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: leafAABBs } },
                { binding: 1, resource: { buffer: sortedIds } },
                { binding: 2, resource: { buffer: treeNodes } },
                { binding: 3, resource: { buffer: boundsFlags } },
                { binding: 4, resource: { buffer: parentIndices } },
            ],
        }),
    };

    const destroy = () => {
        treeNodes.destroy();
        sortedIds.destroy();
        mortonCodes.destroy();
        sceneBounds.destroy();
        parentIndices.destroy();
        boundsFlags.destroy();
        for (const buf of radixSort.buffers) buf.destroy();
        for (const buf of radixSort.prefixSum.buffers) buf.destroy();
    };

    return {
        treeNodes,
        sortedIds,
        count,
        destroy,
        _mortonCodes: mortonCodes,
        _sceneBounds: sceneBounds,
        _parentIndices: parentIndices,
        _boundsFlags: boundsFlags,
        _radixSort: radixSort,
        _pipelines,
        _bindGroups,
    };
}

export function dispatchLBVH(lbvh: LBVH, encoder: GPUCommandEncoder, device: GPUDevice): void {
    const { count, _radixSort, _pipelines, _bindGroups, _sceneBounds, _parentIndices, _boundsFlags } = lbvh;

    device.queue.writeBuffer(_sceneBounds, 0, initBounds);
    encoder.clearBuffer(_parentIndices);
    encoder.clearBuffer(_boundsFlags);

    const boundsWG = Math.ceil(count / WORKGROUP_SIZE);
    const mortonWG = Math.ceil(count / WORKGROUP_SIZE);
    const treeWG = Math.ceil(Math.max(count - 1, 1) / WORKGROUP_SIZE);
    const propagateWG = Math.ceil(count / WORKGROUP_SIZE);

    let pass = encoder.beginComputePass();
    pass.setPipeline(_pipelines.bounds);
    pass.setBindGroup(0, _bindGroups.bounds);
    pass.dispatchWorkgroups(boundsWG);
    pass.end();

    pass = encoder.beginComputePass();
    pass.setPipeline(_pipelines.morton);
    pass.setBindGroup(0, _bindGroups.morton);
    pass.dispatchWorkgroups(mortonWG);
    pass.end();

    pass = encoder.beginComputePass();
    const [sx, sy] = _radixSort.workgroups;
    for (const p of _radixSort.passes) {
        pass.setPipeline(p.histogram.pipeline);
        pass.setBindGroup(0, p.histogram.bindGroup);
        pass.dispatchWorkgroups(sx, sy, 1);

        for (const pfx of _radixSort.prefixSum.passes) {
            pass.setPipeline(pfx.pipeline);
            pass.setBindGroup(0, pfx.bindGroup);
            pass.dispatchWorkgroups(pfx.dispatch[0], pfx.dispatch[1], 1);
        }

        pass.setPipeline(p.scatter.pipeline);
        pass.setBindGroup(0, p.scatter.bindGroup);
        pass.dispatchWorkgroups(sx, sy, 1);
    }
    pass.end();

    pass = encoder.beginComputePass();
    pass.setPipeline(_pipelines.tree);
    pass.setBindGroup(0, _bindGroups.tree);
    pass.dispatchWorkgroups(treeWG);
    pass.end();

    pass = encoder.beginComputePass();
    pass.setPipeline(_pipelines.propagate);
    pass.setBindGroup(0, _bindGroups.propagate);
    pass.dispatchWorkgroups(propagateWG);
    pass.end();
}

export function destroyLBVH(lbvh: LBVH): void {
    lbvh.destroy();
}
