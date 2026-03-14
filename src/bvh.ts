interface Vec3 {
    x: number;
    y: number;
    z: number;
}

interface AABB {
    min: Vec3;
    max: Vec3;
}

export interface BVHNode {
    min: Vec3;
    max: Vec3;
    leftChild: number;
    rightChild: number;
}

interface Triangle {
    v0: Vec3;
    e1: Vec3;
    e2: Vec3;
}

const LEAF_FLAG = 0x80000000;
const AABB_SENTINEL = 1e30;
const MORTON_QUANTIZATION = 1023;
const TREE_NODE_STRIDE = 8;

function vec3(x: number, y: number, z: number): Vec3 {
    return { x, y, z };
}

function vec3Min(a: Vec3, b: Vec3): Vec3 {
    return { x: Math.min(a.x, b.x), y: Math.min(a.y, b.y), z: Math.min(a.z, b.z) };
}

function vec3Max(a: Vec3, b: Vec3): Vec3 {
    return { x: Math.max(a.x, b.x), y: Math.max(a.y, b.y), z: Math.max(a.z, b.z) };
}

function vec3Add(a: Vec3, b: Vec3): Vec3 {
    return { x: a.x + b.x, y: a.y + b.y, z: a.z + b.z };
}

function vec3Sub(a: Vec3, b: Vec3): Vec3 {
    return { x: a.x - b.x, y: a.y - b.y, z: a.z - b.z };
}

function extractTriangles(positions: Float32Array, indices: Uint16Array): Triangle[] {
    const triangles: Triangle[] = [];
    const stride = 6;
    for (let i = 0; i < indices.length; i += 3) {
        const i0 = indices[i];
        const i1 = indices[i + 1];
        const i2 = indices[i + 2];
        const v0 = vec3(
            positions[i0 * stride],
            positions[i0 * stride + 1],
            positions[i0 * stride + 2],
        );
        const v1 = vec3(
            positions[i1 * stride],
            positions[i1 * stride + 1],
            positions[i1 * stride + 2],
        );
        const v2 = vec3(
            positions[i2 * stride],
            positions[i2 * stride + 1],
            positions[i2 * stride + 2],
        );
        triangles.push({ v0, e1: vec3Sub(v1, v0), e2: vec3Sub(v2, v0) });
    }
    return triangles;
}

function computeBounds(triangles: Triangle[]): AABB {
    let min = vec3(Infinity, Infinity, Infinity);
    let max = vec3(-Infinity, -Infinity, -Infinity);
    for (const tri of triangles) {
        const v0 = tri.v0;
        const v1 = vec3Add(v0, tri.e1);
        const v2 = vec3Add(v0, tri.e2);
        min = vec3Min(min, vec3Min(v0, vec3Min(v1, v2)));
        max = vec3Max(max, vec3Max(v0, vec3Max(v1, v2)));
    }
    return { min, max };
}

export function uploadTriangles(
    device: GPUDevice,
    positions: Float32Array,
    indices: Uint16Array,
): { triBuffer: GPUBuffer; triAABBBuffer: GPUBuffer; count: number } {
    const tris = extractTriangles(positions, indices);
    const n = tris.length;

    const triData = new Float32Array(n * 12);
    for (let i = 0; i < n; i++) {
        const tri = tris[i];
        const base = i * 12;
        triData[base + 0] = tri.v0.x;
        triData[base + 1] = tri.v0.y;
        triData[base + 2] = tri.v0.z;
        triData[base + 4] = tri.e1.x;
        triData[base + 5] = tri.e1.y;
        triData[base + 6] = tri.e1.z;
        triData[base + 8] = tri.e2.x;
        triData[base + 9] = tri.e2.y;
        triData[base + 10] = tri.e2.z;
    }

    const aabbData = new Float32Array(n * 8);
    for (let i = 0; i < n; i++) {
        const tri = tris[i];
        const v0 = tri.v0;
        const v1 = vec3Add(v0, tri.e1);
        const v2 = vec3Add(v0, tri.e2);
        const mn = vec3Min(vec3Min(v0, v1), v2);
        const mx = vec3Max(vec3Max(v0, v1), v2);
        const base = i * 8;
        aabbData[base + 0] = mn.x;
        aabbData[base + 1] = mn.y;
        aabbData[base + 2] = mn.z;
        aabbData[base + 4] = mx.x;
        aabbData[base + 5] = mx.y;
        aabbData[base + 6] = mx.z;
    }

    const triBuffer = device.createBuffer({
        size: Math.max(triData.byteLength, 48),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(triBuffer, 0, triData);

    const triAABBBuffer = device.createBuffer({
        size: Math.max(aabbData.byteLength, 32),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(triAABBBuffer, 0, aabbData);

    return { triBuffer, triAABBBuffer, count: n };
}

export const BVH_WGSL = /* wgsl */ `
struct BVHNode {
    minX: f32, minY: f32, minZ: f32, leftChild: u32,
    maxX: f32, maxY: f32, maxZ: f32, rightChild: u32,
}

struct BVHTri {
    v0: vec3f, _p0: f32,
    e1: vec3f, _p1: f32,
    e2: vec3f, _p2: f32,
}

const LEAF_FLAG: u32 = 0x80000000u;
const BVH_SENTINEL: f32 = 1e30;
const BVH_EPSILON: f32 = 1e-7;
const BVH_MAX_STACK: u32 = 24u;

fn bvhIsLeaf(child: u32) -> bool {
    return (child & LEAF_FLAG) != 0u;
}

fn bvhLeafIndex(child: u32) -> u32 {
    return child & ~LEAF_FLAG;
}

fn bvhSafeInverse(d: f32) -> f32 {
    return select(1.0 / d, BVH_SENTINEL, abs(d) < 1e-10);
}

fn bvhIntersectAABB(origin: vec3f, invDir: vec3f, nodeMin: vec3f, nodeMax: vec3f) -> f32 {
    let t1 = (nodeMin - origin) * invDir;
    let t2 = (nodeMax - origin) * invDir;
    let tNear = min(t1, t2);
    let tFar = max(t1, t2);
    let tEnter = max(max(tNear.x, tNear.y), tNear.z);
    let tExit = min(min(tFar.x, tFar.y), tFar.z);
    if (tEnter <= tExit && tExit >= 0.0) {
        return max(tEnter, 0.0);
    }
    return BVH_SENTINEL;
}

fn bvhIntersectTriShadow(origin: vec3f, dir: vec3f, tri: BVHTri) -> f32 {
    let h = cross(dir, tri.e2);
    let a = dot(tri.e1, h);
    if (a > -BVH_EPSILON && a < BVH_EPSILON) { return -1.0; }
    let f = 1.0 / a;
    let s = origin - tri.v0;
    let u = f * dot(s, h);
    if (u < 0.0 || u > 1.0) { return -1.0; }
    let q = cross(s, tri.e1);
    let v = f * dot(dir, q);
    if (v < 0.0 || u + v > 1.0) { return -1.0; }
    let t = f * dot(tri.e2, q);
    if (t > BVH_EPSILON) { return t; }
    return -1.0;
}

fn traceAnyShadow(origin: vec3f, dir: vec3f, tMax: f32) -> bool {
    let triCount = arrayLength(&bvhTriIds);
    if (triCount == 0u) { return false; }

    if (triCount == 1u) {
        let triIdx = bvhTriIds[0];
        let tri = bvhTris[triIdx];
        let t = bvhIntersectTriShadow(origin, dir, tri);
        return t > 0.0 && t < tMax;
    }

    let invDir = vec3f(bvhSafeInverse(dir.x), bvhSafeInverse(dir.y), bvhSafeInverse(dir.z));
    var stack: array<u32, BVH_MAX_STACK>;
    var stackPtr = 0u;
    stack[stackPtr] = 0u;
    stackPtr++;

    var iterations = 0u;
    let maxIterations = min(triCount * 3u, 10000u);

    while (stackPtr > 0u && iterations < maxIterations) {
        iterations++;
        stackPtr--;
        let nodeIdx = stack[stackPtr];
        let node = bvhNodes[nodeIdx];
        let left = node.leftChild;
        let right = node.rightChild;

        if (bvhIsLeaf(left)) {
            let triIdx = bvhTriIds[bvhLeafIndex(left)];
            let tri = bvhTris[triIdx];
            let t = bvhIntersectTriShadow(origin, dir, tri);
            if (t > 0.0 && t < tMax) { return true; }
        } else {
            let leftNode = bvhNodes[left];
            let dist = bvhIntersectAABB(origin, invDir,
                vec3f(leftNode.minX, leftNode.minY, leftNode.minZ),
                vec3f(leftNode.maxX, leftNode.maxY, leftNode.maxZ));
            if (dist < tMax && stackPtr < BVH_MAX_STACK) {
                stack[stackPtr] = left;
                stackPtr++;
            }
        }

        if (bvhIsLeaf(right)) {
            let triIdx = bvhTriIds[bvhLeafIndex(right)];
            let tri = bvhTris[triIdx];
            let t = bvhIntersectTriShadow(origin, dir, tri);
            if (t > 0.0 && t < tMax) { return true; }
        } else {
            let rightNode = bvhNodes[right];
            let dist = bvhIntersectAABB(origin, invDir,
                vec3f(rightNode.minX, rightNode.minY, rightNode.minZ),
                vec3f(rightNode.maxX, rightNode.maxY, rightNode.maxZ));
            if (dist < tMax && stackPtr < BVH_MAX_STACK) {
                stack[stackPtr] = right;
                stackPtr++;
            }
        }
    }
    return false;
}
`;
