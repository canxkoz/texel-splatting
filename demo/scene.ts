interface Geo {
    positions: number[];
    normals: number[];
    indices: number[];
}

const STONE_SUBDIVS = 8;
const TAPER_STRENGTH = 0.15;
const TAPER_CURVE = 1.5;
const WOBBLE_FREQ = 3.0;
const WOBBLE_AMP = 0.12;
const WOBBLE2_FREQ = 7.0;
const WOBBLE2_AMP = 0.04;

function frac(v: number) {
    return v - Math.floor(v);
}

function noise3(x: number, y: number, z: number): number {
    const ix = Math.floor(x),
        iy = Math.floor(y),
        iz = Math.floor(z);
    const fx = x - ix,
        fy = y - iy,
        fz = z - iz;
    const ux = fx * fx * (3 - 2 * fx);
    const uy = fy * fy * (3 - 2 * fy);
    const uz = fz * fz * (3 - 2 * fz);
    const h = (a: number, b: number, c: number) =>
        frac(Math.sin(a * 127.1 + b * 311.7 + c * 74.7) * 43758.5);
    const mix = (a: number, b: number, t: number) => a + (b - a) * t;
    return mix(
        mix(
            mix(h(ix, iy, iz), h(ix + 1, iy, iz), ux),
            mix(h(ix, iy + 1, iz), h(ix + 1, iy + 1, iz), ux),
            uy,
        ),
        mix(
            mix(h(ix, iy, iz + 1), h(ix + 1, iy, iz + 1), ux),
            mix(h(ix, iy + 1, iz + 1), h(ix + 1, iy + 1, iz + 1), ux),
            uy,
        ),
        uz,
    );
}

function createStone(subdivs: number, seed: number): Geo {
    const sx = seed * 17.3,
        sy = seed * 31.7,
        sz = seed * 53.1;
    const positions: [number, number, number][] = [];
    const indexList: number[] = [];

    const faces: [number[], number[], number[]][] = [
        [
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0],
        ],
        [
            [0, 0, -1],
            [-1, 0, 0],
            [0, 1, 0],
        ],
        [
            [1, 0, 0],
            [0, 0, -1],
            [0, 1, 0],
        ],
        [
            [-1, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
        ],
        [
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0],
        ],
        [
            [0, -1, 0],
            [0, 0, -1],
            [1, 0, 0],
        ],
    ];

    for (const [normal, right, up] of faces) {
        const base = positions.length;
        for (let j = 0; j <= subdivs; j++) {
            for (let i = 0; i <= subdivs; i++) {
                const u = i / subdivs - 0.5;
                const v = j / subdivs - 0.5;
                positions.push([
                    normal[0] * 0.5 + right[0] * u + up[0] * v,
                    normal[1] * 0.5 + right[1] * u + up[1] * v,
                    normal[2] * 0.5 + right[2] * u + up[2] * v,
                ]);
            }
        }
        const stride = subdivs + 1;
        for (let j = 0; j < subdivs; j++) {
            for (let i = 0; i < subdivs; i++) {
                const a = base + j * stride + i;
                const b = a + 1;
                const c = a + stride;
                const d = c + 1;
                indexList.push(a, b, d, a, d, c);
            }
        }
    }

    for (const p of positions) {
        const localY = Math.max(0, Math.min(1, p[1] + 0.5));
        const taper = 1 - TAPER_STRENGTH * localY ** TAPER_CURVE;
        p[0] *= taper;
        p[2] *= taper;

        const len = Math.hypot(p[0], p[1], p[2]);
        if (len > 0) {
            const dx = p[0] / len,
                dy = p[1] / len,
                dz = p[2] / len;
            const w1 = noise3(
                p[0] * WOBBLE_FREQ + sx,
                p[1] * WOBBLE_FREQ + sy,
                p[2] * WOBBLE_FREQ + sz,
            );
            const w2 = noise3(
                p[0] * WOBBLE2_FREQ + sx,
                p[1] * WOBBLE2_FREQ + sy,
                p[2] * WOBBLE2_FREQ + sz,
            );
            const disp = (w1 - 0.5) * WOBBLE_AMP + (w2 - 0.5) * WOBBLE2_AMP;
            p[0] += dx * disp;
            p[1] += dy * disp;
            p[2] += dz * disp;
        }
    }

    const normals = positions.map((): [number, number, number] => [0, 0, 0]);
    for (let i = 0; i < indexList.length; i += 3) {
        const a = indexList[i],
            b = indexList[i + 1],
            c = indexList[i + 2];
        const pa = positions[a],
            pb = positions[b],
            pc = positions[c];
        const e1x = pb[0] - pa[0],
            e1y = pb[1] - pa[1],
            e1z = pb[2] - pa[2];
        const e2x = pc[0] - pa[0],
            e2y = pc[1] - pa[1],
            e2z = pc[2] - pa[2];
        const nx = e1y * e2z - e1z * e2y;
        const ny = e1z * e2x - e1x * e2z;
        const nz = e1x * e2y - e1y * e2x;
        normals[a][0] += nx;
        normals[a][1] += ny;
        normals[a][2] += nz;
        normals[b][0] += nx;
        normals[b][1] += ny;
        normals[b][2] += nz;
        normals[c][0] += nx;
        normals[c][1] += ny;
        normals[c][2] += nz;
    }
    for (const n of normals) {
        const len = Math.hypot(n[0], n[1], n[2]);
        if (len > 0) {
            n[0] /= len;
            n[1] /= len;
            n[2] /= len;
        }
    }

    const posFlat: number[] = [];
    const normFlat: number[] = [];
    for (let i = 0; i < positions.length; i++) {
        posFlat.push(positions[i][0], positions[i][1], positions[i][2]);
        normFlat.push(normals[i][0], normals[i][1], normals[i][2]);
    }

    return { positions: posFlat, normals: normFlat, indices: indexList };
}

const DEG = Math.PI / 180;

function transform(geo: Geo, opts: { position?: number[]; scale?: number[]; rot?: number[] }): Geo {
    const s = opts.scale ?? [1, 1, 1];
    const p = opts.position ?? [0, 0, 0];
    const [rx, ry, rz] = (opts.rot ?? [0, 0, 0]).map((d) => d * DEG);

    const cx = Math.cos(rx),
        sx = Math.sin(rx);
    const cy = Math.cos(ry),
        sy = Math.sin(ry);
    const cz = Math.cos(rz),
        sz = Math.sin(rz);

    const m00 = cy * cz;
    const m01 = -cy * sz;
    const m02 = sy;
    const m10 = cx * sz + sx * sy * cz;
    const m11 = cx * cz - sx * sy * sz;
    const m12 = -sx * cy;
    const m20 = sx * sz - cx * sy * cz;
    const m21 = sx * cz + cx * sy * sz;
    const m22 = cx * cy;

    const positions: number[] = [];
    const normals: number[] = [];

    for (let i = 0; i < geo.positions.length; i += 3) {
        const vx = geo.positions[i] * s[0];
        const vy = geo.positions[i + 1] * s[1];
        const vz = geo.positions[i + 2] * s[2];
        positions.push(
            m00 * vx + m01 * vy + m02 * vz + p[0],
            m10 * vx + m11 * vy + m12 * vz + p[1],
            m20 * vx + m21 * vy + m22 * vz + p[2],
        );

        const nx = geo.normals[i];
        const ny = geo.normals[i + 1];
        const nz = geo.normals[i + 2];
        normals.push(
            m00 * nx + m01 * ny + m02 * nz,
            m10 * nx + m11 * ny + m12 * nz,
            m20 * nx + m21 * ny + m22 * nz,
        );
    }

    return { positions, normals, indices: geo.indices };
}

function merge(...geos: Geo[]): Geo {
    const positions: number[] = [];
    const normals: number[] = [];
    const indices: number[] = [];
    let offset = 0;

    for (const g of geos) {
        positions.push(...g.positions);
        normals.push(...g.normals);
        for (const idx of g.indices) indices.push(idx + offset);
        offset += g.positions.length / 3;
    }

    return { positions, normals, indices };
}

function trilithon(
    leftGeo: Geo,
    rightGeo: Geo,
    lintelGeo: Geo,
    groupPos: number[],
    groupRotY: number,
    colSize: number[],
    colSpacing: number,
    colY: number,
    lintelSize: number[],
    lintelY: number,
): Geo {
    const left = transform(leftGeo, {
        scale: colSize,
        rot: [0, groupRotY, 0],
        position: [
            groupPos[0] + Math.cos(groupRotY * DEG) * -colSpacing + Math.sin(groupRotY * DEG) * 0,
            colY,
            groupPos[2] + -Math.sin(groupRotY * DEG) * -colSpacing + Math.cos(groupRotY * DEG) * 0,
        ],
    });
    const right = transform(rightGeo, {
        scale: colSize,
        rot: [0, groupRotY, 0],
        position: [
            groupPos[0] + Math.cos(groupRotY * DEG) * colSpacing,
            colY,
            groupPos[2] + -Math.sin(groupRotY * DEG) * colSpacing,
        ],
    });
    const lintel = transform(lintelGeo, {
        scale: lintelSize,
        rot: [0, groupRotY, 0],
        position: [groupPos[0], lintelY, groupPos[2]],
    });
    return merge(left, right, lintel);
}

export function createSphere(segments: number): {
    vertices: Float32Array<ArrayBuffer>;
    indices: Uint16Array<ArrayBuffer>;
} {
    const positions: number[] = [];
    const normals: number[] = [];
    const indexList: number[] = [];

    for (let lat = 0; lat <= segments; lat++) {
        const theta = (lat * Math.PI) / segments;
        const sinT = Math.sin(theta);
        const cosT = Math.cos(theta);
        for (let lon = 0; lon <= segments; lon++) {
            const phi = (lon * 2 * Math.PI) / segments;
            const x = sinT * Math.cos(phi);
            const y = cosT;
            const z = sinT * Math.sin(phi);
            positions.push(x, y, z);
            normals.push(x, y, z);
        }
    }

    const stride = segments + 1;
    for (let lat = 0; lat < segments; lat++) {
        for (let lon = 0; lon < segments; lon++) {
            const a = lat * stride + lon;
            const b = a + stride;
            indexList.push(a, b, a + 1, a + 1, b, b + 1);
        }
    }

    const vertCount = positions.length / 3;
    const vertices = new Float32Array(vertCount * 6);
    for (let i = 0; i < vertCount; i++) {
        vertices[i * 6] = positions[i * 3];
        vertices[i * 6 + 1] = positions[i * 3 + 1];
        vertices[i * 6 + 2] = positions[i * 3 + 2];
        vertices[i * 6 + 3] = normals[i * 3];
        vertices[i * 6 + 4] = normals[i * 3 + 1];
        vertices[i * 6 + 5] = normals[i * 3 + 2];
    }

    return { vertices, indices: new Uint16Array(indexList) };
}

export function createStones(): {
    vertices: Float32Array<ArrayBuffer>;
    indices: Uint16Array<ArrayBuffer>;
} {
    const stoneA = createStone(STONE_SUBDIVS, 0);
    const stoneB = createStone(STONE_SUBDIVS, 1);
    const parts: Geo[] = [];

    // Hero arch — pos: 9 0 8, rot: 0 30 0
    parts.push(
        trilithon(
            stoneA,
            stoneB,
            stoneA,
            [9, 0, 8],
            30,
            [1.6, 4.5, 1],
            1.85,
            2.25,
            [5.3, 0.8, 1.1],
            4.9,
        ),
    );

    // Outer sarsen trilithons
    // tri-n — pos: 0 0 -15
    parts.push(
        trilithon(
            stoneB,
            stoneA,
            stoneB,
            [0, 0, -15],
            0,
            [1.8, 5.2, 1.2],
            2,
            2.6,
            [5.8, 0.9, 1.3],
            5.6,
        ),
    );
    // tri-nw — pos: -6.3 0 9, rot: 0 288 0
    parts.push(
        trilithon(
            stoneB,
            stoneA,
            stoneB,
            [-6.3, 0, 9],
            288,
            [1.5, 5.4, 1.2],
            1.8,
            2.7,
            [5.2, 0.9, 1.3],
            5.8,
        ),
    );

    // Inner standing stones
    parts.push(
        transform(stoneA, { scale: [1.2, 3.2, 0.9], position: [7.6, 1.6, -2.5], rot: [0, 55, 0] }),
    );
    parts.push(
        transform(stoneB, { scale: [1.3, 3.5, 0.9], position: [-1, 1.75, 7], rot: [0, 205, 0] }),
    );
    parts.push(
        transform(stoneA, { scale: [1.1, 2.6, 0.85], position: [-4, 1.3, 6.9], rot: [0, 260, 0] }),
    );
    parts.push(
        transform(stoneB, { scale: [1.2, 3.0, 0.9], position: [2.5, 1.5, -7.6], rot: [0, 340, 0] }),
    );

    // Fallen stone
    parts.push(
        transform(stoneA, { scale: [1.4, 3.8, 0.9], position: [11.3, 0.45, 7], rot: [0, 60, 80] }),
    );

    // Heel stone
    parts.push(transform(stoneB, { scale: [2.0, 2.8, 1.4], position: [0, 1.4, -20] }));

    const merged = merge(...parts);
    const vertCount = merged.positions.length / 3;
    const vertices = new Float32Array(vertCount * 6);
    for (let i = 0; i < vertCount; i++) {
        vertices[i * 6] = merged.positions[i * 3];
        vertices[i * 6 + 1] = merged.positions[i * 3 + 1];
        vertices[i * 6 + 2] = merged.positions[i * 3 + 2];
        vertices[i * 6 + 3] = merged.normals[i * 3];
        vertices[i * 6 + 4] = merged.normals[i * 3 + 1];
        vertices[i * 6 + 5] = merged.normals[i * 3 + 2];
    }

    return { vertices, indices: new Uint16Array(merged.indices) };
}
